

# Reverse for row-major to column-major transformation
_shape(sz::NTuple{N, Int64}) where {N} = Lib.make_shape(reverse(collect(sz)))
_shape(::Tuple{}) = Lib.make_shape(Int64[])
param(x::AbstractArray{T,N}) where {T,N} = Node{N}(Lib.op_parameter(_element(T), _shape(size(x))), "Param")
param(x::T) where {T} = Node{0}(Lib.op_parameter(_element(T), _shape(())), "Param")
param(x::Node) = x

Base.:+(a::Node{N}, b::Node{N}) where {N} = Node{N}(Lib.op_add(a.ptr, b.ptr), "Add")

# nGraph defines element-wise multiply as a "Multiply" op. In julia semantics, this is the
# same as a broadcasted elementwise multiply. To make this compatible with julia semantics,
# overload the "broadcasted" function.
broadcasted(::typeof(*), a::Node{N}, b::Node{N}) where {N} = Node{N}(Lib.op_mul(a.ptr, b.ptr), "Multiply")

function broadcasted(::typeof(+), a::Node{M}, b::Node{N}) where {M,N}
    @show size(a)
    @show size(b)
    # Get the common axes for this object
    axes = map(last, Base.Broadcast.combine_axes(a, b))

    # Make broadcasts if needed.
    a = (size(a) == axes) ? a : _broadcast(a, axes)
    b = (size(b) == axes) ? b : _broadcast(b, axes)

    return a + b
end

function _broadcast(a::Node{M}, shape::NTuple{N,Int}, axes = nothing) where {M,N}
    # Construct the final shape from `shapw`
    final_shape = Lib.make_shape(collect(reverse(shape)))

    # Construct the axis set from the dims that need to be broadcast
    sz = size(a)

    # Always broadcast over trailing timensions
    if axes === nothing 
        axis_set = Lib.make_axisset([i-1 for i in 1:(N - M)])
    else
        axis_set = Lib.make_axisset(collect(axes))
    end

    return Node{N}(Lib.op_broadcast(a.ptr, final_shape, axis_set), "Broadcast")
end

# Fully Connected
Base.:*(w::Node{N}, x::Node{M}) where {N,M} = Node{M}(Lib.op_dot(x.ptr, w.ptr, UInt64(1)), "Dot")

# Relu
broadcasted(::typeof(Flux.relu), a::Node{N}) where {N} = Node{N}(Lib.op_relu(a.ptr), "Relu")
broadcasted(::typeof(identity), a::Node) = a

#####
##### Convolution
#####

# NOTE: nGraph's "convolution" is NNlib's crosscorrelation
#

function flip_kernel(x::AbstractArray{T,N}) where {T,N} 
    view(x, size(x, 1):-1:1, size(x, 2):-1:1, ntuple(_->:, N-2)...)
end

# Make the signature match the Flux signature
expand(N, i::Tuple) = collect(i)
expand(N, i::Integer) = collect(ntuple(_ -> i, N))
function Flux.conv(x::Node{N}, w::Node{N}; stride = 1, pad = 0, dilation = 1) where {N}
    # Construct the convolution node.  
    strides = Lib.make_strides(expand(N-2, stride)) 
    padding = Lib.make_coordinatediff(expand(N-2, pad))
    dilations = Lib.make_strides(expand(N-2, dilation))

    node = Lib.op_convolution(x.ptr, w.ptr, strides, dilations, padding, padding)
    return Node{N}(node, "Convolution")
end

#####
##### Reshape
#####


