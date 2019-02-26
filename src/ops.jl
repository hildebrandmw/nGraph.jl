

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
    # Get the common axes for this object
    axes = map(last, Base.Broadcast.combine_axes(a, b))

    # Make broadcasts if needed.
    a = (size(a) == axes) ? a : _broadcast(a, axes)
    b = (size(b) == axes) ? b : _broadcast(b, axes)

    return a + b
end

function _broadcast(a::Node{M}, axes::NTuple{N,Int}) where {M,N}
    # Construct the final shape from `axes`
    final_shape = Lib.make_shape(collect(axes))

    # Construct the axis set from the dims that need to be broadcast
    sz = size(a) 

    # Always broadcast over trailing timensions
    axis_set = Lib.make_axisset([i-1 for i in 1:(N - M)])
    return Node{N}(Lib.op_broadcast(a.ptr, final_shape, axis_set), "Broadcast")
end

# Fully Connected
Base.:*(w::Node{N}, x::Node{N}) where {N} = Node{N}(Lib.op_dot(x.ptr, w.ptr, UInt64(1)), "Dot")

# Relu
function relu end
broadcasted(::typeof(relu), a::Node{N}) where {N} = Node{N}(Lib.op_relu(a.ptr), "Relu")
