# #####
# ##### Helper Functions
# #####
#
# Make the signature match the Flux signature
expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

function expand(a::NodeTyped{T1}, b::NodeTyped{T2}) where {T1,T2}
    # Promote types if needed
    T3 = promote_type(T1, T2)
    a = convert_eltype(T3, a)
    b = convert_eltype(T3, b)

    # Get the common axes for this object
    shape = map(last, Base.Broadcast.combine_axes(a, b))

    # Make broadcasts if needed.
    a = (size(a) == shape) ? a : broadcast(a, shape)
    b = (size(b) == shape) ? b : broadcast(b, shape)

    return a, b
end

# Define _forward to allow dispatching to alternative implementations of common base
# operations
_forward(f) = f
_forward(::typeof(*)) = multiply
_forward(::typeof(NNlib.σ)) = sigmoid
_forward(::typeof(/)) = divide

Base.broadcasted(f, x::NodeTyped, y::NodeTyped) = _forward(f)(expand(x,y)...)
Base.broadcasted(f, x::NodeTyped, y) = _forward(f)(expand(x, NodeTyped(y))...)
Base.broadcasted(f, x, y::NodeTyped) = _forward(f)(expand(NodeTyped(x), y)...)
Base.broadcasted(f, x::NodeTyped) = _forward(f)(x)

# Special case element-wise copy - this gets around an issue in Metalhead's ResNet
# implementation.
Base.broadcasted(::typeof(copy), x::NodeTyped) = x

############################################################################################
# Here, we do the actual definitions of the ops.

#####
##### Add
#####

add(a::T, b::T) where {T <: NodeTyped} = T(@op Add(a, b))
Base.:+(a::NodeTyped, b::NodeTyped) = add(a,b)

#####
##### AvgPool
#####

# function avgpool(x::Node{T,N}, shape::Tuple; pad = 0, stride = shape) where {T,N}
#     # Convert to nGraph types
#     window_shape = Shape(shape)
#     strides = Strides(expand(N-2, stride))
#     padding_below = Shape(expand(N-2, pad))
#     padding_above = Shape(expand(N-2, pad))
#
#     ptr = Lib.op_avgpool(getpointer(x), window_shape, strides, padding_below, padding_above)
#     return Node{T,N}(ptr)
# end
# Flux.meanpool(x::Node, args...; kw...) = avgpool(x, args...; kw...)

#####
##### BatchMatrixMultiply
#####

bmm(A::T, B::T) where {T <: Node} = T(@op BatchDot(a, b, false, false))

#####
##### BatchNorm
#####

function batchnorm_training(BN::Flux.BatchNorm, x::NodeTyped)
    # Extract parameters from BN
    λ = NodeTyped(BN.λ)
    β = NodeTyped(BN.β)
    γ = NodeTyped(BN.γ)
    node = NodeTyped(@op BatchNormTraining(x, γ, β))

    # When running under the compiler, this register will make these outputs implicit outputs
    # of the nGraph network.
    #
    # Without this, I've seen mysterious segaults
    __register(getoutput(node, 2))
    __register(getoutput(node, 3))
    return λ(getoutput(node, 1))
end

#####
##### Broadcast
#####

# TODO: Rework to make fully compliant with Base broadcasting.
_broadcast_trailing(M,N) = [i for i in (M+1):N]
function Base.broadcast(
        x::NodeTyped{T,M},
        shape::NTuple{N,Int};
        axes = _broadcast_trailing(M,N)
   ) where {T,M,N}

    shape = Shape(shape)
    axisset = AxisSet(axes, N)
    return NodeTyped{T,N}(@op Broadcast(x, shape, axisset))
end

#####
##### Concat
#####

function Base.cat(x::NodeTyped{T,N}...; dims::Integer = 1) where {T,N}
    return NodeTyped(@op Concat(NodeVector(x), N - dims))
end

#####
##### Constants
#####

function constant(x::T) where {T}
    v = convert(cxxt"std::vector<$T>", [x])
    NodeTyped{T,0}(@op Constant(Element(T), Shape(), v))
end

function constant(x::AbstractArray{T,N}) where {T,N}
    v = convert(cxxt"std::vector<$T>", reshape(x, :))
    NodeTyped{T,N}(@op Constant(Element(T), Shape(size(x)), v))
end

#####
##### Convert
#####

convert_eltype(::Type{T}, x::NodeTyped{T}) where {T} = x
function convert_eltype(::Type{T}, x::NodeTyped{U,N}) where {T,U,N}
    return NodeTyped{T,N}(@op Convert(x, Element(T)))
end

#####
##### Convolution
#####

# Only support the 4d version for now.
function NNlib.conv(x::NodeTyped{T,4}, w::NodeTyped{T,4}, ddims::NNlib.DenseConvDims) where {T}
    # Check if we have to flip the weights
    # nGraph's "convolution" is Flux's "cross-correlation" - so we have to reverse the usual
    # flipping logic
    if !NNlib.flipkernel(ddims)
        w = reverse_axes(w, (1, 2))
    end

    # Now, translate this to ngraph
    stride = Strides(NNlib.stride(ddims))
    window_dilation = Strides(NNlib.dilation(ddims))

    padding = NNlib.padding(ddims)
    padding_below = CoordinateDiff(padding[range(1; step = 2, stop = length(padding))])
    padding_above = CoordinateDiff(padding[range(2; step = 2, stop = length(padding))])

    return NodeTyped{T,4}(
        @op Convolution(x, w, stride, window_dilation, padding_below, padding_above)
    )
end

#####
##### Divide
#####

divide(a::T, b::T) where {T <: NodeTyped} = T(@op Divide(a, b))

# Only support `/` and `//` for 0-d arrays
# Let the broadcasting logic above do the appropriate forwarding for `./`
Base.:/(a::NodeTyped{T,0}, b::NodeTyped{T,0}) where {T} = divide(a, b)
Base.://(a::NodeTyped{T,0}, b::NodeTyped{T,0}) where {T} = divide(a, b)

#####
##### Dot
#####

# Reverse the order in the call to `Lib.op_dot` to account for row major/col major
# differences
# dot(a::Node{T}, b::Node{T}, n) where {T,N,M} =
#     Node(Lib.op_dot(getpointer(b), getpointer(a), convert(UInt, n)))
#
# Fully Connected
# Base.:*(w::Node, x::Node) = dot(w, x, 1)

# Base.:*(w::Node, x::AbstractArray) = w * Node(x)
# Base.:*(w::AbstractArray, x::Node) = Node(w) * x

# Methods defined to avoid method ambiguity in julia's dispatch
# Base.:*(x::Node{T,2}, y::Node{T,2}) where {T} = dot(x, y, 1)
# Base.:*(x::Node{T,2}, y::Node{T,1}) where {T} = dot(x, y, 1)

# Base.:*(x::AbstractArray{T,2}, y::Node{T,1}) where {T} = Node(x) * y
# Base.:*(x::AbstractArray{T,2}, y::Node{T,2}) where {T} = Node(x) * y
# Base.:*(x::Node{T,1}, y::AbstractArray{T,2}) where {T} = x * Node(y)
# Base.:*(x::Node{T,2}, y::AbstractArray{T,2}) where {T} = x * Node(y)

#####
##### Indexing
#####

# _lb(i) = i
# _lb(::Colon) = 1
#
# _ub(bound, i) = i
# _ub(bound, ::Colon) = bound
# function Base.getindex(n::Node{T,N}, args...) where {T,N}
#     sz = size(n)
#     # Subtract 1 from the lower bound since the lower bounds are inclusive in ngraph.
#     # Leave the upper bounds as is since the upper-bounds are exclusive.
#     lb = Coordinate(map(_lb, args) .- 1)
#     ub = Coordinate(ntuple(i -> _ub(sz[i], args[i]), length(args)))
#
#     return Node(Lib.op_slice(getpointer.((n, lb, ub))... ))
# end

#####
##### GetOutput
#####

getoutput(x::NodeTyped, n) = NodeTyped(@op GetOutputElement(x, convert(UInt, n-1)))

#####
##### Log
#####

Base.log(x::T) where {T <: NodeTyped} = T(@op Log(x))

#####
##### Max
#####

# # The semantics between max and maximum are flipped around beween Julia and nGraph
Base.max(a::T, b::T) where {T <: Node} = T(@op Maximum(a, b))

#####
##### MaxPool
#####

# function Flux.maxpool(x::Node{T,N}, shape::Tuple; pad = 0, stride = shape) where {T,N}
#     # Convert to nGraph types
#     window_shape = Shape(shape)
#     strides = Strides(expand(N-2, stride))
#     padding_below = Shape(expand(N-2, pad))
#     padding_above = Shape(expand(N-2, pad))
#
#     ptr = Lib.op_maxpool(getpointer(x), window_shape, strides, padding_below, padding_above)
#     return Node{T,N}(ptr)
# end

# stride_size(c::NNlib.PoolDims{N,K,S,P,D}) where {N,K,S,P,D} = S
# pad_size(c::NNlib.PoolDims{N,K,S,P,D}) where {N,K,S,P,D} = P
# function NNlib.maxpool(x::Node{T,N}, dims::NNlib.PoolDims) where {T,N}
#     ptr = Lib.op_maxpool(
#         getpointer(x),
#         Shape(NNlib.kernel_size(dims)),             # window_shape
#         Strides(stride_size(dims)),                 # strides (same as window_shape)
#         Shape(pad_size(dims)[1:div(N,2)]),          # padding_below
#         Shape(pad_size(dims)[(div(N,2) + 1):N]),    # padding_above
#     )
#     return Node{T,N}(ptr)
# end

#####
##### Multiply
#####

# multiply(a::Node{T,N}, b::Node{T,N}) where {T,N} = Node{T,N}(Lib.op_mul(getpointer(a), getpointer(b)))

# function Base.:*(a::Node{T,0}, b::U) where {T, U <: Number}
#     R = promote_type(T,U)
#     a = convert_eltype(R, a)
#     b = constant(convert(R, b))
#     return multiply(a, b)
# end
# Base.:*(b::U, a::Node{T,0}) where {U <: Number, T} = *(a, b)

#####
##### Minimum
#####

# The `min` and `minimum` semantics are swapped between Julia and nGraph.
# Base.minimum(a::N, b::N) where {N <: Node} = N(Lib.op_minimum(getpointer(a), getpointer(b)))
# _forward(::typeof(min)) = minimum

#####
##### Negative
#####

# negative(a::Node{T,N}) where {T,N} = Node{T,N}(Lib.op_negative(getpointer(a)))

# Base.:-(a::Node) = negative(a)

#####
##### One Hot
#####

# function onehot(x::Node{T,N}, max_index, onehot_index) where {T,N}
#     # Create the output size from `max_index` and `onehot_index`
#     sz = size(x)
#     output_sz = collect(splicein(sz, max_index, onehot_index))
#
#     return Node{T,N+1}(Lib.op_onehot(
#         getpointer(x .- one(T)),
#         Shape(output_sz),
#         convert(UInt, N + 1 - onehot_index)
#     ))
# end

#####
##### Parameter
#####

# parameter(x::AbstractArray{T,N}) where {T,N} = Node(x)
# parameter(::Type{T}, dims...) where {T} = parameter(T, convert.(Int, dims))
# parameter(::Type{T}, dims::NTuple{N,Int}) where {T,N} = Node{T,N}(Lib.op_parameter(Element(T), Shape(dims)))
# parameter(x::T) where {T} = Node{T,0}(Lib.op_parameter(Element(T), Shape(())))
# parameter(x::Node) = x

#####
##### permutedims
#####

function Base.permutedims(x::T, permutation) where {T <: NodeTyped}
    # Quick debug to make sure we're passing things along correctly.
    @assert ndims(x) == length(permutation)
    av = AxisVector(permutation, length(permutation))
    sz = size(x)
    shape = Shape([sz[i] for i in permutation])
    return T(@op Reshape(x, av, shape))
end

#####
##### Power
#####

power(a::T, b::T) where {T <: NodeTyped} = T(@op Power(a, b))
Base.:^(a::T, b::T) where {T <: NodeTyped} = power(a, b)

#####
##### Relu
#####

Flux.relu(a::T) where {T <: NodeTyped} = T(@op Relu(a))

#####
##### Reshape
#####

# NOTE:We're hijacking an internal Base function here to do all of the `Base.Colon`
# preprocessing for us
function Base._reshape(x::NodeTyped{T,N}, dims::NTuple{M,Int}) where {T,N,M}
    av = AxisVector(1:N, N)
    shape = Shape(dims)
    return NodeTyped{T,M}(@op Reshape(x, av, shape))
end

#####
##### Reverse
#####

function reverse_axes(x::NodeTyped{T,N}, axes = ()) where {T,N}
    axes = AxisSet(axes, N)
    return NodeTyped{T,N}(@op Reverse(x, axes))
end


#####
##### Result
#####

result(x::T) where {T <: NodeTyped} = T(@op Result(x))

#####
##### Sigmoid
#####

Flux.sigmoid(x::T) where {T <: NodeTyped} = T(@op Sigmoid(x))

#####
##### Softmax
#####

# function Flux.softmax(x::Node{T,N}; axes = 1) where {T,N}
#     av = AxisSet(axes, N)
#     node = Lib.op_softmax(getpointer(x), av)
#     return Node{T,N}(node)
# end

#####
##### Sqrt
#####

Base.sqrt(x::T) where {T <: NodeTyped} = T(@op Sqrt(x))

#####
##### Subtract
#####

subtract(a::T, b::T) where {T <: NodeTyped} = T(@op Subtract(a, b))
Base.:-(a::T, b::T) where {T <: NodeTyped} = subtract(a, b)

#####
##### Sum
#####

# Default to reducing along all dimensions
# function Base.sum(x::Node{T,N}; axes = 1:N ) where {T,N}
#     as = AxisSet(axes, N)
#     node = Lib.op_sum(getpointer(x), as)
#     return Node{T, N - length(axes)}(node)
# end

#####
##### Tanh
#####

Base.tanh(x::T) where {T <: Node} = T(@op Tanh(x))

#####
##### Transpose
#####

function Base.transpose(a::NodeTyped{T,N}) where {T,N}
    av = AxisVector(N:-1:1, N)
    shape = Shape(reverse(size(a)))
    return NodeTyped{T,N}(@op Reshape(a, av, shape))
end

# #######################################################################################
# #
# # Custom ops
# move(x::T, output = 1) where {T <: Node} = T(Lib.op_move(getpointer(x), convert(UInt, output-1)))
#
# moveasync(x::T, across::Node) where {T <: Node} = moveasync(x, 1, across)
# moveasync(x::T, output, across::Node) where {T <: Node} =
#     T(Lib.op_moveasync(getpointer(x), convert(UInt, output-1), getpointer(across)))
#
# convert_layout_to(x::Node, y::Node, i) =
#     Node(Lib.op_cpu_convert_layout_to(getpointer(x), getpointer(y), convert(Int, i-1)))
