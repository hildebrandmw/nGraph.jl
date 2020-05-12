#####
##### Helper Functions
#####

# Exected transformation:
#
# @op f(a,b,c) -> Node(@ngraphcall f(a,b,c))
# @op T f(a,b,c) -> Node{T}(@ngraphcall f(a,b,c))
# @op T N f(a,b,c) -> Node(T,N)(@ngraphcall f(a,b,c))
"""
    @op [T] [N] fn(args)

Make an Op call into the nGraph library.
Automatically returns a `Node`, with optional type parameters `T` and `N`.
"""
macro op(x...)
    if length(x) > 3
        error("Expected at most 3 arguments to @op")
    end
    # Get the Optional Type and Dimensionality Parameters
    T = length(x) > 1 ? x[1] : nothing
    N = length(x) > 2 ? x[2] : nothing
    expr = x[end]

    # Make sure this is a function call.
    # wrap the `Lib` module around the op.
    if expr.head != :call
        error("@Op only support function calls.")
    end

    # Escape all but the function name.
    # Since we forward this to the `@ngraphcall` macro, we apparently need to do this.
    for i in 2:length(expr.args)
        expr.args[i] = esc(expr.args[i])
    end

    # Prefix the node constructor.
    params = []
    isnothing(T) || push!(params, T)
    isnothing(N) || push!(params, N)
    prefix = isempty(params) ? :(Node) : :(Node{$(esc.(params)...)})
    return :($prefix(@ngraphcall $expr))
end

#####
##### Shape and element type promotion
#####

promote_eltype(x::Node{T}, y::Node{T}) where {T} = (x,y)
function promote_eltype(x::Node{T1}, y::Node{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    return convert_eltype.(T, (x,y))
end

function autobroadcast(x::Node, y::Node)
    # Get the common axes for this object
    shape = map(last, Base.Broadcast.combine_axes(x, y))

    # Make broadcasts if needed.
    x = (size(x) == shape) ? x : broadcast(x, shape)
    y = (size(y) == shape) ? y : broadcast(y, shape)
    return (x, y)
end

function homogenize(x::Node, y::Node)
    x, y = promote_eltype(x, y)
    return autobroadcast(x, y)
end

#####
##### Broadcasting machinery
#####

# Define _forward to allow dispatching to alternative implementations of common base
# operations.
#
# We basically shortcircuit the lazy base implementations.
_forward(f) = f
_forward(::typeof(*)) = multiply
#_forward(::typeof(NNlib.σ)) = _sigmoid
_forward(::typeof(/)) = divide

Base.broadcasted(f, x::Node, y::Node) = _forward(f)(homogenize(x,y)...)
Base.broadcasted(f, x::Node, y::AbstractArray) = _forward(f)(homogenize(x, Node(y))...)
Base.broadcasted(f, x::AbstractArray, y::Node) = _forward(f)(homogenize(Node(x), y)...)
Base.broadcasted(f, x::Node) = _forward(f)(x)

function Base.broadcasted(f, x::Node{T}, y::Number) where {T}
    return _forward(f)(homogenize(x, Node{T}(convert(T, y)))...)
end
function Base.broadcasted(f, x::Number, y::Node{T}) where {T}
    return _forward(f)(homogenize(Node{T}(convert(T, x)), y)...)
end

Base.convert(::Type{Node{T,0}}, x::S) where {T,S <: Number} = Node{T,0}(convert(T, x))

# Special case element-wise copy - this gets around an issue in Metalhead's ResNet
# implementation.
Base.broadcasted(::typeof(copy), x::Node) = x

#####
##### Parameter
#####

parameter(x::T) where {T} = @op T 0 op_parameter(T, Int64[])
function parameter(::Type{T}, dims::NTuple{N,Int}) where {T,N}
    _shape = shape(dims)
    GC.@preserve _shape begin
        node = @op T N op_parameter(T, _shape)
    end
    return node
end

parameter(x::AbstractArray{T,N}) where {T,N} = parameter(T, size(x))
parameter(::Type{T}, dims...) where {T} = parameter(T, dims)
parameter(x::Node) = x

############################################################################################

#####
##### Add
#####

add(a::U, b::U) where {T,N,U <: Node{T,N}} = @op T N op_add(a, b)
Base.:+(a::Node, b::Node) = add(a,b)

#####
##### AvgPool
#####

function avgpool(x::Node{T,N}, kernel; pad = 0, stride = kernel) where {T,N}
    return @op T N op_avgpool(
        x,
        strides(N-2, stride),
        shape(N-2, pad),
        shape(N-2, pad),
        shape(N-2, kernel),
        false,
    )
end
#Flux.meanpool(x::Node, args...; kw...) = avgpool(x, args...; kw...)

# #####
# ##### BatchNorm
# #####
#
# function batchnorm_training(input::Node, γ::Node, β::Node, ϵ)
#     return Node(Lib.op_batchnorm_training(
#         getpointer(input),
#         getpointer(γ),
#         getpointer(β),
#         convert(Float64, ϵ)
#     ))
# end

#####
##### Broadcast
#####

function Base.broadcast(a::Node{T,M}, like::Node{Int64,1}) where {T,M,N}
    return @op T op_broadcast(a, like)
end

function Base.broadcast(a::Node{T}, dims::NTuple{N,Int64}) where {T,N}
    return Base.broadcast(a, constant(shape(dims)))
end

#####
##### Concat
#####

function concat(nodes::Vector{Node{T,N}}; dims::Integer = 1) where {T,N}
    # Flip dims for column -> row
    return @op T N op_concat(nodes, N - dims)
end
Base.cat(x::Node...; kw...) = concat(collect(x); kw...)

#####
##### Constants
#####

function constant(x::T) where {T}
    A = Array{T,0}(undef)
    A[] = x
    return constant(A)
end

function constant(x::AbstractArray{T,N}) where {T,N}
    # Reinterpret the `x` as a byte array
    _x = collect(reinterpret(UInt8, reshape(x,:)))
    GC.@preserve _x begin
        node = @op T N op_constant(Element(T)[], shape(size(x)), _x)
    end
    return node
end

#####
##### Convert
#####

convert_eltype(::Type{T}, x::Node{T}) where {T} = x
function convert_eltype(::Type{T}, x::Node{U,N}) where {T,U,N}
    @op T N op_convert(x, T)
end

#####
##### Divide
#####

divide(a::Node{T,N}, b::Node{T,N}) where {T,N} = @op T N op_divide(a, b)
Base.:/(a::Node{T,0}, b::Node{T,0}) where {T} = divide(a, b)
Base.://(a::Node{T,0}, b::Node{T,0}) where {T} = divide(a, b)

#####
##### Dot
#####

# Reverse the order in the call to `Lib.op_dot` to account for row major/col major
# differences
dot(a::Node{T}, b::Node{T}, n) where {T} = @op T op_dot(b, a, UInt(n))
Base.:*(x::Node, y::Node) = dot(x, y, 1)

# #####
# ##### Indexing
# #####
#
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

goe(x::Node, n) = @op op_goe(x, convert(UInt64, n-1))

#####
##### Log
#####

Base.log(a::Node{T,N}) where {T,N} = @op T N op_log(a)

#####
##### Max
#####

# The semantics between max and maximum are flipped around beween Julia and nGraph
Base.max(a::U, b::U) where {T,N, U <: Node{T,N}} = @op T N op_maximum(a, b)

# #####
# ##### MaxPool
# #####
#
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
#
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

multiply(a::Node{T,N}, b::Node{T,N}) where {T,N} = @op T N op_mul(a, b)

#####
##### Minimum
#####

# The `min` and `minimum` semantics are swapped between Julia and nGraph.
Base.min(a::Node{T,N}, b::Node{T,N}) where {T,N} = @op T N op_minimum(a, b)

#####
##### Negative
#####

negative(a::Node{T,N}) where {T,N} = @op T N op_negative(a)
Base.:-(a::Node) = negative(a)

# #####
# ##### One Hot
# #####
#
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

# #####
# ##### permutedims
# #####
#
# function Base.permutedims(x::N, perm) where {N <: Node}
#     av = AxisVector(perm, length(perm))
#     shape = Shape([size(x)[i] for i in perm])
#     return N(Lib.op_reshape(getpointer(x), av, shape))
# end
#
# #####
# ##### Power
# #####
#
# power(a::N, b::N) where {N <: Node} = N(Lib.op_parameter(getpointer(a), getpointer(b)))
# Base.:^(a::N, b::N) where {N <: Node} = power(a, b)
#
# #####
# ##### Relu
# #####
#
# Flux.relu(a::Node{T,N}) where {T,N} = Node{T,N}(Lib.op_relu(getpointer(a)))
#
# #####
# ##### Reshape
# #####
#
# # NOTE:We're hijacking an internal Base function here to do all of the `Base.Colon`
# # preprocessing for us
# function Base._reshape(x::Node{T,N}, dims::NTuple{M,Int}) where {T,N,M}
#     av = AxisVector(1:N, N)
#     shape = Shape(dims)
#     node = Lib.op_reshape(getpointer(x), av, shape)
#     return Node{T,M}(node)
# end

# #####
# ##### Sigmoid
# #####
#
# _sigmoid(x::N) where {N <: Node} = N(Lib.op_sigmoid(getpointer(x)))
#
# #####
# ##### Softmax
# #####
#
# function Flux.softmax(x::Node{T,N}; axes = 1) where {T,N}
#     av = AxisSet(axes, N)
#     node = Lib.op_softmax(getpointer(x), av)
#     return Node{T,N}(node)
# end
#
# #####
# ##### Sqrt
# #####
#
# Base.sqrt(x::N) where {N <: Node} = N(Lib.op_sqrt(getpointer(x)))
#
# #####
# ##### Subtract
# #####
#
# subtract(a::N, b::N) where {N <: Node} = N(Lib.op_subtract(getpointer(a), getpointer(b)))
# Base.:-(a::N, b::N) where {N <: Node} = subtract(a, b)
#
# #####
# ##### Sum
# #####
#
# # Default to reducing along all dimensions
# function Base.sum(x::Node{T,N}; axes = 1:N ) where {T,N}
#     as = AxisSet(axes, N)
#     node = Lib.op_sum(getpointer(x), as)
#     return Node{T, N - length(axes)}(node)
# end
#
# #####
# ##### Tanh
# #####
#
# Base.tanh(a::N) where {N <: Node} = N(Lib.op_tanh(getpointer(a)))
#
# #####
# ##### Transpose
# #####
#
# function Base.transpose(a::Node{T,N}) where {T,N}
#     av = AxisVector(N:-1:1, N)
#     shape = Shape(reverse(size(a)))
#     node = Lib.op_reshape(getpointer(a), av, shape)
#     return Node{T,N}(node)
# end

