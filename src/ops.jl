#####
##### Helper Functions
#####

# Make the signature match the Flux signature
expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

function expand(a::Node{T,M}, b::Node{T,N}) where {T,M,N}
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
_forward(::typeof(NNlib.σ)) = _sigmoid

Base.broadcasted(f, x::Node, y::Node) = _forward(f)(expand(x,y)...)
Base.broadcasted(f, x::Node, y::AbstractArray) = _forward(f)(expand(x, Node(y))...)
Base.broadcasted(f, x::AbstractArray, y::Node) = _forward(f)(expand(Node(x), y)...)
Base.broadcasted(f, x::Node) = _forward(f)(x)

Base.broadcasted(f, x::Node{T}, y::Number) where {T} = _forward(f)(expand(x, Node{T}(convert(T, y)))...)
Base.broadcasted(f, x::Number, y::Node{T}) where {T} = _forward(f)(expand(Node{T}(convert(T, x)), y)...)

Base.convert(::Type{Node{T,0}}, x::S) where {T,S <: Number} = Node{T,0}(convert(T, x))

# Special case element wise copy - this gets around an issue in Metalhead's ResNet
# implementation.
Base.broadcasted(::typeof(copy), x::Node) = x

# TODO: Add this
#(::Flux.BatchNorm)(x::Node) = x

#####
##### Add
#####

add(a::N, b::N) where {N <: Node} = N(Lib.op_add(getpointer(a), getpointer(b)))
Base.:+(a::Node, b::Node) = add(a,b)

#####
##### AvgPool
#####

function avgpool(x::Node{T,N}, shape::Tuple; pad = 0, stride = shape) where {T,N}
    # Convert to nGraph types
    window_shape = Shape(shape)
    strides = Strides(expand(N-2, stride))
    padding_below = Shape(expand(N-2, pad))
    padding_above = Shape(expand(N-2, pad))

    ptr = Lib.op_avgpool(getpointer(x), window_shape, strides, padding_below, padding_above)
    return Node{T,N}(ptr)
end
Flux.meanpool(x::Node, args...; kw...) = avgpool(x, args...; kw...)

#####
##### BatchNorm
#####

function batchnorm_training(input::Node, γ::Node, β::Node, ϵ)
    return Node(Lib.op_batchnorm_training(
        getpointer(input),
        getpointer(γ),
        getpointer(β),
        convert(Float64, ϵ)
    ))
end

#####
##### Broadcast
#####

_broadcast_trailing(M,N) = [i for i in (M+1):N]
function Base.broadcast(
        a::Node{T,M},
        shape::NTuple{N,Int};
        axes = _broadcast_trailing(M,N)
    ) where {T,M,N}

    # Construct the final shape from `shape`
    final_shape = Shape(shape)
    axis_set = AxisSet(axes, N)

    return Node{T,N}(Lib.op_broadcast(getpointer(a), final_shape, axis_set))
end


#####
##### Concat
#####

function concat(nodes::Vector{Node{T,N}}; dims::Integer = 1) where {T,N}
    # Flip dims for column -> row
    node = Lib.op_concat(NodeVector(nodes), N - dims)
    return Node{T,N}(node)
end

Base.cat(x::Node...; kw...) = concat(collect(x); kw...)

#####
##### Constants
#####

constant(x::T) where {T} = Node{T,0}(Lib.op_constant(Element(T), Shape(), [x]))
constant(x::AbstractArray{T,N}) where {T,N} =
    Node{T,N}(Lib.op_constant(Element(T), Shape(size(x)), reshape(x, :)))

#####
##### Convolution
#####

function NNlib.conv(x::Node{T,N}, w::Node{T,N}; stride = 1, pad = 0, dilation = 1) where {T,N}
    # Construct the convolution node.
    strides = Strides(expand(N-2, stride))
    padding = CoordinateDiff(expand(N-2, pad))
    dilations = Strides(expand(N-2, dilation))

    node = Lib.op_convolution(getpointer(x), getpointer(w), strides, dilations, padding, padding)
    return Node{T,N}(node)
end

#####
##### Deconvolution
#####

function deconvolution(x::Node{T,N}, w::Node{T,N}, out_shape;
        stride = 1,
        pad = 0,
        dilation = 1
    ) where {T,N}

    # see https://github.com/NervanaSystems/ngraph-mxnet-bridge/blob/master/src/ops/deconvolution.cc
    # for inspiration about how this thing came about.
    out_shape = Shape(out_shape)
    strides = Strides(expand(N-2, stride))
    padding = CoordinateDiff(expand(N-2, pad))
    dilations = Strides(expand(N-2, dilation))
    data_dilation = Strides(ntuple(i -> 1, N-2))

    node = Node{T,N}(Lib.op_convolution_backprop_data(
        getpointer(out_shape),
        getpointer(w),
        getpointer(x),
        getpointer(strides),
        getpointer(dilations),
        getpointer(padding),
        getpointer(padding),
        getpointer(data_dilation),
    ))
end

#####
##### Divide
#####

divide(a::Node{T,N}, b::Node{T,N}) where {T,N} =
    Node{T,N}(Lib.op_divide(getpointer(a), getpointer(b)))

Base.:/(a::Node{T,0}, b::Node{T,0}) where {T} = divide(a, b)
Base.://(a::Node{T,0}, b::Node{T,0}) where {T} = divide(a, b)

#####
##### Dot
#####

# Reverse the order in the call to `Lib.op_dot` to account for row major/col major
# differences
dot(a::Node{T,N}, b::Node{T,M}, n) where {T,N,M} =
    Node{T,M}(Lib.op_dot(getpointer(b), getpointer(a), convert(UInt, n)))

# Fully Connected
Base.:*(w::Node, x::Node) = dot(w, x, 1)
Base.:*(w::Node, x::AbstractArray) = w * Node(x)
Base.:*(w::AbstractArray, x::Node) = Node(w) * x

#####
##### Embedding
#####

embedding(data::Node, weights::Node) = 
    Node(Lib.op_embedding(getpointer(data), getpointer(weights)))

#####
##### Indexing
#####

_lb(i) = i
_lb(::Colon) = 1

_ub(bound, i) = i
_ub(bound, ::Colon) = bound
function Base.getindex(n::Node{T,N}, args...) where {T,N}
    sz = size(n)
    # Subtract 1 from the lower bound since the lower bounds are inclusive in ngraph.
    # Leave the upper bounds as is since the upper-bounds are exclusive.
    lb = Coordinate(map(_lb, args) .- 1)
    ub = Coordinate(ntuple(i -> _ub(sz[i], args[i]), length(args)))

    return Node(Lib.op_slice(getpointer.((n, lb, ub))... ))
end

#####
##### GetOutput
#####

get_output_element(x::Node, n) = Node(Lib.op_get_output_element(getpointer(x), convert(UInt, n-1)))

#####
##### Log
#####

Base.log(a::Node{T,N}) where {T,N} = Node{T,N}(Lib.op_log(getpointer(a)))

#####
##### Max
#####

# The semantics between max and maximum are flipped around beween Julia and nGraph
Base.max(a::T, b::T) where {T <: Node} = T(Lib.op_maximum(getpointer(a), getpointer(b)))

#####
##### MaxPool
#####

function Flux.maxpool(x::Node{T,N}, shape::Tuple; pad = 0, stride = shape) where {T,N}
    # Convert to nGraph types
    window_shape = Shape(shape)
    strides = Strides(expand(N-2, stride))
    padding_below = Shape(expand(N-2, pad))
    padding_above = Shape(expand(N-2, pad))

    ptr = Lib.op_maxpool(getpointer(x), window_shape, strides, padding_below, padding_above)
    return Node{T,N}(ptr)
end

#####
##### Multiply
#####

multiply(a::Node{T,N}, b::Node{T,N}) where {T,N} = Node{T,N}(Lib.op_mul(getpointer(a), getpointer(b)))

#####
##### Minimum
#####

# The `min` and `minimum` semantics are swapped between Julia and nGraph.
Base.minimum(a::N, b::N) where {N <: Node} = N(Lib.op_minimum(getpointer(a), getpointer(b)))
_forward(::typeof(min)) = minimum

#####
##### Negative
#####

negative(a::Node{T,N}) where {T,N} = Node{T,N}(Lib.op_negative(getpointer(a)))

Base.:-(a::Node) = negative(a)

#####
##### Parameter
#####

parameter(x::AbstractArray{T,N}) where {T,N} = Node(x)
parameter(x::T) where {T} = Node{T,0}(Lib.op_parameter(Element(T), Shape(())))
parameter(x::Node) = x

#####
##### Power
#####

power(a::N, b::N) where {N <: Node} = N(Lib.op_parameter(getpointer(a), getpointer(b)))
Base.:^(a::N, b::N) where {N <: Node} = power(a, b)

#####
##### Relu
#####

Flux.relu(a::Node{T,N}) where {T,N} = Node{T,N}(Lib.op_relu(getpointer(a)))

#####
##### Reshape
#####

# NOTE:We're hijacking an internal Base function here to do all of the `Base.Colon`
# preprocessing for us
function Base._reshape(x::Node{T,N}, dims::NTuple{M,Int}) where {T,N,M}
    av = AxisVector(1:N, N)
    shape = Shape(dims)
    node = Lib.op_reshape(getpointer(x), av, shape)
    return Node{T,M}(node)
end

#####
##### Result
#####

result(x::T) where {T <: Node} = T(Lib.op_result(getpointer(x)))

#####
##### Sigmoid
#####

_sigmoid(x::N) where {N <: Node} = N(Lib.op_sigmoid(getpointer(x)))

#####
##### Softmax
#####

function Flux.softmax(x::Node{T,N}; axes = 1) where {T,N}
    av = AxisSet(axes, N)
    node = Lib.op_softmax(getpointer(x), av)
    return Node{T,N}(node)
end

#####
##### Subtract
#####

subtract(a::N, b::N) where {N <: Node} = N(Lib.op_subtract(getpointer(a), getpointer(b)))
Base.:-(a::N, b::N) where {N <: Node} = subtract(a, b)

#####
##### Sum
#####

# Default to reducing along all dimensions
function Base.sum(x::Node{T,N}; axes = 1:N ) where {T,N}
    as = AxisSet(axes, N)
    node = Lib.op_sum(getpointer(x), as)
    return Node{T, N - length(axes)}(node)
end

#####
##### Tanh
#####

Base.tanh(a::N) where {N <: Node} = N(Lib.op_tanh(getpointer(a)))

#####
##### Transpose
#####

function Base.transpose(a::Node{T,N}) where {T,N}
    av = AxisVector(N:-1:1, N)
    shape = Shape(reverse(size(a)))
    node = Lib.op_reshape(getpointer(a), av, shape)
    return Node{T,N}(node)
end

#######################################################################################
#
# Custom ops
move(x::T, output = 1) where {T <: Node} = T(Lib.op_move(getpointer(x), convert(UInt, output-1)))

moveasync(x::T, across::Node) where {T <: Node} = moveasync(x, 1, across)
moveasync(x::T, output, across::Node) where {T <: Node} = 
    T(Lib.op_moveasync(getpointer(x), convert(UInt, output-1), name(across)))

convert_layout_to(x::Node, y::Node, i) =
    Node(Lib.op_cpu_convert_layout_to(getpointer(x), getpointer(y), convert(Int, i-1)))
