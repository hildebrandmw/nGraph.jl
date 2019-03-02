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

# Default broadcasting to erroring to catch errors.
#
# We will extend certain valid ones later. 
#Base.broadcasted(::T, a::Node, b::Node) where {T} = error("Cannot braodcast $T over nodes")
#Base.broadcasted(::T, a, b::Node) where {T}= error("Cannot braodcast $T over nodes")
#Base.broadcasted(::T, a::Node, b) where {T}= error("Cannot braodcast $T over nodes")

_forward(f) = f
_forward(::typeof(*)) = multiply
Base.broadcasted(f, x::Node, y::Node) = _forward(f)(expand(x,y)...)
Base.broadcasted(f, x::Node, y::AbstractArray) = _forward(f)(expand(x, Node(y))...)
Base.broadcasted(f, x::AbstractArray, y::Node) = _forward(f)(expand(Node(x), y)...)
Base.broadcasted(f, x::Node) = f(x)

Base.broadcasted(f, x::Node{T}, y::Number) where {T} = _forward(f)(expand(x, Node{T}(convert(T, y)))...)
Base.broadcasted(f, x::Number, y::Node{T}) where {T} = _forward(f)(expand(Node{T}(convert(T, x)), y)...)

Base.convert(::Type{Node{T,0}}, x::S) where {T,S} = Node{T,0}(convert(T, x))

#####
##### Add
#####

add(a::N, b::N) where {N <: Node} = N(Lib.op_add(a.ptr, b.ptr), "Add")
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

    ptr = Lib.op_avgpool(x.ptr, window_shape, strides, padding_below, padding_above)
    return Node{T,N}(ptr, "AvgPool")
end
Flux.meanpool(x::Node, args...; kw...) = avgpool(x, args...; kw...)

#####
##### Broadcast
#####

# This defies the normal broadcast semantics, but in practice shouldn't be an issue.
_broadcast_trailing(M,N) = [i-1 for i in 1:(N-M)]
function Base.broadcast(
        a::Node{T,M}, 
        shape::NTuple{N,Int};
        axes = _broadcast_trailing(M,N)
    ) where {T,M,N}

    # Construct the final shape from `shapw`
    final_shape = Shape(shape)
    axis_set = AxisSet(axes)

    return Node{T,N}(Lib.op_broadcast(a.ptr, final_shape, axis_set), "Broadcast")
end


#####
##### Concat
#####

function concat(nodes::Vector{Node{T,N}}; dims::Integer = 1) where {T,N}
    # Flip dims for column -> row
    node = Lib.op_concat(NodeVector(nodes), N - dims)
    return Node{T,N}(node, "Concat", nodes...)
end

Base.cat(x::Node...; kw...) = concat(collect(x); kw...)

#####
##### Constants
#####

constant(x::T) where {T} = Node{T,0}(Lib.op_constant(Element(T), Shape(), [x]), "Constant")

#####
##### Convolution
#####

function NNlib.conv(x::Node{T,N}, w::Node{T,N}; stride = 1, pad = 0, dilation = 1) where {T,N}
    # Construct the convolution node.  
    strides = Strides(expand(N-2, stride)) 
    padding = CoordinateDiff(expand(N-2, pad))
    dilations = Strides(expand(N-2, dilation))

    node = Lib.op_convolution(x.ptr, w.ptr, strides, dilations, padding, padding)
    return Node{T,N}(node, "Convolution")
end

#####
##### Divide
#####

divide(a::Node{T,N}, b::Node{T,N}) where {T,N} = Node{T,N}(Lib.op_divide(a.ptr, b.ptr), "Divide")

Base.:/(a::Node{T,0}, b::Node{T,0}) where {T} = divide(a, b)
Base.://(a::Node{T,0}, b::Node{T,0}) where {T} = divide(a, b)

#####
##### Dot
#####

# Reverse the order in the call to `Lib.op_dot` to account for row major/col major
# differences
dot(a::Node{T,N}, b::Node{T,M}, n) where {T,N,M} = Node{T,M}(Lib.op_dot(b.ptr, a.ptr, UInt(n)), "Dot")

# Fully Connected
Base.:*(w::Node, x::Node) = dot(w, x, 1)
Base.:*(w::Node, x::AbstractArray) = w * Node(x)
Base.:*(w::AbstractArray, x::Node) = Node(w) * x

#####
##### Log
#####

Base.log(a::Node{T,N}) where {T,N} = Node{T,N}(Lib.op_log(a.ptr), "Log")

#####
##### MaxPool
#####

function Flux.maxpool(x::Node{T,N}, shape::Tuple; pad = 0, stride = shape) where {T,N}
    # Convert to nGraph types    
    window_shape = Shape(shape)
    strides = Strides(expand(N-2, stride)) 
    padding_below = Shape(expand(N-2, pad))
    padding_above = Shape(expand(N-2, pad))

    ptr = Lib.op_maxpool(x.ptr, window_shape, strides, padding_below, padding_above)
    return Node{T,N}(ptr, "MaxPool")
end

#####
##### Multiply
#####

multiply(a::Node{T,N}, b::Node{T,N}) where {T,N} = Node{T,N}(Lib.op_mul(a.ptr, b.ptr), "Multiply")

#####
##### Minimum
#####

# The `min` and `minimum` semantics are swapped between Julia and nGraph.
Base.minimum(a::N, b::N) where {N <: Node} = N(Lib.op_minimum(a.ptr, b.ptr), "Minimum")
_forward(::typeof(min)) = minimum

#####
##### Negative
#####

negative(a::Node{T,N}) where {T,N} = Node{T,N}(Lib.op_negative(a.ptr), "Negative")

Base.:-(a::Node) = negative(a)
#Base.broadcasted(::typeof(-), a::Node) = negative(a)

#####
##### Parameter
#####

parameter(x::AbstractArray{T,N}) where {T,N} = Node(x)
parameter(x::T) where {T} = Node{T,0}(Lib.op_parameter(_element(T), shape(())), IdSet{Node}(), "Param")
parameter(x::Node) = x

#####
##### Power
#####

power(a::N, b::N) where {N <: Node} = N(Lib.op_parameter(a.ptr, b.ptr), "Power")
Base.:^(a::N, b::N) where {N <: Node} = power(a, b)

#####
##### Relu
#####

Flux.relu(a::Node{T,N}) where {T,N} = Node{T,N}(Lib.op_relu(a.ptr), "Relu")

#####
##### Reshape
#####

# We're hijacking an internal Base function here to do all of the `Base.Colon` preprocessing
# for us
function Base._reshape(x::Node{T,N}, dims::NTuple{M,Int}) where {T,N,M}
    av = AxisVector(ntuple(i -> i-1, N))
    shape = Shape(dims)
    node = Lib.op_reshape(x.ptr, av, shape)
    return Node{T,M}(node, "Reshape")
end

#####
##### Softmax
#####

function Flux.softmax(x::Node{T,N}; axes = N-1) where {T,N}
    av = AxisSet(axes)
    node = Lib.op_softmax(x.ptr, av)
    return Node{T,N}(node, "Softmax")
end

#####
##### Subtract
#####

subtract(a::N, b::N) where {N <: Node} = N(Lib.op_subtract(a.ptr, b.ptr), "Subtract")
Base.:-(a::N, b::N) where {N <: Node} = subtract(a, b)

#####
##### Sum
#####

# Default to reducing along all dimensions
function Base.sum(x::Node{T,N}; axes = ntuple(identity, N) ) where {T,N}
    as = AxisSet(axes .- 1)
    node = Lib.op_sum(x.ptr, as) 
    return Node{T, N - length(axes)}(node, "Sum")
end
