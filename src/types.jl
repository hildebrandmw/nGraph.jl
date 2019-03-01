# This is kind of a gross way of mapping Julia types to ngraph types.
# TODO: Think of a better way of doing this.
const TYPEMAPS = (
    Bool => "boolean",
    Float32 => "f32",
    Float64 => "f64",
    Int64 => "i64",
)

const Element = Lib.NGraphTypeRef

Element(::Type{T}) where {T} = error("No translation defined for $T")
for (T,S) in TYPEMAPS
    @eval Element(::Type{$T}) = Lib.gettype($S)
end

# This is horrible :(
# I could at least move this to a compile time auto-generator
function back(x::Element)
    str = Lib.c_type_name(x)
    d = Dict([
        "char"      => Bool,
        "float"     => Float32,
        "double"    => Float64,
        "int64_t"   => Int64, 
    ])
    return get(d, str, Any)
end

#####
##### Shape
#####

const Shape = Lib.ShapeAllocated

# Reverse shape for column major -> row major translation
Shape(x::Vector) = Lib.make_shape(reverse(x))
Shape(x::Tuple) = Shape(collect(x))
Shape() = Shape(Int64[])
Shape(::Tuple{}) = Shape()

#####
##### Strides
#####

const Strides = Lib.StridesAllocated

Strides(x::Vector) = Lib.make_strides(x)
Strides(x::Tuple) = Strides(collect(x))

#####
##### CoordinateDiff
#####

const CoordinateDiff = Lib.CoordinateDiffAllocated

CoordinateDiff(x::Vector) = Lib.make_coordinatediff(x)
CoordinateDiff(x::Tuple) = CoordinateDiff(collect(x))

#####
##### AxisSet
#####

const AxisSet = Lib.AxisSetAllocated

AxisSet(x::Vector) = Lib.make_axisset(x)
AxisSet(x::Tuple) = AxisSet(collect(x))
AxisSet(x) = AxisSet([x])

#####
##### AxisVector
#####

const AxisVector = Lib.AxisVectorAllocated

AxisVector(x::Vector) = Lib.make_axisvector(x)
AxisVector(x::Tuple) = AxisVector(collect(x))
AxisVector(x) = AxisVector([x])


#####
##### Node
#####

struct Node{T,N} <: AbstractArray{T, N}
    # CxxWrap std::shared_ptr to the actual backing node
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.Node,:St10shared_ptrIiE}
    op::String
    # Optional data if the node was created from an array
    data::AbstractArray
end

Node{T,N}(ptr, op, parents::Node...) where {T,N} = Node{T,N}(ptr, op, T[])

Node{T,N}(ptr, op) where {T,N} = Node{T,N}(ptr, op, T[])

Node(x::AbstractArray{T,N}) where {T,N} = Node{T,N}(x)
function Node{T,N}(x::AbstractArray{T,N}) where {T,N}
    return Node{T,N}(Lib.op_parameter(Element(T), Shape(size(x))), "Param", copy(x))
end


function Node{T}(x::T) where {T}
    Node{T,0}(Lib.op_constant(Element(T), Shape(), [x]), "Constant")
end

#Node(x::T) where {T <: Number} = Node{T,0}(x)

Node(x::Node) = x
Base.getindex(n::Node{T,N}, inds...) where {T,N} = zero(T)

function Base.size(n::Node{T,N}) where {T,N}
    shape = Lib.get_output_shape(n.ptr, zero(UInt64))
    @assert N == length(shape)

    return reverse(ntuple(i -> shape[i], N))
end

Base.axes(n::Node) = map(Base.OneTo, size(n))
Base.ndims(n::Node{T,N}) where {T,N} = N

_tab(n) = " "^(4 * n)
function Base.show(io::IO, N::Node)
    # Print out metadata associated with the node.
    println(io, "nGraph Node: $(N.op)")
    num_outputs = Lib.get_output_size(N.ptr)
    for i in 0:num_outputs-1
        println(io, _tab(1), "Output $i: ")
        println(io, 
            _tab(2), 
            "Element Type: ", 
            Lib.c_type_name(Lib.get_output_element_type(N.ptr, i))
        )

        println(io, _tab(2), "Shape: $(size(N))")
    end
end

#####
##### Tensor
#####

struct Tensor{T,N} <: AbstractArray{T,N}
    ptr::CxxWrap.SmartPointerWithDeref{nGraph.Lib.Tensor,:St10shared_ptrIiE}

    function Tensor{T}(::UndefInitializer, backend, inds::Vararg{Int,N}) where {T,N} 
        shape = Shape(inds)
        element = Element(T)
        ptr = Lib.create_tensor(backend, element, shape)

        return new{T,N}(ptr)
    end

    function Tensor(backend, param::Node{T,N}) where {T,N}
        shape = Shape(size(param))
        element = Lib.get_output_element_type(param.ptr, UInt(0))
        ptr = Lib.create_tensor(backend, Element(T), shape)

        A = new{T,N}(ptr)
        # Check if the node have any data attached to it. If so, copy it into the tensor
        if size(param.data) == size(param)
            A .= param.data
        end
        return A
    end
end

function Tensor(backend, x::T) where {T} 
    t = Tensor{T}(undef, backend)
    t[] = x
    return t
end

function Tensor(backend, v::AbstractArray{T,N}) where {T,N}
    t = Tensor{T}(undef, backend, size(v)...)
    t .= v
    return t
end

function Base.size(t::Tensor{T,N}) where {T,N}
    shape = Lib.get_shape(t.ptr)
    @assert N == length(shape)

    return reverse(ntuple(i -> shape[i], N))
end

function Base.getindex(t::Tensor{T,N}, i) where {T,N} 
    x = [zero(T)]
    GC.@preserve x Lib.tensor_read(t.ptr, Ptr{Cvoid}(pointer(x)), sizeof(T) * UInt64(i-1), UInt64(sizeof(T)))
    return first(x)
end

# Need to define this to get around the MKL buffer-overflow bug
function Base.collect(t::Tensor{T,N}) where {T,N}
    x = Array{T}(undef, size(t)...)
    GC.@preserve x Lib.tensor_read(t.ptr, Ptr{Cvoid}(pointer(x)), UInt64(0), UInt64(sizeof(x)))
    return x
end

function Base.setindex!(t::Tensor{T,N}, v, i) where {T,N}
    x = [convert(T, v)]
    GC.@preserve x Lib.tensor_write(t.ptr, Ptr{Cvoid}(pointer(x)), sizeof(T) * UInt64(i-1), UInt64(sizeof(x)))
    return nothing
end

Base.IndexStyle(::Tensor) = Base.IndexLinear()

#####
##### Adjoints
#####

const Adjoints = Lib.AdjointsAllocated

Adjoints(x, y) = Lib.Adjoints(NodeVector(x), NodeVector(y))
backprop_node(A::Adjoints, x::T) where {T <: Node} = T(Lib.backprop_node(A, x.ptr), "Backprop", x)

#####
##### Parameters
#####

function ParameterVector(args::Node...)
    p = Lib.ParameterVector()
    for arg in args
        Lib.push!(p, arg.ptr)
    end
    return p
end

##### 
##### Nodes
#####

NodeVector(x::Node, args::Node...) = NodeVector((x,args...))
function NodeVector(args)
    p = Lib.NodeVector()
    for arg in args
        Lib.push!(p, arg.ptr)
    end
    return p
end
