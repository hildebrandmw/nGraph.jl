# Trait if a type defined here is just a pure CxxWrap pointer
struct IsPointer end

# Trait if type has a field that is a pointer to a CxxWrap pointer
struct HasPointer{Field} end
HasPointer() = HasPointer{:ptr}()

wraptype(x) = error("No wrap type defined for $(typeof(x))")

getpointer(x) = _ptr(x, wraptype(x))
_ptr(x, ::IsPointer) = x
_ptr(x, ::HasPointer{Field}) where {Field} = getfield(x, Field)

# This is kind of a gross way of mapping Julia types to ngraph types.
# TODO: Think of a better way of doing this.
const TYPEMAPS = (
    Bool => "boolean",
    Float32 => "f32",
    Float64 => "f64",
    Int32 => "i32",
    Int64 => "i64",
)

const Element = Lib.NGraphTypeRef
wraptype(::Element) = IsPointer()

Element(::Type{T}) where {T} = error("No translation defined for $T")
for (T,S) in TYPEMAPS
    @eval Element(::Type{$T}) = Lib.gettype($S)
end

# This is horrible :(
# I could at least move this to a compile time auto-generator or something
function back(x::Element)
    str = Lib.c_type_name(x)
    d = Dict([
        "char"      => Bool,
        "float"     => Float32,
        "double"    => Float64,
        "int32_t"   => Int32,
        "int64_t"   => Int64,
    ])
    return d[str]
end

#####
##### Coordinate
#####
const Coordinate = Lib.CoordinateAllocated
wraptype(::Coordinate) = IsPointer()

Coordinate(x::Vector) = Lib.Coordinate(reverse(x))
Coordinate(x::Tuple) = Coordinate(collect(x))

#####
##### CoordinateDiff
#####

const CoordinateDiff = Lib.CoordinateDiffAllocated
wraptype(::CoordinateDiff) = IsPointer()

CoordinateDiff(x::Vector) = Lib.CoordinateDiff(x)
CoordinateDiff(x::Tuple) = CoordinateDiff(collect(x))

#####
##### Shape
#####

const Shape = Union{Lib.ShapeAllocated, Lib.ShapeRef}
wraptype(::Shape) = IsPointer()

# Reverse shape for column major -> row major translation
Shape(x::Vector)    = Lib.Shape(reverse(x))
Shape(x::Tuple)     = Shape(collect(x))
Shape()             = Shape(Int64[])
Shape(::Tuple{})    = Shape()

Base.length(x::Shape) = Lib._length(x)
Base.getindex(x::Shape, i) = Lib._getindex(x, convert(Int64, length(x) - i))

Shape(x::Lib.CxxWrap.SmartPointer) = Lib.get_shape(x)

#####
##### Strides
#####

const Strides = Lib.StridesAllocated
wraptype(::Strides) = IsPointer()

Strides(x::Vector) = Lib.Strides(x)
Strides(x::Tuple) = Strides(collect(x))

#####
##### Nodes
#####

const NodeVector = Union{Lib.NodeVectorAllocated, Lib.NodeVectorRef}
wraptype(::NodeVector) = IsPointer()

NodeVector(args...) = NodeVector(args)
function NodeVector(args::Union{Tuple,Vector})
    p = Lib.NodeVector()
    for arg in args
        Lib.push!(p, getpointer(arg))
    end
    return p
end

#####
##### AxisSet
#####

const AxisSet = Lib.AxisSetAllocated
wraptype(::AxisSet) = IsPointer()

# Take second `n` argument to do the axis reversal. Also convert from 1 based indexing
# to zero based indexing.
AxisSet(x::Vector, n) = Lib.AxisSet(n .- x)
AxisSet(x::Union{Tuple, AbstractRange}, n) = AxisSet(collect(x), n)
AxisSet(x, n) = AxisSet([x], n)

#####
##### AxisVector
#####

const AxisVector = Lib.AxisVectorAllocated
wraptype(::AxisVector) = IsPointer()

# Convert from col major to row major ordering - make sure to reverse the input array
# to preserve the ordering semantics.
AxisVector(x::Vector, n) = Lib.AxisVector(n .- reverse(x))
AxisVector(x::Union{Tuple, AbstractRange}, n) = AxisVector(collect(x), n)
AxisVector(x, n) = AxisVector([x], n)

#####
##### Backend
#####

# For dispatching backend treatment
abstract type AbstractBackendType end
struct CPU <: AbstractBackendType end
struct GPU <: AbstractBackendType end

# Can rely on constant propagation to make this type stable in many circumnstances.
function backend_type(s::String)
    if s == "CPU"
        return CPU
    elseif s == "GPU"
        return GPU
    else
        error("Unrecognized backend type: $s")
    end
end

struct Backend{T <: AbstractBackendType}
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.Backend,:St10unique_ptrIiSt14default_deleteIiEE}
end
wraptype(::Backend) = HasPointer()

# Pass `false` to the "must_support_dynamic" flag for now.
Backend(str::String = "CPU") = Backend{backend_type(str)}(Lib.create(str))

#####
##### Nodes
#####

### Typed nodes for automatic sizing and typing
# Subtype AbstractArray to get the AbstractArray fallbacks
struct Node{T,N} <: AbstractArray{T, N}
    # CxxWrap std::shared_ptr to the actual backing node
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.Node,:St10shared_ptrIiE}
end
wraptype(::Node) = HasPointer()

function Node(ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.Node})
    # Get the element type and shape from the node.
    N = length(Lib.get_output_shape(ptr, zero(UInt64)))
    T = back(Lib.get_output_element_type(ptr, zero(UInt64)))
    return Node{T,N}(ptr)
end

Node(x::AbstractArray{T,N}) where {T,N} = Node{T,N}(x)
function Node{T,N}(x::AbstractArray{T,N}) where {T,N}
    return Node{T,N}(Lib.op_parameter(Element(T), Shape(size(x))))
end
Node{T}(x::T) where {T} = constant(x)

Node(x::Node) = x
Node{T,N}(x::Node{T,N}) where {T,N} = x

Base.display(n::Node) = show(stdout, n)
Base.show(io::IO, n::Node{T,N}) where {T,N} =
    println(io, "Node{$T, $N} with size: $(size(n))")

function Base.size(n::Node{T,N}) where {T,N}
    shape = Lib.get_output_shape(getpointer(n), zero(UInt64))
    @assert N == length(shape)

    return ntuple(i -> shape[i], N)
end
Base.IndexStyle(::Node) = Base.IndexLinear()

# Base Methods
Base.axes(n::Node) = map(Base.OneTo, size(n))
Base.ndims(n::Node{T,N}) where {T,N} = N

### Untyped Node Descriptor for just manipulating nodes
struct NodeDescriptor
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.Node,:St10shared_ptrIiE}
end
wraptype(::NodeDescriptor) = HasPointer()
NodeDescriptor(N::Node) = NodeDescriptor(getpointer(N))
Node(N::NodeDescriptor) = Node(getpointer(N))

Base.show(io::IO, n::NodeDescriptor) = print(io, name(n))

### Operations on Node and NodeDescriptor
# Forwards
const NodeLike = Union{Node, NodeDescriptor}
name(N::NodeLike) = Lib.get_name(getpointer(N))
description(N::NodeLike) = Lib.description(getpointer(N))

rawptr(n::NodeLike) = getpointer(n)[]

Base.:(==)(n::N, m::N) where {N <: NodeLike} = rawptr(n) == rawptr(m)
Base.hash(n::NodeLike, h::UInt = UInt(0x4029388)) = hash(rawptr(n), h)

# Output sizes etc. are dealt with in the node's type signature.
# Here, we deal with inputs
get_input_size(N::NodeLike) = Lib.get_input_size(getpointer(N))
get_input_element_type(N::NodeLike, i) =
    back(Lib.get_input_element_type(getpointer(N), convert(UInt, i-1)))
function get_input_shape(N::NodeLike, i)
    shape = Lib.get_input_shape(getpointer(N), convert(UInt, i-1))
    return ntuple(i -> shape[i], length(shape))
end

# Get input and output nodes.
get_input(N::T, i) where {T <: NodeLike} = T(Lib.get_input_node(getpointer(N), convert(Int, i-1)))
get_inputs(N::NodeLike) = [get_input(N,i) for i in 1:Lib.get_input_size(getpointer(N))]

get_output_size(N::NodeLike) = Lib.get_output_size(getpointer(N))
get_output_element_type(N::NodeLike, i) =
    back(Lib.get_output_element_type(getpointer(N), convert(UInt, i-1)))

function get_output_shape(N::NodeLike, i)
    shape = Lib.get_output_shape(getpointer(N), convert(UInt, i-1))
    return ntuple(i -> shape[i], length(shape))
end

# Convert things to either all <:Node or NodeDescriptors
_as(::Type{<:Node}, x) = Node(x)
_as(::Type{NodeDescriptor}, x) = NodeDescriptor(x)

function get_output(N::T, i) where {T <: NodeLike}
    return _as.(T, (Lib.get_output_nodes(getpointer(N), convert(Int, i-1))))
end

get_outputs(N::NodeLike) =
    [get_output(N, i) for i in 1:Lib.get_output_size(getpointer(N))] |>
    Iterators.flatten |>
    unique

get_control_deps(N::T) where {T <: NodeLike} = T.(Lib.get_control_deps(getpointer(N)))

"""
    copy(node::Node, args::NodeVector)

Construct a copy of `N` with `args` as input arguments.
"""
Base.copy(node::T, args) where {T <: NodeLike} =
    T(Lib.copy_with_new_args(getpointer(node), convert(NodeVector, args)))

Base.convert(::Type{NodeVector}, v::Vector{Node}) = NodeVector(v)


# Get TensorDescriptors
outputs(N::NodeLike) = [output(N, i) for i in 1:get_output_size(N)]
output(N::NodeLike, i) =
    TensorDescriptor(Lib.get_output_tensor_ptr(getpointer(N), convert(Int, i-1)))

inputs(N::NodeLike) = [input(N, i) for i in 1:get_input_size(N)]
input(N::NodeLike, i) =
    TensorDescriptor(Lib.get_input_tensor_ptr(getpointer(N), convert(Int, i-1)))

#copy_with_new_args(n::T, args) where {T <: Node} = T(Lib.copy_with_new_args(getpointer(n), args))
#copy_with_new_args(n::Node, args::Vector) = copy_with_new_args(n, NodeVector(args))

is_mkldnn(n::NodeLike) = Lib.node_is_mkldnn_op(getpointer(n))
set_mkldnn(n::NodeLike) = Lib.node_set_mkldnn_op(getpointer(n))

splice(source::NodeLike, source_output, dest::NodeLike, dest_input, x::NodeLike) =
    Lib.special_insert_new_node_between(
        getpointer(source),
        convert(UInt, source_output - 1),
        getpointer(dest),
        convert(UInt, dest_input - 1),
        getpointer(x)
   )

input_needs_conversion(node::NodeLike, i) =
    Lib.input_needs_conversion(getpointer(node), convert(UInt, i-1))

get_sync(node::NodeLike) = Lib.get_sync(getpointer(node))
set_sync(node::NodeLike) = Lib.set_sync(getpointer(node))
clear_sync(node::NodeLike) = Lib.clear_sync(getpointer(node))

## Associates
set_priority(node::NodeLike, p::Integer) = Lib.set_priority(getpointer(node), convert(Int64, p))
get_priority(node::NodeLike) = Lib.get_priority(getpointer(node))

#####
##### Tensor
#####

revdims(::Val{N}) where {N} = ntuple(i -> N + 1 - i, N)
revdims(::AbstractArray{T,N}) where {T,N} = revdims(Val{N}())

# Hook point for dispatch - we need this because we need to turn regular Arrays into GPU
# Arrays when a `TensorView` is constructed.
transport(::Type{CPU}, x::AbstractArray) = x
transport(::Type{GPU}, x::AbstractArray) = CuArrays.cu(x)
_pointer(x::AbstractArray) = pointer(x)

# This is all scary ... but CuArrays doesn't provide a way of getting a pointer, and nGraph
# wants a GPU pointer ...
_pointer(x::CuArrays.CuArray) = Ptr{Nothing}(UInt(pointer(CuArrays.buffer((x)))))
struct TensorView
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.RuntimeTensor,:St10shared_ptrIiE}
    backend::Backend
    # Keep track of the base array to avoid GC
    base::AbstractArray

    function TensorView(backend::Backend{C}, v::Array{T,N}) where {C,T,N}
        v = transport(C, v)

        # This is kind of scary - we ... just have to make sure that the parent array
        # doesn't get moved (i.e. resized ... )
        vptr = Base.unsafe_convert(Ptr{Cvoid}, _pointer(v))
        ptr = Lib.create_tensor(
            getpointer(backend),
            Element(T),
            Shape(size(v)),
            vptr
        )

        return new(ptr, backend, v)
    end
end
rawptr(T::TensorView) = getpointer(T)[]

function Base.display(TV::TensorView)
    printstyled("Tensor View\n"; color = :yellow)
    display(TV.base)
end

Node(x::TensorView) = Node(Lib.op_parameter(
    Lib.get_element_type(getpointer(x)),
    Lib.get_shape(getpointer(x)),
))

wraptype(::TensorView) = HasPointer()

Base.eltype(T::TensorView) = back(Lib.get_element_type(getpointer(T)))
Base.sizeof(t::TensorView) = convert(Int, Lib.get_size_in_bytes(getpointer(t)))

TensorView(backend, x::TensorView) = x
function TensorView(backend, x::T) where {T}
    A = Array{T,0}(undef)
    A[] = x
    return TensorView(backend, A)
end

# If we're trying to create a tensor view from a node - create an undefined array and return
# a view of that.
function TensorView(backend::Backend, v::Node{T,N}) where {T,N}
    A = Array{T}(undef, size(v))
    return TensorView(backend, A)
end

function Base.size(t::TensorView)
    shape = Shape(getpointer(t))
    return ntuple(i -> shape[i], length(shape))
end

Base.parent(t::TensorView) = t.base

#####
##### Adjoints
#####

const Adjoints = Lib.AdjointsAllocated
wraptype(::Adjoints) = IsPointer()

make_adjoints(x, y) = Lib.make_adjoints(NodeVector(x), NodeVector(y))
backprop_node(A::Adjoints, x::T) where {T <: Node} = T(Lib.backprop_node(A, getpointer(x)))

#####
##### Parameters
#####

const ParameterVector = Union{Lib.ParameterVectorAllocated, Lib.ParameterVectorRef}
wraptype(::ParameterVector) = IsPointer()

ParameterVector(args...) = ParameterVector(args)
function ParameterVector(args::Union{Tuple,Vector})
    p = Lib.ParameterVector()
    for arg in args
        Lib.push!(p, getpointer(arg))
    end
    return p
end

Base.length(P::ParameterVector) = Lib._length(P)
Base.getindex(P::ParameterVector, i) = Node(Lib._getindex(P, convert(Int64, i-1)))
Base.iterate(P, s = 1) = (s > length(P)) ? nothing : (P[s], s+1)

#####
##### NodeWrapper
#####

const NodeWrapper = Lib.NodeWrapperAllocated
wraptype(::NodeWrapper) = IsPointer()

Base.length(n::NodeWrapper) = Lib._length(n)
Base.getindex(n::NodeWrapper, i) = Node(Lib._getindex(n, convert(Int, i-1)))
Base.iterate(n::NodeWrapper, s = 1) = (s > length(n)) ? nothing : (n[s], s+1)

mutable struct NFunction
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.NFunction,:St10shared_ptrIiE}

    # List of node operations in the function. Represented as a Lib.NodeWrapperAlocated
    # because this is generated on the C++ side of things and will get cleaned up if we
    # let it go out of scope.
    #
    # TODO: Find a way to make this not happen.
    ops::Lib.NodeWrapperAllocated
    callback::Any

    function NFunction(nodes::Lib.NodeVectorAllocated, params::Lib.ParameterVectorAllocated)
        ptr = Lib.make_function(nodes, params)
        ops = Lib.get_ordered_ops(ptr)
        return new(ptr, ops, nothing)
    end

    function NFunction(ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.NFunction,:St10shared_ptrIiE})
        ops = Lib.get_ordered_ops(ptr)
        return new(ptr, ops, nothing)
    end
end
wraptype(::NFunction) = HasPointer()

get_ordered_ops!(f::NFunction) = f.ops = Lib.get_ordered_ops(getpointer(f))
get_results(f::NFunction) = Lib.get_results(getpointer(f))
get_parameters(f::NFunction) = Lib.get_parameters(getpointer(f))
get_temporary_pool_size(f::NFunction) = convert(Int, Lib.get_temporary_pool_size(getpointer(f)))
get_pmem_pool_size(f::NFunction) = Lib.get_remote_pool_size(getpointer(f))
get_constants(f::NFunction) = collect(Iterators.filter(x -> description(x) == "Constant", f))

Base.length(f::NFunction) = Lib._length(f.ops)
Base.getindex(f::NFunction, i) = Node(Lib._getindex(f.ops, convert(Int64, i-1)))
name(f::NFunction) = Lib.get_name(getpointer(f))

function Base.iterate(f::NFunction)
    # Make sure everything is ordered
    get_ordered_ops!(f)
    s = 1
    return s <= length(f) ? (f[s], s+1) : nothing
end
Base.iterate(f::NFunction, s) = (s <= length(f)) ? (f[s], s+1) : nothing

# Allow reverse iterations
Base.reverse(f::NFunction) = Iterators.reverse(f)
Base.iterate(f::Iterators.Reverse{NFunction}, s = length(f.itr)) =
    (s == 0) ? nothing : (f.itr[s], s-1)

Base.copy(f::NFunction) = NFunction(Lib.clone_function(getpointer(f)))

#####
##### Low level handles for dealing with objects
#####

# Tensor Descriptor
struct TensorDescriptor
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.DescriptorTensor,:St10shared_ptrIiE}
end
wraptype(::TensorDescriptor) = HasPointer()

# Here, we take advantage (hope) of the unique name for each tensor to get a unique
# identifies
rawptr(a::TensorDescriptor) = getpointer(a)[]
Base.:(==)(a::TensorDescriptor, b::TensorDescriptor) = rawptr(a) == rawptr(b)
Base.hash(x::TensorDescriptor, h::UInt = UInt(0x10984)) = hash(rawptr(x), h)

function Base.show(io::IO, T::TensorDescriptor)
    println(io, "Tensor Descriptor")
    println(io, "    Name: $(name(T))")
    println(io, "    Is Persistent: $(is_persistent(T))")
end

make_persistent(T::TensorDescriptor) = Lib.make_persistent(getpointer(T))
make_volatile(T::TensorDescriptor) = Lib.make_volatile(getpointer(T))
is_persistent(T::TensorDescriptor) = Lib.is_persistent(getpointer(T))
Base.sizeof(T::TensorDescriptor) = convert(Int64, Lib._sizeof(getpointer(T)))
function Base.size(T::TensorDescriptor)
    shape = Lib.get_shape(getpointer(T))
    return ntuple(i -> shape[i], length(shape))
end

Base.eltype(T::TensorDescriptor) = back(Lib.get_element_type(getpointer(T)))
name(T::TensorDescriptor) = Lib.get_name(getpointer(T))
Tensor(backend, t::TensorDescriptor) = Tensor(eltype(t), nothing, backend, size(t)...)
PersistentTensor(backend, t::TensorDescriptor) = Tensor(eltype(t), Persistent(), backend, size(t)...)

# Set pool offsets back to zero
reset_offset(T::TensorDescriptor) = Lib.set_pool_offset(getpointer(T), convert(UInt, 0))
get_pool_offset(T::TensorDescriptor) = Lib.get_pool_offset(getpointer(T))
set_pool_number(T::TensorDescriptor, i) = Lib.set_pool_number(getpointer(T), convert(Int, i))

