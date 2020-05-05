#####
##### Utility Functions
#####

# Construct a `shape` vector from `x`.
# Since Julia is column major while C++ is row major, we need to be careful about the order
# we pass things.
#
# In general, the C++ code will not do any reversing schenanigans.
# Thus, it's up to the Julia interfacing code to make sure the appropriate conversions
# between column major and row major are made.
shape(x) = [Int64(i) for i in reverse(x)]

# TODO: Swap out this whole trait thing for something that deals with the new CxxWrap
# pointer types better.

# Trait if a type defined here is just a pure CxxWrap pointer
struct IsPointer end

# Trait if type has a field that is a pointer to a CxxWrap pointer
struct HasPointer{Field} end
HasPointer() = HasPointer{:ptr}()

wraptype(x) = error("No wrap type defined for $(typeof(x))")

unwrap(x) = _ptr(x, wraptype(x))
_ptr(x, ::IsPointer) = x
_ptr(x, ::HasPointer{Field}) where {Field} = getfield(x, Field)

#####
##### Element Type Maps
#####

# This should be kept inline with `ngraph/type/element_type.hpp`
@enum NGElements::Int32 begin
    ng_undefined=0
    ng_dynamic
    ng_boolean
    ng_bf16
    ng_f16
    ng_f32
    ng_f64
    ng_i8
    ng_i16
    ng_i32
    ng_i64
    ng_u1
    ng_u8
    ng_u16
    ng_u32
    ng_u64
end

# This is kind of a gross way of mapping Julia types to ngraph types.
# TODO: Think of a better way of doing this.
const TYPEMAPS = Dict(
   ng_boolean => Bool,
   ng_f32     => Float32,
   ng_f64     => Float64,
   ng_i8      => Int8,
   ng_i16     => Int16,
   ng_i32     => Int32,
   ng_i64     => Int64,
   ng_u8      => UInt8,
   ng_u16     => UInt16,
   ng_u32     => UInt32,
   ng_u64     => UInt64,
)

const Element = Lib.CxxWrap.CxxWrapCore.ConstCxxPtr{Lib.Element}

# Mapping from Julia => nGraph
Element(::Type{T}) where {T} = error("No translation defined for $T")
for (S,T) in TYPEMAPS
    @eval Element(::Type{$T}) = Lib.ngraph_type($(Int32(S)))
end

back(x) = TYPEMAPS[NGElements(Lib.Type_t(x[]))]

#####
##### Nodes
#####

const NodeCppType = Lib.CxxWrap.StdLib.SharedPtrAllocated{nGraph.Lib.Node}
Base.ndims(x::NodeCppType) = length(Lib.get_output_shape(x))
Base.eltype(x::NodeCppType) = back(Lib.get_output_element_type(x, 0))

# Subtype AbstractArray to get all of the nice fallbacks.
struct Node{T,N} <: AbstractArray{T,N}
    obj::NodeCppType
end
unwrap(x::Node) = x.obj

Base.show(io::IO, x::Node{T,N}) where {T,N} = println(io, "Node{$T,$N} - $(name(x))")
Base.display(x::Node) = show(stdout, x)

Node{T}(x::NodeCppType) where {T} = NodeTyped{T,ndims(x)}(x)
Node(x::NodeCppType) = Node{eltype(x),ndims(x)}(x)

Node(x::AbstractArray{T,N}) where {T,N} = Node{T,N}(x)
Node{T,N}(x::AbstractArray{T,N}) where {T,N} = parameter(T, size(x))

# Array style arguments
Base.ndims(x::Node) = ndims(unwrap(x))

function Base.size(x::Node)
    nd = ndims(x)
    dims = Lib.get_output_shape(unwrap(x))
    # Reversing is done on the ngraph side.
    return Tuple(Int.(reverse(dims)))
end

Base.size(x::Node, i::Integer) = size(x)[i]
Base.length(x) = prod(size(x))
Base.eltype(x::Node{T}) where {T} = T
Base.IndexStyle(::Node) = Base.IndexLinear()

name(x::Node) = String(Lib.get_name(unwrap(x)))
description(x::Node) = String(Lib.description(unwrap(x)))

# So these can be used as keys in a Dict
Base.:(==)(x::T, y::T) where {T <: Node} = name(x) == name(y)
Base.hash(x::Node, h::UInt = UInt(0x209348)) = hash(name(x), h)

function Base.getindex(::Node, i::Int)
    error("Yeah, yeah, I know \"Node\" is an AbstractArray ... but please don't index into it.")
end

#####
##### NGFunction
#####

mutable struct NGFunction
    ptr::Lib.CxxWrap.StdLib.SharedPtrAllocated{nGraph.Lib.NGFunction}
end

function NGFunction(parameters::Vector, results::Vector)
    ptr = make_function(parameters, results)
    return NGFunction(ptr)
end

function make_function(inputs::Vector, outputs::Vector)
    # Unwrap and create references.
    parameters = Lib.CxxWrap.CxxRef.(unwrap.(inputs))
    results = Lib.CxxWrap.CxxRef.(unwrap.(outputs))
    return Lib.make_function(results, parameters)
end

name(f::NGFunction) = String(Lib.get_name(f.ptr))
poolsize(f::NGFunction) = Int(Lib.get_temporary_pool_size(f.ptr))

#####
##### Backend
#####

struct Backend
    ptr::Lib.CxxWrap.StdLib.SharedPtrAllocated{nGraph.Lib.Backend}
end
wraptype(::Backend) = HasPointer()

# Pass `false` to the "must_support_dynamic" flag for now.
Backend(str::String = "CPU") = Backend(Lib.create(str))
version(x::Backend) = String(Lib.get_version(x.ptr))

#####
##### Tensor
#####

struct TensorView
    ptr::Lib.CxxWrap.StdLib.SharedPtrAllocated{nGraph.Lib.RuntimeTensor}
    backend::Backend

    # Keep track of the base array to avoid GC
    # TODO: Rethink the whole - "casting to pointer" thing.
    parent::AbstractArray

    function TensorView(backend::Backend, v::Array{T}) where {T}
        # This is kind of scary - we ... just have to make sure that the parent array
        # doesn't get moved (i.e. resized ... )
        vptr = Base.unsafe_convert(Ptr{Cvoid}, pointer(v))
        ptr = Lib.create_tensor(
            backend.ptr,
            Element(T)[],
            shape(size(v)),
            vptr
        )

        return new(ptr, backend, v)
    end
end

Base.eltype(x::TensorView) = back(Lib.get_element_type(x.ptr))
Base.sizeof(x::TensorView) = convert(Int, Lib.get_size_in_bytes(x.ptr))

TensorView(backend, x::TensorView) = x

# Capture scalars in a 0-dimensional array
function TensorView(backend, x::T) where {T}
    A = Array{T,0}(undef)
    A[] = x
    return TensorView(backend, A)
end

# If we're trying to create a tensor view from a node
# create an undefined array and return a view of that.
function TensorView(backend::Backend, v::Node{T,N}) where {T,N}
    A = Array{T}(undef, size(v))
    return TensorView(backend, A)
end

function Base.size(t::TensorView)
    shape = Lib.get_shape(t.ptr)
    return Tuple(reverse(shape))
end

Base.parent(t::TensorView) = t.parent

function show(io::IO, TV::TensorView)
    printstyled(io, "Tensor View\n"; color = :yellow)
    println(io, parent(TV))
end

