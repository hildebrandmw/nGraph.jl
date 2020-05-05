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
##### Tensor Descriptor
#####

# struct TensorDescriptor
#     ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.DescriptorTensor,:St10shared_ptrIiE}
# end
# wraptype(::TensorDescriptor) = HasPointer()
#
# # Here, we take advantage (hope) of the unique name for each tensor to get a unique
# # identifies
# rawptr(a::TensorDescriptor) = getpointer(a)[]
# Base.:(==)(a::TensorDescriptor, b::TensorDescriptor) = rawptr(a) == rawptr(b)
# Base.hash(x::TensorDescriptor, h::UInt = UInt(0x10984)) = hash(rawptr(x), h)
#
# Base.show(io::IO, T::TensorDescriptor) = println(io, "Tensor Descriptor: $(name(T))")
#
# Base.sizeof(T::TensorDescriptor) = convert(Int64, Lib._sizeof(getpointer(T)))
# function Base.size(T::TensorDescriptor)
#     shape = Lib.get_shape(getpointer(T))
#     return ntuple(i -> shape[i], length(shape))
# end
#
# Base.eltype(T::TensorDescriptor) = back(Lib.get_element_type(getpointer(T)))
# name(T::TensorDescriptor) = Lib.get_name(getpointer(T))

#####
##### Backend
#####

struct Backend
    ptr::Lib.CxxWrap.StdLib.SharedPtrAllocated{nGraph.Lib.Backend}
end
wraptype(::Backend) = HasPointer()

# Pass `false` to the "must_support_dynamic" flag for now.
Backend(str::String = "CPU") = Backend{backend_type(str)}(Lib.create(str))

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
    return Tuple(Int.(reverse(dims)))
end

Base.size(x::Node, i::Integer) = size(x)[i]
Base.length(x) = prod(size(x))

Base.eltype(x::Node{T}) where {T} = T

Base.IndexStyle(::Node) = Base.IndexLinear()
Base.axes(x::Node) = map(Base.OneTo, size(x))

name(x::Node) = String(Lib.get_name(unwrap(x)))
description(x::Node) = String(Lib.description(unwrap(x)))

# So these can be used as keys in a Dict
Base.:(==)(x::T, y::T) where {T <: Node} = name(x) == name(y)
Base.hash(x::Node, h::UInt = UInt(0x209348)) = hash(name(x), h)

# #####
# ##### Tensor
# #####
#
# revdims(::Val{N}) where {N} = ntuple(i -> N + 1 - i, N)
# revdims(::AbstractArray{T,N}) where {T,N} = revdims(Val{N}())
#
# # Hook point for dispatch - we need this because we need to turn regular Arrays into GPU
# # Arrays when a `TensorView` is constructed.
# transport(::Type{CPU}, x::AbstractArray) = x
# _pointer(x::AbstractArray) = pointer(x)
#
# struct TensorView
#     ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.RuntimeTensor,:St10shared_ptrIiE}
#     backend::Backend
#     # Keep track of the base array to avoid GC
#     base::AbstractArray
#
#     function TensorView(backend::Backend{C}, v::Array{T,N}) where {C,T,N}
#         v = transport(C, v)
#
#         # This is kind of scary - we ... just have to make sure that the parent array
#         # doesn't get moved (i.e. resized ... )
#         vptr = Base.unsafe_convert(Ptr{Cvoid}, _pointer(v))
#         ptr = Lib.create_tensor(
#             getpointer(backend),
#             Element(T),
#             Shape(size(v)),
#             vptr
#         )
#
#         return new(ptr, backend, v)
#     end
# end
# rawptr(T::TensorView) = getpointer(T)[]
#
# function Base.display(TV::TensorView)
#     printstyled("Tensor View\n"; color = :yellow)
#     display(TV.base)
# end
#
# Node(x::TensorView) = Node(Lib.op_parameter(
#     Lib.get_element_type(getpointer(x)),
#     Lib.get_shape(getpointer(x)),
# ))
#
# wraptype(::TensorView) = HasPointer()
#
# Base.eltype(T::TensorView) = back(Lib.get_element_type(getpointer(T)))
# Base.sizeof(t::TensorView) = convert(Int, Lib.get_size_in_bytes(getpointer(t)))
#
# TensorView(backend, x::TensorView) = x
# function TensorView(backend, x::T) where {T}
#     A = Array{T,0}(undef)
#     A[] = x
#     return TensorView(backend, A)
# end
#
# # If we're trying to create a tensor view from a node - create an undefined array and return
# # a view of that.
# function TensorView(backend::Backend, v::Node{T,N}) where {T,N}
#     A = Array{T}(undef, size(v))
#     return TensorView(backend, A)
# end
#
# function Base.size(t::TensorView)
#     shape = Shape(getpointer(t))
#     return ntuple(i -> shape[i], length(shape))
# end
#
# Base.parent(t::TensorView) = t.base
# fetch(t::TensorView) = collect(parent(t))
#
# #####
# ##### Adjoints
# #####
#
# const Adjoints = Lib.AdjointsAllocated
# wraptype(::Adjoints) = IsPointer()
#
# make_adjoints(x, y) = Lib.make_adjoints(NodeVector(x), NodeVector(y))
# backprop_node(A::Adjoints, x::T) where {T <: Node} = T(Lib.backprop_node(A, getpointer(x)))
#
# #####
# ##### Parameters
# #####
#
# const ParameterVector = Union{Lib.ParameterVectorAllocated, Lib.ParameterVectorRef}
# wraptype(::ParameterVector) = IsPointer()
#
# ParameterVector(args...) = ParameterVector(args)
# function ParameterVector(args::Union{Tuple,Vector})
#     p = Lib.ParameterVector()
#     for arg in args
#         Lib.push!(p, getpointer(arg))
#     end
#     return p
# end
#
# Base.length(P::ParameterVector) = Lib._length(P)
# Base.getindex(P::ParameterVector, i) = Node(Lib._getindex(P, convert(Int64, i-1)))
# Base.iterate(P, s = 1) = (s > length(P)) ? nothing : (P[s], s+1)
#
# #####
# ##### NodeWrapper
# #####
#
# const NodeWrapper = Lib.NodeWrapperAllocated
# wraptype(::NodeWrapper) = IsPointer()
#
# Base.length(n::NodeWrapper) = Lib._length(n)
# Base.getindex(n::NodeWrapper, i) = Node(Lib._getindex(n, convert(Int, i-1)))
# Base.iterate(n::NodeWrapper, s = 1) = (s > length(n)) ? nothing : (n[s], s+1)
#
# mutable struct NFunction
#     ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.NFunction,:St10shared_ptrIiE}
#
#     # List of node operations in the function. Represented as a Lib.NodeWrapperAlocated
#     # because this is generated on the C++ side of things and will get cleaned up if we
#     # let it go out of scope.
#     #
#     # TODO: Find a way to make this not happen.
#     ops::Lib.NodeWrapperAllocated
#     callback::Any
#
#     function NFunction(nodes::Lib.NodeVectorAllocated, params::Lib.ParameterVectorAllocated)
#         ptr = Lib.make_function(nodes, params)
#         ops = Lib.get_ordered_ops(ptr)
#         return new(ptr, ops, nothing)
#     end
#
#     function NFunction(ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.NFunction,:St10shared_ptrIiE})
#         ops = Lib.get_ordered_ops(ptr)
#         return new(ptr, ops, nothing)
#     end
# end
# wraptype(::NFunction) = HasPointer()
#
# get_ordered_ops!(f::NFunction) = f.ops = Lib.get_ordered_ops(getpointer(f))
# get_results(f::NFunction) = Lib.get_results(getpointer(f))
# get_parameters(f::NFunction) = Lib.get_parameters(getpointer(f))
# get_temporary_pool_size(f::NFunction) = convert(Int, Lib.get_temporary_pool_size(getpointer(f)))
# get_pmem_pool_size(f::NFunction) = Lib.get_remote_pool_size(getpointer(f))
# get_constants(f::NFunction) = collect(Iterators.filter(x -> description(x) == "Constant", f))
#
# Base.length(f::NFunction) = Lib._length(f.ops)
# Base.getindex(f::NFunction, i) = Node(Lib._getindex(f.ops, convert(Int64, i-1)))
# name(f::NFunction) = Lib.get_name(getpointer(f))
#
# function Base.iterate(f::NFunction)
#     # Make sure everything is ordered
#     get_ordered_ops!(f)
#     s = 1
#     return s <= length(f) ? (f[s], s+1) : nothing
# end
# Base.iterate(f::NFunction, s) = (s <= length(f)) ? (f[s], s+1) : nothing
#
# # Allow reverse iterations
# Base.reverse(f::NFunction) = Iterators.reverse(f)
# Base.iterate(f::Iterators.Reverse{NFunction}, s = length(f.itr)) =
#     (s == 0) ? nothing : (f.itr[s], s-1)
#
# Base.copy(f::NFunction) = NFunction(Lib.clone_function(getpointer(f)))
#
# #####
# ##### Low level handles for dealing with objects
# #####
#
