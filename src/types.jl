# This is kind of a gross way of mapping Julia types to ngraph types.
# TODO: Think of a better way of doing this.
const TYPEMAPS = (
    Bool => "boolean",
    Float32 => "f32",
    Float64 => "f64",
    Int64 => "i64",
)

_element(::Type{T}) where {T} = error("No translation defined for $T")
for (T,S) in TYPEMAPS
    @eval _element(::Type{$T}) = Lib.gettype($S)
end

# This is horrible :(
# I could at least move this to a compile time auto-generator
function _back_element(x)
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
##### Node
#####

struct Node{T,N} <: AbstractArray{T, N}
    # CxxWrap std::shared_ptr to the actual backing node
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.Node,:St10shared_ptrIiE}
    op::String
    # Optional data if the node was created from an array
    data::AbstractArray
end

Node{T,N}(ptr, op) where {T,N} = Node{T,N}(ptr, op, T[])
function Node(x::AbstractArray{T,N}) where {T,N}
    Node{T,N}(Lib.op_parameter(_element(T), _shape(size(x))), "Param", copy(x))
end

Node(x::Node) = x

Base.getindex(n::Node{T,N}, inds...) where {T,N} = zero(T)
function Base.size(n::Node{T,N}) where {T,N}
    shape = Lib.get_output_shape(n.ptr, zero(UInt64))
    @assert N == length(shape)

    return (__getindex(shape, Val{N-1}())...,)
end

__getindex(shape, ::Val{N}) where {N} = (shape[N], __getindex(shape, Val{N-1}())...)
__getindex(shape, ::Val{-1}) = ()

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

Base.iterate(n::Node) = (n, nothing)
Base.iterate(n::Node, ::Nothing) = nothing
Base.length(n::Node) = 1

#####
##### Tensor
#####

struct Tensor{T,N} <: AbstractArray{T,N}
    ptr::CxxWrap.SmartPointerWithDeref{nGraph.Lib.Tensor,:St10shared_ptrIiE}

    function Tensor{T}(::UndefInitializer, backend, inds::Vararg{Int,N}) where {T,N} 
        shape = _shape(inds)
        element = _element(T)
        ptr = Lib.create_tensor(backend, element, shape)

        return new{T,N}(ptr)
    end

    function Tensor(backend, param::Node{T,N}) where {T,N}
        shape = _shape(size(param))
        element = Lib.get_output_element_type(param.ptr, UInt(0))
        ptr = Lib.create_tensor(backend, _element(T), shape)

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

    return (__getindex(shape, Val{N-1}())...,)
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
##### Parameters
#####

function parameters(args::Node...)
    p = Lib.ParameterVector()
    for arg in args
        Lib.push!(p, arg.ptr)
    end
    return p
end

##### 
##### Nodes
#####

function nodes(args::Node...)
    p = Lib.NodeVector()
    for arg in args
        Lib.push!(p, arg.ptr)
    end
    return p
end
