# This is kind of a gross way of mapping Julia types to ngraph types.
# TODO: Think of a better way of doing this.
_element(::Type{Bool}) = Lib.gettype("boolean")
_element(::Type{Float32}) = Lib.gettype("f32")
_element(::Type{Float64}) = Lib.gettype("f64")
_element(::Type{Int64}) = Lib.gettype("i64")
_element(::Type{T}) where {T} = error("No translation defined for $T")

#####
##### Node
#####

struct Node{N}
    # CxxWrap std::shared_ptr to the actual backing node
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.Node,:St10shared_ptrIiE}
    op::String
end

__getindex(shape, ::Val{N}) where {N} = (shape[N], __getindex(shape, Val{N-1}())...)
__getindex(shape, ::Val{-1}) = ()

function Base.size(n::Node{N}) where {N}
    shape = Lib.get_output_shape(n.ptr, zero(UInt64))
    @assert N == length(shape)

    return (__getindex(shape, Val{N-1}())...,)
end
Base.axes(n::Node) = map(Base.OneTo, size(n))

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
end

Tensor{T}(::UndefInitializer, backend, inds::Vararg{Int,N}) where {T,N} = Tensor{T,N}(Lib.create_tensor(backend, _element(T), _shape(inds)))

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
    Lib.tensor_read(t.ptr, Ptr{Cvoid}(pointer(x)), sizeof(T) * UInt64(i-1), UInt64(sizeof(x)))
    return first(x)
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
