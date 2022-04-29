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
#
# nGraph Ordered Types shadowed
_reversed(x) = Int64[Int64(i) for i in reverse(x)]
shape(x) = _reversed(x)
strides(x) = _reversed(x)

#shape(::Tuple{}) = Int64[]
#strides(::Tuple{}) = Int64[]

# It is convenient to construct Shape, Strides etc. from Tuples, Numbers, Vectors etc.
# Here, is the machinery that does the dispatch.
_expand(N, x) = fill(Int(x), N)

# Numbers
shape(N, x::Number) = _expand(N, x)
strides(N, x::Number) = _expand(N, x)

# Tuples and Vectors
maybecollect(x::Vector) = x
maybecollect(x) = collect(x)

function ordered(N, x)
    # Determine how many repetitions we need
    repitions, remainder = divrem(length(x), N)
    if !iszero(remainder)
        error("The length of `x` must be divisible by $N")
    end

    return _reversed(repeat(maybecollect(x), repitions))
end
shape(N, x::Union{Tuple,Vector}) = ordered(N, x)
strides(N, x::Union{Tuple,Vector}) = ordered(N, x)

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

back(x) = TYPEMAPS[NGElements(@ngraphcall Type_t(x[]))]
ngraph_convert(::Type{T}) where {T <: Number} = Element(T)[]

#####
##### Nodes
#####

const NodeCppType = Lib.CxxWrap.StdLib.SharedPtrAllocated{nGraph.Lib.Node}
Base.ndims(x::NodeCppType) = length(@ngraphcall get_output_shape(x, 0))
Base.eltype(x::NodeCppType) = back(@ngraphcall get_output_element_type(x, 0))

# Subtype AbstractArray to get all of the nice fallbacks.
struct Node{T,N} <: AbstractArray{T,N}
    obj::NodeCppType
end
ngraph_convert(x::Node) = x.obj

Base.show(io::IO, x::Node{T,N}) where {T,N} = print(io, "Node{$T,$N} - $(name(x))")
Base.display(x::Node) = show(stdout, x)

Node{T}(x::NodeCppType) where {T} = Node{T,ndims(x)}(x)
Node(x::NodeCppType) = Node{eltype(x),ndims(x)}(x)

Node(x::AbstractArray{T,N}) where {T,N} = Node{T,N}(x)
Node{T,N}(x::AbstractArray{T,N}) where {T,N} = parameter(T, size(x))

Node(x::T) where {T <: Number} = Node{T}(x)
Node{T}(x::T) where {T <: Number} = Node{T,0}(fill(x))

# Array style arguments
Base.ndims(x::Node) = ndims(x.obj)

function Base.size(x::Node)
    nd = ndims(x)
    dims = @ngraphcall get_output_shape(x, 0)
    # Reversing is done on the ngraph side.
    return Tuple(Int.(reverse(dims)))
end

Base.size(x::Node, i::Integer) = size(x)[i]
Base.length(x) = prod(size(x))
Base.eltype(x::Node{T}) where {T} = T
Base.IndexStyle(::Node) = Base.IndexLinear()

name(x::Node) = String(@ngraphcall get_name(x))
description(x::Node) = String(@ngraphcall description(x))

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
ngraph_convert(x::NGFunction) = x.ptr

function NGFunction(parameters::Vector, results::Vector)
    ptr = make_function(parameters, results)
    return NGFunction(ptr)
end

function make_function(inputs::Vector, outputs::Vector)
    return @ngraphcall make_function(outputs, inputs)
end

name(f::NGFunction) = String(@ngraphcall get_name(f))
poolsize(f::NGFunction) = Int(@ngraphcall get_temporary_pool_size(f))

#####
##### Backend
#####

struct Backend{T}
    ptr::Lib.CxxWrap.StdLib.SharedPtrAllocated{nGraph.Lib.Backend}
end
ngraph_convert(x::Backend) = x.ptr

#Backend(str::String = "CPU") = Backend(Lib.create(str))
Backend{T}() where {T} = Backend{T}(@ngraphcall create(string(T)))
version(x::Backend) = String(Lib.get_version(ngraph_convert(x)))

#####
##### Tensor
#####

struct Tensor
    ptr::Lib.CxxWrap.StdLib.SharedPtrAllocated{nGraph.Lib.RuntimeTensor}
    backend::Backend

    # Keep track of the base array to avoid GC
    # TODO: Rethink the whole - "casting to pointer" thing.
    parent::AbstractArray

    function Tensor(backend::Backend, v::Array{T}) where {T}
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
ngraph_convert(x::Tensor) = x.ptr

Base.eltype(x::Tensor) = back(Lib.get_element_type(ngraph_convert(x)))
Base.sizeof(x::Tensor) = convert(Int, Lib.get_size_in_bytes(ngraph_convert(x)))

Tensor(backend, x::Tensor) = x

# Capture scalars in a 0-dimensional array
function Tensor(backend, x::T) where {T}
    A = Array{T,0}(undef)
    A[] = x
    return Tensor(backend, A)
end

# If we're trying to create a tensor view from a node
# create an undefined array and return a view of that.
function Tensor(backend::Backend, v::Node{T,N}) where {T,N}
    A = Array{T}(undef, size(v))
    return Tensor(backend, A)
end

function Base.size(t::Tensor)
    shape = Lib.get_shape(ngraph_convert(t))
    return Tuple(reverse(shape))
end

Base.parent(t::Tensor) = t.parent

function Base.show(io::IO, TV::Tensor)
    printstyled(io, "Tensor View\n"; color = :yellow)
    println(io, parent(TV))
end

#####
##### Executable
#####

mutable struct Executable
    ptr::Lib.CxxWrap.StdLib.SharedPtrAllocated{nGraph.Lib.Executable}
    ngraph_function::NGFunction
    backend::Backend

    function Executable(ptr, ngraph_function::NGFunction, backend::Backend)
        ex = new(ptr, ngraph_function, backend)

        # On cleanup - remove the compiled function.
        finalizer(ex) do x
            Lib.remove_compiled_function(x.backend.ptr, x.ptr)
        end
        return ex
    end
end
ngraph_convert(ex::Executable) = ex.ptr

function (ex::Executable)(
        inputs::Vector{Tensor},
        outputs::Vector{Tensor}
    )
    return @ngraphcall call(ex, outputs, inputs)
end

function compile(backend::Backend, inputs::Vector, outputs::Vector; kw...)
    # Gather up the raw shared pointers
    fn = NGFunction(inputs, outputs)
    compile(backend::Backend, fn; kw...)
end

function compile(
        backend::Backend,
        ngraph_function::NGFunction;
        emit_timing::Bool = false,
    )

    @time pointer = Lib.compile(backend.ptr, ngraph_function.ptr, emit_timing)

    # Indicate that the compiler has been invoked.
    return Executable(pointer, ngraph_function, backend)
end

# Conversions for vectors of Nodes and Tensors
ngraph_convert(x::Vector{<:Node}) = Lib.CxxWrap.CxxRef.(ngraph_convert.(x))
ngraph_convert(x::Vector{<:Tensor}) = Lib.CxxWrap.CxxRef.(ngraph_convert.(x))

#####
##### Extract performance data
#####

# TODO: Fix This
# function get_performance(ex::Executable)
#     # Construct a PerfCounterTranslator
#     translator = Lib.PerfCounterTranslator(getpointer(ex))
#
#     # Create a dictionary of timing results. Iterate through the CounterTranslator to
#     # construct the dict.
#     times = Dict{String,Int}()
#     for i in 1:Lib._length(translator)
#         name, time = Lib._getindex(translator, i-1)
#         times[name] = convert(Int, time)
#     end
#     return times
# end


