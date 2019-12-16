# NOTE: Don't hold a reference to the backend nGraph types because Cxx likes to destroy them
# when exiting Julia - which causes a segfault.
const NAMEMAP = (
    Bool => "boolean",
    Float32 => "f32",
    Float64 => "f64",
    Int32 => "i32",
    Int64 => "i64",
)

for (t,s) in NAMEMAP
    str = "ngraph::element::Type_t::$s;"
    @eval function ngraph_type(::Type{$t})
        enum = @icxx_str($str)
        return @cxx ngraph::element::Type(enum)
    end
end
ngraph_type(x) = ngraph_type(typeof(x))
Element(x) = ngraph_type(x)

# map the c++ enum back to the Julia type
const BACK = Dict{UInt,DataType}(
    2 => Bool,
    5 => Float32,
    6 => Float64,
    9 => Int32,
    10 => Int64,
)

@noinline function julia_type(x)
    enum = icxx"$(x).get_type_enum();"
    return BACK[enum.val]
end

#####
##### Misc Types
#####

function AxisSet(itr, ndims)
    s = icxx"ngraph::AxisSet();"
    for i in itr
        # Remember to reverse dimensions
        # This also performs the conversion to Index Zero
        v = ndims - i
        icxx"$s.insert($v);"
    end
    return s
end

function AxisVector(itr, ndims)
    x = icxx"ngraph::AxisVector();"
    for i in reverse(itr)
        v = ndims - i
        icxx"$x.push_back($v);"
    end
    return x
end

function CoordinateDiff(itr)
    x = icxx"ngraph::CoordinateDiff();"
    for i in itr
        icxx"$x.push_back($i);"
    end
    return x
end

function NodeVector(itr)
    x = icxx"ngraph::NodeVector();"
    for i in itr
        v = unwrap(i)
        icxx"$x.push_back($v);"
    end
    return x
end

function ParameterVector(itr)
    x = icxx"ngraph::ParameterVector();"
    for i in itr
        if description(i) != "Parameter"
            throw(ArgumentError("Arguments to Parameter Vector must be Parameters"))
        end
        v = unwrap(i)
        icxx"$x.push_back(std::dynamic_pointer_cast<ngraph::op::Parameter>($v));"
    end
    return x
end

function Shape(itr = ())
    x = icxx"ngraph::Shape(0);"

    # Need to reverse in order because C++ is row-major while Julia is column-major.
    for i in reverse(itr)
        icxx"$(x).push_back($i);"
    end
    return x
end

function Strides(itr)
    x = icxx"ngraph::Strides();"
    for i in itr
        icxx"$(x).push_back($i);"
    end
    return x
end

#####
##### Node
#####

# Could make this an AbstractArray - but I think I'll try not doing that ...
const NodeCppType = cxxt"std::shared_ptr<ngraph::Node>"
Base.ndims(x::NodeCppType) = icxx"$(x)->get_shape().size();"

# Define a typed and untyped version of the same thing.
struct Node
    obj::NodeCppType
end

# Subtype AbstractArray to get all of the nice fallbacks.
struct NodeTyped{T,N} <: AbstractArray{T,N}
    obj::NodeCppType
end
Base.show(io::IO, x::NodeTyped{T,N}) where {T,N} = println(io, "NodeTyped{$T, $N} - $(name(x))")
Base.display(x::NodeTyped) = show(stdout, x)
NodeTyped{T}(obj::NodeCppType) where {T} = NodeTyped{T,ndims(obj)}(obj)

const NodeLike = Union{Node, <:NodeTyped}
unwrap(x::NodeLike) = x.obj

# Conversions between the two
Node(x::NodeTyped) = Node(unwrap(x))
function NodeTyped(x::Node)
    N = ndims(x)
    et = icxx"""$(x.obj)->get_element_type();"""
    T = julia_type(et)
    return NodeTyped{T,N}(unwrap(x))
end
NodeTyped(x::NodeCppType) = NodeTyped(Node(x))

# Construction from Julia objects
Node(x::AbstractArray{T}) where {T} = Node(@op Parameter(Element(T), Shape(size(x))))

function NodeTyped{T,N}(x::AbstractArray{T,N}) where {T,N}
    return NodeTyped{T,N}(@op Parameter(Element(T), Shape(size(x))))
end
NodeTyped{T}(x::AbstractArray{T,N}) where {T,N} = NodeTyped{T,N}(x)
NodeTyped(x::AbstractArray{T,N}) where {T,N} = NodeTyped{T,N}(x)

NodeTyped(x::T) where {T <: Number} = NodeTyped{T,0}(x)
function NodeTyped{T,0}(x::T) where {T <: Number}
    v = Array{T}(undef)
    v[] = x
    return NodeTyped(v)
end

# Array style arguments
Base.ndims(x::Node) = convert(Int, ndims(unwrap(x)))
Base.ndims(::NodeTyped{T,N}) where {T,N} = N

function Base.size(x::NodeLike)
    nd = ndims(x)
    shape = icxx"$(x.obj)->get_shape();"
    dims = ntuple(i -> convert(UInt, icxx"$(shape).at($(i-1));"), nd)
    return convert.(Int, reverse(dims))
end
Base.size(x::NodeLike, i::Integer) = size(x)[i]
Base.length(x) = prod(size(x))

Base.eltype(x::Node) = julia_type(icxx"$(x.obj)->get_element_type();")
Base.eltype(x::NodeTyped{T}) where {T} = T

Base.IndexStyle(::NodeLike) = Base.IndexLinear()
Base.axes(x::NodeLike) = map(Base.OneTo, size(x))

name(x::NodeLike) = convert(String, icxx"$(x.obj)->get_name();")
description(x::NodeLike) = convert(String, icxx"$(x.obj)->description();")

# So these can be used as keys in a Dict
Base.:(==)(x::T, y::T) where {T <: NodeLike} = name(x) == name(y)
Base.hash(x::NodeLike, h::UInt = UInt(0x209348)) = hash(name(x), h)

numinputs(x::NodeLike) = convert(Int, icxx"$(x.obj)->get_input_size();")
function getinputnode(x::T, i) where {T <: NodeLike}
    node = icxx"""
        auto i = $(x.obj)->get_inputs().at($(i-1)).get_output();
        x.get_node();
        """
    return T(node)
end
getinputnodes(x::NodeLike) = [getinput(i) for i in 1:numinputs(x)]

numoutputs(x::NodeLike) = convert(Int, icxx"$(x.obj)->get_output_size();")
# function getoutputnodes(x::T, i) where {T <: NodeLike}
#     i = convert(Int, i-1)
#     return T.collect(icxx"$(x.obj)->get_output_nodes
# end

@deprecate get_input_size numinputs

function output(x::NodeLike, i)
    shared_ptr = icxx"$(x.obj)->get_output_tensor_ptr($(i-1));"
    return TensorDescriptor(shared_ptr)
end
outputs(x::NodeLike) = [output(x, i) for i in 1:numoutputs(x)]

function input(x::NodeLike, i)
    shared_ptr = icxx"$(x.obj)->get_inputs().at($(i-1)).get_output().get_tensor_ptr();"
    return TensorDescriptor(shared_ptr)
end
inputs(x::NodeLike) = [input(x, i) for i in 1:numoutputs(x)]

#####
##### TensorDescriptor
#####

const TensorDescriptorCppType = cxxt"std::shared_ptr<ngraph::descriptor::Tensor>"

struct TensorDescriptor
    obj::TensorDescriptorCppType
end

name(x::TensorDescriptor) = convert(String, icxx"$(x.obj)->get_name();")
Base.:(==)(a::TensorDescriptor, b::TensorDescriptor) = name(a) == name(b)
Base.hash(x::TensorDescriptor, h::UInt = UInt(0x20983)) = hash(x, h)

getpool(x::TensorDescriptor) = convert(Int, icxx"$(x.obj)->get_pool_number();")
function setpool!(x::TensorDescriptor, pool::Integer)
    pool = convert(UInt, pool)
    icxx"$(x.obj)->set_pool_number($pool);"
    return nothing
end

Base.sizeof(x::TensorDescriptor) = convert(Int, icxx"$(x.obj)->size();")
function Base.size(x::TensorDescriptor)
    shape = icxx"$(x.obj)->get_shape();"
    len = icxx"$(shape).size();"
    dims = ntuple(i -> convert(UInt, icxx"$(shape).at($(i-1));"), len)
    return convert.(Int, reverse(dims))
end

#####
##### Backend
#####

const BackendType = cxxt"std::shared_ptr<ngraph::runtime::Backend>"

struct Backend{T <: AbstractBackendType}
    obj::BackendType

    function nGraph.Backend{T}() where {T <: AbstractBackendType}
        obj = icxx"ngraph::runtime::Backend::create($(string(T)));"
        return new{T}(obj)
    end
end
unwrap(x::Backend) = x.obj

#####
##### TensorView
#####

const TensorViewCppType = cxxt"std::shared_ptr<ngraph::runtime::Tensor>"

struct TensorView
    obj::TensorViewCppType
    backend::Backend
    parent::AbstractArray

    function TensorView(backend::Backend{C}, v::AbstractArray{T,N}) where {C,T,N}

        # This is kind of scary - we ... just have to make sure that the parent array
        # doesn't get moved (i.e. resized ... )
        vptr = Base.unsafe_convert(Ptr{Cvoid}, pointer(v))
        backend_obj = unwrap(backend)
        element = Element(T)
        shape = Shape(size(v))

        obj = icxx"$(backend_obj)->create_tensor($element, $shape, $vptr);"
        return new(obj, backend, v)
    end
end

function TensorView(backend, x::T) where {T <: Number}
    v = Array{T}(undef)
    v[] = x
    return TensorView(backend, v)
end
unwrap(x::TensorView) = x.obj

Base.parent(x::TensorView) = x.parent
Base.collect(x::TensorView) = collect(parent(x))

#####
##### nGraph Function
#####

const nGraphFunctionCxxType = cxxt"std::shared_ptr<ngraph::Function>"
mutable struct NFunction
    obj::nGraphFunctionCxxType
    callback::Any

    function NFunction(nodes, parameters)
        obj = icxx"std::make_shared<ngraph::Function>($nodes, $parameters);"
        return new(obj, nothing)
    end
end
unwrap(x::NFunction) = x.obj

