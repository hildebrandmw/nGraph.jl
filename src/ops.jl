_shape(sz::NTuple{N, Int64}) where {N} = Lib.makeshape(collect(UInt64.(sz)))
_shape(::Tuple{}) = Lib.makeshape(UInt64[])

# This is kind of a gross way of mapping Julia types to ngraph types.
# TODO: Think of a better way of doing this.
_element(::Type{Float32}) = Lib.gettype("f32")

struct NGraphParam{T}
    val::T
end

param(x::Array{T}) where {T} = Lib.op_parameter(_element(T), _shape(size(x)))
param(x::T) where {T} = Lib.op_parameter(_element(T), _shape(()))

const Node = Lib.CxxWrap.SmartPointerWithDeref{Lib.Node}

Base.:+(a::Node, b::Node) = Lib.op_add(a, b)
Base.:*(a::Node, b::Node) = Lib.op_mul(a, b)
