# Some tuple utilities
astuple(x::Tuple) = x
astuple(x) = (x,)

untuple(x::Tuple) = x
untuple(x::Tuple{T}) where {T} = first(x)

#####
##### Executable
#####

struct Executable
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.Executable,:St10shared_ptrIiE}
    ngraph_function::NFunction
end

function compile(backend::Backend, inputs::ParameterVector, outputs::NodeVector)
    ngraph_function = NFunction(outputs, inputs)
    ptr = Lib.compile(backend.ptr, ngraph_function.ptr, false)
    return Executable(ptr, ngraph_function)
end

(ex::Executable)(inputs::Vector{Any}, outputs::Vector{Any}) = Lib.call(ex.ptr, outputs, inputs)
(ex::Executable)(inputs::TensorWrapper, outputs::TensorWrapper) = Lib.call(ex.ptr, outputs, inputs)
