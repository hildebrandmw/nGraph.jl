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

    # Get the post-compiled ops for the function.
    get_ordered_ops!(ngraph_function)
    return Executable(ptr, ngraph_function)
end

(ex::Executable)(inputs::Vector{Any}, outputs::Vector{Any}) = Lib.call(ex.ptr, outputs, inputs)

function recompile(backend::Backend, ex::Executable)
    # Delete the executable from the backend
    Lib.remove_compiled_function(backend.ptr, ex.ptr)
    # Assume we're working in the same directory as the "cpu_codegen" directory.
    #
    # This is a brittle assumption, but lets work with it for now.
    rm("./cpu_codegen"; recursive = true) 
    # Recompile the function 
    ptr = Lib.compile(backend.ptr, ex.ngraph_function.ptr, false)
    get_ordered_ops!(ex.ngraph_function)
    return Executable(ptr, ex.ngraph_function)
end
