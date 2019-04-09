# Some tuple utilities
astuple(x::Tuple) = x
astuple(x) = (x,)

untuple(x::Tuple) = x
untuple(x::Tuple{T}) where {T} = first(x)

#####
##### Executable
#####

mutable struct Executable
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.Executable,:St10shared_ptrIiE}
    ngraph_function::NFunction
    backend::Backend
    iscleared::Bool

    function Executable(ptr, ngraph_function::NFunction, backend::Backend)
        ex = new(ptr, ngraph_function, backend, false)
        return ex
    end
end

function compile(backend::Backend, inputs::ParameterVector, outputs::NodeVector)
    ngraph_function = NFunction(outputs, inputs)
    ptr = Lib.compile(backend.ptr, ngraph_function.ptr, false)

    # Get the post-compiled ops for the function.
    get_ordered_ops!(ngraph_function)
    return Executable(ptr, ngraph_function, backend)
end

function (ex::Executable)(inputs::Vector{Any}, outputs::Vector{Any}) 
    ex.iscleared && error("Executable is cleared")
    Lib.call(ex.ptr, outputs, inputs)
end

function recompile(backend::Backend, ex::Executable)
    # Delete the executable from the backend
    Lib.remove_compiled_function(backend.ptr, ex.ptr)

    # Make sure that this executable is not called again, because that would be
    # very wrong
    ex.iscleared = true

    # If this executable was holding onto some large buffers, call the GC now.
    GC.gc()

    # Assume we're working in the same directory as the "cpu_codegen" directory.
    #
    # This is a brittle assumption, but lets work with it for now.
    ispath("./cpu_codegen") && rm("./cpu_codegen"; recursive = true) 

    ptr = withenv("NGRAPH_PASS_HACK" => true) do
        Lib.compile(backend.ptr, ex.ngraph_function.ptr, false)
    end
    get_ordered_ops!(ex.ngraph_function)

    return Executable(ptr, ex.ngraph_function, backend)
end
