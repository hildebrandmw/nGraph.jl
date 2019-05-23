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

    function Executable(ptr, ngraph_function::NFunction, backend::Backend)
        ex = new(ptr, ngraph_function, backend)

        # Immediately clear this from the saved functions
        #
        # This avoids needing to clean it up later.
        Lib.remove_compiled_function(getpointer(ex.backend), getpointer(ex))

        return ex
    end
end
wraptype(::Executable) = HasPointer()

function compile(backend::Backend, inputs::ParameterVector, outputs::NodeVector)
    ngraph_function = NFunction(outputs, inputs)
    pointer = Lib.compile(getpointer(backend), getpointer(ngraph_function), false)

    # Get the post-compiled ops for the function.
    get_ordered_ops!(ngraph_function)
    return Executable(pointer, ngraph_function, backend)
end

function (ex::Executable)(inputs::Vector{Any}, outputs::Vector{Any}) 
    Lib.call(getpointer(ex), outputs, inputs)
end

"""
    recompile(ex::Executable)

WARNING: Don't call ANY previous executables after calling this function.
"""
function recompile(ex::Executable)
    backend = ex.backend

    # Assume we're working in the same directory as the "cpu_codegen" directory.
    #
    # This is a brittle assumption, but lets work with it for now.
    ispath("./cpu_codegen") && rm("./cpu_codegen"; recursive = true) 

    pointer = withenv("NGRAPH_PASS_HACK" => true) do
        Lib.compile(getpointer(backend), getpointer(ex.ngraph_function), false)
    end
    get_ordered_ops!(ex.ngraph_function)

    return Executable(pointer, ex.ngraph_function, backend)
end
