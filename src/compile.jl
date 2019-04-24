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
        #finalizer(_cleanup, ex)
        return ex
    end
end
wraptype(::Executable) = HasPointer()

# Free resources held by the executable
function _cleanup(ex::Executable)
    # Don't try to remove an executable more than once if it was cleaned up elsewhere
    if ex.iscleared == false
        ex.iscleared = true
        Lib.remove_compiled_function(getpointer(ex.backend), getpointer(ex))
    end
end


function compile(backend::Backend, inputs::ParameterVector, outputs::NodeVector)
    ngraph_function = NFunction(outputs, inputs)
    pointer = Lib.compile(getpointer(backend), getpointer(ngraph_function), false)

    # Get the post-compiled ops for the function.
    get_ordered_ops!(ngraph_function)
    return Executable(pointer, ngraph_function, backend)
end

function (ex::Executable)(inputs::Vector{Any}, outputs::Vector{Any}) 
    ex.iscleared && error("Executable is cleared")
    Lib.call(getpointer(ex), outputs, inputs)
end

function recompile(ex::Executable)
    backend = ex.backend

    # Make sure that this executable is not called again, because that would be
    # very wrong. The executable should go out of scope and be cleared up by the 
    # finalizer.
    _cleanup(ex)

    # If this executable was holding onto some large buffers, call the GC now.
    GC.gc()

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
