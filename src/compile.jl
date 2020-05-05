#####
##### Executable
#####

mutable struct Executable
    ptr::Lib.CxxWrap.StdLib.SharedPtrAllocated{nGraph.Lib.Executable}
    ngraph_function::NGFunction
    backend::Backend

    function Executable(ptr, ngraph_function::NGFunction, backend::Backend)
        ex = new(ptr, ngraph_function, backend)

        # Immediately clear this from the saved functions
        #
        # This avoids needing to clean it up later.
        Lib.remove_compiled_function(ex.backend.ptr, ex.ptr)
        return ex
    end
end

# convenience unwrapper
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

    pointer = Lib.compile(backend.ptr, ngraph_function.ptr, emit_timing)

    # Indicate that the compiler has been invoked.
    return Executable(pointer, ngraph_function, backend)
end

function (ex::Executable)(
        inputs::Vector{TensorView},
        outputs::Vector{TensorView}
    )

    # Convert these vectors into shared_pointer references
    inputs = Lib.CxxWrap.CxxRef.(unwrap.(inputs))
    outputs = Lib.CxxWrap.CxxRef.(unwrap.(outputs))

    return Lib.call(ex.ptr, outputs, inputs)
end

#####
##### Extract performance data
#####

# TODO: Fix This
#
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
