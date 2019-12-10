# Some tuple utilities
astuple(x::Tuple) = x
astuple(x) = (x,)

untuple(x::Tuple) = x
untuple(x::Tuple{T}) where {T} = first(x)

#####
##### Executable
#####

const ExecutableCxxType = cxxt"std::shared_ptr<ngraph::runtime::Executable>"
mutable struct Executable{T}
    obj::ExecutableCxxType
    ngraph_function::NFunction
    backend::Backend{T}

    function Executable(obj, ngraph_function::NFunction, backend::Backend{T}) where {T}
        ex = new{T}(obj, ngraph_function, backend)

        # Immediately clear this from the saved functions
        #
        # This avoids needing to clean it up later.
        backend_obj = unwrap(backend)
        icxx"$(backend_obj)->remove_compiled_function($obj);"
        return ex
    end
end
unwrap(x::Executable) = x.obj

#reset_counters(exe::Executable) = Lib.reset_counters(getpointer(exe))

# convenience unwrapper
function ng_compile(backend::Backend, inputs, outputs; kw...)
    fn = NFunction(outputs, inputs)
    ng_compile(backend, fn; kw...)
end

function ng_compile(
        backend::Backend,
        ngraph_function::NFunction;
        emit_timing::Bool = false,
        callback = nothing
    )

    #apply_callback!(ngraph_function, callback)
    backend_obj = unwrap(backend)
    function_obj = unwrap(ngraph_function)
    obj = icxx"$(backend_obj)->compile($function_obj, $emit_timing);"

    # Indicate that the compiler has been invoked.
    global __HAVE_COMPILED[] = true
    return Executable(obj, ngraph_function, backend)
end

#apply_callback!(f::NFunction, ::Nothing) = nothing
#function apply_callback!(f::NFunction, cb)
#    CB = @cfunction($(() -> cb(f)), Cvoid, ())
#
#    # Save the callback with the NFunction object to avoid it being garbage collected
#    f.callback = CB
#
#    # Go through the c++ library to attach the callback to the underlying nGraph function
#    Lib.set_jl_callback(getpointer(f), Base.unsafe_convert(Ptr{Cvoid}, CB))
#    @debug Lib.get_jl_callback(getpointer(f))
#end
#
#(ex::Executable)(inputs::Vector{Any}, outputs::Vector{Any}) = Lib.call(getpointer(ex), outputs, inputs)

function (ex::Executable)(inputs, outputs)
    i = convert(cxxt"std::vector<std::shared_ptr<ngraph::runtime::Tensor>>", inputs)
    o = convert(cxxt"std::vector<std::shared_ptr<ngraph::runtime::Tensor>>", outputs)
    ex_obj = unwrap(ex)
    return icxx"$(ex_obj)->call($o, $i);"
end

#####
##### Extract performance data
#####

"""
    get_performance(ex::Executable) -> Dict{String,Int}

Return the runtime in microseconds of each kernel in `ex` as a dictionary keyed by kernel
name.
"""
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
