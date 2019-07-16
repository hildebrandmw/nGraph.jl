# Some tuple utilities
astuple(x::Tuple) = x
astuple(x) = (x,)

untuple(x::Tuple) = x
untuple(x::Tuple{T}) where {T} = first(x)

#####
##### Executable
#####

mutable struct Executable{T}
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.Executable,:St10shared_ptrIiE}
    ngraph_function::NFunction
    backend::Backend{T}

    function Executable(ptr, ngraph_function::NFunction, backend::Backend{T}) where {T}
        ex = new{T}(ptr, ngraph_function, backend)

        # Immediately clear this from the saved functions
        #
        # This avoids needing to clean it up later.
        Lib.remove_compiled_function(getpointer(ex.backend), getpointer(ex))

        return ex
    end
end
wraptype(::Executable) = HasPointer()

compile(backend::Backend, inputs::ParameterVector, outputs::NodeVector; kw...) = 
    compile(backend::Backend, NFunction(outputs, inputs); kw...)

function compile(backend::Backend, ngraph_function::NFunction; emit_timing::Bool = false, callback = nothing)
    apply_callback!(ngraph_function, callback)
    pointer = Lib.compile(getpointer(backend), getpointer(ngraph_function), emit_timing)

    # Get the post-compiled ops for the function.
    get_ordered_ops!(ngraph_function)

    # Indicate that the compiler has been invoked.
    global __HAVE_COMPILED[] = true
    return Executable(pointer, ngraph_function, backend)
end

apply_callback!(f::NFunction, ::Nothing) = nothing
function apply_callback!(f::NFunction, cb) 
    CB = @cfunction($(() -> cb(f)), Cvoid, ())

    # Save the callback with the NFunction object to avoid it being garbage collected
    f.callback = CB 

    Lib.set_jl_callback(getpointer(f), Base.unsafe_convert(Ptr{Cvoid}, CB))
    @info Lib.get_jl_callback(getpointer(f))
end

function (ex::Executable)(inputs::Vector{Any}, outputs::Vector{Any}) 
    Lib.call(getpointer(ex), outputs, inputs)
end

"""
    recompile(ex::Executable)

WARNING: Don't call ANY previous executables after calling this function.
"""
function recompile(ex::Executable, fn = ex.ngraph_function)
    backend = ex.backend

    # Assume we're working in the same directory as the "cpu_codegen" directory.
    #
    # This is a brittle assumption, but lets work with it for now.
    ispath("./cpu_codegen") && rm("./cpu_codegen"; recursive = true) 

    pointer = withenv("NGRAPH_PASS_HACK" => true) do
        Lib.compile(getpointer(backend), getpointer(fn), false)
    end
    return Executable(pointer, fn, backend)
end

#####
##### Extract performance data
#####

"""
    get_performance(ex::Executable) -> Dict{String,Int}

Return the runtime in microseconds of each kernel in `ex` as a dictionary keyed by kernel 
name.
"""
function get_performance(ex::Executable)
    # Construct a PerfCounterTranslator
    translator = Lib.PerfCounterTranslator(getpointer(ex))  

    # Create a dictionary of timing results. Iterate through the CounterTranslator to
    # construct the dict.
    times = Dict{String,Int}() 
    for i in 1:Lib._length(translator)
        name, time = Lib._getindex(translator, i-1)
        times[name] = convert(Int, time)
    end
    return times
end
