_print(args...) = println("Well, this is something")

function show_stuff(f::NFunction)
    println("Hello")
    for op in f
        if Lib.can_select_algo(getpointer(op))
            algos = UInt32[]
            timings = Float32[]
            memories = UInt64[]
            Lib.get_algo_options(getpointer(op), algos, timings, memories)

            @show algos
            @show timings
            @show Int.(memories)

            println("Workspace Size: ", convert(Int, first(memories)))

            # Try settng an algorithm, lets see if we can get codegen working
            Lib.set_algo(getpointer(op), convert(UInt, first(algos)), convert(UInt, first(memories)))
        end
    end
end

# Methods for testing the new GPU code callback mechanism
function gpu_callback_test()
    backend = nGraph.Backend("GPU")
    fex, args = nGraph.test_model(backend; callback = show_stuff, emit_timing = true)
    return fex
end
