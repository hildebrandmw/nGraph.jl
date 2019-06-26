_print(args...) = println("Well, this is something")

function show_stuff(f::NFunction)
    println("Hello")
    for op in f
        println(name(op))
    end
end

# Methods for testing the new GPU code callback mechanism
function gpu_callback_test()
    fex, args = nGraph.test_model(nGraph.Backend("GPU"); callback = show_stuff)
    return fex
end
