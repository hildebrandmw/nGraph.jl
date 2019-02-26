struct Executable
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.Executable,:St10shared_ptrIiE}
end

function compile(backend, f, inputs)
    # Wrap all of the inputs/outputs as parameters
    #
    # Call the function "f" with the wrapped parameters, this will trace an execution and
    # construct the nGraph graph in the background. Outputs should be a tuple of nodes
    wrapped_inputs = param.(inputs)
    outputs = f(wrapped_inputs...) 

    # Make sure we only get "nodes" as outputs
    @assert all(x -> isa(x, Node), outputs)

    # Create the formal ngraph function
    ngraph_function = Lib.make_function(nodes(outputs...), parameters(wrapped_inputs...))

    # Compile the executable
    ex = Lib.compile(backend, ngraph_function, false)

    return Executable(ex)
end

# TODO: Trouble with broadcasting:
# https://github.com/FluxML/Zygote.jl/issues/73
function trainable(backend, f, inputs)
    wrapped_inputs = param.(inputs)

    # Get the gradient of this
    y, back = Zygote.forward(f, wrapped_inputs...)
    grads = back(y)

    ngraph_function = Lib.make_function(nodes(y..., grads...), parameters(wrapped_inputs...))
    ex = Lib.compile(backend, ngraph_function, false)

    return Executable(ex)
end

(ex::Executable)(outputs, inputs) = Lib.call(ex.ptr, Any[o.ptr for o in outputs], Any[i.ptr for i in inputs])
