# Experiments for pulling out the individual kernels from a function and timing the runtime
# performance of each of them.
function runner(ex::FluxExecutable)
    # Executables have their function attached to them.
    #
    # Since nGraph functions are basically just a bunch of node operations, first ensure
    # the the node list is up to date in the function, then begin iterating through the
    # kernels.
    #
    # There will be some operations to skip (such as "Parameter", "Result" etc.)
    func = ex.ex.ngraph_function
    get_ordered_ops!(func)
    backend = Backend()

    skiplist = (
        "Parameter",
        "Result",
    )

    total_time = 0.0

    # Keep Track of nodes and sizes that we've seen so far
    node_times = Dict{Tuple,Float64}()

    progress_meter = ProgressMeter.Progress(length(func), 1)
    for i in 1:length(func)
        # Get a node type from the function
        node = func[i]
        ProgressMeter.next!(progress_meter)

        # Skip List
        in(description(node), skiplist) && continue

        # Create a new set of inputs
        ninputs = get_input_size(node)
        backend = Backend()

        tensors = map(1:get_input_size(node)) do j
            T = get_input_element_type(node, j)
            sz = get_input_shape(node, j)
            return Tensor(backend, rand(T, sz...))
        end

        sizes = Tuple(size.(tensors))
        key = (description(node), sizes)
        if false
        #if haskey(node_times, key)
            total_time += node_times[key]
        else
            # Create input parameters from the new tensors
            node_version_of_tensors = Node.(tensors) 
            parameters = ParameterVector(node_version_of_tensors...)

            # Copy the node in question, provide it with the new inputs
            output = copy(node, NodeVector(node_version_of_tensors))

            # Construct a node vector from the output
            nodes = NodeVector(output)

            # Compile a new function
            executable = compile(backend, parameters, nodes)

            # Time New function execution
            inputs = TensorWrapper(tensors) 
            outputs = TensorWrapper([Tensor(backend, output)])
           
            executable(inputs, outputs)

            time = gettime(executable, inputs, outputs)
            total_time += time

            # Log this node as already seen
            node_times[key] = time
        end

    end
    @show total_time

end

function gettime(executable, inputs, outputs)
    runtime = Second(3)
    iterations = 1000

    mintime = typemax(Float64)
    starttime = now()
    while now() < starttime + runtime
        time = @elapsed for _ in 1:iterations
            executable(inputs, outputs)
        end
        mintime = min(mintime, time / iterations)
    end
    return mintime
end
