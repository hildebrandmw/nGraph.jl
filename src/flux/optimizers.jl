#####
##### Optimizers
#####

struct Inference
    tensors::Vector{TensorView}
end

function apply!(f, backend::Backend, ::Type{Inference}, trace)
    # Wrap all of the parameters in a TensorView.
    tensors = [TensorView(backend, trace.node_to_param[p]) for p in trace.parameter_nodes]
    return Inference(tensors), trace.parameter_nodes, ()
end

getinputs(I::Inference) = I.tensors
getoutputs(I::Inference) = ()
update!(I::Inference) = nothing

#####
##### Just get the Gradients
#####

struct Gradient
    params::Vector
    gradients::Vector
    _id::IdDict
end

function create(::Type{Gradient}, args::NamedTuple)
    # Unwrap everything and hijack the ID Dict mapping Tracked Arrays to nodes so
    # we can create a new ID Dict mapping Tracked Arrays to Tensors.
    #
    # This will let us compare the gradients calculated by nGraph and those calculated by
    # Flux.
    inputs = args.inputs
    outputs = args.outputs
    params = args.params
    data = args.data

    @show params
    # Map params nodes back to their parent TrackedArray
    @info "Reversing ID Dict"
    id_rev = IdDict()
    for (k,v) in args._id
        id_rev[v] = k
    end

    # Create a backprop node for each parameter
    @info "Inserting Backprop Nodes"
    adjoints = make_adjoints(first(outputs), constant(one(Float32)))
    gradients = [backprop_node(adjoints, n) for n in params]

    args = (
        data = data,
        params = params,
        gradients = gradients,
        id_rev = id_rev,
    )

    return args, params, gradients
end

getinputs(G::Gradient) = G.params
getoutputs(G::Gradient) = G.gradients
update!(::Gradient) = nothing

function instantiate(::Type{Gradient}, backend, ::Nothing, data, params, gradients, id_rev)
    param_tensors = Tensor[]
    gradient_tensors = Tensor[]
    grad_map = IdDict()
    for (d, n, g) in zip(data, params, gradients)
        @assert size(n) == size(g)
        @assert size(d) == size(n)

        pt = Tensor(backend, n)
        write(pt, d)
        gt = Tensor(backend, g)

        grad_map[id_rev[n]] = (pt, gt)

        push!(param_tensors, pt)
        push!(gradient_tensors, gt)
    end
    return Gradient(param_tensors, gradient_tensors, grad_map)
end

#####
##### Standard SGD
#####

struct SGD{T <: Number}
    learning_rate::T
end

mutable struct SGDState
    inputs::Vector{TensorView}
    outputs::Vector{TensorView}
end

function apply!(f, backend::Backend, sgd::SGD, trace)
    # Unpack `trace` arguments
    parameter_nodes = trace.parameter_nodes

    # Create backprop nodes for each parameters.
    adjoints = make_adjoints(first(trace.outputs), -constant(sgd.learning_rate))

    backprop_nodes = [backprop_node(adjoints, n) for n in parameter_nodes]
    update_nodes = map(zip(parameter_nodes, backprop_nodes)) do x
        n, bn = x
        return n + bn
    end

    # Wrap the input parameters in a TensorView
    parameter_tensors = map(parameter_nodes) do node
        data = trace.node_to_param[node]
        return TensorView(backend, f(data))
    end

    # Copy the input arrays to achieve double buffering.
    update_tensors = map(parameter_nodes) do node
        data = trace.node_to_param[node]
        return TensorView(backend, f(copy(data)))
    end

    state = SGDState(parameter_tensors, update_tensors)
    return state, parameter_nodes, update_nodes
end

getinputs(S::SGDState) = S.inputs
getoutputs(S::SGDState) = S.outputs

# Swap
update!(S::SGDState) = (S.inputs, S.outputs) = (S.outputs, S.inputs)
