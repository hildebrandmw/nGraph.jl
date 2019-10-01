#####
##### Optimizers
#####

struct Inference
    tensors::Vector{TensorView}
end

function apply!(f, backend::Backend, ::Type{Inference}, trace)
    # Call the higher order `f` on each element of the `data` array
    tensors = TensorView.(Ref(backend), f.(trace.data))
    return Inference(tensors), trace.parameters, ()
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
    params = trace.parameters
    inplace_nodes = trace.inplace

    # Create backprop nodes for each parameters.
    adjoints = make_adjoints(first(trace.outputs), -constant(sgd.learning_rate))

    backprop_nodes = [backprop_node(adjoints, n) for n in params]
    updates = map(zip(params, backprop_nodes)) do x
        n, bn = x

        # Backprop through LSTMs results in things being transposed - which is not helpful.
        # Untranspose them here ...
        if size(n) == reverse(size(bn))
            bn = transpose(bn)
        end

        # TODO: Make this embedding specific instead of just inplace_nodes.
        if haskey(inplace_nodes, n)
            return bn
        else
            return n + bn
        end
    end

    param_tensors = TensorView.(Ref(backend), f.(trace.data))
    update_tensors = map(zip(updates, params, param_tensors, trace.data)) do x
        # Unpack argument
        node = x[1]
        param = x[2]
        param_tensor = x[3]
        datum = x[4]

        # If this is an inplace node - copy over the parameter tensor.
        if haskey(inplace_nodes, param)
            @info "Creating an Inplace Node"
            @assert size(node) == size(param) == size(param_tensor)
            return param_tensor
        else
            # Need to make sure that we create a copy of the underlying data to create
            # a new TensorView - otherwise we'll accidentally alias with an existing
            # view ... unless that's what we want of course :D
            return TensorView(backend, f(copy(datum)))
        end
    end

    state = SGDState(param_tensors, update_tensors)
    return state, params, updates
end

getinputs(S::SGDState) = S.inputs
getoutputs(S::SGDState) = S.outputs

# Swap
update!(S::SGDState) = (S.inputs, S.outputs) = (S.outputs, S.inputs)
