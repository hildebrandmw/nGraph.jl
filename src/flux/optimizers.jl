#####
##### Optimizers
#####

create(f::Any, args::NamedTuple) = create(f, args.inputs, args.outputs, args.params, args.data)

# Inference 'Optimizer'
struct Inference end
instantiate(::Inference, args...) = InferenceState(args...)

struct InferenceState
    tensors::Vector{Tensor}
end

InferenceState(backend, x::Any, v::Vector) = InferenceState(Tensor.(Ref(backend), v))

create(::Inference, inputs, outputs, params, data) = (data,), params, ()
getinputs(I::InferenceState) = I.tensors
getoutputs(I::InferenceState) = ()
update!(I::InferenceState) = nothing

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
instantiate(::SGD, backend, args...) = SGDState(backend, args...)

mutable struct SGDState
    inputs::Vector{Tensor}
    outputs::Vector{Tensor}
    input_descriptors::Vector{TensorDescriptor}
    output_descriptors::Vector{TensorDescriptor}
end

function SGDState(
        backend::Backend, 
        isremote, 
        param_nodes, 
        param_data::Vector, 
        update_nodes,
        update_data::Vector,
        inplace_nodes
    )

    param_tensors = totensor.(Ref(backend), param_nodes, Ref(isremote))
    for (t, i) in zip(param_tensors, param_data)
        write(t, i)
    end

    # println("Enumerating inplace nodes")
    # for n in keys(inplace_nodes)
    #     println(name(n))
    # end
    # println()
    # println("Enumerating Param Nodes")

    update_tensors = map(zip(update_nodes, param_nodes, param_tensors)) do x
        # Unpack argument
        node = x[1]
        param = x[2]
        param_tensor = x[3]
        # println(name(param))

        # If this is an inplace node - copy over the parameter tensor.
        if haskey(inplace_nodes, param)
            @info "Creating an Inplace Node"
            @assert size(node) == size(param) == size(param_tensor)
            return param_tensor
        end

        return totensor(backend, node, isremote)
    end

    for (t, i) in zip(update_tensors, update_data)
        write(t, i)
    end

    return SGDState(
        param_tensors,
        update_tensors,
        first.(outputs.(param_nodes)),
        first.(outputs.(update_nodes)),
    )
end

function create(sgd::SGD, nt::NamedTuple)
    # Unpace argument.
    inputs = nt.inputs
    outputs = nt.outputs
    params = nt.params
    data = nt.data
    inplace_nodes = nt.inplace

    # Create a backprop node for each parameter
    adjoints = make_adjoints(first(outputs), -constant(sgd.learning_rate))

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

    # Create tensors for the parameters and gradients
    param_tensors = data
    update_tensors = data

    args = (params, param_tensors, updates, update_tensors, inplace_nodes)
    return (
        args,
        params,
        updates
    )
end

getinputs(S::SGDState) = S.inputs
getoutputs(S::SGDState) = S.outputs

# Swap
update!(S::SGDState) = (S.inputs, S.outputs) = (S.outputs, S.inputs)
