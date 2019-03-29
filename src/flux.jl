# Because julia-vim and YouCompleteMe don't get along
_getsigma(x) = x.σ

function _conv_impl(c::Flux.Conv{N}, x::Node) where {N}
    # We flip standard arrays since nGraph really perform cross-correlation
    if flip_kernel(c.weight)
        n = Node(c.weight)
        flip!(n)
        cn = NNlib.conv(
            x, 
            n;
            stride = reverse(c.stride), 
            pad = reverse(c.pad),
            dilation = reverse(c.dilation))
    else
        cn = NNlib.conv(x, c.weight; stride = c.stride, pad = c.pad, dilation = c.dilation)
    end
    # Broadcast the bias along the first `N` dimensions and the last
    axis_set = [collect(1:N); N+2]
    bb = broadcast(Node(c.bias), size(cn); axes = axis_set)

    node =  _getsigma(c).(cn .+ bb)
    return node
end

# Again, getting around a Cassette issue
_dense_impl(d::Flux.Dense, x::Node) = _getsigma(d).(d.W * x .+ d.b)

# Extend to unwrap tracked arrays
function Node{T,N}(x::Flux.Tracker.TrackedArray{T,N}) where {T,N}
    return Node{T,N}(Lib.op_parameter(Element(T), Shape(size(x))), copy(Flux.data(x)))
end

# Methods defined to avoid ambiguity
Base.:*(x::TrackedArray{T,2}, y::Node{T,1}) where {T} = Node(x) * y
Base.:*(x::TrackedArray{T,2}, y::Node{T,2}) where {T} = Node(x) * y

# Need to flip the convolution kernels
# NOTE: nGraph's "convolution" is NNlib's crosscorrelation
#
# Need to flip the W and H dimensions of the filters
flip!(n::Node) = flip!(n.data)
function flip!(x::AbstractArray{T,N}) where {T,N}
    x .= view(x, size(x, 1):-1:1, size(x, 2):-1:1, ntuple(_->:, N-2)...)
end

flip_kernel(x::AbstractArray) = true
flip_kernel(x) = false

Cassette.@context SnoopCtx

# Hijack Node constructors from TrackedArrays
function Cassette.overdub(ctx::SnoopCtx, f::Type{Node{T,N}}, x::Flux.Tracker.TrackedArray) where {T,N}
    return get!(ctx.metadata, x, Cassette.recurse(ctx, f, x))::Node{T,N}
end

# Get around Cassette bug with `reverse`
Cassette.overdub(::SnoopCtx, ::typeof(reverse), args...) = reverse(args...)

# Hijack these layers
Cassette.overdub(ctx::SnoopCtx, f::Flux.Dense, args...) = 
    Cassette.overdub(ctx, _dense_impl, f, args...)

Cassette.overdub(ctx::SnoopCtx, f::Flux.Conv, args...) = 
    Cassette.overdub(ctx, _conv_impl, f, args...)


"""
    compile(backend, f, args..; optimizer = Inference()) -> Executable

Trace and compile a Flux model `f` with `args`.
"""
function compile(backend::Backend, f, args...; optimizer = Inference())
    ctx = SnoopCtx(metadata = IdDict{Any,Node}())

    # Extract the parameter from all the inputs
    inputs = Node.(args)

    # Perform traced execution on the function.
    outputs = astuple(Cassette.overdub(ctx, f, inputs...))
    @assert all(x -> isa(x, Node), outputs)

    # Get all of the implicit parameters that were instantiated during traced execution.
    params = collect(values(ctx.metadata))

    arg_tuple = (
        inputs = inputs,
        outputs = outputs,
        params = params,
        _id = ctx.metadata,
    )
    opt, opt_inputs, opt_outputs = create(optimizer, backend, arg_tuple)

    # Compile the executable
    ex = compile(
        backend, 
        ParameterVector(inputs..., opt_inputs...),
        NodeVector(outputs..., opt_outputs...)
    )

    # TODO: Need to actually iterate over the result vector since nGraph converts
    # the NodeVector to a ResultVector

    # Create tensors for the outputs
    tensors = map(x -> Tensor(backend, x), outputs) 

    return FluxExecutable(ex, opt, tensors)
end

struct FluxExecutable{T,V}
    ex::Executable
    optimizer::T
    outputs::V
end

function recompile(backend::Backend, fex::FluxExecutable)
    # All of the tensors and nodes we've defined earlier ... should still be valid?
    #   -- question - what if we change the state of one of the inputs?
    #  
    # TODO: I think I need to work on the persistent memory implementation in nGraph to
    # allow basically anything to be allocated in persistent memory.
    ex = recompile(backend, fex.ex)
    return FluxExecutable(ex, fex.optimizer, fex.outputs)
end

function (ex::FluxExecutable)(args...)
    inputs = Any[i.ptr for i in Iterators.flatten((args, getinputs(ex.optimizer)))]
    outputs = Any[o.ptr for o in Iterators.flatten((ex.outputs, getoutputs(ex.optimizer)))]
    
    # Since we're passing wrapped type to C++, we have to cast them to Any's
    ex.ex(inputs, outputs)

    update!(ex.optimizer)

    return untuple(ex.outputs)
end

#####
##### Optimizers
#####

create(f::Any, backend, args::NamedTuple) = create(f, backend, args.inputs, args.outputs, args.params)

# Inference 'Optimizer'
struct Inference end

struct InferenceState
    tensors::Vector
end

function create(::Inference, backend, inputs, outputs, params)
    I = InferenceState(map(x -> Tensor(backend, x), params))
    return I, params, ()
end

getinputs(I::InferenceState) = I.tensors
getoutputs(I::InferenceState) = ()
update!(I::InferenceState) = nothing

# Just get the Gradients
struct Gradient 
    params::Vector
    gradients::Vector
    _id::IdDict
end

function create(::Type{Gradient}, backend, args::NamedTuple)
    # Unwrap everything and hijack the ID Dict mapping Tracked Arrays to nodes so
    # we can create a new ID Dict mapping Tracked Arrays to Tensors.
    #
    # This will let us compare the gradients calculated by nGraph and those calculated by
    # Flux.
    inputs = args.inputs
    outputs = args.outputs
    params = args.params

    # Map params nodes back to their parent TrackedArray
    @info "Reversing ID Dict"
    id_rev = IdDict()
    for (k,v) in args._id
        id_rev[v] = k
    end

    # Create a backprop node for each parameter
    @info "Inserting Backprop Nodes"
    adjoints = Adjoints(first(outputs), constant(Float32(1)))
    gradients = [backprop_node(adjoints, n) for n in params]

    # Create tensors for the parameters and gradients
    param_tensors = Tensor[] 
    gradient_tensors = Tensor[]
    grad_map = IdDict()
    @info "Creating Tensors"
    for n in params
        pt = Tensor(backend, n)
        gt = Tensor(backend, n)
        
        grad_map[id_rev[n]] = (pt, gt)

        push!(param_tensors, pt)
        push!(gradient_tensors, gt)
    end
    G = Gradient(param_tensors, gradient_tensors, grad_map)

    return G, params, gradients
end

getinputs(G::Gradient) = G.params
getoutputs(G::Gradient) = G.gradients
update!(::Gradient) = nothing

# Standard SGD
struct SGD{T <: Number}
    learning_rate::T
end

mutable struct SGDState
    inputs::Vector
    outputs::Vector
    #backprops::Vector
    #delta::Any
    #output::Any
end

function create(sgd::SGD, backend, inputs, outputs, params)

    # Create a backprop node for each parameter
    delta = -constant(sgd.learning_rate) .* first(outputs)
    adjoints = Adjoints(first(outputs), delta)

    backprop_nodes = [backprop_node(adjoints, n) for n in params]
    updates = [n + bn for (n, bn) in zip(params, backprop_nodes)]

    # Create tensors for the parameters and gradients
    param_tensors = map(x -> Tensor(backend, x), params)
    update_tensors = map(x -> Tensor(backend, x), updates)
    #backprop_tensors = map(x -> Tensor(backend, x), backprop_nodes)
    #delta_tensor = Tensor(backend, delta)
    #output_tensor = Tensor(backend, first(outputs))

    S = SGDState(
        param_tensors, 
        update_tensors, 
        #backprop_tensors, 
        #delta_tensor, 
        #output_tensor
    )

    return (
        S, 
        params,
        updates
        #Iterators.flatten((updates, backprop_nodes, (delta, first(outputs))))
    )
end

getinputs(S::SGDState) = S.inputs
#getoutputs(S::SGDState) = Iterators.flatten((S.outputs, S.backprops, (S.delta, S.output)))
getoutputs(S::SGDState) = S.outputs

# Swap
function update!(S::SGDState) 
    # @show S.delta
    # @show typeof(S.delta)
    # @show S.output
    # @show typeof(S.output)

    # # Print out the average update amount
    # for b in S.backprops
    #     println(sum(abs, collect(b)))
    # end
    # println()
    # for (i,o) in zip(S.inputs, S.outputs)
    #     println(sum(abs, collect(i)))
    #     println(sum(abs, collect(o)))
    # end

    (S.inputs, S.outputs) = (S.outputs, S.inputs)
end
