

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

# TODO: ngraph makes the dictinction between training and inference. For now, we will 
# assume training, but eventually I can add a parameter to SnoopCtx that will determine
# if we're training or inferring and pass that information here.
function _batchnorm_impl(BN::Flux.BatchNorm, x::Node)
    # Create the batchnorm op and then do activation.
    γ = Node(BN.γ) 
    β = Node(BN.β)
    
    n = batchnorm_training(x, γ, β, BN.ϵ)

    # The batchnorm_training op in ngraph returns a tuple
    # (normalized, gamma, beta). We apply the activation function to the normalized output.
    #
    # We also have call `get_output_element` on the other two outputs so that downstream
    # graph rewriting in ngraph works correctly.
    #
    # Also note that we have to `__register` the nodes so they become hidden outputs of the 
    # compiled ngraph graph. Otherwide, things break horribly
    a = get_output_element(n, 1)
    __register(get_output_element(n, 2))
    __register(get_output_element(n, 3))

    return BN.λ.(a)
end


# Extend to unwrap tracked arrays
function Node{T,N}(x::Flux.Tracker.TrackedArray{T,N}) where {T,N}
    return Node{T,N}(Lib.op_parameter(Element(T), Shape(size(x))), Flux.data(x))
end

# Methods defined to avoid method ambiguity in julia's dispatch
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

#####
##### Cassette Magic
#####

Cassette.@context SnoopCtx

"""
    __register(x::Node) -> Nothing

By default, becomes a no-op. When executed under [`SnoopCtx`](@ref), registers `x` as a 
hidden output of a ngraph function.
"""
__register(x::Node) = nothing

struct SnoopMeta
    id::IdDict{Any,Node}
    hidden_outputs::Vector{Node}
end
SnoopMeta() = SnoopMeta(IdDict{Any,Node}(), Node[])

# Hijack Node constructors from TrackedArrays
function Cassette.overdub(ctx::SnoopCtx, f::Type{Node{T,N}}, x::Flux.Tracker.TrackedArray) where {T,N}
    return get!(ctx.metadata.id, x, Cassette.recurse(ctx, f, x))::Node{T,N}
end

Cassette.overdub(ctx::SnoopCtx, ::typeof(__register), x::Node) = 
    push!(ctx.metadata.hidden_outputs, x)

# Get around Cassette bug with `reverse`
Cassette.overdub(::SnoopCtx, ::typeof(reverse), args...) = reverse(args...)

# Hijack these layers
Cassette.overdub(ctx::SnoopCtx, f::Flux.Dense, args...) = 
    Cassette.overdub(ctx, _dense_impl, f, args...)

Cassette.overdub(ctx::SnoopCtx, f::Flux.Conv, args...) = 
    Cassette.overdub(ctx, _conv_impl, f, args...)

Cassette.overdub(ctx::SnoopCtx, f::Flux.BatchNorm, args...) = 
    Cassette.overdub(ctx, _batchnorm_impl, f, args...)


compile(f, args...; kw...) = compile(Backend(), f, args...; kw...)
"""
    compile(backend, f, args..; optimizer = Inference()) -> Executable

Trace and compile a Flux model `f` with `args`.
"""
function compile(backend::Backend, f, args...; optimizer = Inference())
    ctx = SnoopCtx(metadata = SnoopMeta())

    # Extract the parameter from all the inputs
    inputs = Node.(args)

    # Perform traced execution on the function.
    outputs = astuple(Cassette.overdub(ctx, f, inputs...))
    @assert all(x -> isa(x, Node), outputs)

    # Get all of the implicit parameters that were instantiated during traced execution.
    params = collect(values(ctx.metadata.id))
    println("Found $(length(params)) params")

    arg_tuple = (
        inputs = inputs,
        outputs = outputs,
        params = params,
        _id = ctx.metadata.id,
    )
    opt, opt_inputs, opt_outputs = create(optimizer, backend, arg_tuple)

    # Compile the executable
    hidden_outputs = ctx.metadata.hidden_outputs
    ex = compile(
        backend, 
        ParameterVector(inputs..., opt_inputs...),
        NodeVector(outputs..., hidden_outputs..., opt_outputs...)
    )

    # Create tensors for the outputs
    tensors = map(x -> Tensor(backend, x), outputs) 
    hidden = map(x -> Tensor(backend, x), hidden_outputs)

    return FluxExecutable(ex, opt, tensors, hidden)
end

struct FluxExecutable{T,V,H}
    ex::Executable
    optimizer::T
    outputs::V
    hidden_outputs::H
end

recompile(fex::FluxExecutable) = 
    FluxExecutable(recompile(fex.ex), fex.optimizer, fex.outputs, fex.hidden_outputs)

function (ex::FluxExecutable)(args...)
    inputs = Any[getpointer(i) for i in Iterators.flatten((args, getinputs(ex.optimizer)))]
    outputs = Any[getpointer(o) for o in Iterators.flatten((ex.outputs, ex.hidden_outputs, getoutputs(ex.optimizer)))]
    
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

    S = SGDState(
        param_tensors, 
        update_tensors, 
    )

    return (
        S, 
        params,
        updates
    )
end

getinputs(S::SGDState) = S.inputs
getoutputs(S::SGDState) = S.outputs

# Swap
update!(S::SGDState) = (S.inputs, S.outputs) = (S.outputs, S.inputs)
