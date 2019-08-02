# Because julia-vim and YouCompleteMe don't get along
_getsigma(x) = x.σ

function _conv_impl(c::Flux.Conv{N}, x::Node) where {N}
    # We flip standard arrays since nGraph really perform cross-correlation
    n = Node(c.weight)
    __flip(n)
    cn = NNlib.conv(
        x,
        n;
        stride = reverse(c.stride),
        pad = reverse(c.pad),
        dilation = reverse(c.dilation))

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
Node{T,N}(x::Flux.Tracker.TrackedArray{T,N}) where {T,N} = Node{T,N}(x.data)

# Methods defined to avoid method ambiguity in julia's dispatch
Base.:*(x::TrackedArray{T,2}, y::Node{T,1}) where {T} = Node(x) * y
Base.:*(x::TrackedArray{T,2}, y::Node{T,2}) where {T} = Node(x) * y

# Need to flip the convolution kernels
# NOTE: nGraph's "convolution" is NNlib's crosscorrelation
#
# Need to flip the W and H dimensions of the filters
function flip!(x::AbstractArray{T,N}) where {T,N}
    x .= view(x, size(x, 1):-1:1, size(x, 2):-1:1, ntuple(_->:, N-2)...)
end

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

"""
    __flip(x::Node) -> Nothing

By default, becomes a no-op. When executed under [`SnoopCtx`](@ref), registers that the
data for `x` should be flipped
"""
__flip(x::Node) = nothing

struct SnoopMeta
    parameters::IdDict{Any,Node}
    data::IdDict{Node,AbstractArray}
    # Used for keeping track of the order that Nodes are created for deterministic ordering
    # of inputs and outputs
    primary::Vector{Node}
    secondary::Vector{Node}
end
SnoopMeta() = SnoopMeta(IdDict{Any,Node}(), IdDict{Node,AbstractArray}(), Node[], Node[])

# Hijack Node constructors from TrackedArrays
function Cassette.overdub(ctx::SnoopCtx, f::Type{Node{T,N}}, x::Flux.Tracker.TrackedArray) where {T,N}
    if haskey(ctx.metadata.parameters, x)
        node = ctx.metadata.parameters[x]::Node{T,N}
    else
        node = Cassette.recurse(ctx, f, x)::Node{T,N}
        ctx.metadata.parameters[x] = node
        push!(ctx.metadata.primary, node)
    end

    # Save the data in the tracked array for constructing tensors later.
    get!(ctx.metadata.data, node, copy(x.data))
    return node
end

Cassette.overdub(ctx::SnoopCtx, ::typeof(__register), x::Node) =
    push!(ctx.metadata.secondary, x)

# If this was not an implicitly created kernel, it won't have a registered entry in 
# ctx.metadata.data, so do nothing
Cassette.overdub(ctx::SnoopCtx, ::typeof(__flip), x::Node) = haskey(ctx.metadata.data, x) && flip!(ctx.metadata.data[x])

# Get around Cassette bug with `reverse`
Cassette.overdub(::SnoopCtx, ::typeof(reverse), args...) = reverse(args...)

# Hijack these layers
Cassette.overdub(ctx::SnoopCtx, f::Flux.Dense, args...) =
    Cassette.overdub(ctx, _dense_impl, f, args...)

Cassette.overdub(ctx::SnoopCtx, f::Flux.Conv, args...) =
    Cassette.overdub(ctx, _conv_impl, f, args...)

Cassette.overdub(ctx::SnoopCtx, f::Flux.BatchNorm, args...) =
    Cassette.overdub(ctx, _batchnorm_impl, f, args...)

# Skip recursing initialization calls - recursing turns out to take a very, very long time.
Cassette.overdub(ctx::SnoopCtx, f::typeof(Flux.glorot_normal), args...) = f(args...)
Cassette.overdub(ctx::SnoopCtx, f::typeof(Flux.glorot_uniform), args...) = f(args...)

"""
    compile(backend, f, args..; optimizer = Inference()) -> Executable

Trace and compile a Flux model `f` with `args`.
"""
function compile(backend::Backend, f, args...; optimizer = Inference(), kw...)
    ctx = SnoopCtx(metadata = SnoopMeta())

    # Extract the parameter from all the inputs
    inputs = Node.(args)

    # Perform traced execution on the function.
    outputs = astuple(Cassette.overdub(ctx, f, inputs...))
    @assert all(x -> isa(x, Node), outputs)

    # Get all of the implicit parameters that were instantiated during traced execution.
    #params = collect(values(ctx.metadata.parameters))
    params = ctx.metadata.primary
    data = [ctx.metadata.data[p] for p in params]

    println("Found $(length(params)) params")

    arg_tuple = (
        inputs = inputs,
        outputs = outputs,
        params = params,
        data = data,
        _id = ctx.metadata.parameters,
    )
    opt_args, opt_inputs, opt_outputs = create(optimizer, arg_tuple)

    # Compile the executable
    secondary_outputs = ctx.metadata.secondary
    ex = compile(
        backend,
        ParameterVector(inputs..., opt_inputs...),
        NodeVector(outputs..., secondary_outputs..., opt_outputs...);
        kw...
    )

    # Instantiate the optimizer struct AFTER compilation to provide more memory for
    # profiling
    opt = instantiate(optimizer, backend, opt_args...)

    # Create tensors for the outputs
    input_tensors = map(x -> Tensor(backend, x), args)
    output_tensors = map(x -> Tensor(backend, x), outputs)
    secondary_tensors = map(x -> Tensor(backend, x), secondary_outputs)
    length(secondary_tensors) == 0 && (secondary_tensors = Tensor[])

    return FluxExecutable(ex, opt, input_tensors, output_tensors, secondary_tensors)
end

struct FluxExecutable{B,T,M,N}
    ex::Executable{B}
    optimizer::T
    inputs::NTuple{M,Tensor}
    outputs::NTuple{N,Tensor}
    secondary::Vector{Tensor}
end

recompile(fex::FluxExecutable) = FluxExecutable(
    recompile(fex.ex),
    fex.optimizer,
    fex.inputs,
    fex.outputs,
    fex.secondary,
)

_splat_inputs(fex::FluxExecutable) = Iterators.flatten((fex.inputs, getinputs(fex.optimizer)))
_splat_outputs(fex::FluxExecutable) = Iterators.flatten((fex.outputs, fex.secondary, getoutputs(fex.optimizer)))

function (ex::FluxExecutable)()
    inputs = Any[getpointer(i) for i in _splat_inputs(ex)]
    outputs = Any[getpointer(o) for o in _splat_outputs(ex)]

    # Since we're passing wrapped type to C++, we have to cast them to Any's
    ex.ex(inputs, outputs)

    update!(ex.optimizer)
    return untuple(ex.outputs)
end

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

InferenceState(backend, v::Vector) = InferenceState(Tensor.(Ref(backend), v))

create(::Inference, inputs, outputs, params, data) = (data,), params, ()
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
instantiate(::SGD, backend, args...) = SGDState(backend, args...)

mutable struct SGDState
    inputs::Vector{Tensor}
    outputs::Vector{Tensor}
end

function SGDState(backend::Backend, inputs::Vector, outputs::Vector)
    return SGDState(
        map(x -> Tensor(backend, x), inputs),
        map(x -> Tensor(backend, x), outputs)
    )
end

function create(sgd::SGD, inputs, outputs, params, data)
    # Create a backprop node for each parameter
    adjoints = Adjoints(first(outputs), -constant(sgd.learning_rate))

    backprop_nodes = [backprop_node(adjoints, n) for n in params]
    updates = map(zip(params, backprop_nodes)) do x
        n, bn = x

        # Backprop through LSTMs results in things being transposed - which is helpful.
        # Untranspose them here ...
        if size(n) == reverse(size(bn))
            bn = transpose(bn)
        end

        return n + bn
    end

    # Create tensors for the parameters and gradients
    param_tensors = data
    update_tensors = data

    args = (param_tensors, update_tensors)
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
