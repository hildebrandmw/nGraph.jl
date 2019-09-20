include("flux_ops.jl")
include("optimizers.jl")

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

"""
    __inplace(x) -> Nothing

Annotate a node as a parameter requiring inplace updates.
"""
__inplace(x) = nothing

struct SnoopMeta
    parameters::IdDict{Any,Node}
    data::IdDict{Node,AbstractArray}
    # Used for keeping track of the order that Nodes are created for deterministic ordering
    # of inputs and outputs
    primary::Vector{Node}
    secondary::Vector{Node}

    # Markers for inplace nodes.
    inplace_nodes::IdDict{Node,Nothing}
end

SnoopMeta() = SnoopMeta(
    IdDict{Any,Node}(),
    IdDict{Node,AbstractArray}(),
    Node[],
    Node[],
    IdDict{Node,Nothing}(),
)

#####
##### Cassette Overdubs
#####

# Hijack Node constructors from TrackedArrays
function Cassette.overdub(ctx::SnoopCtx, f::Type{Node{T,N}}, x::Flux.Tracker.TrackedArray) where {T,N}
    if haskey(ctx.metadata.parameters, x)
        node = ctx.metadata.parameters[x]::Node{T,N}
    else
        # Don't recurse into the Node Creation since we implicitly turn Arrays into
        # constants, which will happen if we do recurse
        node = f(x)::Node{T,N}
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

# Mark objects as requiring inplace updates.
function Cassette.overdub(ctx::SnoopCtx, ::typeof(__inplace), x)
    node = get(ctx.metadata.parameters, x, nothing)
    if !isnothing(node)
        ctx.metadata.inplace_nodes[node] = nothing
    end
end

# Get around Cassette bug with `reverse`
Cassette.overdub(::SnoopCtx, ::typeof(reverse), args...) = reverse(args...)

# Hijack these layers
Cassette.overdub(ctx::SnoopCtx, f::Flux.Dense, args...) =
    Cassette.overdub(ctx, _dense_impl, f, args...)

Cassette.overdub(ctx::SnoopCtx, f::Flux.Conv, args...) =
    Cassette.overdub(ctx, _conv_impl, f, args...)

Cassette.overdub(ctx::SnoopCtx, f::Flux.BatchNorm, args...) =
    Cassette.overdub(ctx, _batchnorm_impl, f, args...)

# Slurp up constructors to Nodes from standard Arrays to become constants.
Cassette.overdub(ctx::SnoopCtx, f::Type{Node{T,N}}, x::Array{T,N}) where {T,N} = constant(x)

# Skip recursing initialization calls - recursing turns out to take a very, very long time.
Cassette.overdub(ctx::SnoopCtx, f::typeof(rand), args...) = f(args...)
Cassette.overdub(ctx::SnoopCtx, f::typeof(Flux.glorot_normal), args...) = f(args...)
Cassette.overdub(ctx::SnoopCtx, f::typeof(Flux.glorot_uniform), args...) = f(args...)

#####
##### Main `compile` entrypoint
#####

"""
    compile(backend, f, args..; optimizer = Inference()) -> Executable

Trace and compile a Flux model `f` with `args`.
"""
function compile(
        backend::Backend,
        f,
        args...;
        optimizer = Inference(),
        # if !isnothing, should be a callback that returns tensor descriptors that should
        # be assigned into remote memory
        isremote = nothing,
        kw...
    )

    ctx = SnoopCtx(metadata = SnoopMeta())

    # Extract the parameter from all the inputs
    inputs = Node.(args)

    # Perform traced execution on the function.
    outputs = astuple(Cassette.overdub(ctx, f, inputs...))
    @assert all(x -> isa(x, Node), outputs)

    # Get all of the implicit parameters that were instantiated during traced execution.
    params = ctx.metadata.primary
    data = [ctx.metadata.data[p] for p in params]

    println("Found $(length(params)) params")

    # Pack data gathered by the Cassette run into a NamedTuple.
    arg_tuple = (
        inputs = inputs,
        outputs = outputs,
        params = params,
        data = data,
        inplace = ctx.metadata.inplace_nodes,
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
    opt = instantiate(optimizer, backend, isremote, opt_args...)

    # Create tensors for the outputs
    input_tensors = totensor.(Ref(backend), inputs, Ref(isremote))
    for (t,i) in zip(input_tensors, args)
        write(t, i)
    end

    output_tensors = totensor.(Ref(backend), outputs, Ref(isremote))
    secondary_tensors = totensor.(Ref(backend), secondary_outputs, Ref(isremote))
    length(secondary_tensors) == 0 && (secondary_tensors = Tensor[])

    return FluxExecutable(ex, opt, input_tensors, output_tensors, secondary_tensors)
end

totensor(backend, A, ::Nothing) = Tensor(backend, A)
totensor(backend, A, f) = f(A) ? PersistentTensor(backend, A) : Tensor(backend, A)

struct FluxExecutable{B,T,M,N}
    ex::Executable{B}
    optimizer::T
    inputs::NTuple{M,Tensor}
    outputs::NTuple{N,Tensor}
    secondary::Vector{Tensor}
end

splat_inputs(fex::FluxExecutable) = Iterators.flatten((fex.inputs, getinputs(fex.optimizer)))
splat_outputs(fex::FluxExecutable) = Iterators.flatten((fex.outputs, fex.secondary, getoutputs(fex.optimizer)))

function (ex::FluxExecutable)()
    inputs = Any[getpointer(i) for i in splat_inputs(ex)]
    outputs = Any[getpointer(o) for o in splat_outputs(ex)]

    # Since we're passing wrapped type to C++, we have to cast them to Any's which is
    # kind of gross - but w/e
    ex.ex(inputs, outputs)

    # Perform any updates required by the optimizer.
    update!(ex.optimizer)
    return untuple(ex.outputs)
end

