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
    compile(backend, f, args..; optimizer = Inference) -> Executable

Trace and compile a Flux model `f` with `args`.
"""
function compile(
        backend::Backend,
        f,
        args...;
        optimizer = Inference,
        # if !isnothing, should be a callback that returns tensor descriptors that should
        # be assigned into remote memory
        mutator = identity,
        kw...
    )

    # Build the nGraph computation graph
    trace = snoop(f, args...)

    # build the optimizer and get the implicit input and output parameters
    opt, opt_inputs, opt_outputs = apply!(mutator, backend, optimizer, trace)

    # pass the trace as well as the optimizer arguments to create the executable.
    return make(backend, trace, opt, opt_inputs, opt_outputs, mutator; kw...)
end

"""
Traced parameters from a compiled function: `f(args...)`.

* `inputs`: The nodes that are explicitly used as an input to `f`. Corresponds to
    `args...` in the signature of `f`.
* `outputs`: The nodes that represent the results of `f(args...)`.
* `implicit_outputs::Vector{Node}`: Outputs that were implicitly created during tracing -
    usually the result of BatchNorms being wierd.
* `parameters::Vector{Node}`: Vector of implicit parameters to the function. One is created
    for each `TrackedArray` that is converted to a `Node` during the traced execution of `f`.
* `data::Vector{Array}`: The array data that makes up each `parameter` node. These items are
    index related to the `parameters` field - so, for example, the array at index `2`
    corresponds to the parameter at index `2`.
* `args::Vector`: The original arguments to the function.
* `inplace::IdDict{Node,Nothing}`: Nodes that have been annotated as `inplace`. Kind of an
    awkward interface - essentially an `IdSet`.
* `_id::IdDict{Any,Node}`: The `IdDict` corresponding `TrackedArrays` to `Nodes`.
"""
struct Trace
    inputs::Vector{Node}
    outputs::Vector{Node}
    implicit_outputs::Vector{Node}
    parameters::Vector{Node}
    data::Vector
    args::Vector
    inplace::IdDict{Node,Nothing}
    _id::IdDict{Any,Node}
end

# Step 1 of compilation - trace the provided function with the provided arguments to
# construct an nGraph graph - apply the requested optimizer to finish graph construction.
#
# Returns an argument named tuple that is given to the next stage of compilation.
function snoop(f, args...)
        #optimizer = Inference,
        #isremote = nothing
    ctx = SnoopCtx(metadata = SnoopMeta())

    # Extract the parameter from all the inputs
    inputs = collect(Node.(args))

    # Perform traced execution on the function.
    outputs = collect(astuple(Cassette.overdub(ctx, f, inputs...)))
    @assert all(x -> isa(x, Node), outputs)

    # Get all of the implicit parameters that were instantiated during traced execution.
    params = ctx.metadata.primary
    data = [ctx.metadata.data[p] for p in params]

    println("Found $(length(params)) params")

    return Trace(
        inputs,
        outputs,
        ctx.metadata.secondary,
        params,
        data,
        collect(args),
        ctx.metadata.inplace_nodes,
        ctx.metadata.parameters,
    )
end

function make(backend::Backend, trace, opt, opt_inputs, opt_outputs, mutator; kw...)
    # Create an nGraph Executable
    ex = compile(
        backend,
        ParameterVector(trace.inputs..., opt_inputs...),
        NodeVector(trace.outputs..., trace.implicit_outputs..., opt_outputs...);
        kw...
    )

    # Create TensorViews for each of the inputs and outputs
    input_tensors = Tuple(TensorView.(Ref(backend), mutator.(trace.args)))
    output_tensors = Tuple(TensorView.(Ref(backend), mutator.(trace.outputs)))
    secondary_tensors = TensorView.(Ref(backend), mutator.(trace.implicit_outputs))

    return FluxExecutable(ex, opt, input_tensors, output_tensors, secondary_tensors)
end

struct FluxExecutable{B,T,M,N}
    ex::Executable{B}
    optimizer::T
    inputs::NTuple{M,TensorView}
    outputs::NTuple{N,TensorView}
    secondary::Vector{TensorView}
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

