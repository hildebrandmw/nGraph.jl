include("flux_ops.jl")
include("optimizers.jl")

#####
##### Cassette Magic
#####

Cassette.@context SnoopCtx

"""
    __register(x::NodeTyped) -> Nothing

By default, becomes a no-op. When executed under [`SnoopCtx`](@ref), registers `x` as a
hidden output of a ngraph function.
"""
__register(x::NodeTyped) = nothing

struct SnoopMeta
    # The Julia containers for the parameters of this network
    parameters::IdDict{Any,Nothing}

    # Mapping of Parameter to nGraph Node
    param_to_node::IdDict{Any,Node}
    node_to_param::IdDict{Node,Any}

    # Used for keeping track of the order that Nodes are created for deterministic ordering
    # of inputs and outputs
    primary::Vector{Node}
    secondary::Vector{Node}
end

SnoopMeta() = SnoopMeta(
    IdDict{Any,Nothing}(),
    IdDict{Any,Node}(),
    IdDict{Node,Any}(),
    Node[],
    Node[],
)

function getnode!(S::SnoopMeta, x)
    node = get(S.param_to_node, x, nothing)
    if isnothing(node)
        node = f(x)::Node

        # Update internal data structures
        ctx.metadata.param_to_node[x] = node
        ctx.metadata.node_to_param[node] = x
        push!(ctx.metadata.primary, node)
    end

    return NodeTyped(node)
end

#####
##### Cassette Overdubs
#####

# Hijack Node constructors from Arrays
function Cassette.overdub(ctx::SnoopCtx, f::Type{NodeTyped{T,N}}, x::AbstractArray) where {T,N}
    # Check if this array is a parameter - if so,make it an nGraph input.
    # Otherwise, make it a constant.
    if haskey(ctx.metadata.parameters, x)
        # Check if there is an entry for this parameter already.
        # If so, just return that.
        #
        # Otherwise, we have to create one.
        return getnode!(ctx.metadata, x)::NodeTyped{T,N}
    end

    # Otherwise, make this a constant
    return constant(x)
end

# Do not hijack creating nodes from nodes.
Cassette.overdub(ctx::SnoopCtx, f::Type{<:NodeTyped}, x::NodeTyped) = f(x)

Cassette.overdub(ctx::SnoopCtx, ::typeof(__register), x::NodeTyped) =
    push!(ctx.metadata.secondary, x)

# Get around Cassette bug with `reverse`
Cassette.overdub(::SnoopCtx, ::typeof(reverse), args...) = reverse(args...)

# Hijack these layers
Cassette.overdub(ctx::SnoopCtx, f::Flux.Dense, args...) =
    Cassette.overdub(ctx, _dense_impl, f, args...)

Cassette.overdub(ctx::SnoopCtx, f::Flux.Conv, args...) =
    Cassette.overdub(ctx, _conv_impl, f, args...)

Cassette.overdub(ctx::SnoopCtx, f::Flux.CrossCor, args...) =
    Cassette.overdub(ctx, _conv_impl, f, args...)

Cassette.overdub(ctx::SnoopCtx, f::Flux.BatchNorm, args...) =
    Cassette.overdub(ctx, _batchnorm_impl, f, args...)

# The implementation of Conv reshapes the bias before the broadcasting addition.
# So, lets hijack reshape to test if one of our parameters is being passed as an argument
# and convert it to a Node right away.
function Cassette.overdub(ctx::SnoopCtx, f::reshape, x::AbstractArray, args...)
    if haskey(ctx.metadata.parameters, x)
        return Cassette.overdub(ctx, f, getnode!(ctx.metadata, x), args...)
    else
        return Cassette.recurse(ctx, f, x, args...)
    end
end

# Don't hijack nodes passed into `reshape`
function Cassette.overdub(ctx::SnoopCtx, f::reshape, x::NodeTyped, args...)
    return Cassette.recurse(ctx, f, x, args...)
end

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


"""
struct Trace
    inputs::Vector{Node}
    outputs::Vector{Node}
    implicit_outputs::Vector{Node}

    # Record of what is a parameter
    parameters::IdDict{Any,Nothing}
    parameter_nodes::Vector{Node}

    param_to_node::IdDict{Any,Node}
    node_to_param::IdDict{Node,Any}

    args::Vector
end

# Step 1 of compilation - trace the provided function with the provided arguments to
# construct an nGraph graph - apply the requested optimizer to finish graph construction.
#
# Returns an argument named tuple that is given to the next stage of compilation.
function snoop(f, args...)
    ctx = SnoopCtx(metadata = SnoopMeta())

    # Record the parameters
    flux_parameters = Flux.params(f)
    for p in Flux.params(f)
        ctx.metadata.parameters[p] = nothing
    end

    # Extract the inputs
    inputs = collect(NodeTyped.(args))

    # Perform traced execution on the function.
    outputs = collect(astuple(Cassette.overdub(ctx, f, inputs...)))
    @assert all(x -> isa(x, NodeTyped), outputs)

    return Trace(
        Node.(inputs),
        Node.(outputs),
        ctx.metadata.secondary,
        ctx.metadata.parameters,
        ctx.metadata.primary,
        ctx.metadata.param_to_node,
        ctx.metadata.node_to_param,
        collect(args),
    )
end

function make(backend::Backend, trace, opt, opt_inputs, opt_outputs, mutator; kw...)
    # Create an nGraph Executable
    ex = ng_compile(
        backend,
        ParameterVector(Iterators.flatten((trace.inputs, opt_inputs))),
        NodeVector(Iterators.flatten((trace.outputs, trace.implicit_outputs, opt_outputs)));
        kw...
    )

    # Create TensorViews for each of the inputs and outputs
    input_tensors = Tuple(TensorView.(Ref(backend), mutator.(trace.args)))

    # Create arrays for each output tensor
    output_arrays = map(trace.outputs) do node
        T = eltype(node)
        sz = size(node)
        return Array{T}(undef, sz)
    end
    output_tensors = Tuple(TensorView.(Ref(backend), mutator.(output_arrays)))

    if isempty(trace.implicit_outputs)
        secondary_tensors = TensorView[]
    else
        secondary_tensors = TensorView.(Ref(backend), mutator.(trace.implicit_outputs))
    end

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
    inputs = Any[unwrap(i) for i in splat_inputs(ex)]
    outputs = Any[unwrap(o) for o in splat_outputs(ex)]

    # Since we're passing wrapped type to C++, we have to cast them to Any's which is
    # kind of gross - but w/e
    ex.ex(inputs, outputs)

    # Perform any updates required by the optimizer.
    update!(ex.optimizer)
    return untuple(ex.outputs)
end

