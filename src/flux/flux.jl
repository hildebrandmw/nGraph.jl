# Implements conversion from some high-level Flux ops to their equivalent nGraph
# representations when running under Cassette.
include("flux_ops.jl")

# TODO: remove `optimizers`.
include("optimizers.jl")

# Some tuple utilities
astuple(x::Tuple) = x
astuple(x) = (x,)

untuple(x::Tuple) = x
untuple(x::Tuple{T}) where {T} = first(x)

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

#####
##### Cassette Overdubs
#####

# Hijack Node constructors from Arrays
function Cassette.overdub(ctx::SnoopCtx, f::Type{Node{T,N}}, x::AbstractArray) where {T,N}
    # Check if this array is a parameter - if so,make it an nGraph input.
    # Otherwise, make it a constant.
    if haskey(ctx.metadata.parameters, x)
        # Check if there is an entry for this parameter already.
        # If so, just return that.
        #
        # Otherwise, we have to create one.
        node = get(ctx.metadata.param_to_node, x, nothing)
        if isnothing(node)
            node = f(x)::Node{T,N}

            # Update internal data structures
            ctx.metadata.param_to_node[x] = node
            ctx.metadata.node_to_param[node] = x
            push!(ctx.metadata.primary, node)
        else
            @assert isa(node, Node{T,N})
        end

        return node
    end

    # Otherwise, make this a constant
    return constant(x)
end

# Do not hijack creating nodes from nodes.
Cassette.overdub(ctx::SnoopCtx, f::Type{<:Node}, x::Node) = f(x)

Cassette.overdub(ctx::SnoopCtx, ::typeof(__register), x::Node) =
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
        kw...
    )

    # Build the nGraph computation graph
    trace = snoop(f, args...)

    # build the optimizer and get the implicit input and output parameters
    opt, opt_inputs, opt_outputs = apply!(backend, optimizer, trace)

    # pass the trace as well as the optimizer arguments to create the executable.
    return make(backend, trace, opt, opt_inputs, opt_outputs; kw...)
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
    inputs = collect(Node.(args))

    # Perform traced execution on the function.
    outputs = collect(astuple(Cassette.overdub(ctx, f, inputs...)))
    @assert all(x -> isa(x, Node), outputs)

    return Trace(
        inputs,
        outputs,
        ctx.metadata.secondary,
        ctx.metadata.parameters,
        ctx.metadata.primary,
        ctx.metadata.param_to_node,
        ctx.metadata.node_to_param,
        collect(args),
    )
end

function make(backend::Backend, trace, request, opt_inputs, opt_outputs; kw...)
    # Create an nGraph Executable
    ex = compile(
        backend,
        ParameterVector(trace.inputs..., opt_inputs...),
        NodeVector(trace.outputs..., trace.implicit_outputs..., opt_outputs...);
        kw...
    )

    # Create TensorViews for each of the inputs and outputs
    input_tensors = Tuple(TensorView.(Ref(backend), trace.args))
    output_tensors = Tuple(TensorView.(Ref(backend), trace.outputs))
    secondary_tensors = TensorView.(Ref(backend), trace.implicit_outputs)

    # Instantiate to optimizer, including its Tensor Views
    opt = fulfill(backend, request)

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

