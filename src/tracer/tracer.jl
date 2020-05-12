# Implements conversion from some high-level Flux ops to their equivalent nGraph
# representations when running under Cassette.
#include("lib.jl")

# Some tuple utilities
astuple(x::Tuple) = x
astuple(x) = (x,)

untuple(x::Tuple) = x
untuple(x::Tuple{T}) where {T} = first(x)

#####
##### Cassette Magic
#####

Cassette.@context TraceCtx

# NOTE: Register was required in the past for some nodes like `BatchNorm` that left
# dangling nodes in the computation graph that then led ngraph to segfault
#
# TODO: Does this still happen?
"""
    __register(x::Node) -> Nothing

By default, becomes a no-op. When executed under [`TraceCtx`](@ref), registers `x` as a
hidden output of a ngraph function.
"""
__register(x::Node) = nothing

struct TraceMetadata
    # The Julia containers for the parameters of this network
    parameters::IdSet{Any}

    # Mapping of Parameter to nGraph Node
    param_to_node::IdDict{Any,Node}
    node_to_param::IdDict{Node,Any}

    # Keep track of the constants we have created.
    constants::IdDict{Any,Node}

    # Used for keeping track of the order that Nodes are created for deterministic ordering
    # of inputs and outputs
    primary::Vector{Node}
    secondary::Vector{Node}
end

TraceMetadata() = TraceMetadata(
    IdSet{Any}(),
    IdDict{Any,Node}(),
    IdDict{Node,Any}(),
    IdDict{Any,Node}(),
    Node[],
    Node[],
)

#####
##### Cassette Overdubs
#####

# Hijack Node constructors from Arrays
function Cassette.overdub(ctx::TraceCtx, f::Type{Node{T,N}}, x::AbstractArray) where {T,N}
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

    # Check if we've already made a constant for this object.
    node = get(ctx.constants, x, nothing)
    if isnothing(node)
        node = constant(x)
        ctx.constants[x] = node
    end
    return node
end

# Do not hijack creating nodes from nodes.
Cassette.overdub(ctx::TraceCtx, f::Type{<:Node}, x::Node) = f(x)

Cassette.overdub(ctx::TraceCtx, ::typeof(__register), x::Node) =
    push!(ctx.metadata.secondary, x)

# Get around Cassette bug with `reverse`
Cassette.overdub(::TraceCtx, ::typeof(reverse), args...) = reverse(args...)

# # Hijack these layers
# Cassette.overdub(ctx::TraceCtx, f::Flux.Dense, args...) =
#     Cassette.overdub(ctx, _dense_impl, f, args...)
#
# Cassette.overdub(ctx::TraceCtx, f::Flux.Conv, args...) =
#     Cassette.overdub(ctx, _conv_impl, f, args...)
#
# Cassette.overdub(ctx::TraceCtx, f::Flux.CrossCor, args...) =
#     Cassette.overdub(ctx, _conv_impl, f, args...)
#
# Cassette.overdub(ctx::TraceCtx, f::Flux.BatchNorm, args...) =
#     Cassette.overdub(ctx, _batchnorm_impl, f, args...)

# Skip recursing initialization calls - recursing turns out to take a very, very long time.
Cassette.overdub(ctx::TraceCtx, f::typeof(rand), args...) = f(args...)
#Cassette.overdub(ctx::TraceCtx, f::typeof(Flux.glorot_normal), args...) = f(args...)
#Cassette.overdub(ctx::TraceCtx, f::typeof(Flux.glorot_uniform), args...) = f(args...)

#####
##### Main `compile` entrypoint
#####

"""
    compile(backend, f, args..; optimizer = Inference) -> Executable

Trace and compile a Flux model `f` with `args`.
"""
function compile(backend::Backend, f, args...; kw...)
    # Build the nGraph computation graph
    trace = snoop(f, args...)

    # pass the trace as well as the optimizer arguments to create the executable.
    return make(backend, trace; kw...)
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
    ctx = TraceCtx(metadata = TraceMetadata())

    # TODO: Reword parameters to how it works in Flux 10

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

function make(backend::Backend, trace; kw...)
    # Create an nGraph Executable
    ex = compile(
        backend,
        [trace.inputs; opt_inputs]
        [trace.outputs; trace.implicit_outputs; opt_outputs],
        kw...
    )

    # Create TensorViews for each of the inputs and outputs
    input_tensors = Tensor.(Ref(backend), trace.args)
    output_tensors = Tuple(Tensor.(Ref(backend), trace.outputs))
    secondary_tensors = Tensor.(Ref(backend), trace.implicit_outputs)

    return TracedExecutable(
        ex,
        opt,
        input_tensors,
        output_tensors,
        secondary_tensors
    )
end

struct TracedExecutable{O}
    ex::Executable
    inputs::Vector{Tensor}
    outputs::NTuple{O,Tensor}
    secondary::Vector{Tensor}
end

allinputs(x::TracedExecutable) = x.inputs
alloutputs(x::TracedExecutable) = Iterators.flatten(x.outputs, x.secondary)

function (ex::FluxExecutable)()
    inputs = Any[getpointer(i) for i in splat_inputs(ex)]
    outputs = Any[getpointer(o) for o in splat_outputs(ex)]

    # Since we're passing wrapped type to C++, we have to cast them to Any's which is
    # kind of gross - but w/e
    ex.ex(inputs, outputs)
    return untuple(ex.outputs)
end

