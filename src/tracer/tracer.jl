# Implements conversion from some high-level Flux ops to their equivalent nGraph
# representations when running under Cassette.
#include("lib.jl")

# Some tuple utilities
astuple(x::Tuple) = x
astuple(x) = (x,)

untuple(x::Tuple) = x
untuple(x::Tuple{T}) where {T} = first(x)

# NOTE: Register was required in the past for some nodes like `BatchNorm` that left
# dangling nodes in the computation graph that then led ngraph to segfault
#
# TODO: Does this still happen?
"""
    __register(x::Node) -> Nothing

By default, becomes a no-op. When executed under [`TraceCtx`](@ref), registers `x` as a
hidden output of a ngraph function.
"""
__register(::Node) = nothing

#####
##### Cassette Magic
#####

Cassette.@context TraceCtx

struct TraceMetadata
    training::Bool

    # Mapping of a Parameter array to nGraph Node
    array_to_node::IdDict{Any,Node}

    # Keep track of the constants we have created.
    constants::IdDict{Any,Node}

    # Implicitly created outputs
    implicit_outputs::Vector{Node}
end

function TraceMetadata(array_to_node, training::Bool)
    return TraceMetadata(training, array_to_node, IdDict{Any,Node}(), Node[])
end

get_implicit_outputs(metadata::TraceMetadata) = metadata.implicit_outputs
get_implicit_outputs(ctx::TraceCtx) = get_implicit_outputs(ctx.metadata)

# Flag to indicate if we are training or not.
#
# If so, we need to add another pass to compute the gradients of all parameters.
istraining(x::TraceCtx) = x.metadata.training

#####
##### Cassette Overdubs
#####

function Base.get(ctx::TraceCtx, x::AbstractArray) where {T,N}
    # Check if this array is a parameter.
    # If so, get our cached input for it.
    node = get(ctx.metadata.array_to_node, x, nothing)
    if !isnothing(node)
        return node
    end

    # Check if we've already made a constant for this object.
    node = get(ctx.metadata.constants, x, nothing)
    if isnothing(node)
        node = constant(x)
        ctx.metadata.constants[x] = node
    end
    return node
end

function Base.get(::Type{Node{T,N}}, ctx::TraceCtx, x::AbstractArray) where {T,N}
    node = get(ctx, x)
    @assert isa(node, Node{T,N})
    return node
end

# Hijack Node constructors from Arrays
function Cassette.overdub(ctx::TraceCtx, ::Type{Node{T,N}}, x::AbstractArray) where {T,N}
    return get(Node{T,N}, ctx, x)
end

# Do not hijack creating nodes from nodes.
Cassette.overdub(ctx::TraceCtx, f::Type{<:Node}, x::Node) = f(x)

# Register implicit outputs
# I'm looking at YOU batch-norm training!
function Cassette.overdub(ctx::TraceCtx, ::typeof(__register), x::Node)
    return push!(ctx.metadata.secondary, x)
end

# Get around Cassette bug with `reverse`
Cassette.overdub(::TraceCtx, ::typeof(reverse), args...) = reverse(args...)

# # Hijack these layers
# Cassette.overdub(ctx::TraceCtx, f::Flux.Dense, args...) =
#     Cassette.overdub(ctx, _dense_impl, f, args...)

function Cassette.overdub(ctx::TraceCtx, f::T, x) where {T<:Union{Flux.Conv,Flux.CrossCor}}
    return convolution_implementation(
        x,
        get(ctx, f.weight),
        get(ctx, f.bias);
        f.stride,
        f.pad,
        f.dilation,
        activation = f.σ,
    )
end

# Cassette.overdub(ctx::TraceCtx, f::Flux.CrossCor, args...) =
#     Cassette.overdub(ctx, _conv_impl, f, args...)
#
# Cassette.overdub(ctx::TraceCtx, f::Flux.BatchNorm, args...) =
#     Cassette.overdub(ctx, _batchnorm_impl, f, args...)

# Short circuits to reduce compile times
Cassette.overdub(::TraceCtx, f::typeof(rand), args...) = f(args...)
#Cassette.overdub(ctx::TraceCtx, f::typeof(Flux.glorot_normal), args...) = f(args...)
#Cassette.overdub(ctx::TraceCtx, f::typeof(Flux.glorot_uniform), args...) = f(args...)

#####
##### Main `compile` entrypoint
#####

"""
    compile(backend, f, x...; [training]) -> Executable

Trace and compile a Flux model `f` with `args`.
"""
function compile(backend::Backend, f, x...; kw...)
    # Build the nGraph computation graph
    trace, _ = snoop(f, x...; kw...)

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
    input_arrays::Vector{Any}
    input_nodes::Vector{Node}
    output_nodes::Vector{Node}
    implicit_output_nodes::Vector{Node}

    # Record of what is a parameter
    parameter_arrays::Vector{Any}
    parameter_nodes::Vector{Node}

    array_to_node::IdDict{Any,Node}
    node_to_array::IdDict{Node,Any}
end

# Step 1 of compilation - trace the provided function with the provided arguments to
# construct an nGraph graph - apply the requested optimizer to finish graph construction.
#
# Returns an argument named tuple that is given to the next stage of compilation.
snoop(f, x...; kw...) = snoop(f, Flux.Params(), x...; kw...)
function snoop(f, parameters::Flux.Params, xs...; training = false, kw...)
    # Extract the inputs and other parameters
    input_arrays = Any[x for x in xs]
    input_nodes = [Node(x) for x in xs]
    parameter_nodes = isempty(parameters) ? Node[] : [Node(p) for p in parameters]

    array_to_node = IdDict{Any,Node}()
    node_to_array = IdDict{Node,Any}()

    # Construct parameter-to-node mappings.
    for (node, array) in zip(parameter_nodes, parameters)
        array_to_node[array] = node
        node_to_array[node] = array
    end

    metadata = TraceMetadata(array_to_node, training)
    ctx = TraceCtx(; metadata)

    # Perform traced execution on the function.
    output_nodes = collect(astuple(Cassette.overdub(ctx, f, input_nodes...)))
    @assert all(x -> isa(x, Node), output_nodes)

    trace = Trace(
        input_arrays,
        input_nodes,
        output_nodes,
        get_implicit_outputs(ctx),
        collect(parameters),
        parameter_nodes,
        array_to_node,
        node_to_array,
    )

    return (trace, metadata)
end

function make(backend::Backend, trace::Trace; training = false, kw...)
    (;
        input_arrays,
        input_nodes,
        output_nodes,
        implicit_output_nodes,
        parameter_arrays,
        parameter_nodes,
    ) = trace
    # Sanity check on inputs
    @assert length(parameter_arrays) == length(parameter_nodes)
    for (_array, _node) in zip(parameter_arrays, parameter_nodes)
        @assert size(_array) == size(_node)
    end

    # TODO: Reimplement training

    # Create an nGraph Executable
    allinputs = [input_nodes; parameter_nodes]
    alloutputs = [output_nodes; implicit_output_nodes]
    ex = compile(backend, allinputs, alloutputs, kw...)

    # Create TensorViews for each of the inputs and outputs
    input_tensors = Tensor[
        Tensor(backend, x) for x in Iterators.flatten((input_arrays, parameter_arrays))
    ]

    output_tensors = ([Tensor(backend, x) for x in output_nodes]...,)
    implicit_output_tensors = Tensor[Tensor(backend, x) for x in implicit_output_nodes]

    return CallableFunction(ex, input_tensors, output_tensors, implicit_output_tensors)
end

#####
##### Executable
#####

struct CallableFunction{O}
    ex::Executable
    inputs::Vector{Tensor}
    outputs::NTuple{O,Tensor}
    implicit_outputs::Vector{Tensor}
end

allinputs(x::CallableFunction) = x.inputs
alloutputs(x::CallableFunction) =
    collect(Iterators.flatten((x.outputs, x.implicit_outputs)))

function (ex::CallableFunction)()
    inputs = allinputs(ex)
    outputs = alloutputs(ex)

    # Since we're passing wrapped type to C++, we have to cast them to Any's which is
    # kind of gross - but w/e
    ex.ex(inputs, outputs)
    return untuple(ex.outputs)
end

