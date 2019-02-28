# Because julia-vim and YouCompleteMe don't get along
_getsigma(x) = x.σ

# Forward to `_conv_impl` so we can intercept normal `Conv` invocations from Cassette
(c::Flux.Conv{N,F,A,V})(x::Node) where {N,F,A <: Node, V <: Node} = _conv_impl(c, x)

function _conv_impl(c::Flux.Conv{N}, x::Node) where {N}
    sigma = _getsigma(c)

    # Perform the convolution
    cn = Flux.conv(x, c.weight; stride = c.stride, pad = c.pad, dilation = c.dilation)

    # Broadcast the bias along the first `N` dimensions and the last
    axis_set = (N + 2) .- [collect(1:N); N+2]
    bb = broadcast(c.bias, size(cn); axes = axis_set)

    node =  sigma.(cn .+ bb)
    return node
end

# Again, getting around a Cassette issue
_dense_impl(d::Flux.Dense, x::Node) = _getsigma(d).(d.W * x .+ d.b)

# Need to flip the convolution kernels
# NOTE: nGraph's "convolution" is NNlib's crosscorrelation
#
# Need to flip the W and H dimensions of the filters
function flip(x::AbstractArray{T,N}) where {T,N} 
    collect(view(x, size(x, 1):-1:1, size(x, 2):-1:1, ntuple(_->:, N-2)...))
end

#####
##### Executable
#####

mutable struct Executable{V}
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.Executable,:St10shared_ptrIiE}
    outputs::V
    implicit_inputs::Vector
    implicit_outputs::Vector
    train::Bool
end

istracked(x) = Flux.Tracker.istracked(x)
untrack(x) = istracked(x) ? Flux.data(x) : x

astuple(x::Tuple) = x
astuple(x) = (x,)

untuple(x::Tuple) = x
untuple(x::Tuple{T}) where {T} = first(x)


# Setup our Cassette passes.
#
# Basically, whenever we see a tracked object, we want to convert that object to a parameter
# and record the use of that parameter.
Cassette.@context SnoopCtx

# Intercept all methods - look for "tracked" objects, create a Node for them and register
# their existence.
# function register(ctx, x)
#     # If this is a tracked object, register it with the metadata if it isn't already
#     # registered
#     if istracked(x) && !haskey(ctx.metadata, x)
#         @show typeof(x)
#         @show typeof(Flux.data(x))
#         ctx.metadata[x] = Node(Flux.data(x))
#     end
#     return nothing
# end

register(ctx, x::Flux.Tracker.TrackedArray) = get!(ctx.metadata, x, Node(Flux.data(x)))
register(ctx, x) = nothing

function Cassette.prehook(ctx::SnoopCtx, f, args...) 
    #println(f)
    #println("    ", typeof.(args))
    map(x -> register(ctx, x), args)
end

#Cassette.prehook(ctx::SnoopCtx, f::T, args...) where {T <: Flux.Tracker.TrackedArray} = nothing

_untrack(ctx, x) = get(ctx.metadata, x, x)
untrack(ctx, x) = _untrack.(Ref(ctx), x)

Cassette.overdub(ctx::SnoopCtx, f, args...) = Cassette.recurse(ctx, f, untrack(ctx, args)...)

# Get around a Cassette issue with it hating reverse
Cassette.overdub(ctx::SnoopCtx, f::typeof(reverse), args...) = f(args...)
Cassette.overdub(ctx::SnoopCtx, f::Flux.Dense, args...) = Cassette.overdub(ctx, _dense_impl, f, args...)
function Cassette.overdub(ctx::SnoopCtx, f::Flux.Conv{N,F,A,V}, args...) where {N,F,A <: TrackedArray, V <: TrackedArray}
    println("Overloading Conv")
    # We need to flip the kernels for this to work - the easiest way to do this is to just
    # create the Conv object we want and manually register the parameters.
    weight = get!(ctx.metadata, f.weight, Node(flip(Flux.data(f.weight))))::Node
    bias = get!(ctx.metadata, f.bias, Node(Flux.data(f.bias)))::Node

    c = Flux.Conv(_getsigma(f), weight, bias, f.stride, f.pad, f.dilation)
    Cassette.overdub(ctx, _conv_impl, c, args...)
end

# Skip Convolution constructors
#
# This is required when a Conv((3,3), 10 => 20 ... happens. During the construction of
# the Conv type, a lot of tracking occurs. Here, we just skip that and hijack the results.
function Cassette.overdub(ctx::SnoopCtx, f::UnionAll, args...)
    println("Hijacking Convolution")
    y = f(args...)
    @show typeof(y)
    return y
end

function compile(backend, f, args...; training = false, learning_rate = Float32(1))
    ctx = SnoopCtx(metadata = IdDict{Any,Any}())

    # Extract the parameter from all the inputs
    inputs = Node.(args)

    outputs = astuple(Cassette.overdub(ctx, f, inputs...))
    @assert all(x -> isa(x, Node), outputs)

    implicit_inputs = collect(values(ctx.metadata))
    
    # Make sure we only get "nodes" as outputs
    #@show length(implicit_inputs) 
    #@show typeof.(implicit_inputs)
    #@show typeof.(outputs)

    # If we're training, we need to insert backprop nodes
    if training   
        # Assume the first output is the loss
        loss = first(outputs) 
        if size(loss) != ()
            error("Expected Loss to be a Scalar.")
        end

        ### TODO: For now, just make the learning rate an extra implicit parameter
        # However, I really need to think through an API that allows multiple optimizers.
        delta = -constant(convert(Float32, learning_rate))
        adjoints = Adjoints(loss, delta)
        implicit_outputs = [P + backprop_node(adjoints, P) for P in implicit_inputs]
    else
        implicit_outputs = Node[]
    end

    # Create the formal ngraph function
    ngraph_function = Lib.make_function(
        NodeVector(outputs..., implicit_outputs...), 
        ParameterVector(inputs..., implicit_inputs...)
    )

    # Compile the executable
    ex = Lib.compile(backend, ngraph_function, false)

    # Create tensors for the outputs
    output_tensors = map(x -> Tensor(backend, x), outputs) 
    implicit_input_tensors = map(x -> Tensor(backend, x), implicit_inputs)
    implicit_output_tensors = map(x -> Tensor(backend, x), implicit_outputs)

    return Executable(
        ex, 
        output_tensors, 
        implicit_input_tensors,
        implicit_output_tensors,
        training
    )
end

function (ex::Executable)(inputs...) 
    all_inputs = Any[i.ptr for i in [collect(inputs); ex.implicit_inputs]]
    outputs = Any[o.ptr for o in [collect(ex.outputs); ex.implicit_outputs]]
    
    # Since we're passing wrapped type to C++, we have to cast them to Any's
    Lib.call(ex.ptr, outputs, all_inputs)

    # If training, swap the implicit parameters for the next run.
    if ex.train
        ex.implicit_inputs, ex.implicit_outputs = ex.implicit_outputs, ex.implicit_inputs
    end

    return untuple(ex.outputs)
end

