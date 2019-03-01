# Because julia-vim and YouCompleteMe don't get along
_getsigma(x) = x.σ

function _conv_impl(c::Flux.Conv{N}, x::Node) where {N}
    # We flip standard arrays since nGraph really perform cross-correlation
    cn = Flux.conv(x, Node(flip(c.weight)); stride = c.stride, pad = c.pad, dilation = c.dilation)
    # Broadcast the bias along the first `N` dimensions and the last
    axis_set = (N + 2) .- [collect(1:N); N+2]
    bb = broadcast(Node(c.bias), size(cn); axes = axis_set)

    node =  _getsigma(c).(cn .+ bb)
    return node
end

# Again, getting around a Cassette issue
_dense_impl(d::Flux.Dense, x::Node) = _getsigma(d).(d.W * x .+ d.b)

# Extend to unwrap tracked arrays
function Node{T,N}(x::Flux.Tracker.TrackedArray{T,N}) where {T,N}
    return Node{T,N}(
        Lib.op_parameter(Element(T), Shape(size(x))), 
        "Param", 
        copy(Flux.data(x))
    )
end

# Methods defined to avoid ambiguity
Base.:*(x::TrackedArray{T,2}, y::Node{T,1}) where {T} = Node(x) * y
Base.:*(x::TrackedArray{T,2}, y::Node{T,2}) where {T} = Node(x) * y

# Need to flip the convolution kernels
# NOTE: nGraph's "convolution" is NNlib's crosscorrelation
#
# Need to flip the W and H dimensions of the filters
flip(x::Node) = x
function flip(x::AbstractArray{<:Any,N}) where {N}
    collect(view(x, size(x, 1):-1:1, size(x, 2):-1:1, ntuple(_->:, N-2)...))
end

Cassette.@context SnoopCtx

function Cassette.overdub(ctx::SnoopCtx, f::Type{Node{T,N}}, x) where {T,N}
    return get!(ctx.metadata, x, Cassette.recurse(ctx, f, x))::Node{T,N}
end

# Get around Cassette bug with `reverse`
Cassette.overdub(::SnoopCtx, ::typeof(reverse), args...) = reverse(args...)

# Hijack these layers
Cassette.overdub(ctx::SnoopCtx, f::Flux.Dense, args...) = 
    Cassette.overdub(ctx, _dense_impl, f, args...)

Cassette.overdub(ctx::SnoopCtx, f::Flux.Conv, args...) = 
    Cassette.overdub(ctx, _conv_impl, f, args...)

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

function compile(backend, f, args...; training = false, learning_rate = Float32(1))
    ctx = SnoopCtx(metadata = IdDict{Any,Node}())
    # Extract the parameter from all the inputs
    inputs = Node.(args)

    outputs = astuple(Cassette.overdub(ctx, f, inputs...))
    @assert all(x -> isa(x, Node), outputs)

    implicit_inputs = collect(values(ctx.metadata))
    
    #@show length(implicit_inputs) 
    #@show typeof.(implicit_inputs)

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

    #@show length(implicit_outputs)
    #@show typeof.(implicit_outputs)

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

