# Because julia-vim and YouCompleteMe don't get along
_getsigma(x) = x.σ

function (c::Flux.Conv{N,F,A,V})(x::Node) where {N,F,A <: Node, V <: Node}
    sigma = _getsigma(c)

    # Perform the convolution
    cn = Flux.conv(x, c.weight; stride = c.stride, pad = c.pad, dilation = c.dilation)

    # Broadcast the bias along the first `N` dimensions and the last
    axis_set = (N + 2) .- [collect(1:N); N+2]
    bb = broadcast(c.bias, size(cn); axes = axis_set)

    node =  sigma.(cn .+ bb)
    return node
end

# Need to flip the convolution kernels
# NOTE: nGraph's "convolution" is NNlib's crosscorrelation
#
# Need to flip the W and H dimensions of the filters
function flip_kernel(x::AbstractArray{T,N}) where {T,N} 
    view(x, size(x, 1):-1:1, size(x, 2):-1:1, ntuple(_->:, N-2)...)
end

flip(c) = c
flip(c::Flux.Conv) = c.weight .= flip_kernel(c.weight)

toparam(backend, x) = x
toparam(backend, x::AbstractArray) = parameter(x)

function find_implicit(x)
    ts = Node[]
    Flux.prefor(t -> isa(t, Node) && push!(ts, t), x)
    return ts
end

mutable struct Executable{V}
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.Executable,:St10shared_ptrIiE}
    outputs::V
    implicit_inputs::Vector
    implicit_outputs::Vector
    train::Bool
end

untrack(x) = Flux.Tracker.istracked(x) ? Flux.data(x) : x

astuple(x::Tuple) = x
astuple(x) = (x,)

unwrap(x::Tuple) = x
unwrap(x::Tuple{T}) where {T} = first(x)

function compile(backend, @nospecialize(f), args...; training = false, learning_rate = Float32(1))
    # Flip convolutions kernels
    g = Flux.mapleaves(untrack, f)
    Flux.prefor(flip, g)

    # Convert all leaf arrays to tensors
    h = Flux.mapleaves(x -> toparam(backend, x), g)

    # flip back
    Flux.prefor(flip, g)

    # Extract the parameter from all the inputs
    inputs = Node.(args)
    implicit_inputs = find_implicit(h)
    outputs = astuple(h(inputs...))
    
    # Make sure we only get "nodes" as outputs
    @show length(implicit_inputs) 
    @show typeof.(outputs)
    @assert all(x -> isa(x, Node), outputs)

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

    return unwrap(ex.outputs)
end

