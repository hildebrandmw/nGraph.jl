# Because julia-vim and YouCompleteMe don't get along
_getsigma(x) = x.σ

function (c::Flux.Conv{N,F,A,V})(x::Node) where {N,F,A <: Node, V <: Node}
    sigma = _getsigma(c)

    # Perform the convolution
    cn = Flux.conv(x, c.weight; stride = c.stride, pad = c.pad, dilation = c.dilation)

    # Broadcast the bias along the first `N` dimensions and the last
    axis_set = (N + 2) .- [collect(1:N); N+2]
    bb = _broadcast(c.bias, size(cn), axis_set)

    return sigma.(cn .+ bb)
end

# Need to flip the convolution kernels
flip(c) = c
flip(c::Flux.Conv) = c.weight .= flip_kernel(c.weight)

toparam(backend, x) = x
toparam(backend, x::AbstractArray) = param(x)

function ng_params(x)
    ts = Node[]
    Flux.prefor(t -> isa(t, Node) && push!(ts, t), x)
    return ts
end

struct Executable{T, V}
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.Executable,:St10shared_ptrIiE}
    implicit::T
    outputs::V
end

astuple(x::Tuple) = x
astuple(x) = (x,)

unwrap(x::Tuple) = x
unwrap(x::Tuple{T}) where {T} = first(x)

function compile(backend, f, args...)
    # Flip convolutions kernels
    Flux.prefor(flip, f)

    # Convert all leaf arrays to tensors
    g = Flux.mapleaves(x -> toparam(backend, x), f)

    # Flip back the original
    Flux.prefor(flip, f) 

    # Extract the parameter from all the inputs
    inputs = Node.(args)
    params = ng_params(g)
    outputs = astuple(g(inputs...))

    # Make sure we only get "nodes" as outputs
    @assert all(x -> isa(x, Node), outputs)

    # Create the formal ngraph function
    ngraph_function = Lib.make_function(
        nodes(outputs...), 
        parameters(inputs..., params...)
    )

    # Compile the executable
    ex = Lib.compile(backend, ngraph_function, false)

    # Create tensors for the outputs
    output_tensors = map(x -> Tensor(backend, x), outputs) 
    implicit_tensors = map(x -> Tensor(backend, x), params)

    return Executable(ex, implicit_tensors, output_tensors)
end

function (ex::Executable)(inputs...) 
    all_inputs = Any[i.ptr for i in [collect(inputs); ex.implicit]]
    outputs = Any[o.ptr for o in ex.outputs]
    Lib.call(ex.ptr, outputs, all_inputs)

    return unwrap(ex.outputs)
end
