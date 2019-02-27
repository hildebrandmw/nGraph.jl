# Because julia-vim and YouCompleteMe don't get along
_getsigma(x) = x.σ

function (c::Flux.Conv{N,F,A,V})(x::Node) where {N,F,A <: Tensor, V <: Tensor}
    sigma = _getsigma(c)

    # Perform the convolution
    cn = Flux.conv(x, c.weight.param; stride = c.stride, pad = c.pad, dilation = c.dilation)

    # Broadcast the bias along the first `N` dimensions and the last
    axis_set = (N + 2) .- [collect(1:N); N+2]
    bb = _broadcast(c.bias.param, size(cn), axis_set)

    return sigma.(cn .+ bb)
end

(a::Flux.Dense)(x::Node) = _getsigma(a).(a.W.param * x .+ a.b.param)

# Need to flip the convolution kernels
flip(c) = c
flip(c::Flux.Conv) = c.weight .= flip_kernel(c.weight)

totensor(backend, x) = x
totensor(backend, x::AbstractArray) = Tensor(backend, x)

function tensors(x)
    ts = Tensor[]
    Flux.prefor(t -> isa(t, Tensor) && push!(ts, t), x)
    return ts
end

struct Executable{T <: Vector{Tensor}, V}
    ptr::Lib.CxxWrap.SmartPointerWithDeref{nGraph.Lib.Executable,:St10shared_ptrIiE}
    implicit::T
    outputs::V
end

astuple(x::Tuple) = x
astuple(x) = (x,)

unwrap(x::Tuple) = x
unwrap(x::Tuple{T}) where {T} = first(x)

getparam(x::Tensor) = x.param
getparam(x::Node) = x

function compile(backend, f, args...)
    # Flip convolutions kernels
    Flux.prefor(flip, f)

    # Convert all leaf arrays to tensors
    g = Flux.mapleaves(x -> totensor(backend, x), f)

    # Flip back the original
    Flux.prefor(flip, f) 

    # Extract the parameter from all the inputs
    inputs = getparam.(args)
    implicit = tensors(g)
    params = getparam.(implicit)

    @show length(params)
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

    return Executable(ex, implicit, output_tensors)
end

function (ex::Executable)(inputs...) 
    all_inputs = Any[i.ptr for i in [collect(inputs); ex.implicit]]
    outputs = Any[o.ptr for o in ex.outputs]

    println("Calling function")
    Lib.call(ex.ptr, outputs, all_inputs)
    println("Function done?")

    return unwrap(ex.outputs)
end
