# Because julia-vim and YouCompleteMe don't get along
_getsigma(x) = x.σ

_dims(c::Flux.Conv{N}) where {N} = N
_dims(c::Flux.CrossCor{N}) where {N} = N

function _conv_impl(c, x::Node)
    N = _dims(c)

    # We flip standard arrays since nGraph really perform cross-correlation
    n = Node(c.weight)
    cn = NNlib.conv(
        x,
        n;
        stride = reverse(c.stride),
        pad = reverse(c.pad),
        dilation = reverse(c.dilation)
    )

    # Broadcast the bias along the first `N` dimensions and the last
    axis_set = [collect(1:N); N+2]
    bb = broadcast(Node(c.bias), size(cn); axes = axis_set)

    node = _getsigma(c).(cn .+ bb)
    return node
end

# Again, getting around a Cassette issue
_dense_impl(d::Flux.Dense, x::Node) = (d.σ).(d.W * x .+ d.b)

# TODO: ngraph makes the dictinction between training and inference. For now, we will
# assume training, but eventually I can add a parameter to SnoopCtx that will determine
# if we're training or inferring and pass that information here.
function _batchnorm_impl(BN::Flux.BatchNorm, x::Node)
    # Create the batchnorm op and then do activation.
    γ = Node(BN.γ)
    β = Node(BN.β)

    n = batchnorm_training(x, γ, β, BN.ϵ)

    # The batchnorm_training op in ngraph returns a tuple
    # (normalized, gamma, beta). We apply the activation function to the normalized output.
    #
    # We also have call `get_output_element` on the other two outputs so that downstream
    # graph rewriting in ngraph works correctly.
    #
    # Also note that we have to `__register` the nodes so they become hidden outputs of the
   # compiled ngraph graph. Otherwide, things break horribly
    a = get_output_element(n, 1)
    __register(get_output_element(n, 2))
    __register(get_output_element(n, 3))

    return BN.λ.(a)
end

# Need to flip the convolution kernels
# NOTE: nGraph's "convolution" is NNlib's crosscorrelation
#
# Need to flip the W and H dimensions of the filters
function flip!(x::AbstractArray{T,N}) where {T,N}
    x .= view(x, size(x, 1):-1:1, size(x, 2):-1:1, ntuple(_->:, N-2)...)
end

