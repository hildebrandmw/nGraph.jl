# # Because julia-vim and YouCompleteMe don't get along
# _getsigma(x) = x.σ
#
# _dims(c::Flux.Conv{N}) where {N} = N
# _dims(c::Flux.CrossCor{N}) where {N} = N

function convolution_implementation(
    x::Node,
    weights::Node,
    bias::Node;
    stride = 1,
    pad = 0,
    dilation = 1,
    activation = identity,
)
    conv = convolution(x, weights; stride, pad, dilation)
    bias_reshaped = reshape(bias, 1, 1, :, 1)
    @show size(conv),size(bias_reshaped)
    return activation.(conv .+ bias_reshaped)
end

# # Again, getting around a Cassette issue
# _dense_impl(d::Flux.Dense, x::Node) = (d.σ).(d.W * x .+ d.b)
#
# # TODO: ngraph makes the dictinction between training and inference. For now, we will
# # assume training, but eventually I can add a parameter to SnoopCtx that will determine
# # if we're training or inferring and pass that information here.
# function _batchnorm_impl(BN::Flux.BatchNorm, x::Node)
#     # Create the batchnorm op and then do activation.
#     γ = Node(BN.γ)
#     β = Node(BN.β)
#
#     n = batchnorm_training(x, γ, β, BN.ϵ)
#
#     # The batchnorm_training op in ngraph returns a tuple
#     # (normalized, gamma, beta). We apply the activation function to the normalized output.
#     #
#     # We also have call `get_output_element` on the other two outputs so that downstream
#     # graph rewriting in ngraph works correctly.
#     #
#     # Also note that we have to `__register` the nodes so they become hidden outputs of the
#    # compiled ngraph graph. Otherwide, things break horribly
#     a = get_output_element(n, 1)
#     __register(get_output_element(n, 2))
#     __register(get_output_element(n, 3))
#
#     return BN.λ.(a)
# end
#
# # Need to flip the convolution kernels
# # NOTE: nGraph's "convolution" is NNlib's crosscorrelation
# #
# # Need to flip the W and H dimensions of the filters
# function flip!(x::AbstractArray{T,N}) where {T,N}
#     x .= view(x, size(x, 1):-1:1, size(x, 2):-1:1, ntuple(_->:, N-2)...)
# end
#
