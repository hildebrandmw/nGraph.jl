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
