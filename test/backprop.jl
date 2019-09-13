# Test the embedding table backprop against Zygote
#
# This shadows the embedding operating defined in nGraph.jl so standard vectors
# will be dispatched here.
nGraph.embedding(indices::Vector, weights::Matrix) = view(weights, :, indices)

@testset "Testing Embedding Backprop" begin
    weights_size = (10, 10)
    indices_size = 5

    bias_size = (first(weights_size), indices_size)
    bias = randn(Float32, bias_size)

    # Construct a function returning a scalar
    f = (indices, weights) -> sum(bias .* embedding(indices, weights))

    # Make the weights and indices
    #weights = randn(Float32, weights_size)
    weights = randn(Float32, weights_size)

    # Zygote has a bug with how it deals with repeated elements - here we just make sure
    # all are unique to get around it.
    indices = shuffle(Int32(1):Int32(last(weights_size)))[1:indices_size]
    backend = nGraph.Backend("CPU") 

    # Make sure embedding lookup works
    F = nGraph.compile(backend, embedding, indices, weights)
    @test embedding(indices, weights) == read(F())

    # Make sure round trip of "f" works
    F = nGraph.compile(backend, f, indices, weights) 
    @test isapprox(f(indices, weights), read(F())[])

    # Now for the tricky bit - check if the gradients are the same.
    #
    # First - we take the Zygote gradient for the weights - which will be the second return
    # value from Zygote.gradient
    zygote_gradient = Zygote.gradient(f, indices, weights)[2]

    # Next - we compile the nGraph function with the `Gradient` optimizer
    #
    # For now, we have to wrap `weights` in a `param` so the nGraph converter will actually
    # take a gradient with respect to it.
    P = Flux.param(weights)
    F = nGraph.compile(backend, x -> f(x, P), indices; optimizer = nGraph.Gradient)

    display(zygote_gradient)
    println()

    # Run the function - extract the gradient from F
    F() 
    ngraph_gradient = read(F.optimizer._id[P][2]::nGraph.Tensor)
    display(ngraph_gradient)
    println()

    @test isapprox(zygote_gradient, ngraph_gradient)
end
