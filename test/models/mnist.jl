@testset "Testing MNIST Conv" begin
    model = Chain(
        # First convolution, operating upon a 28x28 image
        CrossCor((3, 3), 1=>16, pad=(1,1,1,1), relu),
        MaxPool((2,2)),

        # Second convolution, operating upon a 14x14 image
        CrossCor((3, 3), 16=>32, pad=(1,1,1,1), relu),
        MaxPool((2,2)),

        # Third convolution, operating upon a 7x7 image
        CrossCor((3, 3), 32=>32, pad=(1,1,1,1), relu),
        MaxPool((2,2)),

        # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
        # which is where we get the 288 in the `Dense` layer below:
        x -> reshape(x, :, size(x, 4)),
        Dense(288, 10),

        # Finally, softmax to get nice probabilities
        softmax,
    )


    backend = nGraph.Backend()
    x = rand(Float32, 28, 28, 1, 100)
    expected = model(x)
    f = nGraph.compile(backend, model, x)

    # Test that the forward passes match
    @test isapprox(expected, parent(f()))

    @time model(x)
    @time model(x)
    @time f()
    @time f()
end

