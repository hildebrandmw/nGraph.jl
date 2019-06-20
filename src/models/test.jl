function _network(x)
    chain = Chain(
        Conv((3, 3), size(x, 3) => 128, relu; pad = (1, 1)),
        BatchNorm(128),
        x -> reshape(x, :,  size(x, 4))
    )

    y = chain(x)

    return softmax(Dense(size(y, 1), 10, relu)(y))
end

function test_model(backend = nGraph.Backend())
    batchsize = 8
    nchannels = 16

    # Create a nGraph tensor
    X = nGraph.Tensor(backend, rand(Float32, 20, 20, nchannels, batchsize))
    Y = nGraph.Tensor(backend, rand(Float32, 10))

    g = (x, y) -> Flux.crossentropy(_network(x), y)

    f = nGraph.compile(backend, g, X, Y; optimizer = nGraph.SGD(Float32(0.000001)))

    # Return the arguments as a tuple so in the future, we can return multiple compiled
    # function arguments and still have downstream code work.
    return f, (X,Y)
end
