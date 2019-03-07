@testset "Testing Inception" begin
    batchsize = 16
    x = rand(Float32, 299, 299, 3, batchsize)

    backend = nGraph.Backend()
    X = nGraph.Tensor(backend, x)
    f = nGraph.compile(backend, nGraph.inception_v4, X)

    @show size(f(X))

    for _ in 1:10
        @time f(X)
    end

    f(x,y) = Flux.crossentropy(nGraph.inception_v4(x), y)

    y = rand(Float32, 1000, batchsize)
    Y = nGraph.Tensor(backend, y)
    g = nGraph.compile(backend, f, X, Y; optimizer = nGraph.SGD(Float32(0.001)))

    @show length(g.outputs)
    @show length(g.optimizer.inputs)
    @show length(g.optimizer.outputs)

    for _ in 1:5
        @time g(X,Y)
    end

    #@test isapprox(z, f(X))
end
