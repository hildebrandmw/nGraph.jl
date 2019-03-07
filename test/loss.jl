# Testsets for Loss Functions
@testset "Cross Entropy" begin
    x = rand(Float32, 10)
    y = rand(Float32, 10)

    z = Flux.crossentropy(x, y)

    backend = nGraph.Backend()
    X = nGraph.Tensor(backend, x)
    Y = nGraph.Tensor(backend, y)

    f = nGraph.compile(backend, Flux.crossentropy, X, Y)

    @test isapprox(z, collect(f(X,Y))[])
end
