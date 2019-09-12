# Testsets for Loss Functions
@testset "Cross Entropy" begin
    x = rand(Float32, 10)
    y = rand(Float32, 10)

    z = Flux.crossentropy(x, y)

    backend = nGraph.Backend()
    f = nGraph.compile(backend, Flux.crossentropy, x, y)

    @test isapprox(z, read(f())[])
end
