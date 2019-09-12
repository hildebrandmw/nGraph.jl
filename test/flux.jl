@testset "Testing ConvConversion" begin
    # Test the Conv flux layer
    w = rand(Float32, 3, 3, 128, 256)
    b = rand(Float32, 256)
    x = rand(Float32, 14, 14, 128, 10)

    C = Conv(Flux.param(w), param(b))
    expected = C(x)

    backend = nGraph.Backend()
    f = nGraph.compile(backend, C, x)

    Z = f()

    collected_Z = read(Z)

    @test size(expected) == size(collected_Z)
    @test isapprox(expected, collected_Z)
end

@testset begin
    m = Chain(
        Dense(28^2, 32, relu),
        Dense(32, 10)
    )

    # Construct a dummy input.
    x = rand(Float32, 28^2)

    expected = m(x)

    # nGraph compile
    backend = nGraph.Backend()
    f = nGraph.compile(backend, m, x)

    @test isapprox(expected, read(f()))
end
