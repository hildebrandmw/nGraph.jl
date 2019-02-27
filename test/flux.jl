@testset "Testing ConvConversion" begin
    # Test the Conv flux layer
    w = rand(Float32, 3, 3, 128, 256)
    b = rand(Float32, 256)
    x = rand(Float32, 14, 14, 128, 10)

    C = Conv(w, b)
    expected = C(x)

    backend = nGraph.Lib.create("CPU") 
    X = nGraph.Tensor(backend, x)
    f = nGraph.compile(backend, C, X)

    Z = f(X)

    collected_Z = collect(Z)

    @test size(expected) == size(collected_Z)
    @test isapprox(expected, collected_Z)
end

@testset begin
    m = Chain(
        Dense(rand(Float32, 32, 28^2), rand(Float32, 32), relu),
        Dense(rand(Float32, 10, 32), rand(Float32, 10))
    )

    # Construct a dummy input.
    x = rand(Float32, 28^2)

    expected = m(x)

    # nGraph compile
    backend = nGraph.Lib.create("CPU") 
    X = nGraph.Tensor(backend, x)
    f = nGraph.compile(backend, m, X)
    Z = f(X)

    @test isapprox(expected, collect(Z))
end
