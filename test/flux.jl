@testset "Testing ConvConversion" begin
    # Test the Conv flux layer

    input_channels = (10, 16, 20, 64)
    output_channels = (10, 16, 20, 64)
    batch_sizes = (16,)
    backend = nGraph.Backend("CPU")

    for (ic, oc, bs) in Iterators.product(input_channels, output_channels, batch_sizes)
        C = CrossCor((3,3), ic => oc, Flux.relu)
        x = rand(Float32, 14, 14, ic, bs)

        E = C(x)
        f = nGraph.compile(backend, C, x)
        Z = parent(f())

        approx = isapprox(Z, E)
        color = approx ? :green : :red
        printstyled("$ic $oc $bs\n"; color = color)
        @test approx
    end
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

    @test isapprox(expected, parent(f()))
end
