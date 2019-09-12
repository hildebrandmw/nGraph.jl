using Flux

@testset "Constant" begin
    x = rand(Float32, 10, 10)
    X = nGraph.Node(x)

    y = X .+ 1
    @test isa(X .+ 1, nGraph.Node{Float32,2})
    @test isa(1 .+ X, nGraph.Node{Float32,2})
end

@testset "Reshape" begin
    backend = nGraph.Backend()

    x = rand(Float32, 100)
    f = nGraph.compile(backend, x -> reshape(x, 1, :), x)
    Z = f()
    @test reshape(x, 1, :) == read(Z)

    # More extravagent reshape
    x = rand(Float32, 1, 2, 3, 4, 5, 6) 
    g = x -> reshape(x, 6, 5, :, 3, 2)

    N = nGraph.Node(x)
    M = g(N)
    @test size(M) == (6, 5, 4, 3, 2)
    f = nGraph.compile(backend, g, x)

    @test g(x) == read(f())
end

@testset "Softmax" begin
    backend = nGraph.Backend()

    # 1D case
    x = rand(Float32, 100)
    z = softmax(x)
    f = nGraph.compile(backend, softmax, x)
    @test isapprox(z, read(f()))

    # 2D case
    x = rand(Float32, 100, 100) 
    z = softmax(x)
    f = nGraph.compile(backend, softmax, x)
    @test isapprox(z, read(f()))
end
