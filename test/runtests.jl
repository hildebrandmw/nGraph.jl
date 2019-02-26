using nGraph
using Test

@testset "Simple Example" begin
    backend = nGraph.Lib.create("CPU")

    a = Float32.([1,2,3,4])
    b = Float32.([1,2,3,4])
    c = Float32.([1,2,3,4])

    a = nGraph.Tensor(backend, a)
    b = nGraph.Tensor(backend, b)
    c = nGraph.Tensor(backend, c)
    x = nGraph.Tensor{Float32}(undef, backend, 4)

    f(x, y, z) = x .* (y .+ z)

    ex = nGraph.compile(backend, f, (a,b,c))

    ex((x,), (a, b, c))

    x_expected = f(c, a, b)

    @test isa(x, nGraph.Tensor)
    @test x == x_expected
end

##### 
##### Broadcast Test
#####
@testset "Testing Broadcast" begin
    a = [1,2,3,4]
    b = 1

    backend = nGraph.Lib.create("CPU")
    A = nGraph.Tensor(backend, a)
    B = nGraph.Tensor(backend, b)

    f(x, y) = x .+ y

    ex = nGraph.compile(backend, f, (A, B))

    C = nGraph.Tensor{Int64}(undef, backend, 4)
    ex((C,), (A,B))

    @test C == f(a,b)
end

#####
##### Fully connected layer with bias and activation
#####
nGraph.relu(x) = x > zero(x) ? x : zero(x)
@testset "Testing Fully Connected" begin
    ts = 10
    w = rand(Float32, ts, ts) 
    b = rand(Float32, ts)
    x = rand(Float32, ts, ts)

    f(x, w, b) = nGraph.relu.(w*x .+ b)

    # Compile a test function
    backend = nGraph.Lib.create("CPU")
    W = nGraph.Tensor(backend, w)
    B = nGraph.Tensor(backend, b)
    X = nGraph.Tensor(backend, x)
    ex = nGraph.compile(backend, f, (X, W, B))

    Z = nGraph.Tensor{Float32}(undef, backend, ts, ts)
    ex((Z,), (X, W, B))

    expected = f(x, w, b)
    @test isa(Z, nGraph.Tensor)
    @test isapprox(expected, collect(Z))
end
