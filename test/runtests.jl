using nGraph
using Test
using nGraph.Flux

@testset "Testing Elements" begin
    elements = [
        ("boolean", Bool),
        #("bf16",    Float16),
        ("f32",     Float32),
        ("f64",     Float64),
        #("i8",      Int8),
        #("i16",     Int16),
        #("i32",     Int32),
        ("i64",     Int64),
        #("u8",      UInt8),
        #("u16",     UInt16),
        #("u32",     UInt32),
        #("u64",     UInt64),
    ]

    for pair in elements
        @test nGraph._element(pair[2]) == nGraph.Lib.gettype(pair[1])
        println(pair[1], " => ", nGraph.Lib.c_type_name(nGraph._element(pair[2])))

        @test nGraph._back_element(nGraph._element(pair[2])) == pair[2]
    end
end

include("flux.jl")

#=
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

#####
##### Convolution Tests
#####

@testset "Testing Convolutions" begin
    x = rand(Float32, 10, 10, 10, 10)
    w = rand(Float32, 3, 3, 10, 5)

    expected = Flux.conv(x, w)
    @show size(expected)

    # Now, do the same thing as a ngraph function
    backend = nGraph.Lib.create("CPU")
    X = nGraph.Tensor(backend, x)

    # Need to flip the kernel to get nGraph to match flux
    W = nGraph.Tensor(backend, collect(nGraph.flip_kernel(w)))
    ex = nGraph.compile(backend, Flux.conv, (X, W))

    Z = nGraph.Tensor{eltype(x)}(undef, backend, size(expected)...)
    ex((Z,), (X, W))

    @test isa(Z, nGraph.Tensor)
    @test isapprox(expected, collect(Z))

end

@testset "Testing Conv" begin

    # Test the Conv flux layer
    w = rand(Float32, 3, 3, 128, 256)
    b = rand(Float32, 256)
    x = rand(Float32, 14, 14, 128, 10)

    C = Conv(w, b)
    expected = C(x)

    backend = nGraph.Lib.create("CPU")
    W = nGraph.Tensor(backend, collect(nGraph.flip_kernel(w)))
    B = nGraph.Tensor(backend, b)
    X = nGraph.Tensor(backend, x)

    D = Conv(W, B)

    ex = nGraph.compile(backend, D, (X,), (W, B))
    Z = nGraph.Tensor{eltype(x)}(undef, backend, size(expected)...)
    ex((Z,), (X, W, B))

    @test isapprox(expected, collect(Z))
end

@testset "Testing ConvConversion" begin
    # Test the Conv flux layer
    w = rand(Float32, 3, 3, 128, 256)
    b = rand(Float32, 256)
    x = rand(Float32, 14, 14, 128, 10)

    C = Conv(w, b)
    expected = C(x)

    backend = nGraph.Lib.create("CPU") 

    # Flip the convolution kernels
    Flux.prefor(nGraph.flip, C)

    # Convert all leaf arrays to tensors
    D = Flux.mapleaves(x -> nGraph.totensor(backend, x), C)

    # Make sure we extract the correct number of parameters
    @test length(nGraph.nparams(D)) == 2
end
=#
