@testset "Testing Elements" begin
    elements = [
        Bool,
        #Float16,
        Float32,
        Float64,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
    ]

    # Does this survive the round trip?
    for element in elements
        @test nGraph.back(nGraph.Element(element)) == element
    end
end

@testset "Testing Nodes" begin
    # Create a dummy parameter.
    param = nGraph.parameter(Float32, (10, 10))

    @test isa(param, nGraph.Node)

    # Test printing
    println(devnull, param)

    @test ndims(param) == 2
    @test size(param) == (10,10)
    @test size(param, 1) == 10
    @test size(param, 2) == 10
    @test eltype(param) == Float32
    @test eltype(param.obj) == Float32
    @test nGraph.description(param) == "Parameter"
    println(devnull, nGraph.name(param))

    # Go into higher dimensions.
    x = nGraph.parameter(Int32, (10, 20, 30))
    @test size(x) == (10, 20, 30)
    @test size(x, 3) == 30
end

@testset "Testing Function" begin
    # For this test, we need the `+` funcion defined in `ops`.
    a = nGraph.parameter(Float32, (10,10))
    b = nGraph.parameter(Float32, (10,10))

    z = a + b
    @test isa(z, nGraph.Node)
    @test eltype(z) == Float32
    @test size(z) == (10,10)
    @test nGraph.description(z) == "Add"

    fn = nGraph.NGFunction([a,b], [z])

    println(devnull, nGraph.name(fn))

    # At this point, we haven't compiled the function, so the poolsize should just default
    # to zero.
    @test iszero(nGraph.poolsize(fn))
end

@testset "Testing Backend" begin
    backend = nGraph.Backend("CPU")
    @test nGraph.version(backend) == "0.0.0"
end

@testset "Testing Tensor View" begin
    backend = nGraph.Backend("CPU")
    x = Array{Float32}(undef, 10, 10)

    tv = nGraph.TensorView(backend, x)
    @test parent(tv) === x
    @test eltype(tv) == eltype(parent(tv))
    @test sizeof(tv) == sizeof(parent(tv))
    @test size(tv) == size(parent(tv))
    println(devnull, tv)

    # Construct a TensorView from a scalar.
    tv = nGraph.TensorView(backend, 1.0)
    @test size(tv) == ()
    @test eltype(tv) == Float64

    # Construct a TensorView from a Node
    node = nGraph.parameter(Int32, (20, 20))
    tv = nGraph.TensorView(backend, node)
    @test size(node) == size(parent(node)) == (20, 20)
    @test eltype(node) == eltype(parent(node)) == Int32
end

