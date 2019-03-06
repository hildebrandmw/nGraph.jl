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
        @test nGraph.Element(pair[2]) == nGraph.Lib.gettype(pair[1])
        println(pair[1], " => ", nGraph.Lib.c_type_name(nGraph.Element(pair[2])))

        @test nGraph.back(nGraph.Element(pair[2])) == pair[2]
    end
end

@testset "Testing CoordinateDiff" begin
    # Just simple construction
    c = nGraph.CoordinateDiff([1,2,3])
    @test isa(c, nGraph.CoordinateDiff)

    nGraph.CoordinateDiff((1,2,3))
    @test isa(c, nGraph.CoordinateDiff)
end

@testset "Testing Shape" begin
    # No Arg Constructor
    s = nGraph.Shape()
    @test length(s) == 0

    s = nGraph.Shape(())
    @test length(s) == 0

    s = nGraph.Shape((1,2,3))
    @test length(s) == 3
    @test s[1] == 1
    @test s[2] == 2
    @test s[3] == 3

    s = nGraph.Shape([1,2,3])
    @test length(s) == 3
    @test s[1] == 1
    @test s[2] == 2
    @test s[3] == 3
end

@testset "Testing Strides" begin
    s = nGraph.Strides([1,2,3])
    @test isa(s, nGraph.Strides)

    s = nGraph.Strides((1,2,3))
    @test isa(s, nGraph.Strides)
end

@testset "Testing AxisSet" begin
    s = nGraph.AxisSet([1,2,3])
    @test isa(s, nGraph.AxisSet)

    s = nGraph.AxisSet((1,2,3))
    @test isa(s, nGraph.AxisSet)

    s = nGraph.AxisSet(1)
    @test isa(s, nGraph.AxisSet)
end

@testset "Testing AxisVector" begin
    s = nGraph.AxisVector([1,2,3])
    @test isa(s, nGraph.AxisVector)

    s = nGraph.AxisVector((1,2,3))
    @test isa(s, nGraph.AxisVector)

    s = nGraph.AxisVector(1)
    @test isa(s, nGraph.AxisVector)
end
