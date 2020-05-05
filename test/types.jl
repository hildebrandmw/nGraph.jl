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

# @testset "Testing CoordinateDiff" begin
#     # Just simple construction
#     c = nGraph.CoordinateDiff([1,2,3])
#     @test isa(c, nGraph.CoordinateDiff)
#
#     nGraph.CoordinateDiff((1,2,3))
#     @test isa(c, nGraph.CoordinateDiff)
# end
#
# @testset "Testing Shape" begin
#     # No Arg Constructor
#     s = nGraph.Shape()
#     @test length(s) == 0
#
#     s = nGraph.Shape(())
#     @test length(s) == 0
#
#     s = nGraph.Shape((1,2,3))
#     @test length(s) == 3
#     @test s[1] == 1
#     @test s[2] == 2
#     @test s[3] == 3
#
#     s = nGraph.Shape([1,2,3])
#     @test length(s) == 3
#     @test s[1] == 1
#     @test s[2] == 2
#     @test s[3] == 3
# end
#
# @testset "Testing Strides" begin
#     s = nGraph.Strides([1,2,3])
#     @test isa(s, nGraph.Strides)
#
#     s = nGraph.Strides((1,2,3))
#     @test isa(s, nGraph.Strides)
# end
#
# @testset "Testing AxisSet" begin
#     s = nGraph.AxisSet([1,2,3], 3)
#     @test isa(s, nGraph.AxisSet)
#
#     s = nGraph.AxisSet((1,2,3), 4)
#     @test isa(s, nGraph.AxisSet)
#
#     s = nGraph.AxisSet(1, 3)
#     @test isa(s, nGraph.AxisSet)
# end
#
# @testset "Testing AxisVector" begin
#     s = nGraph.AxisVector([1,2,3], 3)
#     @test isa(s, nGraph.AxisVector)
#
#     s = nGraph.AxisVector((1,2,3), 5)
#     @test isa(s, nGraph.AxisVector)
#
#     s = nGraph.AxisVector(1, 1)
#     @test isa(s, nGraph.AxisVector)
# end
