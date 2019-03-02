using nGraph
using Test
using Flux, BenchmarkTools

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

#include("ops.jl")
#include("loss.jl")
#include("flux.jl")
#include("train.jl")
#include("models/mnist.jl")
include("models/inception.jl")
#include("models/resnet.jl")
#include("models/inception_v4.jl")
