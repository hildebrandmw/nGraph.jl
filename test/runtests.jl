using nGraph
using Test
using Random, Distributions
using Flux, BenchmarkTools, Zygote

#nGraph.codegen_debug()

# Set OMP_NUM_THREADS to 1 because the sizes of everything used in the test suite is small
# enough that the overhead of parallelization on large systems hurts performance.
ENV["OMP_NUM_THREADS"] = 1

include("types.jl")
include("ops.jl")
#include("flux.jl")
#include("loss.jl")
#include("backprop.jl")
#include("train.jl")
#include("models/mnist.jl")
#include("models/inception.jl")

# include("models/resnet.jl")
# include("models/inception_v4.jl")
