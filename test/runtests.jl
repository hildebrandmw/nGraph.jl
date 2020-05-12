using nGraph
using CxxWrap
using Test
using Flux
#using Random, Distributions
#using Flux, BenchmarkTools, Zygote

# Testing Strategy
# 1) Type Tests
# 2) Basic op tests without the Flux backend
#    These tests will be pretty basic because constructing graphps without the Flux backend
#    is pretty tedious and error prone.
#
# 3) Flux Backend

# Set OMP_NUM_THREADS to 1 because the sizes of everything used in the test suite is small
# enough that the overhead of parallelization on large systems hurts performance.
ENV["OMP_NUM_THREADS"] = 1

include("types.jl")
include("ops.jl")
# include("loss.jl")
# include("flux.jl")
# include("backprop.jl")
# include("train.jl")
# include("models/mnist.jl")
# include("models/inception.jl")

# include("models/resnet.jl")
# include("models/inception_v4.jl")
