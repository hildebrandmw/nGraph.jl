using nGraph
using Test
using Flux, BenchmarkTools, Zygote

#include("backprop.jl")
include("types.jl")
include("ops.jl")
include("loss.jl")
include("flux.jl")
include("train.jl")
include("models/mnist.jl")
include("models/inception.jl")
# include("models/resnet.jl")
# include("models/inception_v4.jl")
