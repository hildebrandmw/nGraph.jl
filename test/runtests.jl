using nGraph
using Test
using Random, Distributions
using Flux, BenchmarkTools, Zygote

include("types.jl")
include("ops.jl")
include("loss.jl")
include("flux.jl")
include("backprop.jl")
include("train.jl")
include("models/mnist.jl")
include("models/inception.jl")
# include("models/resnet.jl")
# include("models/inception_v4.jl")
