module nGraph

using Flux
using Cassette
using IterTools

using CxxWrap
using BenchmarkTools
using Glob
using MacroTools

const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const DEPSDIR = joinpath(PKGDIR, "deps")
const MODELDIR = joinpath(PKGDIR, "models")

import Base: broadcasted

include("lib.jl"); using .Lib

include("types.jl")
include("ops.jl")
include("flux.jl")
include("models/inception_v4.jl")


end # module
