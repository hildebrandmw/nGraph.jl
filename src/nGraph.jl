module nGraph

using Zygote

using CxxWrap
using BenchmarkTools
using Glob

const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const DEPSDIR = joinpath(PKGDIR, "deps")
const MODELDIR = joinpath(PKGDIR, "models")

include("lib.jl"); using .Lib

include("ops.jl")
include("models.jl")


end # module
