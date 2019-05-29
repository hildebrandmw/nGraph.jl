module nGraph

using Flux, Cassette
import ProgressMeter
import JSON

export embedding

using Dates

const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const DEPSDIR = joinpath(PKGDIR, "deps")
const MODELDIR = joinpath(PKGDIR, "models")

# Enable experimental operations
const EXPERIMENTAL = true

import Base: broadcasted

include("build.jl")
include("env.jl")
include("lib.jl"); using .Lib

include("types.jl")
include("ops.jl")
include("compile.jl")

include("flux.jl")
#include("models/inception_v4.jl")
include("models/resnet.jl")
include("models/test.jl")

end # module
