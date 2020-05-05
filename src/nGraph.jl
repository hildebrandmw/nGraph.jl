module nGraph

# stdlinb
using Random
using Distributions

# external dependcies
using Cassette
#using Flux
import ProgressMeter

using Dates

# Turn on CPU code generation by default.
function __init__()
    enable_codegen()
end

const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const DEPSDIR = joinpath(PKGDIR, "deps")
const MODELDIR = joinpath(PKGDIR, "models")

# For convenient overloading in ops.jl
import Base: broadcasted

include("env.jl")
include("lib.jl")
import .Lib

include("types.jl")
include("ops.jl")
include("compile.jl")

# include("flux/flux.jl")
# include("models/resnet.jl")
# include("models/test.jl")
# include("embed_test.jl")

end # module
