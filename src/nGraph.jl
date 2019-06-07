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

# Flag to determine if the compile has been invoked yet.
#
# This is useful for options like setting the number of threads via environmental varialbes
# such as OMP_NUM_THREADS and KMP_HW_SUBSET that only take effect if set before the first
# compilation.
const __HAVE_COMPILED = Ref(false)
have_compiled() = __HAVE_COMPILED[]

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
