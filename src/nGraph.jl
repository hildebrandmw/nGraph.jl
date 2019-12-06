module nGraph

# stdlinb
using Random
using Distributions

# external dependcies
using Cassette
using Flux
import ProgressMeter
import JSON

export embedding

using Dates

# Turn on CPU code generation by default.
function __init__()
    enable_codegen()
end

const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const DEPSDIR = joinpath(PKGDIR, "deps")
const MODELDIR = joinpath(PKGDIR, "models")

# For dispatching backend treatment
abstract type AbstractBackendType end
struct CPU <: AbstractBackendType end
struct GPU <: AbstractBackendType end

# Check if GPU is required. If so, bring in the GPU code
params = JSON.parsefile(joinpath(DEPSDIR, "build.json"))
if params["GPU"]
    @info "Including CUDA Arrays"
    include("cuarrays.jl")
end

# Enable experimental operations
const EXPERIMENTAL = true

# For convenient overloading in ops.jl
import Base: broadcasted

settings() = JSON.parsefile(joinpath(DEPSDIR, "build.json"))

# Flag to determine if the compiler has been invoked yet.
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

include("flux/flux.jl")
include("models/resnet.jl")
include("models/test.jl")
include("embed_test.jl")

#####
##### Util Functions
#####

embedding(indices::Vector, weights::Matrix) = view(weights, :, indices)

splicein(i::CartesianIndex, v, at) = CartesianIndex(splicein(Tuple(i), v, at))
splicein(i::Tuple, v, at) = (i[1:at-1]..., v, i[at:end]...)

### Testing Onehot
function onehot(input, max_index, onehot_index)
    # Create the output size from `max_index` and `onehot_index`
    sz = size(input)
    output_sz = splicein(sz, max_index, onehot_index)
    output = zeros(eltype(input), output_sz)

    for i in CartesianIndices(input)
        x = input[i]
        output[splicein(i, x, onehot_index)] = one(eltype(input))
    end
    return output
end

end # module
