__precompile__(false)
module nGraph

# stdlinb
using Random
using Distributions

# external dependcies
using Cxx
using Cassette
using Flux
using Libdl
import ProgressMeter
import JSON

export embedding

using Dates

## Turn on CPU code generation by default.
function __init__()
    enable_codegen()
end

const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const DEPSDIR = joinpath(PKGDIR, "deps")
const USRDIR = joinpath(DEPSDIR, "usr")
const MODELDIR = joinpath(PKGDIR, "models")

# Setup cxx
Cxx.addHeaderDir(joinpath(USRDIR, "include"))
cxxinclude("ngraph/ngraph.hpp")
Libdl.dlopen(joinpath(USRDIR, "lib", "libngraph.so"), Libdl.RTLD_GLOBAL)

# For dispatching backend treatment
abstract type AbstractBackendType end
struct CPU <: AbstractBackendType end
struct GPU <: AbstractBackendType end

string(::Type{CPU}) = "CPU"
string(::Type{GPU}) = "GPU"

# For creating ngraph nodes
unwrap(x) = x

# ngraph likes its nodes as shared pointers.
#
# This macro essentially converts
#
# @op OpName(args...)
#
# into
#
# icxx"""
#     auto node = std::make_shared<ngraph::op::$OpName>(args...);
#     std::dynamic_pointer_cast<ngraph::Node>(node);
# """
#
# All args will be passed through the `unwrap` function, so types like NodeTyped can be
# passed directly and things will work.
macro op(ex)
    @assert ex.head == :call
    op = first(ex.args)
    args = esc.(ex.args[2:end])

    # Emit argument evaluation
    evals = [:($(Symbol("x$i")) = unwrap($arg)) for (i, arg) in enumerate(args)]

    # Create the argument list for the @icxx_str macro
    arglist = join(("\$(x$i)" for i in 1:length(args)), ", ")

    str = """
        auto node = std::make_shared<ngraph::op::$op>($arglist);
        std::dynamic_pointer_cast<ngraph::Node>(node);
    """
    return quote
        $(evals...)
        Cxx.@icxx_str $str
    end
end

include("cuarrays.jl")

# Hijack exception displaying
#
# We have to do some tricks with imports because the @exception macro isn't very robust.
import Base: showerror
const NGRAPH_ERRORS = [
    cxxt"ngraph::ngraph_error&",
    cxxt"ngraph::NodeValidationError&",
]

for err in NGRAPH_ERRORS
    @eval Cxx.@exception function showerror(io::IO, e::$err)
        try
            @show e
            print(io, unsafe_string(icxx"$e.what();"))
        catch w
            @show w
        end
    end
end

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
include("types.jl")
include("ops.jl")
include("compile.jl")

include("flux/flux.jl")
#include("models/resnet.jl")
#include("models/test.jl")
#include("embed_test.jl")

#####
##### Util Functions
#####

# embedding(indices::Vector, weights::Matrix) = view(weights, :, indices)
#
# splicein(i::CartesianIndex, v, at) = CartesianIndex(splicein(Tuple(i), v, at))
# splicein(i::Tuple, v, at) = (i[1:at-1]..., v, i[at:end]...)
#
# ### Testing Onehot
# function onehot(input, max_index, onehot_index)
#     # Create the output size from `max_index` and `onehot_index`
#     sz = size(input)
#     output_sz = splicein(sz, max_index, onehot_index)
#     output = zeros(eltype(input), output_sz)
#
#     for i in CartesianIndices(input)
#         x = input[i]
#         output[splicein(i, x, onehot_index)] = one(eltype(input))
#     end
#     return output
# end

end # module
