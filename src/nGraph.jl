__precompile__(false)
module nGraph

# stdlinb
using Random
using Distributions

# external dependcies
using Cxx
import Cassette
import Flux
import NNlib
import Libdl
import JSON

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

Base.string(::Type{CPU}) = "CPU"
Base.string(::Type{GPU}) = "GPU"

# Some types wrap Cxx types - this provides a hook for converting a Wrapper type into the
# unwrapped type.
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

# Hijack exception displaying
#
# We have to do some tricks with imports because the @exception macro isn't very robust.
import Base: showerror
const NGRAPH_ERRORS = [
    cxxt"ngraph::ngraph_error&",
    cxxt"ngraph::NodeValidationFailure&",
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

end # module

