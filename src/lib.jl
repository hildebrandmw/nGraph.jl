module Lib

using CxxWrap, Libdl

const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const DEPSDIR = joinpath(PKGDIR, "deps")
const MODELDIR = joinpath(PKGDIR, "models")

# Setup the LD_LIBRARY_PATH so libpmem can be found.
#
# This is a really annoying hack since libpmemobj cannot, by default, find libpmem.
#ENV["LD_LIBRARY_PATH"] = join((joinpath(DEPSDIR, "usr", "Lib"), get(ENV, "LD_LIBRARY_PATH", "")), ":")
#@show ENV["LD_LIBRARY_PATH"]

# Library opening
Libdl.dlopen(joinpath(DEPSDIR, "usr", "lib", "libpmem.so.1"))
Libdl.dlopen(joinpath(DEPSDIR, "usr", "lib", "libngraph.so"))
@wrapmodule(joinpath(DEPSDIR, "libngraph-julia.so"))

function __init__()
    @initcxx
end

end
