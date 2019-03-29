module Lib

using CxxWrap, Libdl

const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const DEPSDIR = joinpath(PKGDIR, "deps")
const MODELDIR = joinpath(PKGDIR, "models")

# Setup the LD_LIBRARY_PATH so libpmem can be found.
#
# This is a really annoying hack since libpmemobj cannot, by default, find libpmem.
ENV["LD_LIBRARY_PATH"] = joinpath(DEPSDIR, "usr", "lib")
#Libdl.dlopen(joinpath(DEPSDIR, "usr", "lib", "libtbb.so.2"))
#Libdl.dlopen(joinpath(DEPSDIR, "usr", "lib", "libcpu_backend.so"))
Libdl.dlopen(joinpath(DEPSDIR, "usr", "lib", "libngraph.so"))

@wrapmodule(joinpath(DEPSDIR, "libngraph-julia.so"))

function __init__()
    @initcxx
end

end
