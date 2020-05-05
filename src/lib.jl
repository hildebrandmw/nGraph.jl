module Lib

using CxxWrap, Libdl

const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const DEPSDIR = joinpath(PKGDIR, "deps")
const MODELDIR = joinpath(PKGDIR, "models")

# Lib64 vs Lib schenanigans
const _LIB64DIR = joinpath(DEPSDIR, "usr", "lib64")
const _LIBDIR = joinpath(DEPSDIR, "usr", "lib")
const LIBDIR = ispath(_LIB64DIR) ? "lib64" : "lib"

const _flags = Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND | Libdl.RTLD_GLOBAL
Libdl.dlopen(joinpath(DEPSDIR, "usr", LIBDIR, "libngraph.so"), _flags)

@wrapmodule(joinpath(DEPSDIR, "libngraph-julia.so"))

function __init__()
    @initcxx
end

end
