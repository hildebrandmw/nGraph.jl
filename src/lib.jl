module Lib

using CxxWrap, Libdl

const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const DEPSDIR = joinpath(PKGDIR, "deps")
const MODELDIR = joinpath(PKGDIR, "models")

# Library opening
Libdl.dlopen(joinpath(DEPSDIR, "usr", "lib", "libngraph.so"))
Libdl.dlopen(joinpath(DEPSDIR, "usr", "lib", "libcpu_backend.so"))
@wrapmodule(joinpath(DEPSDIR, "libngraph-julia.so"))

function __init__()
    @initcxx
end

# Types wrappers forwarded to "types.jl"

end
