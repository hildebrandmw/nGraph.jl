module Lib

using CxxWrap, Libdl

const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const DEPSDIR = joinpath(PKGDIR, "deps")
const MODELDIR = joinpath(PKGDIR, "models")

# Library opening
Libdl.dlopen(joinpath(DEPSDIR, "usr", "lib", "libngraph.so"))
@wrapmodule(joinpath(DEPSDIR, "libngraph-julia.so"))

function __init__()
    @initcxx
end

# Materialize a nGraph "Shape" as a Julia array
Base.getindex(x::ShapeRef, i) = shape_getindex(x,i)  # defined in c++ code
Base.length(x::ShapeRef) = shape_length(x)           # defined in c++ code
shape(x::ShapeRef) = [x[i] for i in 0:length(x)-1]

end
