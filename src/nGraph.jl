module nGraph

# stdlib
using Random

# external dependcies
import Cassette
#using Distributions
import Flux
import NNlib
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

ngraph_convert(x) = x

# Expected Transformation
#
# f(a,b,c) -> Lib.f(ngraph_convert(a), ngraph_convert(b), ngraph_convert(c))
#
# Pretty straightforward.
#
# If it becomes relevant, I've left code in that will do the transformation
#
# f(a,b,c) ->
#    #genysym1 = ngraph_convert(a)
#    #genysym2 = ngraph_convert(b)
#    #genysym2 = ngraph_convert(c)
#    GC.@preserve #gensym1 #gensym2 @gensym3 begin
#       Lib.f(#gensym1, #gensym2, @gensym3)
#    end
macro ngraphcall(expr)
    if expr.head != :call
        error("Only call `@ngraphcall` on function calls")
    end

    # Prefix "Lib." in front of the function call
    fname = expr.args[1]
    if !isa(fname, Symbol)
        error("Don't put the modlue `Lib` in front of the function call")
    end
    fname = :(Lib.$fname)

    args = expr.args[2:end]
    for i in eachindex(args)
        args[i] = :(ngraph_convert($(esc(args[i]))))
    end

    return :($fname($(args...)))

    # vars = [gensym() for i in 1:length(args)]

    # # Wrap the arguments in ngraph_convert
    # newargs = []
    # for (var, arg) in zip(vars, args)
    #     newarg = :($(esc(var)) = ngraph_convert($(esc(arg))))
    #     push!(newargs, newarg)
    # end

    # newcall = :($fname($(vars...)))

    # return quote
    #     $(newargs...)
    #     GC.@preserve $(vars...) $newcall
    # end
end

include("types.jl")
include("ops.jl")
include("tracer/tracer.jl")
include("tracer/lib.jl")

flip1(x::AbstractArray{T,4}) where {T} = x[size(x, 1):-1:1, :, :, :]
flip2(x::AbstractArray{T,4}) where {T} = x[:, size(x, 2):-1:1, :, :]

function tryconv()
    x = randn(Float32, 16, 16, 128, 128)
    w = randn(Float32, 3, 3, 128, 128)
    backend = nGraph.Backend{:CPU}()

    f = nGraph.compile(backend, convolution, x, w)
    z = parent(f())

    ops = [
        flip1,
        flip2,
        (flip1 ∘ flip2)
    ]

    for (xop, wop) in Iterators.product(ops, ops)
        xp = xop(x)
        wp = wop(w)

        y = NNlib.conv(xp, wp; flipped = true)
        @assert size(y) == size(z)
        if isapprox(z, y)
            @show (xp, wp)
        end
    end
end

end # module
