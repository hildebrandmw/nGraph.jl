using nGraph
using Test

@testset "Simple Example" begin
    a = Float32.([1,2,3,4])
    b = Float32.([1,2,3,4])
    c = Float32.([1,2,3,4])

    na = nGraph.param(a)
    nb = nGraph.param(b)
    nc = nGraph.param(c)

    x = nc * (na + nb)

    # Construct node and parameter vectors
    parameters = nGraph.Lib.ParameterVector() 
    nGraph.Lib.push!(parameters, na)
    nGraph.Lib.push!(parameters, nb)
    nGraph.Lib.push!(parameters, nc)

    nodes = nGraph.Lib.NodeVector()
    nGraph.Lib.push!(nodes, x)

    f = nGraph.Lib.make_function(nodes, parameters)
    backend = nGraph.Lib.create("CPU")

    executable = nGraph.Lib.compile(backend, f, false)

    # Hack for now
    element_type = nGraph._element(eltype(a))
    ta = nGraph.Lib.create_tensor(backend, element_type, UInt[4])
    tb = nGraph.Lib.create_tensor(backend, element_type, UInt[4])
    tc = nGraph.Lib.create_tensor(backend, element_type, UInt[4])
    tx = nGraph.Lib.create_tensor(backend, element_type, UInt[4])

    nGraph.Lib.tensor_write(ta, Ptr{Cvoid}(pointer(a)), zero(UInt64), UInt64(sizeof(a)))
    nGraph.Lib.tensor_write(tb, Ptr{Cvoid}(pointer(b)), zero(UInt64), UInt64(sizeof(b)))
    nGraph.Lib.tensor_write(tc, Ptr{Cvoid}(pointer(c)), zero(UInt64), UInt64(sizeof(c)))

    nGraph.Lib.call(executable, Any[tx], Any[ta, tb, tc])

    x_expected = c .* (a .+ b)
    x_ngraph = similar(x_expected)
    @show x_ngraph
    nGraph.Lib.tensor_read(tx, Ptr{Cvoid}(pointer(x_ngraph)), zero(UInt64), UInt64(sizeof(x_ngraph)))
    @show x_ngraph

    @test x_expected == x_ngraph
end
