struct Quadratic
    P
end

(q::Quadratic)(x) = sum((x .+ q.P) .* (x .+ q.P))

@testset "Testing Simple Training" begin
    # We'd expect P to be forced to zero to minimize this quadratic objective.
    P = param(Float32[1.0, 1.0, 1.0])
    f = Quadratic(P)

    x = Float32[1.0, 2.0, 3.0]
    backend = nGraph.Backend()
    X = nGraph.Tensor(backend, x)

    learning_rate = Float32(0.2)
    ex = nGraph.compile(backend, f, X; optimizer = nGraph.SGD(learning_rate))

    # We should have one implicit input/output: P
    @test length(ex.optimizer.inputs) == 1
    @test length(ex.optimizer.outputs) == 1

    @show read(first(ex.optimizer.inputs))
    @show read(first(ex.optimizer.outputs))

    Z = ex(X)
    @show read(Z)

    @show read(first(ex.optimizer.inputs))
    @show read(first(ex.optimizer.outputs))

    @test all(read(first(ex.optimizer.inputs)) .< read(first(ex.optimizer.outputs)))

    # We can compute what the derivatives should be. Note that the inputs/outputs should have
    # swapped, so now we're looking at the implicit inputs
    dfdx(i) = 2 * (f.P[i] + x[i])

    @test P[1] - learning_rate * dfdx(1) == read(ex.optimizer.inputs[1])[1]
    @test P[2] - learning_rate * dfdx(2) == read(ex.optimizer.inputs[1])[2]
    @test P[3] - learning_rate * dfdx(3) == read(ex.optimizer.inputs[1])[3]

    # Run this for a few more iterations - see if we can get the implicit parameters to be
    # negative the input
    for i in 1:20
        println(read(ex(X)))
    end

    @test read(ex(X))[] < 1E-3
end
