struct Quadratic
    P
end
Flux.@functor Quadratic

(q::Quadratic)(x) = sum((x .+ q.P) .* (x .+ q.P))

@testset "Testing Simple Training" begin
    # We'd expect P to be forced to zero to minimize this quadratic objective.
    P = Float32[1.0, 1.0, 1.0]
    f = Quadratic(P)

    x = Float32[1.0, 2.0, 3.0]
    backend = nGraph.Backend()

    learning_rate = Float32(0.2)
    ex = nGraph.compile(backend, f, x; optimizer = nGraph.SGD(learning_rate))

    # We should have one implicit input/output: P
    @test length(ex.optimizer.inputs) == 1
    @test length(ex.optimizer.outputs) == 1

    @show parent(first(ex.optimizer.inputs))
    @show parent(first(ex.optimizer.outputs))

    Z = ex()
    @show parent(Z)

    @show parent(first(ex.optimizer.inputs))
    @show parent(first(ex.optimizer.outputs))

    @test all(parent(first(ex.optimizer.inputs)) .< parent(first(ex.optimizer.outputs)))

    # We can compute what the derivatives should be. Note that the inputs/outputs should have
    # swapped, so now we're looking at the implicit inputs
    dfdx(i) = 2 * (f.P[i] + x[i])

    @test P[1] - learning_rate * dfdx(1) == parent(ex.optimizer.inputs[1])[1]
    @test P[2] - learning_rate * dfdx(2) == parent(ex.optimizer.inputs[1])[2]
    @test P[3] - learning_rate * dfdx(3) == parent(ex.optimizer.inputs[1])[3]

    # Run this for a few more iterations - see if we can get the implicit parameters to be
    # negative the input
    for i in 1:20
        println(parent(ex()))
    end

    @test parent(ex())[] < 1E-3
end
