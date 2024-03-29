macro tensors(x...)
    return :(nGraph.Tensor.(($(esc(x[1])),), $(esc(x[2]))))
end

@testset "Ops" begin
    # NOTE: Tests are not in alphabetical order because we need the existence of some
    # ops to bootstrap the testing of other ops.
    #
    # I've added tests here in the order I created them, so the ordering should reflect
    # the bootstrap requirement order.

    backend = nGraph.Backend{:CPU}()

    @testset "Add" begin
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)

        Z = X + Y

        # Compile and run.
        ex = nGraph.compile(backend, [X, Y], [Z])
        vX, vY, vZ = @tensors backend (x, y, Z)
        ex([vX, vY], [vZ])
        @test isapprox(parent(vZ), x .+ y)

        # Broadcasting variant
        Z2 = X .+ Y
        ex = nGraph.compile(backend, [X, Y], [Z2])
        vX, vY, vZ2 = @tensors backend (x, y, Z2)
        ex([vX, vY], [vZ2])
        @test isapprox(parent(vZ2), x .+ y)

        # Traced Variant
        f = nGraph.compile(backend, +, x, y)
        @test parent(f()) == x + y

        f = nGraph.compile(backend, (a, b) -> a .+ b, x, y)
        @test parent(f()) == x .+ y
    end

    @testset "AvgPool" begin
        x = randn(Float32, 30, 30, 10, 10)

        X = nGraph.Node(x)
        Z = nGraph.avgpool(X, (3, 3); pad = 0, stride = (3, 3))
        ex = nGraph.compile(backend, [X], [Z])
        vX, vZ = @tensors backend (x, Z)
        ex([vX], [vZ])

        # Do an equivalent Flux MeanPool
        m = Flux.MeanPool((3, 3))
        @test isapprox(parent(vZ), m(x))

        # Traced Variant
        f = nGraph.compile(
            backend,
            x -> nGraph.avgpool(x, (3, 3); pad = 0, stride = (3, 3)),
            x,
        )
        @test isapprox(parent(f()), m(x))
    end

    # Test that constants get slurped up by nGraph
    # TODO: Run over all element types exported by nGraph.
    @testset "Constant" begin
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 10)

        X = nGraph.constant(x)
        Y = nGraph.Node(y)
        Z = X + Y
        ex = nGraph.compile(backend, [Y], [Z])
        vY, vZ = @tensors backend (y, Z)

        ex([vY], [vZ])
        @test isapprox(parent(vZ), x .+ y)
    end

    @testset "Broadcast" begin
        x = randn(Float32, 10)
        dims = (10, 8)

        X = nGraph.Node(x)
        Z = nGraph.broadcast(X, dims)
        @test isa(Z, nGraph.Node{Float32,2})
        @test size(Z) == dims

        ex = nGraph.compile(backend, [X], [Z])
        vX, vZ = @tensors backend (x, Z)
        ex([vX], [vZ])

        # Broadcast `x` into `z` for comparison.
        z = zeros(Float32, dims)
        z .= x

        @test isapprox(parent(vZ), z)

        # Broadcasting for tracing
        x = randn(Float32, 10)
        y = randn(Float32, 10, 10)
        f = nGraph.compile(backend, (a, b) -> (a .+ b), x, y)
        @test parent(f()) == x .+ y
    end

    @testset "Convert Eltype" begin
        x = rand(Int32, 10)

        X = nGraph.Node(x)
        Z = nGraph.convert_eltype(Int64, X)
        ex = nGraph.compile(backend, [X], [Z])
        vX, vZ = @tensors backend (x, Z)
        ex([vX], [vZ])

        @test parent(vX) == parent(vZ)
    end

    @testset "Divide" begin
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)

        Z = X ./ Y
        ex = nGraph.compile(backend, [X, Y], [Z])
        vX, vY, vZ = @tensors backend (x, y, Z)
        ex([vX, vY], [vZ])

        @test isapprox(parent(vZ), x ./ y)
    end

    @testset "Concat" begin
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 10)
        z = randn(Float32, 10, 10)

        X, Y, Z = nGraph.Node.((x, y, z))

        # Concat along dimensions 1 and 2
        A = cat(X, Y, Z; dims = 1)
        @test size(A) == (30, 10)
        ex = nGraph.compile(backend, [X, Y, Z], [A])
        vX, vY, vZ, vA = @tensors backend (x, y, z, A)
        ex([vX, vY, vZ], [vA])
        @test parent(vA) == cat(x, y, z; dims = 1)

        B = cat(X, Y, Z; dims = 2)
        @test size(B) == (10, 30)
        ex = nGraph.compile(backend, [X, Y, Z], [B])
        vX, vY, vZ, vB = @tensors backend (x, y, z, B)
        ex([vX, vY, vZ], [vB])
        @test parent(vB) == cat(x, y, z; dims = 2)
    end

    @testset "Dot" begin
        # 1x1
        x = randn(Float32, 10, 1)
        y = randn(Float32, 1, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)
        Z = X * Y
        ex = nGraph.compile(backend, [X, Y], [Z])
        tX, tY, tZ = @tensors backend (x, y, Z)
        ex([tX, tY], [tZ])

        @test isapprox(parent(tZ), x * y)

        # 2x2
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)
        Z = X * Y
        ex = nGraph.compile(backend, [X, Y], [Z])
        tX, tY, tZ = @tensors backend (x, y, Z)
        ex([tX, tY], [tZ])

        @test isapprox(parent(tZ), x * y)

        # Mixed
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 1)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)
        Z = X * Y
        ex = nGraph.compile(backend, [X, Y], [Z])
        tX, tY, tZ = @tensors backend (x, y, Z)
        ex([tX, tY], [tZ])
        @test isapprox(parent(tZ), x * y)

        x = randn(Float32, 1, 10)
        y = randn(Float32, 10, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)
        Z = X * Y
        ex = nGraph.compile(backend, [X, Y], [Z])
        tX, tY, tZ = @tensors backend (x, y, Z)
        ex([tX, tY], [tZ])
        @test isapprox(parent(tZ), x * y)
    end

    @testset "GOE" begin end

    #####
    ##### Misc Element-Wise + Binary Ops
    #####

    @testset "Log" begin
        # Add 1 to make sure the log is more or less nicely behaved.
        x = rand(Float64, 10, 10) .+ 1
        X = nGraph.Node(x)

        Z = log.(X)
        ex = nGraph.compile(backend, [X], [Z])
        tX, tZ = @tensors backend (x, Z)
        ex([tX], [tZ])
        @test isapprox(parent(tZ), log.(x))
    end

    @testset "Negative" begin
        # Add 1 to make sure the log is more or less nicely behaved.
        x = randn(Float64, 10, 10)
        X = nGraph.Node(x)

        Z = (-).(X)
        ex = nGraph.compile(backend, [X], [Z])
        tX, tZ = @tensors backend (x, Z)
        ex([tX], [tZ])
        @test isapprox(parent(tZ), (-).(x))
    end

    @testset "MaxPool" begin
        x = randn(Float32, 30, 30, 10, 10)

        X = nGraph.Node(x)
        Z = nGraph.maxpool(X, (3, 3); pad = 0, stride = (3, 3))
        ex = nGraph.compile(backend, [X], [Z])
        vX, vZ = @tensors backend (x, Z)
        ex([vX], [vZ])

        # Do an equivalent Flux MeanPool
        m = Flux.MaxPool((3, 3))
        @test isapprox(parent(vZ), m(x))

        # Traced Variant
        f = nGraph.compile(
            backend,
            x -> nGraph.maxpool(x, (3, 3); pad = 0, stride = (3, 3)),
            x,
        )
        @test isapprox(parent(f()), m(x))
    end

    @testset "Maximum" begin
        x = randn(Float64, 10, 10)
        y = randn(Float64, 10, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)

        Z = max.(X, Y)
        ex = nGraph.compile(backend, [X, Y], [Z])
        tX, tY, tZ = @tensors backend (x, y, Z)
        ex([tX, tY], [tZ])
        @test isapprox(parent(tZ), max.(x, y))
    end

    @testset "Minimum" begin
        x = randn(Float64, 10, 10)
        y = randn(Float64, 10, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)

        Z = min.(X, Y)
        ex = nGraph.compile(backend, [X, Y], [Z])
        tX, tY, tZ = @tensors backend (x, y, Z)
        ex([tX, tY], [tZ])
        @test isapprox(parent(tZ), min.(x, y))
    end

    @testset "Multiply" begin
        x = randn(Float64, 10, 10)
        y = randn(Float64, 10, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)

        Z = X .* Y
        ex = nGraph.compile(backend, [X, Y], [Z])
        tX, tY, tZ = @tensors backend (x, y, Z)
        ex([tX, tY], [tZ])
        @test isapprox(parent(tZ), x .* y)

        # Test broadcasting with a number in both directions
        Z = X .* 2
        ex = nGraph.compile(backend, [X], [Z])
        tX, tZ = @tensors backend (x, Z)
        ex([tX, tZ], [tZ])
        @test isapprox(parent(tZ), x .* 2)

        Z = 2 .* X
        ex = nGraph.compile(backend, [X], [Z])
        tX, tZ = @tensors backend (x, Z)
        ex([tX, tZ], [tZ])
        @test isapprox(parent(tZ), x .* 2)
    end

    @testset "Relu" begin
        x = randn(Float32, 100, 100)
        X = nGraph.Node(x)
        Y = Flux.relu.(X)
        ex = nGraph.compile(backend, [X], [Y])
        tX, tY = @tensors backend (x, Y)
        ex([tX], [tY])
        @test parent(tY) ≈ Flux.relu.(x)

        f = nGraph.compile(backend, i -> Flux.relu.(i), x)
        @test parent(f()) ≈ Flux.relu.(x)
    end

    @testset "Reshape" begin
        tests = Any[
            (100) => (1, :),
            (2, 2, 2) => (:,),
            (3, 2, 1) => (1, 2, 3),
            (1, 2, 3, 4, 5, 6) => (6, 5, :, 3, 2),
        ]

        for (oldshape, newshape) in tests
            x = rand(Float32, oldshape...)

            # Direct Invocation
            X = nGraph.Node(x)
            Y = reshape(X, newshape...)
            ex = nGraph.compile(backend, [X], [Y])
            tX, tY = @tensors backend (x, Y)
            ex([tX], [tY])
            @test parent(tY) == reshape(x, newshape...)

            # Traced Invocation
            f = nGraph.compile(backend, i -> reshape(i, newshape...), x)
            @test parent(f()) == reshape(x, newshape...)
        end
    end

    @testset "Sigmoid" begin
        x = randn(Float32, 100, 100)
        X = nGraph.Node(x)
        Y = Flux.σ.(X)
        ex = nGraph.compile(backend, [X], [Y])
        tX, tY = @tensors backend (x, Y)
        ex([tX], [tY])
        @test parent(tY) ≈ Flux.σ.(x)

        f = nGraph.compile(backend, i -> Flux.σ.(i), x)
        @test parent(f()) ≈ Flux.σ.(x)
    end

    @testset "Softmax" begin
        # 1D case
        x = rand(Float32, 100)
        z = softmax(x)
        f = nGraph.compile(backend, softmax, x)
        @test isapprox(z, parent(f()))

        # 2D case
        x = rand(Float32, 100, 100)
        z = softmax(x)
        f = nGraph.compile(backend, softmax, x)
        @test isapprox(z, parent(f()))
    end

    @testset "Subtract" begin
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)

        Z = X - Y

        # Compile and run.
        ex = nGraph.compile(backend, [X, Y], [Z])
        vX, vY, vZ = @tensors backend (x, y, Z)
        ex([vX, vY], [vZ])
        @test isapprox(parent(vZ), x .- y)

        # Broadcasting variant
        Z2 = X .- Y
        ex = nGraph.compile(backend, [X, Y], [Z2])
        vX, vY, vZ2 = @tensors backend (x, y, Z2)
        ex([vX, vY], [vZ2])
        @test isapprox(parent(vZ2), x .- y)

        # Traced Variant
        f = nGraph.compile(backend, -, x, y)
        @test parent(f()) == x - y

        f = nGraph.compile(backend, (a, b) -> a .+ b, x, y)
        @test parent(f()) == x .+ y
    end

    @testset "Sum" begin
        x = randn(Float32, 10, 10)
        functions = [
            sum,
            #x -> sum(x; dims = 1),
            x -> sum(x; dims = 2),
            x -> sum(x; dims = (1, 2)),
            x -> sum(x; dims = (2, 1)),
        ]

        for (i, fn) in enumerate(functions)
            @show i
            g = nGraph.compile(backend, fn, x)
            @test all(isapprox.(parent(g()), fn(x)))
        end
    end
end


