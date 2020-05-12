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
        ex = nGraph.compile(backend, [X,Y], [Z])
        vX, vY, vZ = @tensors backend (x,y,Z)
        ex([vX, vY], [vZ])

        @test isapprox(parent(vZ), x .+ y)
    end

    @testset "AvgPool" begin
        x = randn(Float32, 30, 30, 10, 10)

        X = nGraph.Node(x)
        Z = nGraph.avgpool(X, (3,3); pad = 0, stride = (3,3))
        ex = nGraph.compile(backend, [X], [Z])
        vX, vZ = @tensors backend (x,Z)
        ex([vX], [vZ])

        # Do an equivalent Flux MeanPool
        m = Flux.MeanPool((3,3))
        @test isapprox(parent(vZ), m(x))
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
        vY, vZ = @tensors backend (y,Z)

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
        vX, vZ = @tensors backend (x,Z)
        ex([vX], [vZ])

        # Broadcast `x` into `z` for comparison.
        z = zeros(Float32, dims)
        z .= x

        @test isapprox(parent(vZ), z)
    end

    @testset "Convert Eltype" begin
        x = rand(Int32, 10)

        X = nGraph.Node(x)
        Z = nGraph.convert_eltype(Int64, X)
        ex = nGraph.compile(backend, [X], [Z])
        vX, vZ = @tensors backend (x,Z)
        ex([vX], [vZ])

        @test parent(vX) == parent(vZ)
    end

    @testset "Divide" begin
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)

        Z = X ./ Y
        ex = nGraph.compile(backend, [X,Y], [Z])
        vX, vY, vZ = @tensors backend (x,y,Z)
        ex([vX,vY], [vZ])

        @test isapprox(parent(vZ), x ./ y)
    end

    @testset "Concat" begin
        x = randn(Float32, 10, 1)
        y = randn(Float32, 10, 1)
        z = randn(Float32, 10, 1)

        X,Y,Z = nGraph.Node.((x,y,z))

        # Concat along dimensions 1 and 2
        A = cat(X,Y,Z; dims = 1)
        @test size(A) == (30,1)
        ex = nGraph.compile(backend, [X,Y,Z], [A])
        vX,vY,vZ,vA = @tensors backend (x,y,z,A)
        ex([vX,vY,vZ], [vA])
        @test parent(vA) == cat(x,y,z; dims = 1)

        B = cat(X,Y,Z; dims = 2)
        @test size(B) == (10,3)
        ex = nGraph.compile(backend, [X,Y,Z], [B])
        vX,vY,vZ,vB = @tensors backend (x,y,z,B)
        ex([vX,vY,vZ], [vB])
        @test parent(vB) == cat(x,y,z; dims = 2)
    end

    @testset "Dot" begin
        # 1x1
        x = randn(Float32, 10, 1)
        y = randn(Float32, 1, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)
        Z = X * Y
        ex = nGraph.compile(backend, [X,Y], [Z])
        tX,tY,tZ = @tensors backend (x,y,Z)
        ex([tX,tY], [tZ])

        @test isapprox(parent(tZ), x * y)

        # 2x2
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)
        Z = X * Y
        ex = nGraph.compile(backend, [X,Y], [Z])
        tX,tY,tZ = @tensors backend (x,y,Z)
        ex([tX,tY], [tZ])

        @test isapprox(parent(tZ), x * y)

        # Mixed
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 1)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)
        Z = X * Y
        ex = nGraph.compile(backend, [X,Y], [Z])
        tX,tY,tZ = @tensors backend (x,y,Z)
        ex([tX,tY], [tZ])
        @test isapprox(parent(tZ), x * y)

        x = randn(Float32, 1, 10)
        y = randn(Float32, 10, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)
        Z = X * Y
        ex = nGraph.compile(backend, [X,Y], [Z])
        tX,tY,tZ = @tensors backend (x,y,Z)
        ex([tX,tY], [tZ])
        @test isapprox(parent(tZ), x * y)
    end

    @testset "GOE" begin
    end

    #####
    ##### Misc Element-Wise + Binary Ops
    #####

    @testset "Log" begin
        # Add 1 to make sure the log is more or less nicely behaved.
        x = rand(Float64, 10, 10) .+ 1
        X = nGraph.Node(x)

        Z = log.(X)
        ex = nGraph.compile(backend, [X], [Z])
        tX,tZ = @tensors backend (x,Z)
        ex([tX],[tZ])
        @test isapprox(parent(tZ), log.(x))
    end

    @testset "Negative" begin
        # Add 1 to make sure the log is more or less nicely behaved.
        x = randn(Float64, 10, 10)
        X = nGraph.Node(x)

        Z = (-).(X)
        ex = nGraph.compile(backend, [X], [Z])
        tX,tZ = @tensors backend (x,Z)
        ex([tX],[tZ])
        @test isapprox(parent(tZ), (-).(x))
    end

    @testset "Maximum" begin
        x = randn(Float64, 10, 10)
        y = randn(Float64, 10, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)

        Z = max.(X,Y)
        ex = nGraph.compile(backend, [X,Y], [Z])
        tX, tY, tZ = @tensors backend (x,y,Z)
        ex([tX, tY],[tZ])
        @test isapprox(parent(tZ), max.(x,y))
    end

    @testset "Minimum" begin
        x = randn(Float64, 10, 10)
        y = randn(Float64, 10, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)

        Z = min.(X,Y)
        ex = nGraph.compile(backend, [X,Y], [Z])
        tX, tY, tZ = @tensors backend (x,y,Z)
        ex([tX, tY],[tZ])
        @test isapprox(parent(tZ), min.(x,y))
    end

    @testset "Multiply" begin
        x = randn(Float64, 10, 10)
        y = randn(Float64, 10, 10)

        X = nGraph.Node(x)
        Y = nGraph.Node(y)

        Z = X .* Y
        ex = nGraph.compile(backend, [X,Y], [Z])
        tX, tY, tZ = @tensors backend (x,y,Z)
        ex([tX, tY],[tZ])
        @test isapprox(parent(tZ), x .* y)
    end
end

# @testset "Reshape" begin
#     backend = nGraph.Backend()
#
#     x = rand(Float32, 100)
#     f = nGraph.compile(backend, x -> reshape(x, 1, :), x)
#     Z = f()
#     @test reshape(x, 1, :) == Z.base
#
#     x = rand(Float32, 2, 2, 2)
#     g = x -> reshape(x, :)
#     f = nGraph.compile(backend, g, x)
#     @test g(x) == f().base
#
#     x = rand(Float32, 3, 2, 1)
#     g = x -> reshape(x, 1, 2, 3)
#     f = nGraph.compile(backend, g, x)
#     @test g(x) == f().base
#
#     # More extravagent reshape
#     x = rand(Float32, 1, 2, 3, 4, 5, 6)
#     g = x -> reshape(x, 6, 5, :, 3, 2)
#
#     N = nGraph.Node(x)
#     M = g(N)
#     @test size(M) == (6, 5, 4, 3, 2)
#     f = nGraph.compile(backend, g, x)
#
#     @test g(x) == f().base
# end
#
# @testset "Softmax" begin
#     backend = nGraph.Backend()
#
#     # 1D case
#     x = rand(Float32, 100)
#     z = softmax(x)
#     f = nGraph.compile(backend, softmax, x)
#     @test isapprox(z, f().base)
#
#     # 2D case
#     x = rand(Float32, 100, 100)
#     z = softmax(x)
#     f = nGraph.compile(backend, softmax, x)
#     @test isapprox(z, f().base)
# end
