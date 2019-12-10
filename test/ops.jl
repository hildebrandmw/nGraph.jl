using Flux
using NNlib

# The order of the operations are is determined roughly in the order of implemntation.
# This is to make sure "lower-level" funcitonality is working before higher level functionality.
#
# For example, NNlib.conv may require reversing axes - so we test for axis reversing first.
@testset "Ops" begin
    backend = nGraph.Backend{nGraph.CPU}()

    #####
    ##### Add
    #####

    A = rand(Float32, 2, 2)
    B = rand(Float32, 2, 2)

    F = nGraph.compile(backend, +, A, B)
    @test parent(F()) == A + B

    # Do the broadcasting version of add as well.
    f = (x, y) -> x .+ y
    F = nGraph.compile(backend, f, A, B)
    @test parent(F()) == f(A, B)

    B = rand(Float32, 2)
    F = nGraph.compile(backend, f, A, B)
    @test parent(F()) == f(A, B)

    B = rand(Float32)
    F = nGraph.compile(backend, f, A, B)
    @test parent(F()) == f(A, B)

    # TODO: Make broadcasting fully equivalent to Julia's
    # B = rand(Float32, 1, 2)
    # F = nGraph.compile(backend, f, A, B)
    # @test parent(F()) == f(A, B)

    #####
    ##### Concat
    #####

    A = rand(Float32, 2, 2)

    # Go up to 4 args
    f = (x...) -> cat(x...; dims = 1)
    F = nGraph.compile(backend, f, A)
    @test parent(F()) == f(A)

    F = nGraph.compile(backend, f, A, A)
    @test parent(F()) == f(A, A)

    F = nGraph.compile(backend, f, A, A, A)
    @test parent(F()) == f(A, A, A)

    F = nGraph.compile(backend, f, A, A, A, A)
    @test parent(F()) == f(A, A, A, A)

    # Try cat along another dimension.
    f = (x...) -> cat(x...; dims = 2)
    F = nGraph.compile(backend, f, A)
    @test parent(F()) == f(A)

    F = nGraph.compile(backend, f, A, A)
    @test parent(F()) == f(A, A)

    F = nGraph.compile(backend, f, A, A, A)
    @test parent(F()) == f(A, A, A)

    F = nGraph.compile(backend, f, A, A, A, A)
    @test parent(F()) == f(A, A, A, A)

    #####
    ##### Constants
    #####

    A = rand(Float32, 2, 2)
    y = nGraph.Node(nGraph.constant(10))
    @test eltype(y) == Int64
    @test size(y) == ()
    @test nGraph.description(y) == "Constant"

    y = nGraph.Node(nGraph.constant(A))
    @test eltype(y) == eltype(A)
    @test size(y) == size(A)
    @test nGraph.description(y) == "Constant"

    #####
    ##### Convert Eltype
    #####

    A = rand(Float32, 2, 2)
    N = nGraph.NodeTyped(A)
    y = nGraph.Node(nGraph.convert_eltype(Float64, N))
    @test eltype(y) == Float64
    @test size(y) == size(A)

    y = nGraph.Node(nGraph.convert_eltype(eltype(A), N))
    @test eltype(y) == eltype(A)
    @test size(y) == size(A)

    # When compiling and coming across a wild array, it will be turned into a constant.
    # We verify this behavior and that the array is captured correctly.
    f = x -> x .+ 1
    F = nGraph.compile(backend, f, A)
    @test parent(F()) == f(A)
    @test eltype(parent(F())) == eltype(f(A))

    f = x -> x .+ A
    F = nGraph.compile(backend, f, A)
    @test parent(F()) == f(A)

    #####
    ##### Permute Dims
    #####

    A = rand(Float32, 3, 3, 16, 16)
    f = x -> permutedims(x, (2, 1, 3, 4))

    F = nGraph.compile(backend, f, A)
    @test parent(F()) == f(A)

    #####
    ##### Reverse
    #####

    # This is the equivalent functionality in Julia of the nGraph `Reverse` node.
    function nGraph.reverse_axes(A::AbstractArray, axes)
        i = ntuple(i -> in(i, axes) ? (size(A, i):-1:1) : Base.Colon(), ndims(A))
        return A[i...]
    end

    A = rand(Float32, 2, 2, 2)

    axes = [
        (1,),
        (2,),
        (3,),
        (1,2),
        (1,3),
        (2,3),
        (1,2,3)
    ]

    for ax in axes
        f = x -> nGraph.reverse_axes(x, ax)
        F = nGraph.compile(backend, f, A)
        @test parent(F()) == f(A)
    end

    #####
    ##### Convolution
    #####

    # Try with a few different variations
    pads = (0, 1)
    strides = (1, 2)
    flipkernel = (true, false)

    channels = (10, 16)
    filter = ((3, 3), (7, 7))

    iter = Iterators.product(
        pads,
        strides,
        flipkernel,
        channels,
        filter
    )

    for (p, s, fl, ch, wh) in iter
        x = randn(Float32, 14, 14, ch, ch)
        w = randn(Float32, wh..., ch, ch)
        dims = NNlib.DenseConvDims(size(x), size(w);
            stride = s,
            padding = p,
            flipkernel = fl,
        )

        f = (x, w) -> NNlib.conv(x, w, dims)
        F = nGraph.compile(backend, f, x, w)
        @test isapprox(parent(F()), f(x, w))
    end

    #####
    ##### Divide
    #####

    # Element-wise division
    A = rand(Float32, 2, 2)
    B = rand(Float32, 2, 2)
    f = (a, b) -> a ./ b
    F = nGraph.compile(backend, f, A, B)
    @test parent(F()) == f(A, B)

    # Scalar division
    A = 1
    B = 2
    F = nGraph.compile(backend, /, A, B)
    @test parent(F())[] == A / B

    F = nGraph.compile(backend, //, A, B)
    @test parent(F())[] == A // B

    f = x -> x / 2
    F = nGraph.compile(backend, f, A)
    @test parent(F())[] == f(A)

    #####
    ##### Reshape
    #####

    A = rand(Float32, 100)
    f = x -> reshape(x, 1, :)
    F = nGraph.compile(backend, f, A)
    @test parent(F()) == f(A)

    A = rand(Float32, 2, 2, 2)
    f = x -> reshape(x, :)
    F = nGraph.compile(backend, f, A)
    @test parent(F()) == f(A)

    A = rand(Float32, 3, 2, 1)
    f = x -> reshape(x, 1, 2, 3)
    F = nGraph.compile(backend, f, A)
    @test parent(F()) == f(A)

    # More extravagent reshape
    A = rand(Float32, 1, 2, 3, 4, 5, 6)
    f = x -> reshape(x, 6, 5, :, 3, 2)
    F = nGraph.compile(backend, f, A)
    @test parent(F()) == f(A)
end

