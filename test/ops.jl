using Flux
using NNlib

macro dotest(backend, f, args...)
    backend = esc(backend)
    f = esc(f)
    args = esc.(args)
    return quote
        F = nGraph.compile($backend, $(f), $(args...))
        @test isapprox(parent(F()), $(f)($(args...)))
    end
end

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
    @dotest backend f A

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
        @dotest backend f A
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
        @dotest backend f x w
    end

    #####
    ##### Divide
    #####

    # Element-wise division
    A = rand(Float32, 2, 2)
    B = rand(Float32, 2, 2)
    f = (a, b) -> a ./ b
    @dotest backend f A B

    # Scalar division
    A = 1
    B = 2
    F = nGraph.compile(backend, /, Float32(A), Float32(B))
    @test parent(F())[] == A / B

    F = nGraph.compile(backend, //, Float32(A), Float32(B))
    @test parent(F())[] == A // B

    f = x -> x / Float32(2)
    F = nGraph.compile(backend, f, Float32(A))
    @test parent(F())[] == f(A)

    #####
    ##### Reshape
    #####

    A = rand(Float32, 100)
    f = x -> reshape(x, 1, :)
    @dotest backend f A

    A = rand(Float32, 2, 2, 2)
    f = x -> reshape(x, :)
    @dotest backend f A

    A = rand(Float32, 3, 2, 1)
    f = x -> reshape(x, 1, 2, 3)
    @dotest backend f A

    # More extravagent reshape
    A = rand(Float32, 1, 2, 3, 4, 5, 6)
    f = x -> reshape(x, 6, 5, :, 3, 2)
    @dotest backend f A

    #####
    ##### Log
    #####
    
    A = rand(Float32, 2, 2) 
    f = x -> log.(x)
    @dotest backend f A

    #####
    ##### Max
    #####
    
    A = rand(Float32, 2, 2)
    B = rand(Float32, 2, 2)

    f = (x, y) -> max.(x, y)
    @dotest backend f A B

    #####
    ##### Min
    #####
    
    A = rand(Float32, 2, 2)
    B = rand(Float32, 2, 2)

    f = (x, y) -> min.(x, y)
    @dotest backend f A B

    #####
    ##### Relu
    #####
    
    A = rand(Float32, 2, 2) 
    f = x -> Flux.relu.(x)
    @dotest backend f A

    #####
    ##### Sigmoid
    #####
    
    A = rand(Float32, 2, 2)
    f = x -> Flux.σ.(x)
    @dotest backend f A

    #####
    ##### Sqrt
    #####
    
    A = rand(Float32, 2, 2)
    f = x -> sqrt.(x)
    @dotest backend f A

    #####
    ##### Tanh
    #####
    
    A = rand(Float32, 2, 2)
    f = x -> tanh.(x)
    @dotest backend f A

    #####
    ##### Subtract
    #####

    A = rand(Float32, 2, 2)
    B = rand(Float32, 2, 2)

    f = (x, y) -> x .- y
    @dotest backend f A B

    #####
    ##### Negative
    #####
    
    A = rand(Float32, 2, 2)
    f = x -> -x
    @dotest backend f A

    g(x) = -x
    f = x -> g.(x)
    @dotest backend f A

    #####
    ##### Power
    #####
    
    A = rand(Float32, 2, 2)
    B = rand(Float32)
    f = (x, y) -> x .^ y
    @dotest backend f A B

    #####
    ##### Softmax
    #####
    
    A = rand(Float32, 5, 5)
    f = x -> NNlib.softmax(x)
    @dotest backend f A

    f = x -> NNlib.softmax(reshape(x, :))
    @dotest backend f A

    #####
    ##### Dot
    #####
    
    # Do a series of right and left multiplication.
    cA = rand(Float32, 2, 2) 
    cB = rand(Float32, 2)

    A = rand(Float32, 2, 2)
    B = rand(Float32, 2, 2)
    C = rand(Float32, 2)
    D = rand(Float32, 1, 2)

    # Test ambiguity resolution
    f = x -> x * cA
    @dotest backend f A

    f = x -> x * cB
    @dotest backend f A

    f = x -> cA * x
    @dotest backend f A

    # Just general multiplication
    f = (x, y) -> x * y
    @dotest backend f A B
    @dotest backend f A C
    @dotest backend f D A
    @dotest backend f D C
end

