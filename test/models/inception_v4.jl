function stem(input)
    println("Stem")
    # Stem
    x = Conv((3, 3), 3 => 32, relu; pad = 0, stride = 2)(input)
    x = Conv((3, 3), 32 => 32, relu; pad = 0)(x)
    x = Conv((3, 3), 32 => 64, relu; pad = 1)(x)

    # Split then merge
    b0 = MaxPool((3,3); pad = 0, stride = 2)(x)
    b1 = Conv((3,3), 64 => 96, relu; pad = 0, stride = 2)(x)
    x = cat(b0, b1; dims = 3)

    # Second Split
    b0 = Conv((1,1), 160 => 64, relu; pad = 0, stride = 1)(x)
    b0 = Conv((3,3), 64 => 96, relu; pad = 0)(b0)

    b1 = Conv((1,1), 160 => 64, relu; pad = 0)(x)
    b1 = Conv((7,1), 64 => 64, relu; pad = (3,0))(b1)
    b1 = Conv((1,7), 64 => 64, relu; pad = (0,3))(b1)
    b1 = Conv((3,3), 64 => 96, relu; pad = 0)(b1)

    x = cat(b0, b1; dims = 3)

    # Final Split
    b0 = Conv((3,3), 192 => 192, relu; pad = 0, stride = 2)(x)
    b1 = MaxPool((3,3); pad = 0, stride = 2)(x)
    x = cat(b0, b1; dims = 3)

    return x
end

function inception_a(x)
    println("A Block")
    a = Chain(
        x -> meanpool(x, (3,3); pad = 1, stride = 1),
        Conv((1,1), 384 => 96, relu; pad = 0)
    )(x)

    b = Conv((1,1), 384 => 96, relu; pad = 0)(x)

    c = Chain(
        Conv((1,1), 384 => 64, relu; pad = 0),
        Conv((3,3), 64 => 96, relu; pad = 1)
    )(x)

    d = Chain(
        Conv((1,1), 384 => 64, relu; pad = 0),
        Conv((3,3), 64 => 96, relu; pad = 1),
        Conv((3,3), 96 => 96, relu; pad = 1)
    )(x)

    return cat(a, b, c, d; dims = 3)
end

function inception_b(x)
    println("B Block")
    S = size(x, 3)

    a = Chain(
        x -> meanpool(x, (3,3); pad = 1, stride = 1),
        Conv((1,1), S => 128)
    )(x)

    b = Conv((1,1), S => 384, relu)(x)

    c = Chain(
        Conv((1,1), S => 192, relu),
        Conv((1,7), 192 => 224, relu; pad = (0, 3)),
        Conv((7,1), 224 => 256, relu; pad = (3, 0)),
    )(x)

    d = Chain(
        Conv((1,1), S => 192, relu),
        Conv((1,7), 192 => 192, relu; pad = (0, 3)),
        Conv((7,1), 192 => 224, relu; pad = (3, 0)),
        Conv((1,7), 224 => 224, relu; pad = (0, 3)),
        Conv((7,1), 224 => 256, relu; pad = (3, 0)),
    )(x)

    return cat(a, b, c, d; dims = 3)
end

function inception_c(x)
    println("C Block")
    S = size(x, 3)

    a = Chain(
        x -> meanpool(x, (3,3); pad = 1, stride = 1),
        Conv((1,1), S => 256, relu)
    )(x)

    b = Conv((1,1), S => 256)(x)

    _c = Conv((1,1), S => 384, relu)(x)
    c0 = Conv((1, 3), 384 => 256, relu; pad = (0, 1))(_c)
    c1 = Conv((3, 1), 384 => 256, relu; pad = (1, 0))(_c)

    _d = Chain(
        Conv((1,1), S => 384, relu),
        Conv((1,3), 384 => 448, relu; pad = (0, 1)),
        Conv((3,1), 448 => 512, relu; pad = (1, 0)),
    )(x)

    d0 = Conv((1,3), 512 => 256, relu; pad = (0, 1))(_d)
    d1 = Conv((3,1), 512 => 256, relu; pad = (1, 0))(_d)

    return cat(a, b, c0, c1, d0, d1; dims = 3)
end

function inception_ra(x, k, l, m, n)
    println("A Reduction")
    S = size(x, 3)

    a = maxpool(x, (3,3); pad = 0, stride = 2)
    b = Conv((3,3), S => n, relu; pad = 0, stride = 2)(x)

    c = Chain(
        Conv((1,1), S => k, relu),
        Conv((3,3), k => l, relu; pad = 1),
        Conv((3,3), l => m, relu; pad = 0, stride = 2)
    )(x)

    return cat(a, b, c; dims = 3)
end

function inception_rb(x)
    println("B Reduction")
    S = size(x, 3)

    a = maxpool(x, (3,3); pad = 0, stride = 2)

    b = Chain(
        Conv((1,1), S => 192, relu),
        Conv((3,3), 192 => 192, relu; pad = 0, stride = 2)
    )(x)

    c = Chain(
        Conv((1,1), S => 256, relu; pad = 0),
        Conv((1,7), 256 => 256, relu; pad = (0, 3)),
        Conv((7,1), 256 => 320, relu; pad = (3, 0)),
        Conv((3,3), 320 => 320, relu; pad = 0, stride = 2)
    )(x)

    return cat(a, b, c; dims = 3)
end

function inception_v4(x)
    x = stem(x)
    for _ in 1:4
        x = inception_a(x)
    end
    x = inception_ra(x, 192, 224, 256, 384)

    for _ in 1:7
        x = inception_b(x)
    end
    x = inception_rb(x)

    for _ in 1:3
        x = inception_c(x)
    end

    kernel_size = size.(Ref(x), (1, 2))
    x =  meanpool(x, kernel_size; pad = 0, stride = 1)
    # dropout
    
    x = reshape(x, :, size(x,4))
    x = Dense(1536,1000)(x)

    x = softmax(x)

    return x
end

@testset "Testing Inception" begin
    batchsize = 16
    x = rand(Float32, 299, 299, 3, batchsize)

    backend = nGraph.Lib.create("CPU")
    X = nGraph.Tensor(backend, x)
    f = nGraph.compile(backend, inception_v4, X)

    @show size(f(X))

    for _ in 1:10
        @time f(X)
    end

    f(x,y) = Flux.crossentropy(inception_v4(x), y)

    y = rand(Float32, 1000, batchsize)
    Y = nGraph.Tensor(backend, y)
    g = nGraph.compile(backend, f, X, Y; optimizer = nGraph.SGD(Float32(0.001)))

    @show length(g.outputs)
    @show length(g.optimizer.inputs)
    @show length(g.optimizer.outputs)

    for _ in 1:5
        @time g(X,Y)
    end

    #@test isapprox(z, f(X))
end
