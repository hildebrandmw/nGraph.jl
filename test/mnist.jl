@testset "Testing MNIST Conv" begin
    model = Chain(
        # First convolution, operating upon a 28x28 image
        Conv((3, 3), 1=>16, pad=(1,1), relu),
        MaxPool((2,2)),

        # Second convolution, operating upon a 14x14 image
        Conv((3, 3), 16=>32, pad=(1,1), relu),
        MaxPool((2,2)),

        # Third convolution, operating upon a 7x7 image
        Conv((3, 3), 32=>32, pad=(1,1), relu),
        MaxPool((2,2)),

        # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
        # which is where we get the 288 in the `Dense` layer below:
        x -> reshape(x, :, size(x, 4)),
        Dense(288, 10),

        # Finally, softmax to get nice probabilities
        softmax,
    )

    backend = nGraph.Lib.create("CPU")
    x = rand(Float32, 28, 28, 1, 100)
    X = nGraph.Tensor(backend, x)
    f = nGraph.compile(backend, model, X)

    # Test that the forward passes match
    @test isapprox(model(x), collect(f(X)))

    @time model(x)
    @time f(X)
end

#####
##### Inception
#####

struct InceptionBlock
    path_1
    path_2
    path_3
    path_4
end

Flux.@treelike InceptionBlock

function InceptionBlock(
        in_chs,
        chs_1x1,
        chs_3x3_reduce,
        chs_3x3,
        chs_5x5_reduce,
        chs_5x5,
        pool_proj
    )
    path_1 = Conv((1, 1), in_chs=>chs_1x1, relu)

    path_2 = (Conv((1, 1), in_chs=>chs_3x3_reduce, relu),
            Conv((3, 3), chs_3x3_reduce=>chs_3x3, relu, pad = (1, 1)))

    path_3 = (Conv((1, 1), in_chs=>chs_5x5_reduce, relu),
            Conv((5, 5), chs_5x5_reduce=>chs_5x5, relu, pad = (2, 2)))

    path_4 = (x -> maxpool(x, (3,3), stride = (1, 1), pad = (1, 1)),
            Conv((1, 1), in_chs=>pool_proj, relu))

    InceptionBlock(path_1, path_2, path_3, path_4)
end

function (m::InceptionBlock)(x)
    cat(
        m.path_1(x),
        m.path_2[2](m.path_2[1](x)),
        m.path_3[2](m.path_3[1](x)),
        m.path_4[2](m.path_4[1](x)),
        dims = 3
    )
end

@testset "Testing Inception" begin

    _googlenet() = Chain(Conv((7, 7), 3=>64, stride = (2, 2), relu, pad = (3, 3)),
          x -> maxpool(x, (3, 3), stride = (2, 2), pad = (1, 1)),
          Conv((1, 1), 64=>64, relu),
          Conv((3, 3), 64=>192, relu, pad = (1, 1)),
          x -> maxpool(x, (3, 3), stride = (2, 2), pad = (1, 1)),
          InceptionBlock(192, 64, 96, 128, 16, 32, 32),
          InceptionBlock(256, 128, 128, 192, 32, 96, 64),
          x -> maxpool(x, (3, 3), stride = (2, 2), pad = (1, 1)),
          InceptionBlock(480, 192, 96, 208, 16, 48, 64),
          InceptionBlock(512, 160, 112, 224, 24, 64, 64),
          InceptionBlock(512, 128, 128, 256, 24, 64, 64),
          InceptionBlock(512, 112, 144, 288, 32, 64, 64),
          InceptionBlock(528, 256, 160, 320, 32, 128, 128),
          x -> maxpool(x, (3, 3), stride = (2, 2), pad = (1, 1)),
          InceptionBlock(832, 256, 160, 320, 32, 128, 128),
          InceptionBlock(832, 384, 192, 384, 48, 128, 128),
          x -> maxpool(x, (7, 7), stride = (1, 1), pad = (0, 0)),
          x -> reshape(x, :, size(x, 4)),
          #Dropout(0.4),
          Dense(1024, 1000), softmax)

    model = _googlenet()

    x = rand(Float32, 224, 224, 3, 16)

    backend = nGraph.Lib.create("CPU")
    X = nGraph.Tensor(backend, x)
    f = nGraph.compile(backend, model, X)

    z = model(x)
    @test isapprox(z, collect(f(X)))

    @time model(x)
    @time f(X)

    @info "Compiling Training Pass for Inception"
    y = similar(model(x))
    loss(x, y) = Flux.crossentropy(model(x), y)
    Y = nGraph.Tensor(backend, y)
    g = nGraph.compile(backend, loss, X, Y; training = true)
end
