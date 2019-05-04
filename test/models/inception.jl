
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
          Dense(1024, 1000, relu), 
          x -> log.(max.(x, Float32(1e-9))),
          softmax,
    )

    model = _googlenet()

    x = rand(Float32, 224, 224, 3, 16)

    backend = nGraph.Backend()
    z = model(x)
    X = nGraph.Tensor(backend, x)
    f = nGraph.compile(backend, model, X)

    @test isapprox(z, read(f(X)))

    @info "Compiling Training Pass for Inception"
    y = similar(model(x))
    y .= zero(eltype(y))

    model = _googlenet()
    loss(x, y) = sum(model(x) .- y)
    l = loss(x, y) 
    @show l

    Y = nGraph.Tensor(backend, y)

    @info "Testing nGraph Gradients"
    h = nGraph.compile(backend, loss, X, Y; optimizer = nGraph.Gradient)
    @show read(h(X,Y))

    # Run the Flux back-propagation
    @info "Taking Flux Gradients"
    ps = Flux.params(model)
    @show length(ps)
    Flux.back!(l)

    # Now, we compare.
    # The `Gradient` optimizer has a dict that maps the original tracked array to the 
    # tensors that were built from them.
    #
    # We leverage this to check that
    #
    # - The data was captured correctly when parameters were constructed
    # - The gradients computed by the two frameworks are rougly equal
    gradient_map = h.optimizer._id  

    for p in ps
        @test haskey(gradient_map, p)
    end

    #=
    for p in ps
        # If this is a convolution weight, we need to flip it to compare data and gradients
        if ndims(p) == 4 
            # Get the data tensor from the gradient map. Make sure we copied it correctly.
            x = copy(p.data)
            nGraph.flip!(x)
            @test isapprox(read(gradient_map[p][1]), x)

            # Check that the gradients are the same as well
            x = copy(p.grad) 
            nGraph.flip!(x)
            @test isapprox(read(gradient_map[p][2]), x)
        else
            @test isapprox(read(gradient_map[p][1]), p.data)
            @test isapprox(read(gradient_map[p][2]), p.grad)

        end
    end
    =#
end
