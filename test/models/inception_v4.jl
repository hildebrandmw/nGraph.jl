function inception_v4(input)
    # Stem
    x = Conv((3, 3), 3 => 32; pad = 0, stride = 2)(input)
    x = Conv((3, 3), 32 => 32; pad = 0, stride = 2)(x)
    x = Conv((3, 3), 32 => 64; pad = 1)(x)

    # Split then merge
    b0 = MaxPool((3,3); pad = 0, stride = 2)(x)
    b1 = Conv((3,3), 64 => 96; pad = 0, stride = 2)(x)
    x = cat(b0, b1; dims = 3)

    # Second Split
    b0 = Conv((1,1), 160 => 64; pad = 0, stride = 1)(x)
    b0 = Conv((3,3), 64 => 96; pad = 0)(b0)

    b1 = Conv((1,1), 160 => 64; pad = 0)(x)
    b1 = Conv((7,1), 64 => 64; pad = (3,0))(b1)
    b1 = Conv((1,7), 64 => 64; pad = (0,3))(b1)
    b1 = Conv((3,3), 64 => 92; pad = 0)(b1)

    x = cat(b0, b1; dims = 3)

    # Final Split
    b0 = Conv((3,3), 192 => 192; pad = 0)
    b1 = MaxPool((3,3); pad = 0, stride = 2)
    x = cat(b0, b1; dims = 3)

    return x
end

@testset "Testing Inception" begin
    x = rand(Float32, 299, 299, 3, 1)
    #z = inception_v4(x)

    backend = nGraph.Lib.create("CPU")
    X = nGraph.Tensor(backend, x)
    f = nGraph.compile(backend, inception_v4, X)

    #@test isapprox(z, f(X))
end
