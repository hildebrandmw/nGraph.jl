@testset "Testing Inception" begin
    batchsize = 16

    # Inference
    f, X  = nGraph.inception_v4_inference(batchsize)

    @show size(f(X))
    for _ in 1:10
        @time f(X)
    end

    # Training
    g, X, Y = nGraph.inception_v4_training(batchsize)

    @show length(g.outputs)
    @show length(g.optimizer.inputs)
    @show length(g.optimizer.outputs)

    for _ in 1:5
        @time g(X,Y)
    end

    #@test isapprox(z, f(X))
end
