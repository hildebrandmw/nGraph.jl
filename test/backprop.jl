# Test the embedding table backprop against Zygote
embedded_lookup(data::Vector, weights::Matrix) = 
    [weights[i,j] for i in data, j in 1:size(weights, 2)]
