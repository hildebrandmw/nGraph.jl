unif(b) = (dims...) -> _unif(b, dims...)
function _unif(b::T, dims...) where {T} 
    sampler = Uniform{T}(-b, b)
    return convert.(T, rand(sampler, dims...))
end

function test_embedding()
    context_size = 2
    embedding_dim = 10
    test_sentence = split("""
        When forty winters shall besiege thy brow,
        And dig deep trenches in thy beauty's field,
        Thy youth's proud livery so gazed on now,
        Will be a totter'd weed of small worth held:
        Then being asked, where all thy beauty lies,
        Where all the treasure of thy lusty days;
        To say, within thine own deep sunken eyes,
        Were an all-eating shame, and thriftless praise.
        How much more praise deserv'd thy beauty's use,
        If thou couldst answer 'This fair child of mine
        Shall sum my count, and make my old excuse,'
        Proving his beauty by succession thine!
        This were to be new made when thou art old,
        And see thy blood warm when thou feel'st it cold.
        """)

    # Build vocab and trigrams.
    trigrams = [((test_sentence[i], test_sentence[i+1]), test_sentence[i+2]) for i in 1:length(test_sentence)-2]

    vocab = Set(test_sentence)
    word_to_idx = Dict(word => convert(Int32, i-1) for (i, word) in enumerate(vocab))

    embedding_param = Flux.param(randn(Float32, embedding_dim, length(vocab)))
    insize = context_size * embedding_dim

    d1 = Dense(
        insize,
        128, 
        relu;
        initW = unif(Float32(1/sqrt(insize))),
        initb = unif(Float32(1/sqrt(insize))),
    )
    d2 = Dense(
        128, 
        length(vocab);
        initW = unif(Float32(1/sqrt(128))),
        initb = unif(Float32(1/sqrt(128))),
    )

    f = function(x)
        a = reshape(embedding(x, embedding_param), :)
        # Mark the embedding parameter as an inplace update node.
        nGraph.__inplace(embedding_param)
        b = d1(a)
        c = d2(b)
        return softmax(c)
    end

    loss(a, b) = Flux.crossentropy(f(a), nGraph.onehot(b, length(vocab), 1))

    # Construct inputs and output placeholders
    inputs = zeros(Int32, context_size)
    expected = nGraph.parameter(Int32(1))

    backend = nGraph.Backend("CPU")
    F = nGraph.compile(
        backend, 
        loss, 
        inputs, 
        expected; 
        optimizer = nGraph.SGD(Float32(0.01))
    )
    return F
end
