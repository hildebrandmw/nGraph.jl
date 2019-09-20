# Wrappers for parameters that require special handling.
#
# The main culprate are embedding tables, that can efficiently use an in-place update.
struct EmbeddingWeights{T}
    x::T
end


