# nGraph

*Documentation goes here.*

## Bugs

**MKLDNN Memory Copy Bug**

If nGraph decides to do some optimization and send a tensor to MKLDNN, then when reading,
it has some conversion to do (`runtime/cpu/cpu_tensor_view.cpp`, Function at 108). However,
during this process, it does not respect the size limit passed to it and will thus overflow
the target buffer.

Thus, whenever transferring a Tensor, we probably have to do the whole tensor at once to
avoid getting bitten by this bug.
