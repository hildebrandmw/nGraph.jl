using .CuArrays

transport(::Type{GPU}, x::AbstractArray) = cu(x)

# This is all scary ... but CuArrays doesn't provide a way of getting a pointer, and nGraph
# wants a GPU pointer ...
_pointer(x::CuArray) = Ptr{Nothing}(UInt(pointer(CuArrays.buffer((x)))))
