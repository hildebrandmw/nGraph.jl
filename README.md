# nGraph

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/hildebrandmw/nGraph.jl.svg?branch=master)](https://travis-ci.com/hildebrandmw/nGraph.jl)
[![codecov.io](http://codecov.io/github/hildebrandmw/nGraph.jl/coverage.svg?branch=master)](http://codecov.io/github/hildebrandmw/nGraph.jl?branch=master)

An experimental frontend for https://github.com/NervanaSystems/ngraph.

## Usage Example

```julia
julia> using nGraph, Flux

# We're going to create a simple matrix multiply + bias
julia> W = param(randn(Float32, 2, 10))
Tracked 2×10 Array{Float32,2}:
 -2.32432    0.927459  -0.523398   1.06994   -1.34763    0.929182  0.320528  -0.0786303  1.12172   0.526701
 -0.206099  -0.246847  -0.387398  -0.816688  -0.722219  -0.337349  0.916446  -0.140121   0.143286  1.54653

julia> b = param(randn(Float32, 2))
Tracked 2-element Array{Float32,1}:
 -1.965381f0
  0.06922187f0

julia> y(x) = σ.(W * x .+ b)
y (generic function with 1 method)

julia> x = randn(Float32, 10)
10-element Array{Float32,1}:
  0.18938474
 -1.0258152
  0.094758816
  0.6261743
  0.6724537
  1.3771137
 -0.22054482
 -1.4793428
 -1.1879464
  0.12393876
  
# Demonstrate the results on the Julia side when we call the function
julia> y(x)
Tracked 2-element Array{Float32,1}:
 0.026991807f0
 0.23356533f0
 
# Now, we convert this into an nGraph executable, which will extract 
# all parameters from the Julia function and turnall of the operations
# into nGraph ops.
julia> fex = nGraph.compile(nGraph.Backend("CPU"), y, x);

julia> typeof(fex)
nGraph.FluxExecutable{nGraph.CPU,nGraph.InferenceState,1,1}

# When we run the FluxExecutable, we get the same results!
julia> read(fex())
2-element Array{Float32,1}:
 0.026991807
 0.23356533
```
