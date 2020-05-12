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

## Current State of Affairs

The current state of affairs is something like this:

TLDR: Package is currently broken, but I hope to have it unbroken and registered sometime in the near future (~weeks?).

I originally developed this repo for a project using ngraph (since I very much prefer writing Julia code than C++) and so much of the code in this repo was tied to a custom modified version of ngraph which has since fallen very behind later ngraph released.

I've taken up this project after some months in my spare time with the plans of registering this as an "official" Julia package (because I don't want all the work in this to go to waste). However, changes to ngraph and CxxWrap (and the need to rip out all the custom functionality I added) has lead to a large churn in the code base.

Fortunately, most of the nitty details of getting it to work have already been figured out, it's mostly a matter of refactoring. Once I register it (and add documentation to make it actually usable), I will endeavor to stop hilariously breaking master and focus on cross-platform support, supporting more of the full list of ngraph ops etc.

Stay tuned! 😄

## Usage Example

```julia
julia> using nGraph, Flux

# We're going to create a simple matrix multiply + bias
julia> W = randn(Float32, 2, 10)
2×10 Array{Float32,2}:
 -0.0713108   0.485168  -0.511655  -0.282555   0.152891  -0.490765  0.216486  1.835     -0.694151  -0.270645
  0.0592051  -1.77903    1.73074   -1.50022   -0.530367   1.48821   0.847445  0.190752   1.16327   -0.0605583

julia> b = randn(Float32, 2)
2-element Array{Float32,1}:
 -0.5872276
  1.2322273

julia> f = Dense(W, b, relu)
Dense(10, 2, relu)

julia> x = randn(Float32, 10)
10-element Array{Float32,1}:
  1.2264888
  0.3246307
  0.6522203
 -0.5913128
 -1.3148737
 -0.5557325
 -0.107921995
  0.14666164
  0.09259398
  0.94359696
  
# Demonstrate the results on the Julia side when we call the function
julia> f(x)
2-element Array{Float32,1}:
 0.0
 2.6006393
 
# Now, we convert this into an nGraph executable, which will extract 
# all parameters from the Julia function and turnall of the operations
# into nGraph ops.
julia> fex = nGraph.compile(nGraph.Backend("CPU"), f, x);

julia> typeof(fex)
nGraph.FluxExecutable{nGraph.CPU,nGraph.InferenceState,1,1}

# When we run the FluxExecutable, we get the same results!
julia> fex()
Tensor View
2-element Array{Float32,1}:
 0.0
 2.6006393
 
 # As a bonus, we are free to change weights and biases and the results will be updated the next time the FluxExecutable is run
 julia> f.W .= randn(Float32, 2, 10)
2×10 Array{Float32,2}:
 -0.223597  0.182274  0.661538  -1.0027    -0.00451888  0.179959  -0.438976  -0.118112  -0.334693   0.0516222
 -0.24377   0.949058  0.515212   0.229445   0.74604     0.904801   0.101299  -0.532028   0.253983  -0.172822
 
 julia> f(x)
2-element Array{Float32,1}:
 0.17578578
 0.0
 
 julia> fex()
Tensor View
2-element Array{Float32,1}:
 0.17578584
 0.0
```
