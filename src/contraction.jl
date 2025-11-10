#===
contraction.jl - Tensor train contraction algorithms

Copyright (c) 2025 Hiroshi Shinaoka and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file includes implementations from ITensors.jl and ITensorTDVP.jl
See individual files in contraction/ for detailed attributions.
====#

module ContractionImpl
using ITensors
using ITensorMPS
import ITensors
import ITensorMPS

import ITensors: Algorithm, @Algorithm_str
import ITensorMPS: setleftlim!, setrightlim!
import LinearAlgebra
include("contraction/fitalgorithm.jl")
include("contraction/densitymatrix.jl")
end

"""
    contract(M1::TensorTrain, M2::TensorTrain; alg=Algorithm"fit"(), cutoff::Real=default_cutoff(), maxdim::Int=default_maxdim(), nsweeps::Int=default_nsweeps(), kwargs...)

Contract two TensorTrain objects (tensor network contraction).

This function performs the contraction of two tensor trains, which is equivalent to
computing the product of two matrix product operators. Multiple algorithms are available
for the contraction.

# Arguments
- `M1::TensorTrain`: First tensor train
- `M2::TensorTrain`: Second tensor train

# Keyword Arguments
- `alg`: Algorithm to use for contraction. Options:
  - `Algorithm"fit"()` (default): Variational fitting algorithm
  - `Algorithm"densitymatrix"()`: Density matrix renormalization algorithm
  - `Algorithm"zipup"()`: Zip-up algorithm from ITensorMPS
  - `Algorithm"naive"()`: Naive contraction from ITensorMPS
- `cutoff::Real`: Truncation cutoff for singular values (default: `default_cutoff()`)
- `maxdim::Int`: Maximum bond dimension (default: `default_maxdim()`)
- `nsweeps::Int`: Number of sweeps for the fit algorithm (default: `default_nsweeps()`)
- `kwargs...`: Additional keyword arguments passed to the underlying algorithm

# Returns
- `TensorTrain`: The contracted tensor train (product of M1 and M2)

# Examples
```julia
result = contract(M1, M2)  # Using default fit algorithm
result = contract(M1, M2; alg=Algorithm"densitymatrix"(), maxdim=100)
```
"""
function contract(M1::TensorTrain, M2::TensorTrain; alg=Algorithm"fit"(), cutoff::Real=default_cutoff(), maxdim::Int=default_maxdim(), nsweeps::Int=default_nsweeps(), kwargs...)::TensorTrain
    M1_ = ITensorMPS.MPO(M1)
    M2_ = ITensorMPS.MPO(M2)
    alg = Algorithm(alg)
    if alg == Algorithm"densitymatrix"()
        return TensorTrain(ContractionImpl.contract_densitymatrix(M1_, M2_; cutoff, maxdim, kwargs...))
    elseif alg == Algorithm"fit"()
        return TensorTrain(ContractionImpl.contract_fit(M1_, M2_; cutoff, maxdim, nsweeps, kwargs...))
    elseif alg == Algorithm"zipup"()
        return TensorTrain(ITensorMPS.contract(M1_, M2_; alg=Algorithm"zipup"(), cutoff, maxdim, kwargs...))
    elseif alg == Algorithm"naive"()
        return TensorTrain(ITensorMPS.contract(M1_, M2_; alg=Algorithm"naive"(), cutoff, maxdim, kwargs...))
    else
        error("Unknown algorithm: $alg")
    end
end