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
import ..default_cutoff, ..default_maxdim, ..default_nsweeps
include("contraction/fitalgorithm.jl")
include("contraction/densitymatrix.jl")
include("contraction/fitalgorithm_sum.jl")
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
    # Detect MPS-like vs MPO-like by counting physical indices per site
    is_mps1 = begin
        sites_per_tensor = siteinds(M1)
        length(M1) > 0 && all(length(s) == 1 for s in sites_per_tensor)
    end
    is_mps2 = begin
        sites_per_tensor = siteinds(M2)
        length(M2) > 0 && all(length(s) == 1 for s in sites_per_tensor)
    end
    
    # Convert to ITensorMPS types based on detected type
    M1_ = is_mps1 ? ITensorMPS.MPS(M1) : ITensorMPS.MPO(M1)
    M2_ = is_mps2 ? ITensorMPS.MPS(M2) : ITensorMPS.MPO(M2)
    
    alg = Algorithm(alg)
    
    # Handle MPO * MPS case
    if !is_mps1 && is_mps2
        # MPO * MPS: use ITensorMPS.contract
        # Only pass nsweeps for fit algorithm, otherwise remove it from kwargs
        if alg == Algorithm"fit"()
            result = ITensorMPS.contract(M1_, M2_; alg=alg, cutoff=cutoff, maxdim=maxdim, nsweeps=nsweeps, kwargs...)
        else
            # Remove nsweeps from kwargs for non-fit algorithms
            kwargs_dict = Dict(kwargs)
            delete!(kwargs_dict, :nsweeps)
            result = ITensorMPS.contract(M1_, M2_; alg=alg, cutoff=cutoff, maxdim=maxdim, kwargs_dict...)
        end
        return TensorTrain(result)
    elseif is_mps1 && !is_mps2
        # MPS * MPO: use ITensorMPS.contract (commutative)
        # Only pass nsweeps for fit algorithm, otherwise remove it from kwargs
        if alg == Algorithm"fit"()
            result = ITensorMPS.contract(M2_, M1_; alg=alg, cutoff=cutoff, maxdim=maxdim, nsweeps=nsweeps, kwargs...)
        else
            # Remove nsweeps from kwargs for non-fit algorithms
            kwargs_dict = Dict(kwargs)
            delete!(kwargs_dict, :nsweeps)
            result = ITensorMPS.contract(M2_, M1_; alg=alg, cutoff=cutoff, maxdim=maxdim, kwargs_dict...)
        end
        return TensorTrain(result)
    else
        # MPO * MPO: use T4AITensorCompat algorithms
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
end

"""
    fit(input_states::AbstractVector{TensorTrain}, init::TensorTrain; coeffs::AbstractVector{<:Number}=ones(Int, length(input_states)), cutoff::Real=default_cutoff(), maxdim::Int=default_maxdim(), nsweeps::Int=default_nsweeps(), kwargs...)

Fit a linear combination of multiple TensorTrain objects to approximate their weighted sum.

This function uses a variational fitting algorithm to find a TensorTrain that approximates
the weighted sum of multiple input tensor trains. The algorithm iteratively optimizes the
bond dimensions while maintaining numerical accuracy.

# Arguments
- `input_states::AbstractVector{TensorTrain}`: Vector of tensor trains to sum
- `init::TensorTrain`: Initial guess for the fitted tensor train

# Keyword Arguments
- `coeffs::AbstractVector{<:Number}`: Coefficients for each input state (default: all ones)
- `cutoff::Real`: Truncation cutoff for singular values (default: `default_cutoff()`)
- `maxdim::Int`: Maximum bond dimension (default: `default_maxdim()`)
- `nsweeps::Int`: Number of sweeps for the fit algorithm (default: `default_nsweeps()`)
- `kwargs...`: Additional keyword arguments passed to the underlying algorithm

# Returns
- `TensorTrain`: The fitted tensor train approximating the weighted sum

# Examples
```julia
# Fit a weighted sum of three tensor trains
result = fit([tt1, tt2, tt3], init_tt; coeffs=[1.0, 2.0, 0.5])
```

# Note
FIXME (HS): I observed this function is sometime less accurate than direct sum of the input states.
"""
function fit(
    input_states::AbstractVector{TensorTrain},
    init::TensorTrain;
    coeffs::AbstractVector{<:Number} = ones(Int, length(input_states)),
    cutoff::Real = default_cutoff(),
    maxdim::Int = default_maxdim(),
    nsweeps::Int = default_nsweeps(),
    kwargs...,
)::TensorTrain
    println(stderr, "⚠ Warning: The `fit` function may produce less accurate results than direct sum of the input states. Consider using direct sum (`+`) if accuracy is critical.")
    # Convert TensorTrain objects to ITensorMPS.MPS
    mps_inputs = [ITensorMPS.MPS(tt) for tt in input_states]
    mps_init = ITensorMPS.MPS(init)
    
    # Call the fit function from ContractionImpl
    mps_result = ContractionImpl.fit(mps_inputs, mps_init; coeffs=coeffs, cutoff=cutoff, maxdim=maxdim, nsweeps=nsweeps, kwargs...)
    
    # Convert back to TensorTrain
    return TensorTrain(mps_result, init.llim, init.rlim)
end

"""
    product(A::TensorTrain, Ψ::TensorTrain; alg=Algorithm"fit"(), cutoff=default_cutoff(), maxdim=default_maxdim(), nsweeps=default_nsweeps(), kwargs...)

Multiply an MPO with an MPS or MPO (official API name).

This function multiplies an MPO `A` by a tensor train `Ψ` (MPS or MPO),
using `contract` internally and adjusting prime levels for compatibility.

# Arguments
- `A::TensorTrain`: The MPO to apply
- `Ψ::TensorTrain`: The MPS or MPO to multiply

# Keywords
- `alg`: Algorithm specification (String or Algorithm type). Defaults to "fit".
- `cutoff::Real`: Truncation cutoff. Defaults to `default_cutoff()`.
- `maxdim::Int`: Maximum bond dimension. Defaults to `default_maxdim()`.
- `nsweeps::Int`: Number of sweeps for variational algorithms. Defaults to `default_nsweeps()`.
- `kwargs...`: Additional keyword arguments passed to contract
"""
function product(A::TensorTrain, Ψ::TensorTrain; alg=Algorithm"fit"(), cutoff=default_cutoff(), maxdim=default_maxdim(), nsweeps=default_nsweeps(), kwargs...)
    if :algorithm ∈ keys(kwargs)
        error("keyword argument :algorithm is not allowed")
    end
    
    # Convert alg to Algorithm type (accepts both String and Algorithm)
    alg_ = alg isa Algorithm ? alg : Algorithm(alg)
    
    # Warn if cutoff is too small for densitymatrix algorithm
    if alg_ == Algorithm("densitymatrix") && cutoff <= 1e-10
        @warn "cutoff is too small for densitymatrix algorithm. Use fit algorithm instead."
    end
    
    # Detect MPS-like vs MPO-like by counting physical indices per site:
    # MPS tensors have 1 physical index per site, MPO tensors have 2 per site.
    # This is robust to boundary tensors having fewer link indices.
    is_mps_like = begin
        sites_per_tensor = siteinds(Ψ)
        length(Ψ) > 0 && all(length(s) == 1 for s in sites_per_tensor)
    end
    
    if is_mps_like
        # Apply MPO to MPS: contract(A, ψ) then replaceprime(..., 1 => 0)
        # Use T4AITensorCompat.contract for MPO * MPS
        result_tt = contract(A, Ψ; alg=alg_, cutoff=cutoff, maxdim=maxdim, nsweeps=nsweeps, kwargs...)
        # Adjust prime levels: replaceprime(..., 1 => 0) to get unprimed result
        # Use ITensorMPS.replaceprime by converting to MPS
        result_mps = ITensorMPS.MPS(result_tt)
        result_mps = ITensorMPS.replaceprime(result_mps, 1 => 0)
        return TensorTrain(result_mps)
    else
        # Apply MPO to MPO: contract(A', B) then replaceprime(..., 2 => 1)
        # Use T4AITensorCompat.contract for MPO * MPO (with A' to contract over one set of indices)
        A_primed = ITensors.prime(A)
        result_tt = contract(A_primed, Ψ; alg=alg_, cutoff=cutoff, maxdim=maxdim, nsweeps=nsweeps, kwargs...)
        # Adjust prime levels: replaceprime(..., 2 => 1) to get pairs of primed/unprimed indices
        # Use ITensorMPS.replaceprime by converting to MPO
        result_mpo = ITensorMPS.MPO(result_tt)
        result_mpo = ITensorMPS.replaceprime(result_mpo, 2 => 1)
        return TensorTrain(result_mpo)
    end
end

"""
    apply(A::TensorTrain, Ψ::TensorTrain; kwargs...)

Backwards-compatible alias for [`product`](@ref).

This function multiplies an MPO `A` by a tensor train `Ψ` (MPS or MPO) and
forwards all keyword arguments to [`product`](@ref).
See [`product`](@ref) for the full list of supported keywords and behavior.
"""
apply(A::TensorTrain, Ψ::TensorTrain; kwargs...) = product(A, Ψ; kwargs...)