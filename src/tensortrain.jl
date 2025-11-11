#===
tensortrain.jl - TensorTrain type and operations

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
====#

"""
TensorTrain - A simple tensor train data structure

This struct represents a tensor train (matrix product state/operator) as a vector
of ITensors with left and right boundary indices.
This struct is designed to be compatible with ITensorMPS.MPS and ITensorMPS.MPO.

# Fields
- `data::Vector{ITensor}`: Vector of tensors in the tensor train
- `llim::Int`: Left boundary index
- `rlim::Int`: Right boundary index

# Conversion Functions
This struct supports conversion to/from ITensorMPS types:
- `MPS(stt::TensorTrain)` - Convert to Matrix Product State
- `MPO(stt::TensorTrain)` - Convert to Matrix Product Operator  
- `TensorTrain(mps::MPS)` - Convert from Matrix Product State
- `TensorTrain(mpo::MPO)` - Convert from Matrix Product Operator
"""
mutable struct TensorTrain
    data::Vector{ITensor}
    llim::Int
    rlim::Int
end

"""
    TensorTrain(data::Vector{ITensor})

Construct a TensorTrain from a vector of ITensors.

The left and right limits are automatically set to 0 and length(data) + 1 respectively.

# Arguments
- `data::Vector{ITensor}`: Vector of tensors forming the tensor train

# Returns
- `TensorTrain`: A new TensorTrain object with default limits
"""
function TensorTrain(data::Vector{ITensor})
    return TensorTrain(data, 0, length(data) + 1)
end

# Size-based constructor to allocate an empty tensor train of length `n`.
# Elements are uninitialized ITensors and should be assigned by the caller.
function TensorTrain(n::Int)
    return TensorTrain(Vector{ITensor}(undef, n), 0, n + 1)
end

# Iterator implementation
Base.iterate(stt::TensorTrain) = iterate(stt.data)
Base.iterate(stt::TensorTrain, state) = iterate(stt.data, state)
Base.length(stt::TensorTrain) = length(stt.data)
Base.size(stt::TensorTrain) = size(stt.data)
Base.getindex(stt::TensorTrain, i) = stt.data[i]
Base.setindex!(stt::TensorTrain, value, i) = (stt.data[i] = value)
Base.firstindex(stt::TensorTrain) = 1
Base.lastindex(stt::TensorTrain) = length(stt.data)
Base.eachindex(stt::TensorTrain) = eachindex(stt.data)

"""
Convert TensorTrain to ITensorMPS.MPS

This function takes a TensorTrain and converts it to an ITensorMPS.MPS.
The conversion preserves the tensor structure and indices.
"""
function ITensorMPS.MPS(stt::TensorTrain)
    return ITensorMPS.MPS(stt.data)
end

"""
Convert ITensorMPS.MPS to TensorTrain

This function takes an ITensorMPS.MPS and converts it to a TensorTrain.
The conversion preserves the tensor structure and indices.
"""
function TensorTrain(mps::ITensorMPS.MPS)
    # Extract the tensor data from the MPS
    data = Vector{ITensor}(undef, length(mps))
    for i in 1:length(mps)
        data[i] = mps[i]
    end
    return TensorTrain(data, 0, length(data) + 1)
end

"""
Convert ITensorMPS.MPS to TensorTrain with explicit left and right limits

This function allows specifying the left and right limits explicitly.
"""
function TensorTrain(mps::ITensorMPS.MPS, llim::Int, rlim::Int)
    # Extract the tensor data from the MPS
    data = Vector{ITensor}(undef, length(mps))
    for i in 1:length(mps)
        data[i] = mps[i]
    end
    return TensorTrain(data, llim, rlim)
end

"""
Convert TensorTrain to ITensorMPS.MPO

This function takes a TensorTrain and converts it to an ITensorMPS.MPO.
The conversion preserves the tensor structure and indices.
"""
function ITensorMPS.MPO(stt::TensorTrain)
    return ITensorMPS.MPO(stt.data)
end

"""
Convert ITensorMPS.MPO to TensorTrain

This function takes an ITensorMPS.MPO and converts it to a TensorTrain.
The conversion preserves the tensor structure and indices.
"""
function TensorTrain(mpo::ITensorMPS.MPO)
    # Extract the tensor data from the MPO
    data = Vector{ITensor}(undef, length(mpo))
    for i in 1:length(mpo)
        data[i] = mpo[i]
    end
    return TensorTrain(data, 0, length(data) + 1)
end

"""
Convert ITensorMPS.MPO to TensorTrain with explicit left and right limits

This function allows specifying the left and right limits explicitly.
"""
function TensorTrain(mpo::ITensorMPS.MPO, llim::Int, rlim::Int)
    # Extract the tensor data from the MPO
    data = Vector{ITensor}(undef, length(mpo))
    for i in 1:length(mpo)
        data[i] = mpo[i]
    end
    return TensorTrain(data, llim, rlim)
end

"""
Add two TensorTrain objects using ITensors.Algorithm("directsum")

This function computes the sum of two tensor trains by:
1. Converting both TensorTrain objects to ITensorMPS.MPS
2. Computing the sum using ITensors.Algorithm("directsum") for high precision
3. Converting the result back to TensorTrain

The result preserves the tensor structure while combining the bond dimensions.
Uses Algorithm("directsum") instead of default + operator for better numerical precision.
"""
function Base.:+(stt1::TensorTrain, stt2::TensorTrain)
    # Check that both tensor trains have the same length
    if length(stt1.data) != length(stt2.data)
        throw(ArgumentError("Tensor trains must have the same length for addition. Got lengths $(length(stt1.data)) and $(length(stt2.data))"))
    end
    
    # Convert to MPS
    mps1 = ITensorMPS.MPS(stt1)
    mps2 = ITensorMPS.MPS(stt2)
    
    # Compute sum using ITensors.Algorithm("directsum") for better precision
    alg = Algorithm"directsum"()
    mps_sum = +(alg, mps1, mps2)
    
    # Convert back to TensorTrain
    # Use the left and right limits from the first tensor train
    return TensorTrain(mps_sum, stt1.llim, stt1.rlim)
end

"""
Add multiple TensorTrain objects using ITensors.Algorithm("directsum")

This function computes the sum of multiple tensor trains by:
1. Converting all TensorTrain objects to ITensorMPS.MPS
2. Computing the sum using ITensors.Algorithm("directsum") for high precision
3. Converting the result back to TensorTrain

The result preserves the tensor structure while combining all bond dimensions.
Uses Algorithm("directsum") instead of default + operator for better numerical precision.
"""
function Base.:+(stt1::TensorTrain, stt2::TensorTrain, stts...)
    # Check that all tensor trains have the same length
    lengths = [length(stt.data) for stt in [stt1, stt2, stts...]]
    if !all(l -> l == lengths[1], lengths)
        throw(ArgumentError("All tensor trains must have the same length for addition. Got lengths: $(lengths)"))
    end
    
    # Convert all to MPS
    mps_list = [ITensorMPS.MPS(stt) for stt in [stt1, stt2, stts...]]
    
    # Compute sum of all MPS using ITensors.Algorithm("directsum") for better precision
    alg = Algorithm"directsum"()
    mps_sum = +(alg, mps_list...)
    
    # Convert back to TensorTrain
    # Use the left and right limits from the first tensor train
    return TensorTrain(mps_sum, stt1.llim, stt1.rlim)
end

"""
Scalar multiplication for TensorTrain

This function multiplies a TensorTrain by a scalar value by delegating to ITensorMPS.
"""
function Base.:*(α::Number, stt::TensorTrain)
    # Convert to MPS and delegate to ITensorMPS
    mps = ITensorMPS.MPS(stt)
    scaled_mps = α * mps
    
    # Convert back to TensorTrain
    return TensorTrain(scaled_mps, stt.llim, stt.rlim)
end

"""
Scalar multiplication for TensorTrain (right multiplication)

This function multiplies a TensorTrain by a scalar value by delegating to ITensorMPS.
"""
function Base.:*(stt::TensorTrain, α::Number)
    return α * stt
end

"""
    dist(stt1::TensorTrain, stt2::TensorTrain)

Compute the Euclidean distance between two TensorTrain objects.

This function delegates to ITensorMPS.dist for efficient computation using:
`sqrt(abs(inner(A, A) + inner(B, B) - 2 * real(inner(A, B))))`.

Note that if the tensor trains are not normalized, the normalizations may diverge and
this may not be accurate. For those cases, it is best to use `norm(stt1 - stt2)` directly.
"""
function dist(stt1::TensorTrain, stt2::TensorTrain)
    # Check that both tensor trains have the same length
    if length(stt1.data) != length(stt2.data)
        throw(ArgumentError("Tensor trains must have the same length for distance computation. Got lengths $(length(stt1.data)) and $(length(stt2.data))"))
    end
    
    # Convert to MPS and delegate to ITensorMPS.dist
    mps1 = ITensorMPS.MPS(stt1)
    mps2 = ITensorMPS.MPS(stt2)
    
    return ITensorMPS.dist(mps1, mps2)
end

"""
    isapprox(stt1::TensorTrain, stt2::TensorTrain; kwargs...)

Check if two TensorTrain objects are approximately equal.

This function delegates to ITensorMPS.isapprox for efficient computation.
"""
function Base.isapprox(stt1::TensorTrain, stt2::TensorTrain; kwargs...)
    # Check that both tensor trains have the same length
    if length(stt1.data) != length(stt2.data)
        return false
    end
    
    # Convert to MPS and delegate to ITensorMPS.isapprox
    mps1 = ITensorMPS.MPS(stt1)
    mps2 = ITensorMPS.MPS(stt2)
    
    return ITensorMPS.isapprox(mps1, mps2; kwargs...)
end

"""
    isapprox(x::TensorTrain, y::TensorTrain; atol::Real=0, rtol::Real=Base.rtoldefault(LinearAlgebra.promote_leaf_eltypes(x), LinearAlgebra.promote_leaf_eltypes(y), atol))

Check if two TensorTrain objects are approximately equal using explicit tolerance parameters.

This function computes the distance between two tensor trains and compares it against
absolute and relative tolerances.

# Arguments
- `x::TensorTrain`: First tensor train
- `y::TensorTrain`: Second tensor train

# Keyword Arguments
- `atol::Real`: Absolute tolerance (default: 0)
- `rtol::Real`: Relative tolerance (default: computed from element types)

# Returns
- `Bool`: `true` if `norm(x - y) <= max(atol, rtol * max(norm(x), norm(y)))`, `false` otherwise

# Examples
```julia
tt1 ≈ tt2  # Using default tolerances
isapprox(tt1, tt2; atol=1e-10, rtol=1e-8)  # Using explicit tolerances
```
"""
function isapprox(
    x::TensorTrain,
    y::TensorTrain;
    atol::Real = 0,
    rtol::Real = Base.rtoldefault(
        LinearAlgebra.promote_leaf_eltypes(x), LinearAlgebra.promote_leaf_eltypes(y), atol
    ),
)
    d = norm(x - y)
    if isfinite(d)
        return d <= max(atol, rtol * max(norm(x), norm(y)))
    else
        error("In `isapprox(x::TensorTrain, y::TensorTrain)`, `norm(x - y)` is not finite")
    end
end

# Extend LinearAlgebra.promote_leaf_eltypes for TensorTrain
function LinearAlgebra.promote_leaf_eltypes(tt::TensorTrain)
    return LinearAlgebra.promote_leaf_eltypes(tt.data)
end

"""
    norm(stt::TensorTrain)

Compute the norm (2-norm) of a TensorTrain object.

This function delegates to ITensorMPS.norm for efficient computation.
The norm is computed as the square root of the inner product with itself.
"""
function LinearAlgebra.norm(stt::TensorTrain)
    # Convert to MPS and delegate to ITensorMPS.norm
    mps = ITensorMPS.MPS(stt)
    
    return ITensorMPS.norm(mps)
end

"""
    lognorm(stt::TensorTrain)

Compute the log norm of a TensorTrain object.

This function delegates to ITensorMPS.lognorm for efficient computation.
The log norm is useful when the norm may be very large to avoid overflow.
"""
function lognorm(stt::TensorTrain)
    # Convert to MPS and delegate to ITensorMPS.lognorm
    mps = ITensorMPS.MPS(stt)
    
    return ITensorMPS.lognorm(mps)
end

# Also extend ITensorMPS.lognorm for backward compatibility
ITensorMPS.lognorm(stt::TensorTrain) = lognorm(stt)

"""
Subtract two TensorTrain objects using ITensors.Algorithm("directsum")

This function computes the difference of two tensor trains by:
1. Converting both TensorTrain objects to ITensorMPS.MPS
2. Computing the difference using ITensors.Algorithm("directsum") for high precision
3. Converting the result back to TensorTrain

The result preserves the tensor structure while combining the bond dimensions.
Uses Algorithm("directsum") for optimal numerical precision.
"""
function Base.:-(stt1::TensorTrain, stt2::TensorTrain)
    # Check that both tensor trains have the same length
    if length(stt1.data) != length(stt2.data)
        throw(ArgumentError("Tensor trains must have the same length for subtraction. Got lengths $(length(stt1.data)) and $(length(stt2.data))"))
    end
    
    # Convert to MPS
    mps1 = ITensorMPS.MPS(stt1)
    mps2 = ITensorMPS.MPS(stt2)
    
    # Compute difference using ITensors.Algorithm("directsum") for better precision
    # Subtraction: stt1 - stt2 = stt1 + (-1) * stt2
    alg = Algorithm"directsum"()
    mps_diff = +(alg, mps1, -1 * mps2)
    
    # Convert back to TensorTrain
    # Use the left and right limits from the first tensor train
    return TensorTrain(mps_diff, stt1.llim, stt1.rlim)
end

"""
Subtract multiple TensorTrain objects using ITensors.Algorithm("directsum")

This function computes the difference of multiple tensor trains by:
1. Converting all TensorTrain objects to ITensorMPS.MPS
2. Computing the difference using ITensors.Algorithm("directsum") for high precision
3. Converting the result back to TensorTrain

The result preserves the tensor structure while combining all bond dimensions.
Uses Algorithm("directsum") for optimal numerical precision.
"""
function Base.:-(stt1::TensorTrain, stt2::TensorTrain, stts...)
    # Check that all tensor trains have the same length
    lengths = [length(stt.data) for stt in [stt1, stt2, stts...]]
    if !all(l -> l == lengths[1], lengths)
        throw(ArgumentError("All tensor trains must have the same length for subtraction. Got lengths: $(lengths)"))
    end
    
    # Convert all to MPS
    mps_list = [ITensorMPS.MPS(stt) for stt in [stt1, stt2, stts...]]
    
    # Compute difference using ITensors.Algorithm("directsum") for better precision
    # Subtraction: stt1 - stt2 - stt3 - ... = stt1 + (-1)*stt2 + (-1)*stt3 + ...
    scaled_mps_list = [mps_list[1]]  # First term is positive
    for i in 2:length(mps_list)
        push!(scaled_mps_list, -1 * mps_list[i])  # Remaining terms are negative
    end
    
    alg = Algorithm"directsum"()
    mps_diff = +(alg, scaled_mps_list...)
    
    # Convert back to TensorTrain
    # Use the left and right limits from the first tensor train
    return TensorTrain(mps_diff, stt1.llim, stt1.rlim)
end

"""
Add TensorTrain objects using ITensors.Algorithm("directsum")

This function computes the sum of tensor trains using ITensors.Algorithm("directsum") for high precision.
Note: Algorithm parameter is accepted for interface compatibility but Algorithm("directsum") is always used
for optimal numerical precision.
"""
function Base.:+(alg::Algorithm, stt1::TensorTrain, stts...)
    # Check that all tensor trains have the same length
    lengths = [length(stt.data) for stt in [stt1, stts...]]
    if !all(l -> l == lengths[1], lengths)
        throw(ArgumentError("All tensor trains must have the same length for addition. Got lengths: $(lengths)"))
    end
    
    # Convert all to MPS
    mps_list = [ITensorMPS.MPS(stt) for stt in [stt1, stts...]]
    
    # Compute sum using ITensors.Algorithm("directsum") for better precision
    # Note: Algorithm parameter is ignored, directsum algorithm is always used
    alg = Algorithm"directsum"()
    mps_sum = +(alg, mps_list...)
    
    # Convert back to TensorTrain
    # Use the left and right limits from the first tensor train
    return TensorTrain(mps_sum, stt1.llim, stt1.rlim)
end

"""
Add TensorTrain objects with algorithm keyword argument.

This function accepts an `alg` keyword argument for interface compatibility,
but always uses Algorithm("directsum") for optimal numerical precision.
"""
function Base.:+(stt1::TensorTrain, stts...; alg::Union{String,Algorithm}="directsum", kwargs...)
    alg_obj = alg isa String ? Algorithm(alg) : alg
    return +(alg_obj, stt1, stts...)
end


"""
    truncate!(stt::TensorTrain; cutoff::Real=default_cutoff(), maxdim::Int=default_maxdim(), kwargs...)

Truncate a TensorTrain in-place by removing small singular values.

This function modifies the TensorTrain in-place by converting to MPS,
applying ITensorMPS.truncate!, and updating the tensor data.

# Arguments
- `stt::TensorTrain`: The tensor train to truncate (modified in-place)

# Keyword Arguments
- `cutoff::Real`: Cutoff threshold for singular values (default: `default_cutoff()`)
- `maxdim::Int`: Maximum bond dimension (default: `default_maxdim()`)
- `kwargs...`: Additional keyword arguments passed to ITensorMPS.truncate!

# Returns
- `TensorTrain`: The modified tensor train (same object as input)
"""
function truncate!(stt::TensorTrain; cutoff::Real=default_cutoff(), maxdim::Int=default_maxdim(), kwargs...)::TensorTrain
    mps = ITensorMPS.MPS(stt)
    ITensorMPS.truncate!(mps; cutoff=cutoff, maxdim=maxdim, kwargs...)
    # Update in place
    for i in 1:length(stt)
        stt[i] = mps[i]
    end
    return stt
end

"""
    truncate(stt::TensorTrain; cutoff::Real=default_cutoff(), maxdim::Int=default_maxdim(), kwargs...)

Truncate a TensorTrain by removing small singular values, returning a new object.

This function creates a new TensorTrain by converting to MPS,
applying ITensorMPS.truncate!, and creating a new TensorTrain from the result.

# Arguments
- `stt::TensorTrain`: The tensor train to truncate

# Keyword Arguments
- `cutoff::Real`: Cutoff threshold for singular values (default: `default_cutoff()`)
- `maxdim::Int`: Maximum bond dimension (default: `default_maxdim()`)
- `kwargs...`: Additional keyword arguments passed to ITensorMPS.truncate!

# Returns
- `TensorTrain`: A new truncated tensor train
"""
function truncate(stt::TensorTrain; cutoff::Real=default_cutoff(), maxdim::Int=default_maxdim(), kwargs...)::TensorTrain
    mps = ITensorMPS.MPS(stt)
    ITensorMPS.truncate!(mps; cutoff=cutoff, maxdim=maxdim, kwargs...)
    return TensorTrain(mps)
end

"""
    maxlinkdim(stt::TensorTrain)

Get the maximum link (bond) dimension in a TensorTrain.

This function computes the maximum dimension of the bond indices
connecting adjacent tensors in the tensor train.

# Arguments
- `stt::TensorTrain`: The tensor train to analyze

# Returns
- `Int`: Maximum bond dimension
"""
function maxlinkdim(stt::TensorTrain)
    return ITensorMPS.maxlinkdim(ITensorMPS.MPO(stt))
end

function _extractsite(x::TensorTrain, n::Int)::Vector{Index}
    if n == 1
        return copy(uniqueinds(x[n], x[n + 1]))
    elseif n == length(x)
        return copy(uniqueinds(x[n], x[n - 1]))
    else
        return copy(uniqueinds(x[n], x[n + 1], x[n - 1]))
    end
end

"""
    siteinds(x::TensorTrain)

Extract the site indices from each tensor in a TensorTrain.

This function returns a vector of index vectors, where each element contains
the site (physical) indices for the corresponding tensor in the train.

# Arguments
- `x::TensorTrain`: The tensor train to extract site indices from

# Returns
- `Vector{Vector{Index}}`: Vector of site index vectors, one per tensor
"""
siteinds(x::TensorTrain) = [_extractsite(x, n) for n in eachindex(x)]

"""
    prime(tt::TensorTrain, args...; kwargs...)

Apply `ITensors.prime` to all ITensors in a TensorTrain.

This function applies the `prime` function to each tensor in the tensor train,
returning a new TensorTrain with primed indices.

# Arguments
- `tt::TensorTrain`: The tensor train to prime
- `args...`: Arguments passed to `ITensors.prime`
- `kwargs...`: Keyword arguments passed to `ITensors.prime`

# Returns
- `TensorTrain`: A new TensorTrain with primed indices

# Examples
```julia
tt_primed = prime(tt, 1)  # Prime all indices by 1
tt_primed = prime(tt, 1; inds=sites)  # Prime only specific indices
```
"""
function ITensors.prime(tt::TensorTrain, args...; kwargs...)
    return TensorTrain([ITensors.prime(t, args...; kwargs...) for t in tt.data], tt.llim, tt.rlim)
end

"""
    noprime(tt::TensorTrain, args...; kwargs...)

Apply `ITensors.noprime` to all ITensors in a TensorTrain.

This function applies the `noprime` function to each tensor in the tensor train,
returning a new TensorTrain with unprimed indices.

# Arguments
- `tt::TensorTrain`: The tensor train to unprime
- `args...`: Arguments passed to `ITensors.noprime`
- `kwargs...`: Keyword arguments passed to `ITensors.noprime`

# Returns
- `TensorTrain`: A new TensorTrain with unprimed indices
"""
function ITensors.noprime(tt::TensorTrain, args...; kwargs...)
    return TensorTrain([ITensors.noprime(t, args...; kwargs...) for t in tt.data], tt.llim, tt.rlim)
end

"""
    replaceprime(tt::TensorTrain, p1 => p2; kwargs...)

Apply `ITensors.replaceprime` to all ITensors in a TensorTrain.

This function applies the `replaceprime` function to each tensor in the tensor train,
replacing prime level `p1` with `p2` for all matching indices.

# Arguments
- `tt::TensorTrain`: The tensor train to modify
- `p1 => p2`: Pair specifying the prime level replacement (e.g., `1 => 0` or `2 => 1`)
- `kwargs...`: Keyword arguments passed to `ITensors.replaceprime`

# Returns
- `TensorTrain`: A new TensorTrain with replaced prime levels

# Examples
```julia
tt_replaced = replaceprime(tt, 1 => 0)  # Replace prime level 1 with 0
tt_replaced = replaceprime(tt, 2 => 1; inds=sites)  # Replace only specific indices
```
"""
function ITensors.replaceprime(tt::TensorTrain, p1_p2::Pair; kwargs...)
    return TensorTrain([ITensors.replaceprime(t, p1_p2; kwargs...) for t in tt.data], tt.llim, tt.rlim)
end