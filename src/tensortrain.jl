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
    
    # Internal constructor to prevent ambiguous matches
    TensorTrain(data::Vector{ITensor}, llim::Int, rlim::Int) = new(data, llim, rlim)
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

"""
    TensorTrain(sites::Vector{<:Index})

Construct a TensorTrain (MPS) from a vector of site indices.

This creates an MPS with empty ITensors initialized with the specified site indices.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices

# Returns
- `TensorTrain`: A new TensorTrain (MPS) object
"""
function TensorTrain(sites::Vector{<:Index})
    # Construct an MPS with default link dimensions
    mps = ITensorMPS.MPS(Float64, sites)
    return TensorTrain(mps)
end

"""
    TensorTrain(::Type{T}, sites::Vector{<:Index}; linkdims=1) where {T<:Number}

Construct a TensorTrain (MPS) from a vector of site indices with specified element type.

This creates an MPS with empty ITensors of type `T` initialized with the specified site indices.

# Arguments
- `T::Type{<:Number}`: Element type (e.g., Float64, ComplexF64)
- `sites::Vector{<:Index}`: Vector of site indices
- `linkdims::Union{Integer,Vector{<:Integer}}`: Link dimension(s) (default: 1)

# Returns
- `TensorTrain`: A new TensorTrain (MPS) object
"""
function TensorTrain(::Type{T}, sites::Vector{<:Index}; linkdims::Union{Integer,Vector{<:Integer}}=1) where {T<:Number}
    mps = ITensorMPS.MPS(T, sites; linkdims=linkdims)
    return TensorTrain(mps)
end

"""
    TensorTrain(::Type{T}, sites::Vector{Vector{<:Index}}, linkdims::Union{Integer,Vector{<:Integer}}) where {T<:Number}

Construct a TensorTrain (MPO) from a vector of site index pairs with specified element type and link dimensions.

This creates an MPO by manually constructing ITensor objects for each site. Each site is represented by a pair of indices
(typically the upper and lower indices of the MPO).

# Arguments
- `T::Type{<:Number}`: Element type (e.g., Float64, ComplexF64)
- `sites::Vector{Vector{<:Index}}`: Vector of site index pairs, where each element is a vector of indices for that site
- `linkdims::Union{Integer,Vector{<:Integer}}`: Link dimension(s) between sites

# Returns
- `TensorTrain`: A new TensorTrain (MPO) object

# Examples
```julia
sites = [[Index(2, "Site,n=\$n"), Index(2, "Site,n=\$n")] for n=1:5]
mpo = TensorTrain(ComplexF64, sites, 2)
```
"""
function TensorTrain(::Type{T}, sites::AbstractVector{<:AbstractVector{<:Index}}, linkdims::Union{Integer,Vector{<:Integer}}) where {T<:Number}
    N = length(sites)
    N == 0 && return TensorTrain(Vector{ITensor}())
    
    # Normalize linkdims to a vector
    if linkdims isa Integer
        _linkdims = fill(linkdims, N - 1)
    else
        _linkdims = linkdims
    end
    
    length(_linkdims) == N - 1 || error("Length mismatch: linkdims ($(length(_linkdims))) must have length $(N - 1)")
    
    # Create internal link indices (no boundary links)
    links = N > 1 ? [Index(_linkdims[n], "Link,l=$n") for n in 1:(N - 1)] : Index[]
    
    # Create ITensors for each site
    tensors = Vector{ITensor}(undef, N)
    for n in 1:N
        site_inds = sites[n]
        if length(site_inds) != 2
            error("Each site must have exactly 2 indices (upper and lower), but site $n has $(length(site_inds))")
        end
        
        # For MPO: sites[n] = [lower_index, upper_index]
        # Lower index is unprimed, upper index is primed
        lower_ind = site_inds[1]  # unprimed
        upper_ind = ITensors.prime(site_inds[2])  # primed
        
        # MPO structure:
        # - First site: (upper_site, lower_site, right_link)
        # - Last site:  (left_link, upper_site, lower_site)
        # - Middle:     (left_link, upper_site, lower_site, right_link)
        if n == 1 && n == N
            inds_tuple = (upper_ind, lower_ind)
        elseif n == 1
            inds_tuple = (upper_ind, lower_ind, links[n])
        elseif n == N
            inds_tuple = (links[n - 1], upper_ind, lower_ind)
        else
            inds_tuple = (links[n - 1], upper_ind, lower_ind, links[n])
        end
        
        # Create zero ITensor with appropriate dimensions
        dims = map(ITensors.dim, inds_tuple)
        data = zeros(T, dims...)
        tensors[n] = ITensor(data, inds_tuple...)
    end
    
    return TensorTrain(tensors)
end

"""
    TensorTrain(A::ITensor, sites; kwargs...)

Construct a TensorTrain from an ITensor by decomposing it according to site indices.

This function creates an MPS or MPO by decomposing the ITensor `A` site by site
according to the site indices `sites`. The `sites` can be either `Vector{Index}`
(for MPS) or `Vector{Vector{Index}}` (for MPO).

# Arguments
- `A::ITensor`: The ITensor to decompose
- `sites`: Site indices - either `Vector{Index}` (for MPS) or `Vector{Vector{Index}}` (for MPO)

# Keyword Arguments
- `leftinds=nothing`: Optional left dangling indices
- `orthocenter::Integer=length(sites)`: Desired orthogonality center
- `tags`: Tags for link indices
- `cutoff`: Truncation error at each link
- `maxdim`: Maximum link dimension
- `kwargs...`: Additional keyword arguments passed to the decomposition

# Returns
- `TensorTrain`: A new TensorTrain object (MPS or MPO depending on `sites`)

# Examples
```julia
sites = [Index(2, "Site,n=\$n") for n=1:5]
A = randomITensor(sites...)
tt = TensorTrain(A, sites)
```
"""
function TensorTrain(A::ITensor, sites; kwargs...)
    # Detect if sites is Vector{Index} (MPS) or Vector{Vector{Index}} (MPO)
    if length(sites) > 0 && sites[1] isa Index
        # MPS case: sites is Vector{Index}
        mps = ITensorMPS.MPS(A, sites; kwargs...)
        return TensorTrain(mps)
    else
        # MPO case: sites is Vector{Vector{Index}} or similar
        mpo = ITensorMPS.MPO(A, sites; kwargs...)
        return TensorTrain(mpo)
    end
end

"""
    TensorTrain(A::AbstractArray, sites; kwargs...)

Construct a TensorTrain from an AbstractArray by converting it to an ITensor first.

# Arguments
- `A::AbstractArray`: The array to convert
- `sites`: Site indices - either `Vector{Index}` (for MPS) or `Vector{Vector{Index}}` (for MPO)
- `kwargs...`: Keyword arguments passed to the decomposition

# Returns
- `TensorTrain`: A new TensorTrain object
"""
function TensorTrain(A::AbstractArray, sites; kwargs...)
    # Convert array to ITensor
    if length(sites) > 0 && sites[1] isa Index
        # MPS case: sites is Vector{Index}
        A_itensor = ITensor(A, sites...)
        mps = ITensorMPS.MPS(A_itensor, sites; kwargs...)
        return TensorTrain(mps)
    else
        # MPO case: sites is Vector{Vector{Index}}
        # Flatten sites for ITensor construction
        sites_flat = collect(Iterators.flatten(sites))
        A_itensor = ITensor(A, sites_flat...)
        mpo = ITensorMPS.MPO(A_itensor, sites; kwargs...)
        return TensorTrain(mpo)
    end
end

"""
    TensorTrain(tt_data; sites, kwargs...)

Construct a TensorTrain from tensor train data (like QuanticsTCI/TensorCrossInterpolation) with specified sites.

This constructor is used for converting from other tensor train formats (e.g., QuanticsTCI, TensorCrossInterpolation)
to TensorTrain by specifying the site indices.

# Arguments
- `tt_data`: Tensor train data (e.g., from QuanticsTCI or TensorCrossInterpolation)
- `sites`: Site indices - either `Vector{Index}` (for MPS) or `Vector{Vector{Index}}` (for MPO)

# Keyword Arguments
- `kwargs...`: Additional keyword arguments passed to ITensorMPS constructors

# Returns
- `TensorTrain`: A new TensorTrain object
"""
function TensorTrain(tt_data; sites, kwargs...)
    if length(sites) > 0 && sites[1] isa Index
        # MPS case: sites is Vector{Index}
        mps = ITensorMPS.MPS(tt_data, sites; kwargs...)
        return TensorTrain(mps)
    else
        # MPO case: sites is Vector{Vector{Index}}
        # Check if tt_data is TensorCrossInterpolation.TensorTrain
        # Try to use TensorCrossInterpolation conversion if available
        tt_type = typeof(tt_data)
        if hasmethod(ITensorMPS.MPO, (typeof(tt_data),); kwargs...)
            # Try ITensorMPS.MPO with sites keyword argument (for TensorCrossInterpolation)
            try
                mpo = ITensorMPS.MPO(tt_data; sites=sites, kwargs...)
                return TensorTrain(mpo)
            catch
                # Fallback to generic MPO constructor
                mpo = ITensorMPS.MPO(tt_data, sites; kwargs...)
                return TensorTrain(mpo)
            end
        else
            # Try generic MPO constructor
            mpo = ITensorMPS.MPO(tt_data, sites; kwargs...)
            return TensorTrain(mpo)
        end
    end
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
Add multiple TensorTrain objects using ITensors.Algorithm("directsum").

The sum is computed with Algorithm("directsum") for high precision.
"""
function Base.:+(stt1::TensorTrain, stt2::TensorTrain, stts::Vararg{TensorTrain})
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
function Base.:-(stt1::TensorTrain, stt2::TensorTrain, stts::Vararg{TensorTrain})
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
function Base.:+(alg::Algorithm, stt1::TensorTrain, stts::Vararg{TensorTrain})
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
function Base.:+(stt1::TensorTrain, stts::Vararg{TensorTrain}; alg::Union{String,Algorithm}="directsum", kwargs...)
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

"""
    linkinds(tt::TensorTrain)

Extract the link (bond) indices from a TensorTrain.

This function returns a vector of link indices connecting adjacent tensors
in the tensor train. For a TensorTrain of length N, it returns N-1 link indices.

# Arguments
- `tt::TensorTrain`: The tensor train to extract link indices from

# Returns
- `Vector{Index}`: Vector of link indices connecting adjacent tensors
"""
function linkinds(tt::TensorTrain)
    N = length(tt)
    if N <= 1
        return Index[]
    end
    links = Index[]
    for n in 1:(N - 1)
        # Link index is the common index between tensor n and n+1
        common = commoninds(tt[n], tt[n + 1])
        if length(common) != 1
            error("Expected exactly one common index between tensors $n and $(n+1), got $(length(common))")
        end
        push!(links, only(common))
    end
    return links
end

"""
    linkind(tt::TensorTrain, p::Int)

Get the link index at position p in a TensorTrain.

Position p refers to the link between tensor p and p+1.
Valid positions are 1 to length(tt)-1.

# Arguments
- `tt::TensorTrain`: The tensor train
- `p::Int`: Position of the link (1-indexed, between tensor p and p+1)

# Returns
- `Index`: The link index at position p
"""
function linkind(tt::TensorTrain, p::Int)
    links = linkinds(tt)
    if p < 1 || p > length(links)
        error("Link position $p out of range. Valid range: 1 to $(length(links))")
    end
    return links[p]
end

"""
    findsite(tt::TensorTrain, site::Index)

Find the position of a site index in a TensorTrain.

This function searches for the site index in the tensor train and returns
the position (1-indexed) where it is found. Returns `nothing` if not found.

# Arguments
- `tt::TensorTrain`: The tensor train to search
- `site::Index`: The site index to find

# Returns
- `Union{Int, Nothing}`: Position of the site index, or `nothing` if not found
"""
function findsite(tt::TensorTrain, site::Index)
    sites = siteinds(tt)
    for (pos, site_vec) in enumerate(sites)
        if site in site_vec
            return pos
        end
    end
    return nothing
end

"""
    findsites(tt::TensorTrain, site::Index)

Find all positions of a site index in a TensorTrain.

This function searches for the site index in the tensor train and returns
all positions (1-indexed) where it is found.

# Arguments
- `tt::TensorTrain`: The tensor train to search
- `site::Index`: The site index to find

# Returns
- `Vector{Int}`: Vector of positions where the site index is found
"""
function findsites(tt::TensorTrain, site::Index)
    sites = siteinds(tt)
    positions = Int[]
    for (pos, site_vec) in enumerate(sites)
        if site in site_vec
            push!(positions, pos)
        end
    end
    return positions
end

"""
    isortho(tt::TensorTrain)

Check if a TensorTrain is orthogonal (canonical form).

This function checks whether the tensor train is in orthogonal/canonical form
by delegating to ITensorMPS.isortho after converting to MPS.

# Arguments
- `tt::TensorTrain`: The tensor train to check

# Returns
- `Bool`: `true` if the tensor train is orthogonal, `false` otherwise
"""
function isortho(tt::TensorTrain)
    mps = ITensorMPS.MPS(tt)
    return ITensorMPS.isortho(mps)
end

"""
    orthocenter(tt::TensorTrain)

Get the orthogonality center position of a TensorTrain.

This function returns the position of the orthogonality center in the tensor train
by delegating to ITensorMPS.orthocenter after converting to MPS.

# Arguments
- `tt::TensorTrain`: The tensor train

# Returns
- `Int`: Position of the orthogonality center (1-indexed)
"""
function orthocenter(tt::TensorTrain)
    mps = ITensorMPS.MPS(tt)
    return ITensorMPS.orthocenter(mps)
end

#===
Random tensor train generation functions
====#

import Random
import ITensorMPS

"""
    random_mps(sites::Vector{<:Index}; linkdims=1)

Construct a random TensorTrain (MPS) with link dimension `linkdims` which by
default has element type `Float64`.

`linkdims` can also accept a `Vector{Int}` with
`length(linkdims) == length(sites) - 1` for constructing an
MPS with non-uniform bond dimension.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices
- `linkdims::Union{Integer,Vector{<:Integer}}`: Link dimension(s) (default: 1)

# Returns
- `TensorTrain`: A random tensor train (MPS)

# Examples
```julia
sites = [Index(2, "Qubit,n=\$n") for n = 1:5]
psi = random_mps(sites; linkdims=3)
```
"""
function random_mps(sites::Vector{<:Index}; linkdims::Union{Integer,Vector{<:Integer}}=1)
    return random_mps(Random.default_rng(), sites; linkdims)
end

"""
    random_mps(rng::Random.AbstractRNG, sites::Vector{<:Index}; linkdims=1)

Construct a random TensorTrain (MPS) with link dimension `linkdims` using the specified RNG.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator
- `sites::Vector{<:Index}`: Vector of site indices
- `linkdims::Union{Integer,Vector{<:Integer}}`: Link dimension(s) (default: 1)

# Returns
- `TensorTrain`: A random tensor train (MPS)
"""
function random_mps(rng::Random.AbstractRNG, sites::Vector{<:Index}; linkdims::Union{Integer,Vector{<:Integer}}=1)
    return random_mps(rng, Float64, sites; linkdims)
end

"""
    random_mps(eltype::Type{<:Number}, sites::Vector{<:Index}; linkdims=1)

Construct a random TensorTrain (MPS) with specified element type and link dimension.

# Arguments
- `eltype::Type{<:Number}`: Element type (e.g., Float64, ComplexF64)
- `sites::Vector{<:Index}`: Vector of site indices
- `linkdims::Union{Integer,Vector{<:Integer}}`: Link dimension(s) (default: 1)

# Returns
- `TensorTrain`: A random tensor train (MPS)
"""
function random_mps(eltype::Type{<:Number}, sites::Vector{<:Index}; linkdims::Union{Integer,Vector{<:Integer}}=1)
    return random_mps(Random.default_rng(), eltype, sites; linkdims)
end

"""
    random_mps(rng::Random.AbstractRNG, eltype::Type{<:Number}, sites::Vector{<:Index}; linkdims=1)

Construct a random TensorTrain (MPS) with specified RNG, element type, and link dimension.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator
- `eltype::Type{<:Number}`: Element type (e.g., Float64, ComplexF64)
- `sites::Vector{<:Index}`: Vector of site indices
- `linkdims::Union{Integer,Vector{<:Integer}}`: Link dimension(s) (default: 1)

# Returns
- `TensorTrain`: A random tensor train (MPS)
"""
function random_mps(rng::Random.AbstractRNG, eltype::Type{<:Number}, sites::Vector{<:Index}; linkdims::Union{Integer,Vector{<:Integer}}=1)
    mps = ITensorMPS.random_mps(rng, eltype, sites; linkdims)
    return TensorTrain(mps)
end

"""
    random_mps(sites::Vector{<:Index}, state; linkdims=1)

Construct a random TensorTrain (MPS) with initial state (for quantum number conservation).

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices
- `state`: Initial state specification (function or vector)
- `linkdims::Union{Integer,Vector{<:Integer}}`: Link dimension(s) (default: 1)

# Returns
- `TensorTrain`: A random tensor train (MPS)
"""
function random_mps(sites::Vector{<:Index}, state; linkdims::Union{Integer,Vector{<:Integer}}=1)
    return random_mps(Random.default_rng(), sites, state; linkdims)
end

"""
    random_mps(rng::Random.AbstractRNG, sites::Vector{<:Index}, state; linkdims=1)

Construct a random TensorTrain (MPS) with initial state using the specified RNG.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator
- `sites::Vector{<:Index}`: Vector of site indices
- `state`: Initial state specification (function or vector)
- `linkdims::Union{Integer,Vector{<:Integer}}`: Link dimension(s) (default: 1)

# Returns
- `TensorTrain`: A random tensor train (MPS)
"""
function random_mps(rng::Random.AbstractRNG, sites::Vector{<:Index}, state; linkdims::Union{Integer,Vector{<:Integer}}=1)
    return random_mps(rng, Float64, sites, state; linkdims)
end

"""
    random_mps(eltype::Type{<:Number}, sites::Vector{<:Index}, state; linkdims=1)

Construct a random TensorTrain (MPS) with element type and initial state.

# Arguments
- `eltype::Type{<:Number}`: Element type (e.g., Float64, ComplexF64)
- `sites::Vector{<:Index}`: Vector of site indices
- `state`: Initial state specification (function or vector)
- `linkdims::Union{Integer,Vector{<:Integer}}`: Link dimension(s) (default: 1)

# Returns
- `TensorTrain`: A random tensor train (MPS)
"""
function random_mps(eltype::Type{<:Number}, sites::Vector{<:Index}, state; linkdims::Union{Integer,Vector{<:Integer}}=1)
    return random_mps(Random.default_rng(), eltype, sites, state; linkdims)
end

"""
    random_mps(rng::Random.AbstractRNG, eltype::Type{<:Number}, sites::Vector{<:Index}, state; linkdims=1)

Construct a random TensorTrain (MPS) with RNG, element type, and initial state.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator
- `eltype::Type{<:Number}`: Element type (e.g., Float64, ComplexF64)
- `sites::Vector{<:Index}`: Vector of site indices
- `state`: Initial state specification (function or vector)
- `linkdims::Union{Integer,Vector{<:Integer}}`: Link dimension(s) (default: 1)

# Returns
- `TensorTrain`: A random tensor train (MPS)
"""
function random_mps(rng::Random.AbstractRNG, eltype::Type{<:Number}, sites::Vector{<:Index}, state; linkdims::Union{Integer,Vector{<:Integer}}=1)
    mps = ITensorMPS.random_mps(rng, eltype, sites, state; linkdims)
    return TensorTrain(mps)
end

"""
    random_mpo(sites::Vector{<:Index}, m::Int=1)

Construct a random TensorTrain (MPO) with specified sites.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices
- `m::Int`: Currently only m=1 is supported (default: 1)

# Returns
- `TensorTrain`: A random tensor train (MPO)

# Examples
```julia
sites = [Index(2, "Qubit,n=\$n") for n = 1:5]
mpo = random_mpo(sites)
```
"""
function random_mpo(sites::Vector{<:Index}, m::Int=1)
    return random_mpo(Random.default_rng(), sites, m)
end

"""
    random_mpo(rng::Random.AbstractRNG, sites::Vector{<:Index}, m::Int=1)

Construct a random TensorTrain (MPO) with specified RNG and sites.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator
- `sites::Vector{<:Index}`: Vector of site indices
- `m::Int`: Currently only m=1 is supported (default: 1)

# Returns
- `TensorTrain`: A random tensor train (MPO)
"""
function random_mpo(rng::Random.AbstractRNG, sites::Vector{<:Index}, m::Int=1)
    mpo = ITensorMPS.random_mpo(rng, sites, m)
    return TensorTrain(mpo)
end