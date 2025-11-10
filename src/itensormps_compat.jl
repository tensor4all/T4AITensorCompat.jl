#===
itensormps_compat.jl - Compatibility functions for ITensorMPS API

This file provides compatibility functions that match ITensorMPS API
for use with TensorTrain. These are temporary functions for migration purposes.

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

# Note: MPS and MPO are type aliases for TensorTrain, so MPS([...]) and MPO([...])
# will automatically call TensorTrain([...]) constructor. No explicit constructors needed.

