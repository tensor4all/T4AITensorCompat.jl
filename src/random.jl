#===
random.jl - Random tensor train generation functions

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

