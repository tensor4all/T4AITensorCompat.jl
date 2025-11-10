#===
fitalgorithm.jl - Variational fitting algorithm for tensor train contraction

This file contains code derived from ITensorTDVP.jl
Original source: https://github.com/ITensor/ITensorTDVP.jl
(specifically: https://github.com/shinaoka/ITensorTDVP.jl/commit/23e09395cce66215b256aeeaa993fe2c64a0f1c8)
Copyright (c) ITensorTDVP.jl developers
Copyright (c) 2025 Hiroshi Shinaoka and contributors (modifications)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Note: This is a temporary implementation until performance issues in ITensorNetworks.jl are resolved.
====#
import ITensorMPS: AbstractProjMPO, makeL!, makeR!, set_nsite!, OneITensor, MPO, MPS, linkinds
import ITensors: contract, siteinds, commoninds, sim, dag, replaceinds, @Algorithm_str
import Base: copy

"""
Contract M1 and M2, and return the result as an MPO.
"""
function contract_fit(M1::ITensorMPS.MPO, M2::ITensorMPS.MPO; nsweeps = default_nsweeps(), init = nothing, kwargs...)::ITensorMPS.MPO
    M2_ = ITensorMPS.MPS([M2[v] for v in eachindex(M2)])
    if init === nothing
        init_MPO::ITensorMPS.MPO = ITensors.contract(M1, M2; alg = Algorithm"zipup"(), kwargs...)
        init = ITensorMPS.MPS([init_MPO[v] for v in eachindex(init_MPO)])
    else
        init = ITensorMPS.MPS([init[v] for v in eachindex(M2)])
    end
    M12_ = contract_fit(M1, M2_; init_mps = init, nsweeps = nsweeps, kwargs...)
    M12 = ITensorMPS.MPO([M12_[v] for v in eachindex(M1)])

    return M12
end


# To support MPO-MPO contraction
# Taken from https://github.com/shinaoka/ITensorTDVP.jl/commit/23e09395cce66215b256aeeaa993fe2c64a0f1c8
function contract_fit(A::ITensorMPS.MPO, psi0::ITensorMPS.MPS; init_mps = psi0, nsweeps = 1, kwargs...)::ITensorMPS.MPS
    n = length(A)
    n != length(psi0) && throw(
        DimensionMismatch("lengths of MPO ($n) and MPS ($(length(psi0))) do not match"),
    )
    if n == 1
        return ITensorMPS.MPS([A[1] * psi0[1]])
    end

    any(i -> isempty(i), siteinds(commoninds, A, psi0)) &&
        error("In `contract(A::MPO, x::MPS)`, `A` and `x` must share a set of site indices")

    # In case A and psi0 have the same link indices
    A = sim(linkinds, A)

    # Fix site and link inds of init_mps
    init_mps = deepcopy(init_mps)
    init_mps = sim(linkinds, init_mps)
    Ai = siteinds(A)
    init_mpsi = siteinds(init_mps)
    for j = 1:n
        ti = nothing
        for i in Ai[j]
            if !hasind(psi0[j], i)
                ti = i
                break
            end
        end
        if ti !== nothing
            ci = commoninds(init_mpsi[j], A[j])[1]
            replaceind!(init_mps[j], ci => ti)
        end
    end

    reduced_operator = ITensorMPS.ReducedContractProblem(psi0, A)
    return ITensorMPS.alternating_update(
        reduced_operator,
        init_mps;
        updater = ITensorMPS.contract_operator_state_updater,
        nsweeps = nsweeps,
        kwargs...,
    )
end
