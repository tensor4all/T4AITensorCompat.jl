import T4AITensorCompat: TensorTrain, contract, dist
import ITensors: ITensors, ITensor, Index, random_itensor
import ITensorMPS
import ITensors: Algorithm, @Algorithm_str
import LinearAlgebra: norm
ITensors.disable_warn_order()
using Random

# Helper function to create random MPO with custom sites structure
function _random_mpo(sites::Vector{Vector{Index{T}}}; linkdims = 1) where {T}
    _random_mpo(Random.GLOBAL_RNG, sites; linkdims = linkdims)
end

function _random_mpo(rng, sites::Vector{Vector{Index{T}}}; linkdims = 1) where {T}
    N = length(sites)
    links = [Index(linkdims, "Link,n=$n") for n = 1:N-1]
    M = ITensorMPS.MPO(N)
    M[1] = random_itensor(rng, sites[1]..., links[1])
    M[N] = random_itensor(rng, links[N-1], sites[N]...)
    for n = 2:N-1
        M[n] = random_itensor(rng, links[n-1], sites[n]..., links[n])
    end
    return M
end

function relative_error(tt1::TensorTrain, tt2::TensorTrain)
    sites = T4AITensorCompat.siteinds(tt1)
    tt1_full = Array(reduce(*, tt1), sites)
    tt2_full = Array(reduce(*, tt2), sites)
    return norm(tt1_full - tt2_full) / norm(tt1_full)
end
