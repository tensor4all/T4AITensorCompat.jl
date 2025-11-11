@testitem "apply_tests.jl/MPO_times_MPS" begin
    using Test
    using ITensors
    import ITensors: Algorithm, @Algorithm_str
    import ITensorMPS
    using T4AITensorCompat: random_mps, random_mpo, product, apply, siteinds, TensorTrain

    ITensors.disable_warn_order()

    @testset "product(A::MPO, ψ::MPS) matches ITensorMPS.contract + replaceprime" for R in (2, 3)
        # Build simple qubit sites
        sites = [Index(2, "Qubit, s=$(n)") for n in 1:R]
        ψ = random_mps(sites)
        A = random_mpo(sites)

        # Under test
        f = product(A, ψ; alg="naive", cutoff=1e-25)

        # Reference using ITensorMPS.contract + replaceprime
        A_mpo = ITensorMPS.MPO(A)
        ψ_mps = ITensorMPS.MPS(ψ)
        f_ref_mps = ITensorMPS.contract(A_mpo, ψ_mps; alg=Algorithm("naive"), cutoff=1e-25)
        f_ref_mps = ITensorMPS.replaceprime(f_ref_mps, 1 => 0)
        f_ref = TensorTrain(f_ref_mps)

        # Compare full vectors
        s_order = reverse(sites) # standard order used in other tests
        f_vec = vec(Array(reduce(*, f), s_order))
        f_ref_vec = vec(Array(reduce(*, f_ref), s_order))
        @test f_vec ≈ f_ref_vec atol=1e-12 rtol=1e-12
    end

    # Alias check
    @testset "apply alias equals product (MPO*MPS)" for R in (2, 3)
        sites = [Index(2, "Qubit, s=$(n)") for n in 1:R]
        ψ = random_mps(sites)
        A = random_mpo(sites)
        @test Array(reduce(*, apply(A, ψ; alg="naive")), reverse(sites)) ≈ Array(reduce(*, product(A, ψ; alg="naive")), reverse(sites))
    end
end

@testitem "apply_tests.jl/MPO_times_MPO" begin
    using Test
    using ITensors
    import ITensors: Algorithm, @Algorithm_str
    import ITensorMPS
    using T4AITensorCompat: random_mpo, product, apply, siteinds, TensorTrain

    ITensors.disable_warn_order()

    @testset "product(A::MPO, B::MPO) matches ITensorMPS.contract(A', B) + replaceprime" for R in (2, 3)
        # Build simple qubit sites
        sites = [Index(2, "Qubit, s=$(n)") for n in 1:R]
        A = random_mpo(sites)
        B = random_mpo(sites)

        # Under test (zipup is standard for MPO*MPO)
        C = product(A, B; alg="zipup", cutoff=1e-25)

        # Reference using ITensorMPS.contract(A', B) + replaceprime(2=>1)
        A_mpo = ITensorMPS.MPO(A)
        B_mpo = ITensorMPS.MPO(B)
        C_ref_mpo = ITensorMPS.contract(A_mpo', B_mpo; alg=Algorithm("zipup"))
        C_ref_mpo = ITensorMPS.replaceprime(C_ref_mpo, 2 => 1)
        C_ref = TensorTrain(C_ref_mpo)

        # Compare dense tensors of the MPOs in a consistent index order
        flatten_sites(x) = collect(Iterators.flatten(siteinds(x)))
        C_arr = Array(reduce(*, C), flatten_sites(C))
        C_ref_arr = Array(reduce(*, C_ref), flatten_sites(C_ref))
        @test C_arr ≈ C_ref_arr atol=1e-12 rtol=1e-12
    end

    # Alias check
    @testset "apply alias equals product (MPO*MPO)" for R in (2, 3)
        sites = [Index(2, "Qubit, s=$(n)") for n in 1:R]
        A = random_mpo(sites)
        B = random_mpo(sites)
        flatten_sites(x) = collect(Iterators.flatten(siteinds(x)))
        @test Array(reduce(*, apply(A, B; alg="zipup")), flatten_sites(apply(A, B; alg="zipup"))) ≈ Array(reduce(*, product(A, B; alg="zipup")), flatten_sites(product(A, B; alg="zipup")))
    end
end
