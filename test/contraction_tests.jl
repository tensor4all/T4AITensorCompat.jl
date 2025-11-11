@testitem "contraction.jl" begin
    include("util.jl")

    import T4AITensorCompat: TensorTrain, contract, dist, fit
    import ITensors: ITensors, ITensor, Index, random_itensor
    import ITensorMPS
    import ITensors: Algorithm, @Algorithm_str
    import LinearAlgebra: norm
    ITensors.disable_warn_order()
    using Random

    # Test algorithms
    algs = ["densitymatrix", "fit", "zipup"]
    eps = Dict("densitymatrix" => 1e-6, "fit" => 1e-12, "zipup" => 1e-12)
    linkdims = 3
    R = 5

    for alg in algs
        @testset "MPO-MPO contraction (x-y-z) with $alg" begin
            Random.seed!(1234)

            sitesx = [Index(2, "Qubit,x=$n") for n = 1:R]
            sitesy = [Index(2, "Qubit,y=$n") for n = 1:R]
            sitesz = [Index(2, "Qubit,z=$n") for n = 1:R]

            sitesa = collect(collect.(zip(sitesx, sitesy)))
            sitesb = collect(collect.(zip(sitesy, sitesz)))
            a_mpo = _random_mpo(sitesa; linkdims = linkdims)
            b_mpo = _random_mpo(sitesb; linkdims = linkdims)
            
            # Convert to TensorTrain
            a = TensorTrain(a_mpo)
            b = TensorTrain(b_mpo)
            
            ab_ref = contract(a, b; alg = Algorithm"naive"())
            ab = contract(a, b; alg = Algorithm(alg))
            @test relative_error(ab_ref, ab) < eps[alg]
        end
    end

    for alg in algs
        @testset "MPO-MPO contraction (xk-y-z) with $alg" begin
            Random.seed!(1234)
            sitesx = [Index(2, "Qubit,x=$n") for n = 1:R]
            sitesk = [Index(2, "Qubit,k=$n") for n = 1:R]
            sitesy = [Index(2, "Qubit,y=$n") for n = 1:R]
            sitesz = [Index(2, "Qubit,z=$n") for n = 1:R]

            sitesa = collect(collect.(zip(sitesx, sitesk, sitesy)))
            sitesb = collect(collect.(zip(sitesy, sitesz)))
            a_mpo = _random_mpo(sitesa; linkdims = linkdims)
            b_mpo = _random_mpo(sitesb; linkdims = linkdims)
            
            # Convert to TensorTrain
            a = TensorTrain(a_mpo)
            b = TensorTrain(b_mpo)
            
            ab_ref = contract(a, b; alg = Algorithm"naive"())
            ab = contract(a, b; alg = Algorithm(alg))
            @test relative_error(ab_ref, ab) < eps[alg]
        end
    end

    for alg in algs
        @testset "MPO-MPO contraction (xk-y-zl) with $alg" begin
            Random.seed!(1234)
            sitesx = [Index(2, "Qubit,x=$n") for n = 1:R]
            sitesk = [Index(2, "Qubit,k=$n") for n = 1:R]
            sitesy = [Index(2, "Qubit,y=$n") for n = 1:R]
            sitesz = [Index(2, "Qubit,z=$n") for n = 1:R]
            sitesl = [Index(2, "Qubit,l=$n") for n = 1:R]

            sitesa = collect(collect.(zip(sitesx, sitesk, sitesy)))
            sitesb = collect(collect.(zip(sitesy, sitesz, sitesl)))
            a_mpo = _random_mpo(sitesa; linkdims = linkdims)
            b_mpo = _random_mpo(sitesb; linkdims = linkdims)
            
            # Convert to TensorTrain
            a = TensorTrain(a_mpo)
            b = TensorTrain(b_mpo)
            
            ab_ref = contract(a, b; alg = Algorithm"naive"())
            ab = contract(a, b; alg = Algorithm(alg))

            @test relative_error(ab_ref, ab) < eps[alg]
        end
    end

    for alg in algs
        @testset "MPO-MPO contraction (xk-ym-zl) with $alg" begin
            Random.seed!(1234)
            sitesx = [Index(2, "Qubit,x=$n") for n = 1:R]
            sitesk = [Index(2, "Qubit,k=$n") for n = 1:R]
            sitesy = [Index(2, "Qubit,y=$n") for n = 1:R]
            sitesz = [Index(2, "Qubit,z=$n") for n = 1:R]
            sitesl = [Index(2, "Qubit,l=$n") for n = 1:R]
            sitesm = [Index(2, "Qubit,m=$n") for n = 1:R]

            sitesa = collect(collect.(zip(sitesx, sitesk, sitesm, sitesy)))
            sitesb = collect(collect.(zip(sitesy, sitesm, sitesz, sitesl)))
            a_mpo = _random_mpo(sitesa; linkdims = linkdims)
            b_mpo = _random_mpo(sitesb; linkdims = linkdims)
            
            # Convert to TensorTrain
            a = TensorTrain(a_mpo)
            b = TensorTrain(b_mpo)
            
            ab_ref = contract(a, b; alg = Algorithm"naive"())
            ab = contract(a, b; alg = Algorithm(alg))
            @test relative_error(ab_ref, ab) < eps[alg]
        end
    end

    @testset "fit function for summing multiple TensorTrain objects" begin
        Random.seed!(1234)
        R = 5
        linkdims = 3

        # Create test sites
        sites = [Index(2, "Qubit,s=$n") for n = 1:R]
        
        # Create multiple random MPS
        mps1 = _random_mps(sites; linkdims = linkdims)
        mps2 = _random_mps(sites; linkdims = linkdims)
        mps3 = _random_mps(sites; linkdims = linkdims)
        
        # Convert to TensorTrain
        tt1 = TensorTrain(mps1)
        tt2 = TensorTrain(mps2)
        tt3 = TensorTrain(mps3)
        
        # Create initial guess (use first tensor train)
        init_tt = TensorTrain(mps1)
        
        # Test fit with equal coefficients
        coeffs = [1.0, 1.0, 1.0]
        result = fit([tt1, tt2, tt3], init_tt; coeffs=coeffs, nsweeps=2, cutoff=1e-12, maxdim=100)
        
        # Verify result is a TensorTrain
        @test result isa TensorTrain
        @test length(result) == R
        
        # Test fit with different coefficients
        coeffs2 = [2.0, 0.5, -1.0]
        result2 = fit([tt1, tt2, tt3], init_tt; coeffs=coeffs2, nsweeps=2, cutoff=1e-12, maxdim=100)
        
        @test result2 isa TensorTrain
        @test length(result2) == R
        
        # Test fit with default coefficients (all ones)
        result3 = fit([tt1, tt2], init_tt; nsweeps=2, cutoff=1e-12, maxdim=100)
        @test result3 isa TensorTrain
        @test length(result3) == R
    end

    @testset "fit function accuracy" begin
        Random.seed!(5678)
        R = 4
        linkdims = 2

        # Create test sites
        sites = [Index(2, "Qubit,s=$n") for n = 1:R]
        
        # Create two random MPS
        mps1 = _random_mps(sites; linkdims = linkdims)
        mps2 = _random_mps(sites; linkdims = linkdims)
        
        # Convert to TensorTrain
        tt1 = TensorTrain(mps1)
        tt2 = TensorTrain(mps2)
        
        # Compute exact sum using direct sum algorithm
        exact_sum = tt1 + tt2
        
        # Create initial guess (use exact sum truncated)
        init_tt = exact_sum
        
        # Fit the sum
        coeffs = [1.0, 1.0]
        fitted = fit([tt1, tt2], init_tt; coeffs=coeffs, nsweeps=3, cutoff=1e-15, maxdim=100)
        
        # Check that fitted result is close to exact sum
        # Note: fit is an approximation, so we check for reasonable accuracy
        error = dist(fitted, exact_sum)
        @test error < 1e-5 # I observed this function is sometime less accurate than direct sum of the input states.
    end
end

@testitem "contraction_tests.jl/product_MPO_times_MPS" begin
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

@testitem "contraction_tests.jl/product_MPO_times_MPO" begin
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