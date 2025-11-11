@testitem "tensortrain.jl" begin
    include("util.jl")

    import T4AITensorCompat: TensorTrain, dist, siteinds, random_mps, random_mpo
    import ITensors: ITensor, Index, random_itensor
    import ITensorMPS
    import ITensors: Algorithm, @Algorithm_str
    import LinearAlgebra: norm

    @testset "TensorTrain construction" begin
        # Create some test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")

        # Create test tensors
        t1 = random_itensor(i1, i2)
        t2 = random_itensor(i2, i3)

        # Test construction with default limits
        stt1 = TensorTrain([t1, t2])
        @test length(stt1.data) == 2
        @test stt1.llim == 0
        @test stt1.rlim == 3
    end

    @testset "Conversion from TensorTrain to MPS" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")

        # Create test tensors
        t1 = random_itensor(i1, i2)
        t2 = random_itensor(i2, i3)

        # Create TensorTrain
        stt = TensorTrain([t1, t2])

        # Convert to MPS
        mps = ITensorMPS.MPS(stt)

        # Test that conversion worked
        @test length(mps) == 2
        @test mps[1] == t1
        @test mps[2] == t2
    end

    @testset "Conversion from MPS to TensorTrain" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")

        # Create test tensors
        t1 = random_itensor(i1, i2)
        t2 = random_itensor(i2, i3)

        # Create MPS
        mps = ITensorMPS.MPS([t1, t2])

        # Convert to TensorTrain
        stt1 = TensorTrain(mps)

        # Test that conversion worked
        @test length(stt1.data) == 2
        @test stt1.data[1] == t1
        @test stt1.data[2] == t2
        @test stt1.llim == 0
        @test stt1.rlim == 3

        # Test conversion with explicit limits
        stt2 = TensorTrain(mps, 5, 10)
        @test length(stt2.data) == 2
        @test stt2.data[1] == t1
        @test stt2.data[2] == t2
        @test stt2.llim == 5
        @test stt2.rlim == 10
    end

    @testset "Round-trip conversion" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")

        # Create test tensors
        t1 = random_itensor(i1, i2)
        t2 = random_itensor(i2, i3)

        # Start with TensorTrain
        stt_original = TensorTrain([t1, t2])

        # Convert to MPS and back
        mps = ITensorMPS.MPS(stt_original)
        stt_converted = TensorTrain(mps)

        # Test that round-trip conversion preserves data
        @test length(stt_converted.data) == length(stt_original.data)
        @test stt_converted.data[1] == stt_original.data[1]
        @test stt_converted.data[2] == stt_original.data[2]

        # Note: llim and rlim are not preserved in this round-trip
        # since MPS doesn't store these values
    end

    @testset "Conversion from TensorTrain to MPO" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")
        j1 = Index(2, "j1")
        j2 = Index(3, "j2")
        j3 = Index(2, "j3")

        # Create test tensors (MPO tensors have both physical and auxiliary indices)
        t1 = random_itensor(i1, j1, i2, j2)
        t2 = random_itensor(i2, j2, i3, j3)

        # Create TensorTrain
        stt = TensorTrain([t1, t2])

        # Convert to MPO
        mpo = ITensorMPS.MPO(stt)

        # Test that conversion worked
        @test length(mpo) == 2
        @test mpo[1] == t1
        @test mpo[2] == t2
    end

    @testset "Conversion from MPO to TensorTrain" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")
        j1 = Index(2, "j1")
        j2 = Index(3, "j2")
        j3 = Index(2, "j3")

        # Create test tensors (MPO tensors have both physical and auxiliary indices)
        t1 = random_itensor(i1, j1, i2, j2)
        t2 = random_itensor(i2, j2, i3, j3)

        # Create MPO
        mpo = ITensorMPS.MPO([t1, t2])

        # Convert to TensorTrain
        stt1 = TensorTrain(mpo)

        # Test that conversion worked
        @test length(stt1.data) == 2
        @test stt1.data[1] == t1
        @test stt1.data[2] == t2
        @test stt1.llim == 0
        @test stt1.rlim == 3

        # Test conversion with explicit limits
        stt2 = TensorTrain(mpo, 5, 10)
        @test length(stt2.data) == 2
        @test stt2.data[1] == t1
        @test stt2.data[2] == t2
        @test stt2.llim == 5
        @test stt2.rlim == 10
    end

    @testset "Round-trip conversion with MPO" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")
        j1 = Index(2, "j1")
        j2 = Index(3, "j2")
        j3 = Index(2, "j3")

        # Create test tensors (MPO tensors have both physical and auxiliary indices)
        t1 = random_itensor(i1, j1, i2, j2)
        t2 = random_itensor(i2, j2, i3, j3)

        # Start with TensorTrain
        stt_original = TensorTrain([t1, t2])

        # Convert to MPO and back
        mpo = ITensorMPS.MPO(stt_original)
        stt_converted = TensorTrain(mpo)

        # Test that round-trip conversion preserves data
        @test length(stt_converted.data) == length(stt_original.data)
        @test stt_converted.data[1] == stt_original.data[1]
        @test stt_converted.data[2] == stt_original.data[2]

        # Note: llim and rlim are not preserved in this round-trip
        # since MPO doesn't store these values
    end

    @testset "TensorTrain addition using ITensors.Algorithm(\"directsum\")" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")

        # Create test tensors for first tensor train
        t1a = random_itensor(i1, i2)
        t2a = random_itensor(i2, i3)

        # Create test tensors for second tensor train
        t1b = random_itensor(i1, i2)
        t2b = random_itensor(i2, i3)

        # Create TensorTrain objects
        stt1 = TensorTrain([t1a, t2a])
        stt2 = TensorTrain([t1b, t2b])

        # Test addition
        stt_sum = stt1 + stt2

        # Check that the result has the correct structure
        @test length(stt_sum.data) == 2

        # Check that the result is a valid TensorTrain
        @test stt_sum isa TensorTrain

        # Strict numerical verification: dist(A, B)/norm(A) < 1e-13
        # Test that TensorTrain addition matches MPS directsum calculation
        # Convert to MPS and compute direct sum
        mps1 = ITensorMPS.MPS(stt1)
        mps2 = ITensorMPS.MPS(stt2)
        alg = Algorithm"directsum"()
        mps_direct_sum = +(alg, mps1, mps2)

        # Convert TensorTrain result to MPS
        stt_sum_as_mps = ITensorMPS.MPS(stt_sum)

        # Compare results with high precision
        if norm(mps_direct_sum) > 0
            @test ITensorMPS.dist(stt_sum_as_mps, mps_direct_sum) / norm(mps_direct_sum) < 1e-13
        end
    end

    @testset "Multiple TensorTrain addition" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")

        # Create three tensor trains
        t1a = random_itensor(i1, i2)
        t2a = random_itensor(i2, i3)
        stt1 = TensorTrain([t1a, t2a])

        t1b = random_itensor(i1, i2)
        t2b = random_itensor(i2, i3)
        stt2 = TensorTrain([t1b, t2b])

        t1c = random_itensor(i1, i2)
        t2c = random_itensor(i2, i3)
        stt3 = TensorTrain([t1c, t2c])

        # Test multiple addition
        stt_sum = stt1 + stt2 + stt3

        # Check that the result has the correct structure
        @test length(stt_sum.data) == 2

        # Strict numerical verification: dist(A, B)/norm(A) < 1e-13
        # Test that TensorTrain addition matches MPS directsum calculation
        # Convert to MPS and compute direct sum
        mps1 = ITensorMPS.MPS(stt1)
        mps2 = ITensorMPS.MPS(stt2)
        mps3 = ITensorMPS.MPS(stt3)
        alg = Algorithm"directsum"()
        mps_direct_sum = +(alg, mps1, mps2, mps3)

        # Convert TensorTrain result to MPS
        stt_sum_as_mps = ITensorMPS.MPS(stt_sum)

        # Compare results with high precision
        if norm(mps_direct_sum) > 0
            @test ITensorMPS.dist(stt_sum_as_mps, mps_direct_sum) / norm(mps_direct_sum) < 1e-13
        end
    end

    @testset "TensorTrain scalar multiplication (delegates to ITensorMPS)" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")

        # Create test tensors
        t1 = random_itensor(i1, i2)
        t2 = random_itensor(i2, i3)

        # Create TensorTrain
        stt = TensorTrain([t1, t2])

        # Test scalar multiplication (left)
        α = 2.5
        stt_scaled1 = α * stt

        # Test scalar multiplication (right)
        stt_scaled2 = stt * α

        # Note: Individual tensor scaling tests are removed because
        # scaling each core individually would result in α^2 scaling overall

        # Strict numerical verification: dist(A, B)/norm(A) < 1e-13
        # Test that both left and right multiplication give the same result
        if norm(stt_scaled1) > 0
            @test dist(stt_scaled1, stt_scaled2) / norm(stt_scaled1) < 1e-13
        end

        # Test that TensorTrain scalar multiplication matches MPS calculation
        # Convert to MPS and compute direct scaling
        mps_original = ITensorMPS.MPS(stt)
        mps_direct_scaled = α * mps_original

        # Convert TensorTrain result to MPS
        stt_scaled_as_mps = ITensorMPS.MPS(stt_scaled1)

        # Compare results with high precision
        if norm(mps_direct_scaled) > 0
            @test ITensorMPS.dist(stt_scaled_as_mps, mps_direct_scaled) / norm(mps_direct_scaled) < 1e-13
        end
    end

    @testset "TensorTrain subtraction using ITensors.Algorithm(\"directsum\")" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")

        # Create test tensors for first tensor train
        t1a = random_itensor(i1, i2)
        t2a = random_itensor(i2, i3)

        # Create test tensors for second tensor train
        t1b = random_itensor(i1, i2)
        t2b = random_itensor(i2, i3)

        # Create TensorTrain objects
        stt1 = TensorTrain([t1a, t2a])
        stt2 = TensorTrain([t1b, t2b])

        # Test subtraction
        stt_diff = stt1 - stt2

        # Check that the result has the correct structure
        @test length(stt_diff.data) == 2

        # Check that the result is a valid TensorTrain
        @test stt_diff isa TensorTrain

        # Strict numerical verification: dist(A, B)/norm(A) < 1e-13
        # Test that TensorTrain subtraction matches MPS directsum calculation
        # Convert to MPS and compute direct subtraction (as addition with negative scaling)
        mps1 = ITensorMPS.MPS(stt1)
        mps2 = ITensorMPS.MPS(stt2)
        alg = Algorithm"directsum"()
        mps_direct_diff = +(alg, mps1, -1 * mps2)

        # Convert TensorTrain result to MPS
        stt_diff_as_mps = ITensorMPS.MPS(stt_diff)

        # Compare results with high precision
        if norm(mps_direct_diff) > 0
            @test ITensorMPS.dist(stt_diff_as_mps, mps_direct_diff) / norm(mps_direct_diff) < 1e-13
        end
    end

    @testset "Multiple TensorTrain subtraction" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")

        # Create three tensor trains
        t1a = random_itensor(i1, i2)
        t2a = random_itensor(i2, i3)
        stt1 = TensorTrain([t1a, t2a])

        t1b = random_itensor(i1, i2)
        t2b = random_itensor(i2, i3)
        stt2 = TensorTrain([t1b, t2b])

        t1c = random_itensor(i1, i2)
        t2c = random_itensor(i2, i3)
        stt3 = TensorTrain([t1c, t2c])

        # Test multiple subtraction: stt1 - stt2 - stt3
        stt_diff = stt1 - stt2 - stt3

        # Check that the result has the correct structure
        @test length(stt_diff.data) == 2

        # Strict numerical verification: dist(A, B)/norm(A) < 1e-13
        # Test that TensorTrain subtraction matches MPS directsum calculation
        # Convert to MPS and compute direct subtraction (as addition with negative scaling)
        mps1 = ITensorMPS.MPS(stt1)
        mps2 = ITensorMPS.MPS(stt2)
        mps3 = ITensorMPS.MPS(stt3)
        alg = Algorithm"directsum"()
        mps_direct_diff = +(alg, mps1, -1 * mps2, -1 * mps3)

        # Convert TensorTrain result to MPS
        stt_diff_as_mps = ITensorMPS.MPS(stt_diff)

        # Compare results with high precision
        if norm(mps_direct_diff) > 0
            @test ITensorMPS.dist(stt_diff_as_mps, mps_direct_diff) / norm(mps_direct_diff) < 1e-13
        end
    end

    @testset "Error handling for tensor train subtraction" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")

        # Create tensor trains with different lengths
        t1 = random_itensor(i1, i2)
        t2 = random_itensor(i2, i3)
        stt1 = TensorTrain([t1, t2])  # length 2

        t3 = random_itensor(i1, i2)
        stt2 = TensorTrain([t3])      # length 1

        # Test that subtraction with different lengths throws an error
        @test_throws ArgumentError stt1 - stt2
    end

    @testset "TensorTrain norm, lognorm, and dist functions" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")

        # Create test tensors
        t1a = random_itensor(i1, i2)
        t2a = random_itensor(i2, i3)
        stt1 = TensorTrain([t1a, t2a])

        t1b = random_itensor(i1, i2)
        t2b = random_itensor(i2, i3)
        stt2 = TensorTrain([t1b, t2b])

        # Test norm function
        norm1 = norm(stt1)
        @test norm1 > 0  # Should be positive

        # Test lognorm function
        lognorm1 = ITensorMPS.lognorm(stt1)
        @test lognorm1 isa Real

        # Test dist function
        distance = dist(stt1, stt2)
        @test distance >= 0  # Should be non-negative

        # Test that dist(A, A) = 0
        @test dist(stt1, stt1) < 1e-13

        # Test that norm and lognorm are consistent
        @test abs(norm1 - exp(lognorm1)) / norm1 < 1e-13

        # Test that TensorTrain norm matches MPS calculation
        mps_direct = ITensorMPS.MPS(stt1)
        @test abs(norm(stt1) - ITensorMPS.norm(mps_direct)) / ITensorMPS.norm(mps_direct) < 1e-13
    end

    @testset "TensorTrain isapprox function" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")

        # Create test tensors
        t1a = random_itensor(i1, i2)
        t2a = random_itensor(i2, i3)
        stt1 = TensorTrain([t1a, t2a])

        t1b = random_itensor(i1, i2)
        t2b = random_itensor(i2, i3)
        stt2 = TensorTrain([t1b, t2b])

        # Test that identical tensor trains are approximately equal
        @test isapprox(stt1, stt1)
        @test isapprox(stt1, stt1; atol=1e-10)
        @test isapprox(stt1, stt1; rtol=1e-8)

        # Test that different tensor trains are not approximately equal (with strict tolerance)
        @test !isapprox(stt1, stt2; atol=1e-15, rtol=1e-15)

        # Test with explicit atol parameter
        @test isapprox(stt1, stt1; atol=1e-10)

        # Test with explicit rtol parameter
        @test isapprox(stt1, stt1; rtol=1e-8)

        # Test that isapprox works with Base.isapprox (≈ operator)
        @test stt1 ≈ stt1
        # Note: ≈ operator doesn't accept keyword arguments directly
        # Use isapprox function instead for custom tolerances
        @test !isapprox(stt1, stt2; atol=1e-15, rtol=1e-15)

        # Test that isapprox is consistent with dist
        # If distance is small, they should be approximately equal
        small_diff = stt1 + 1e-12 * stt2
        @test isapprox(stt1, small_diff; atol=1e-10)

        # Test that isapprox handles different lengths correctly
        t3 = random_itensor(i1, i2)
        stt3 = TensorTrain([t3])  # Different length
        @test !isapprox(stt1, stt3)  # Should return false for different lengths

        # Test promote_leaf_eltypes extension
        using LinearAlgebra
        eltype_result = LinearAlgebra.promote_leaf_eltypes(stt1)
        @test eltype_result isa Type
    end

    @testset "Error handling for tensor train addition" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")

        # Create tensor trains with different lengths
        t1 = random_itensor(i1, i2)
        t2 = random_itensor(i2, i3)
        stt1 = TensorTrain([t1, t2])  # length 2

        t3 = random_itensor(i1, i2)
        stt2 = TensorTrain([t3])      # length 1

        # Test that addition with different lengths throws an error
        @test_throws ArgumentError stt1 + stt2
    end

    @testset "TensorTrain addition with Algorithm (uses Algorithm(\"directsum\"))" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(3, "i2")
        i3 = Index(2, "i3")

        # Create test tensors
        t1a = random_itensor(i1, i2)
        t2a = random_itensor(i2, i3)
        stt1 = TensorTrain([t1a, t2a])

        t1b = random_itensor(i1, i2)
        t2b = random_itensor(i2, i3)
        stt2 = TensorTrain([t1b, t2b])

        # Test addition with default algorithm
        alg = Algorithm("default")
        stt_sum = +(alg, stt1, stt2)

        # Check that the result has the correct structure
        @test length(stt_sum.data) == 2
        @test stt_sum isa TensorTrain
    end

    @testset "TensorTrain siteinds function - 2-site MPS" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(2, "i2")
        l1 = Index(3, "Link,l1")

        # Create test tensors
        t1 = random_itensor(i1, l1)
        t2 = random_itensor(l1, i2)

        # Create TensorTrain
        stt = TensorTrain([t1, t2])

        # Test siteinds function
        sites = T4AITensorCompat.siteinds(stt)

        # Check that we get the right number of sites
        @test length(sites) == 2

        # Check first site (should contain i1, not l1)
        @test length(sites[1]) == 1
        @test sites[1][1] == i1

        # Check second site (should contain i2, not l1)
        @test length(sites[2]) == 1
        @test sites[2][1] == i2
    end

    @testset "TensorTrain siteinds function - 3-site MPS" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(2, "i2")
        i3 = Index(2, "i3")
        l1 = Index(3, "Link,l1")
        l2 = Index(3, "Link,l2")

        # Create test tensors
        t1 = random_itensor(i1, l1)
        t2 = random_itensor(l1, i2, l2)
        t3 = random_itensor(l2, i3)

        # Create TensorTrain
        stt = TensorTrain([t1, t2, t3])

        # Test siteinds function
        sites = T4AITensorCompat.siteinds(stt)

        # Check that we get the right number of sites
        @test length(sites) == 3

        # Check first site (should contain i1 only)
        @test length(sites[1]) == 1
        @test sites[1][1] == i1

        # Check middle site (should contain i2 only, not l1 or l2)
        @test length(sites[2]) == 1
        @test sites[2][1] == i2

        # Check last site (should contain i3 only)
        @test length(sites[3]) == 1
        @test sites[3][1] == i3
    end

    @testset "TensorTrain siteinds function - 2-site MPO" begin
        # Create test indices
        i1 = Index(2, "i1")
        i2 = Index(2, "i2")
        j1 = Index(2, "j1")
        j2 = Index(2, "j2")
        l1 = Index(3, "Link,l1")

        # Create MPO tensors
        t1 = random_itensor(i1, j1, l1)
        t2 = random_itensor(l1, i2, j2)

        # Create TensorTrain
        stt = TensorTrain([t1, t2])

        # Test siteinds function
        sites = T4AITensorCompat.siteinds(stt)

        # Check that we get the right number of sites
        @test length(sites) == 2

        # Check first site (should contain i1 and j1, not l1)
        @test length(sites[1]) == 2
        @test i1 in sites[1]
        @test j1 in sites[1]
        @test l1 ∉ sites[1]

        # Check second site (should contain i2 and j2, not l1)
        @test length(sites[2]) == 2
        @test i2 in sites[2]
        @test j2 in sites[2]
        @test l1 ∉ sites[2]
    end

    @testset "TensorTrain truncate and truncate! functions" begin
        # Create a simple 2-site MPS
        i1 = Index(2, "i1")
        i2 = Index(2, "i2")
        l1 = Index(3, "Link,l1")

        t1 = random_itensor(i1, l1)
        t2 = random_itensor(l1, i2)

        # Create TensorTrain
        a = TensorTrain([t1, t2], 1, 5)

        # Test that a + a doubles the bond dimension
        a_plus_a = a + a
        @test T4AITensorCompat.maxlinkdim(a_plus_a) == 6  # 3 * 2 = 6

        # Test truncate! (in-place)
        a_plus_a_truncated = deepcopy(a_plus_a)
        T4AITensorCompat.truncate!(a_plus_a_truncated; maxdim=10)

        @test relative_error(a_plus_a, a_plus_a_truncated) < 1e-13

        # Test truncate (creates new object)
        a_plus_a_truncated_copy = T4AITensorCompat.truncate(a_plus_a; maxdim=10)

        @test relative_error(a_plus_a, a_plus_a_truncated_copy) < 1e-13
    end

    @testset "TensorTrain siteinds function - random_mps" begin
        # Test siteinds with random_mps
        sites = [Index(2, "Site,n=$n") for n in 1:5]
        mps = random_mps(sites; linkdims=3)
        
        # Test siteinds function
        sites_extracted = siteinds(mps)
        
        # Check that we get the right number of sites
        @test length(sites_extracted) == 5
        
        # Check that each site contains the correct index
        for (i, site) in enumerate(sites_extracted)
            @test length(site) == 1
            @test site[1] == sites[i]
        end
    end

    @testset "TensorTrain siteinds function - random_mpo" begin
        # Test siteinds with random_mpo
        sites = [Index(2, "Site,n=$n") for n in 1:4]
        mpo = random_mpo(sites)
        
        # Test siteinds function
        sites_extracted = siteinds(mpo)
        
        # Check that we get the right number of sites
        @test length(sites_extracted) == 4
        
        # Check that each site contains two indices (upper and lower for MPO)
        for site in sites_extracted
            @test length(site) == 2
            # Check that indices have the same dimension
            @test dim(site[1]) == dim(site[2])
        end
    end

    @testset "TensorTrain siteinds function - MPO with multiple site indices" begin
        # Test siteinds with MPO that has multiple site indices per site
        # This simulates the case used in T4AQuantics
        sites1 = [Index(2, "x=$n") for n in 1:3]
        sites2 = [Index(2, "y=$n") for n in 1:3]
        sites = [[x, y] for (x, y) in zip(sites1, sites2)]
        
        # Create MPO manually
        links = [Index(2, "Link,l=$n") for n in 1:2]
        t1 = random_itensor(sites[1]..., links[1])
        t2 = random_itensor(links[1], sites[2]..., links[2])
        t3 = random_itensor(links[2], sites[3]...)
        mpo = TensorTrain([t1, t2, t3])
        
        # Test siteinds function
        sites_extracted = siteinds(mpo)
        
        # Check that we get the right number of sites
        @test length(sites_extracted) == 3
        
        # Check that each site contains the correct indices
        for (i, site) in enumerate(sites_extracted)
            @test length(site) == 2
            @test Set(site) == Set(sites[i])
        end
    end

    @testset "TensorTrain siteinds function - empty TensorTrain" begin
        # Test siteinds with empty TensorTrain
        empty_tt = TensorTrain(Vector{ITensor}())
        
        # Test siteinds function
        sites = siteinds(empty_tt)
        
        # Check that we get an empty vector
        @test length(sites) == 0
    end

    @testset "TensorTrain siteinds function - single site TensorTrain" begin
        # Test siteinds with single site TensorTrain
        i1 = Index(2, "i1")
        t1 = random_itensor(i1)
        stt = TensorTrain([t1])
        
        # Test siteinds function
        sites = siteinds(stt)
        
        # Check that we get one site
        @test length(sites) == 1
        @test length(sites[1]) == 1
        @test sites[1][1] == i1
    end
end
