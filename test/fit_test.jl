@testitem "fit.jl" begin
    include("util.jl")

    import T4AITensorCompat: TensorTrain, fit, dist
    import ITensors: ITensors, ITensor, Index, random_itensor
    import ITensorMPS
    import LinearAlgebra: norm
    ITensors.disable_warn_order()
    using Random

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
        @test error < 1e-10
    end
end

