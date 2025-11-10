@testitem "contraction.jl" begin
    include("util.jl")

    import T4AITensorCompat: TensorTrain, contract, dist
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
end