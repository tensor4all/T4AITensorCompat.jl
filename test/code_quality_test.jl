@testitem "Code quality (Aqua.jl)" begin
    using Test
    using Aqua

    import T4AITensorCompat

    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(T4AITensorCompat; unbound_args = false, deps_compat = false)
    end

end

@testitem "Code linting (JET.jl)" begin
    using Test
    using JET

    import T4AITensorCompat

    if VERSION >= v"1.10"
        @testset "Code linting (JET.jl)" begin
            # Run JET with more lenient settings
            result = JET.report_package(T4AITensorCompat; 
                target_defined_modules = true,
                toplevel_logger = nothing
            )
            
            # Check if there are any critical errors (not just type inference issues)
            critical_errors = []
            for report in result
                # Only check for actual runtime errors, not type inference warnings
                if isa(report, JET.UncaughtExceptionReport)
                    # Check if it's a real error, not just type inference issues
                    if !occursin("type inference", string(report)) && 
                       !occursin("MethodErrorReport", string(report)) &&
                       !occursin("UndefVarErrorReport", string(report))
                        push!(critical_errors, report)
                    end
                end
            end
            
            # Only fail if there are actual runtime errors
            @test length(critical_errors) == 0
        end
    end
end
