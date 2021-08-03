@testset "chainrules.jl" begin
    @testset "BlockDiagonal" begin
        x = [randn(1, 2), randn(2, 2)]
        test_rrule(BlockDiagonal, x)
        test_rrule(BlockDiagonal, x; output_tangent=Tangent{BlockDiagonal}(;blocks=x))
    end

    @testset "Matrix" begin
        D = BlockDiagonal([randn(1, 2), randn(2, 2)])
        test_rrule(Matrix, D)
    end

    @testset "BlockDiagonal * Vector" begin
        D = BlockDiagonal([rand(2, 3), rand(3, 3)])
        v = rand(6)
        test_rrule(*, D, v)
    end
end
