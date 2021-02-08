@testset "chainrules.jl" begin
    @testset "BlockDiagonal" begin
        x = [randn(1, 2), randn(2, 2)]
        x̄ = [randn(1, 2), randn(2, 2)]
        ȳ = Composite{typeof(BlockDiagonal(x))}(blocks=[randn(1, 2), randn(2, 2)])
        rrule_test(BlockDiagonal, ȳ, (x, x̄))
    end

    @testset "Matrix" begin
        D = BlockDiagonal([randn(1, 2), randn(2, 2)])
        D̄ = Composite{typeof(D)}((blocks=[randn(1, 2), randn(2, 2)]), )
        Ȳ = randn(size(D))
        rrule_test(Matrix, Ȳ, (D, D̄))
    end
end
