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

    @testset "ProjectTo" begin
        bd = BlockDiagonal([ones(2, 2), ones(3, 3)])
        project = ProjectTo(bd)
        @test project(ones(5, 5)) == bd
        @test project(reshape(ones(5, 5), 5, 5)) == bd
        @test project(Diagonal(ones(5))) isa BlockDiagonal
        @test project(Diagonal(ones(5))) == Diagonal(ones(5))
    end
end
