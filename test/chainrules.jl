@testset "chainrules.jl" begin
    @testset "BlockDiagonal" begin
        x = [randn(1, 2), randn(2, 2)]
        test_rrule(BlockDiagonal, x; check_inferred=false)
    end

    @testset "Matrix" begin
        D = BlockDiagonal([randn(1, 2), randn(2, 2)])
        test_rrule(Matrix, D)
        D_dense = collect(D) + reverse(collect(D), dims=2)
        @test BlockDiagonals._BlockDiagonal_pullback(D, ProjectTo(D)) ==
            BlockDiagonals._BlockDiagonal_pullback(D_dense, ProjectTo(D))

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
        @test project(adjoint(ones(5, 5))) == bd
        @test project(Diagonal(ones(5))) isa BlockDiagonal
        @test project(Diagonal(ones(5))) == Diagonal(ones(5))
    end
end
