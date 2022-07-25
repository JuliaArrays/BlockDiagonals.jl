@testset "chainrules.jl" begin
    @testset "BlockDiagonal" begin
        x = [randn(1, 2), randn(2, 2)]
        test_rrule(BlockDiagonal, x)
        test_rrule(BlockDiagonal, x; output_tangent=Tangent{BlockDiagonal}(;blocks=x))

        b = BlockDiagonal(x)
        m = Matrix(b)
        y, pb = rrule(BlockDiagonal, x)
        # want to test `output_tangent=m`, but can't do it directly because of
        # https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/199
        # so just compare the tangent to the tangent we get from the same BlockDiagonal
        @test pb(b) == pb(m)
    end

    @testset "Matrix" begin
        D = BlockDiagonal([randn(2, 2), randn(2, 2)])
        test_rrule(Matrix, D)
        test_rrule(Matrix, D, output_tangent=UpperTriangular(rand(4, 4)))
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

        @test ProjectTo(bd)(ChainRulesCore.ZeroTangent()) == ChainRulesCore.ZeroTangent()
    end
end
