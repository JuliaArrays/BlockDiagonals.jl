using BlockDiagonals
using Test

@testset "testblockdiagonals.jl" begin
    rng = MersenneTwister(123456)

    N1, N2, N3 = 4, 5, 6
    N = N1 + N2 + N3
    v1 = [0.5 0.6 0.7]
    v2 = [0.5 0.5 0.5 0.5]
    b1 = BlockDiagonal([rand(rng, N1, N2), rand(rng, N2, N1), rand(rng, N3, N2)])
    b2 = BlockDiagonal([rand(rng, 1, 3), rand(rng, 1, 1), rand(rng, 1, 4)])
    b3 = BlockDiagonal([rand(rng, N1, N3), rand(rng, N2, N3), rand(rng, N2, N1)])
    IndicesMatrix = BlockDiagonal([rand(2, 4), rand(1, 3), rand(2, 5)])
    
    @test blocksizes(b1) == [(N1, N2), (N2, N1), (N3, N2)]
    @test nblocks(b1) == 3
    @test BlockDiagonals.getblock(b1, 3) == blocks(b1)[3]
    @test BlockDiagonals.getblock(b1, 1, 2) == zeros(N1, N1)
    @test BlockDiagonals.setblock!(b2, v1, 1) == BlockDiagonals.getblock(b2, 1)
    @test BlockDiagonals.setblock!(b2, v1, 1, 1) == BlockDiagonals.getblock(b2, 1)
    @test_throws DimensionMismatch BlockDiagonals.setblock!(b2, v2, 1)
    @test_throws ArgumentError BlockDiagonals.setblock!(b2, v2, 3, 1)
    @test typeof(Base.Matrix(b2)) == Matrix{Float64}
          Base.setindex!(b1, 0.5, 1, 1)
    @test first(b1) == 0.5
    @test_throws ArgumentError Base.setindex!(b1, 0.5, 1, 7)
    @test Base.getindex(b1, 1, 1) == 0.5
    @test typeof(Base.convert(BlockDiagonal{Float32,Matrix{Float32}}, b1)) ==
          BlockDiagonal{Float32,Matrix{Float32}}
    @test BlockDiagonals._block_indices(IndicesMatrix, 5, 12) == (3, 2, 0)
    @test_throws BoundsError BlockDiagonals._block_indices(IndicesMatrix, 5, 13)
          b4 = Base.copy(b3)
          Base.setindex!(b3, 0.5, 1, 1)
    @test first(b3) != first(b4)
          Base.copy!(b3, b4)
    @test first(b3) == first(b4)
    @test_throws DimensionMismatch Base.copy!(b4, b2)
end
