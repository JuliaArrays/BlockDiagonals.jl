using BlockDiagonals
using ChainRulesCore
using ChainRulesTestUtils
using Documenter
using FiniteDifferences # For overloading to_vec
using Test
using LinearAlgebra

function FiniteDifferences.to_vec(X::BlockDiagonal)
    x, blocks_from_vec = to_vec(X.blocks)
    BlockDiagonal_from_vec(x_vec) = BlockDiagonal(blocks_from_vec(x_vec))
    return x, BlockDiagonal_from_vec
end

@testset "BlockDiagonals" begin
    # The doctests fail on x86, so only run them on 64-bit hardware
    Sys.WORD_SIZE == 64 && doctest(BlockDiagonals)
    include("blockdiagonal.jl")
    include("base_maths.jl")
    include("chainrules.jl")
    include("linalg.jl")
end  # tests
