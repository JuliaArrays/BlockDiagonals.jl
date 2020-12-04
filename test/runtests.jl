using BlockDiagonals
using ChainRulesCore
using ChainRulesTestUtils
using Documenter
using FiniteDifferences # For overloading to_vec
using Test

@testset "BlockDiagonals" begin
    # The doctests fail on x86, so only run them on 64-bit hardware
    if Sys.WORD_SIZE == 64
        doctest(BlockDiagonals)
    end
    include("blockdiagonal.jl")
    include("base_maths.jl")
    include("linalg.jl")
end  # tests
