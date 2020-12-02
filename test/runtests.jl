using BlockDiagonals
using ChainRulesCore
using ChainRulesTestUtils
using Documenter
using FiniteDifferences # For overloading to_vec
using Test

@testset "BlockDiagonals" begin
    doctest(BlockDiagonals)
    include("blockdiagonal.jl")
    include("base_maths.jl")
    include("linalg.jl")
end  # tests
