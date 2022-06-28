using BlockDiagonals
using ChainRulesCore
using ChainRulesTestUtils
using Documenter
using FiniteDifferences # For overloading to_vec
using PDMats
using Test
using LinearAlgebra

push!(ChainRulesTestUtils.TRANSFORMS_TO_ALT_TANGENTS, x -> @thunk(x))

@testset "BlockDiagonals" begin
    # The doctests fail version other than 64bit julia 1.6.x, due to printing differences
    Sys.WORD_SIZE == 64 && v"1.6" <= VERSION < v"1.7" && doctest(BlockDiagonals)
    include("blockdiagonal.jl")
    include("base_maths.jl")
    include("chainrules.jl")
    include("linalg.jl")
end  # tests
