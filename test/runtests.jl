using BlockDiagonals

using Documenter
using PDMats
using Test
using LinearAlgebra
using ChainRulesCore
using ChainRulesTestUtils

push!(ChainRulesTestUtils.TRANSFORMS_TO_ALT_TANGENTS, x -> @thunk(x))

@testset "BlockDiagonals" begin
    include("blockdiagonal.jl")
    include("base_maths.jl")
    include("linalg.jl")
    include("chainrules.jl")
end  # tests

#
#tests
