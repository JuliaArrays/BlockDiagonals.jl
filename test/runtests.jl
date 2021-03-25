using BlockDiagonals
using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences # For overloading to_vec
using Test
using LinearAlgebra

@testset "BlockDiagonals" begin
    include("blockdiagonal.jl")
    include("base_maths.jl")
    include("chainrules.jl")
    include("finitedifferences.jl")
    include("linalg.jl")
end  # tests
