using BlockDiagonals
using Documenter
using Test

@testset "BlockDiagonals" begin
    doctest(BlockDiagonals)
    include("blockdiagonal.jl")
    include("base_maths.jl")
    include("linalg.jl")
end  # tests
