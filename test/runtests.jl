using Test

@testset "BlockDiagonals" begin
    include("blockdiagonal.jl")
    include("base_maths.jl")
    include("linalg.jl")
end  # tests
