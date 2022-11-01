@testset "deprecate.jl" begin
    blocks = [rand(3, 3), rand(3, 3)]
    @test_deprecated BlockDiagonal{Float64, Matrix{Float64}}(blocks)
    @test BlockDiagonal(blocks) == BlockDiagonal{Float64, Matrix{Float64}}(blocks)
end
