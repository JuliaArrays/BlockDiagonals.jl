function test_to_vec(x::T; check_inferred = true) where {T}
    check_inferred && @inferred to_vec(x)
    x_vec, back = to_vec(x)
    @test x_vec isa Vector
    @test all(s -> s isa Real, x_vec)
    check_inferred && @inferred back(x_vec)
    @test x == back(x_vec)
    return nothing
end

@testset "finitedifferences.jl" begin
    b = BlockDiagonal([rand(2, 2), rand(3, 4)])
    test_to_vec(b)
end
