using Base.Test

@testset "BlockDiagonal" begin
    b1 = BlockDiagonal([rand(3, 3), rand(4, 4), rand(5, 5)])
    b2 = BlockDiagonal([rand(3, 2), rand(4, 4), rand(5, 3)])
    A = rand(size(b1))
    B = rand(size(b2))
    C = B'

    @test b1 ≈ b1
    @test b1 ≈ Matrix(b1)
    @test Matrix(b1) ≈ b1
    @test b1 * b1 ≈ Matrix(b1) * Matrix(b1)
    @test b1' * b1 ≈ Matrix(b1)' * Matrix(b1)
    @test b1 * b1' ≈ Matrix(b1) * Matrix(b1)'
    @test b1 * A ≈ Matrix(b1) * A
    @test b1 * A' ≈ Matrix(b1) * A'
    @test b1' * A ≈ Matrix(b1)' * A
    @test A * b1 ≈ A * Matrix(b1)
    @test A' * b1 ≈ A' * Matrix(b1)
    @test A * b1' ≈ A * Matrix(b1)'
    @test isa(b1 + eye(b1), BlockDiagonal)
    @test diag(b1 + eye(b1)) ≈ diag(b1) + ones(size(b1, 1)) atol = _ATOL_

    @test_throws DimensionMismatch b2 * b1
    @test_throws DimensionMismatch b2 * A
    @test_throws DimensionMismatch B * b1

    @test trace(b1) ≈ trace(Matrix(b1))

    b1 = BlockDiagonal(Hermitian.(
        [(rand(3, 3) + 15eye(3)), (rand(4, 4) +15eye(4)), (rand(5, 5) + 15eye(5))]
    ))
    Ub = chol(b1)
    Um = chol(Matrix(b1))
    @test Ub' * Ub ≈ Matrix(b1) ≈ b1 ≈ Um' * Um
    @test det(b1) ≈ det(Matrix(b1))
    @test eigvals(b1) ≈ eigvals(Matrix(b1))

    eqs = []
    for i in 1:size(b1, 1)
        for j in 1:size(b1, 2)
            push!(eqs, b1[i, j] ≈ Matrix(b1)[i, j])
        end
    end
    @test all(eqs)

    @test 5 * b1 ≈ 5 * Matrix(b1)
    @test b1 * 5 ≈ 5 * Matrix(b1)
    @test b1 / 5 ≈ Matrix(b1) / 5
    @test b1 + b1 ≈ 2 * b1
    @test isa(b1 + b1, BlockDiagonal)
end
