using BlockDiagonals
using BlockDiagonals: blocks, isequal_blocksizes
using LinearAlgebra
using Random
using Test

@testset "BlockDiagonals" begin

    rng = MersenneTwister(123456)
    N1, N2, N3 = 3, 4, 5
    N = N1 + N2 + N3
    b1 = BlockDiagonal([rand(rng, N1, N1), rand(rng, N2, N2), rand(rng, N3, N3)])
    b2 = BlockDiagonal([rand(rng, N1, N1), rand(rng, N3, N3), rand(rng, N2, N2)])
    b3 = BlockDiagonal([rand(rng, N1, N1), rand(rng, N2, N2), rand(rng, N2, N2)])
    A = rand(rng, N, N + N1)
    B = rand(rng, N + N1, N + N2)
    a, b = rand(rng, N), rand(rng, N + N1)
    A′, B′ = A', B'

    @testset "AbstractArray" begin

        @test size(b1) == (N, N)
        @test size(b1, 1) == N && size(b1, 2) == N

        eqs = []
        for i in 1:size(b1, 1)
            for j in 1:size(b1, 2)
                push!(eqs, b1[i, j] ≈ Matrix(b1)[i, j])
            end
        end
        @test all(eqs)
    end

    @testset "isequal_blocksizes" begin
        @test isequal_blocksizes(b1, b1) == true
        @test isequal_blocksizes(b1, similar(b1)) == true
        @test isequal_blocksizes(b1, b2) == false
    end

    @testset "equality" begin

        # Equality
        @test b1 == b1
        @test b1 == Matrix(b1)
        @test Matrix(b1) == b1

        # Inequality
        @test b1 != b2
        @test b1 != Matrix(b2)
        @test Matrix(b1) != b2

        # Approximate equality
        @test b1 ≈ b1
        @test Matrix(b1) ≈ b1
        @test b1 ≈ Matrix(b1)
    end

    @testset "unary" begin

        for f in [adjoint, det, diag, eigvals, transpose, tr]
            @test f(b1) ≈ f(Matrix(b1))
        end

        # Construct a PSD matrix to check logdet
        b1′ = b1'
        b̂1 = b1 * b1′
        @test logdet(b̂1) ≈ logdet(Matrix(b̂1))
        @test cholesky(b̂1).U ≈ cholesky(Matrix(b̂1)).U

        @test similar(b1) isa BlockDiagonal
        @test size(similar(b1)) == size(b1)
        @test size.(blocks(similar(b1))) == size.(blocks(b1))
    end

    @testset "addition" begin

        # BlockDiagonal + BlockDiagonal
        @test b1 + b1 isa BlockDiagonal
        @test Matrix(b1 + b1) == Matrix(b1) + Matrix(b1)
        @test_throws DimensionMismatch b1 + b3

        # BlockDiagonal + Matrix
        @test b1 + Matrix(b1) isa Matrix
        @test b1 + Matrix(b1) == b1 + b1
        @test_throws DimensionMismatch b1 + Matrix(b3)

        # Matrix + BlockDiagonal
        @test Matrix(b1) + b1 isa Matrix
        @test Matrix(b1) + b1 == b1 + b1
        @test_throws DimensionMismatch Matrix(b1) + b3

        # BlockDiagonal + Diagonal
        D, D′ = Diagonal(randn(rng, N)), Diagonal(randn(rng, N + N1))
        @test b1 + D isa BlockDiagonal
        @test b1 + D == Matrix(b1) + D
        @test_throws DimensionMismatch b1 + D′

        # Diagonal + BlockDiagonal
        D, D′ = Diagonal(randn(rng, N)), Diagonal(randn(rng, N + N1))
        @test D + b1 isa BlockDiagonal
        @test D + b1 == D + Matrix(b1)
        @test_throws DimensionMismatch D′ + b1

        # BlockDiagonal + UniformScaling
        @test b1 + 5I isa BlockDiagonal
        @test b1 + 5I == Matrix(b1) + 5I

        # UniformScaling + BlockDiagonal
        @test 5I + b1 isa BlockDiagonal
        @test 5I + b1 == 5I + Matrix(b1)
    end

    @testset "multiplication" begin

        # BlockDiagonal * BlockDiagonal
        @test b1 * b1 isa BlockDiagonal
        @test Matrix(b1 * b1) ≈ Matrix(b1) * Matrix(b1)
        @test_throws DimensionMismatch b3 * b1

        # BlockDiagonal * Vector.
        @test b1 * a isa Vector
        @test b1 * a ≈ Matrix(b1) * a
        @test_throws DimensionMismatch b1 * b

        # BlockDiagonal * Matrix.
        C = randn(size(b1, 1), size(A, 2))
        @test b1 * A isa Matrix
        @test b1 * A ≈ Matrix(b1) * A
        @test_throws DimensionMismatch b1 * B

        # Matrix * BlockDiagonal.
        @test A′ * b1 isa Matrix
        @test A′ * b1 ≈ A′ * Matrix(b1)
        @test_throws DimensionMismatch A * b1
    end

end
