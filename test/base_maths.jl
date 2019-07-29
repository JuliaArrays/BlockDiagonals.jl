using BlockDiagonals
using LinearAlgebra: Diagonal, I
using Random
using Test

@testset "base_maths.jl" begin
    rng = MersenneTwister(123456)
    N1, N2, N3 = 3, 4, 5
    N = N1 + N2 + N3
    b1 = BlockDiagonal([rand(rng, N1, N1), rand(rng, N2, N2), rand(rng, N3, N3)])
    b2 = BlockDiagonal([rand(rng, N1, N1), rand(rng, N3, N3), rand(rng, N2, N2)])
    b3 = BlockDiagonal([rand(rng, N1, N1), rand(rng, N2, N2), rand(rng, N2, N2)])
    A = rand(rng, N, N + N1)
    B = rand(rng, N + N1, N + N2)
    A′, B′ = A', B'
    a = rand(rng, N)
    b = rand(rng, N + N1)

    @testset "Addition" begin
        @testset "BlockDiagonal + BlockDiagonal" begin
            @test b1 + b1 isa BlockDiagonal
            @test Matrix(b1 + b1) == Matrix(b1) + Matrix(b1)
            @test_throws DimensionMismatch b1 + b3
        end

        @testset "BlockDiagonal + Matrix" begin
            @test b1 + Matrix(b1) isa Matrix
            @test b1 + Matrix(b1) == b1 + b1
            @test_throws DimensionMismatch b1 + Matrix(b3)

            # Matrix + BlockDiagonal
            @test Matrix(b1) + b1 isa Matrix
            @test Matrix(b1) + b1 == b1 + b1
            @test_throws DimensionMismatch Matrix(b1) + b3

            # If the AbstractMatrix is diagonal, we should return a BlockDiagonal.
            # Test the StridedMatrix method.
            @test Matrix(Diagonal(b1)) + b1 isa BlockDiagonal  # StridedMatrix
            @test b1 + Matrix(Diagonal(b1)) isa BlockDiagonal

            # Test the AbstractMatrix method.
            @test Matrix(Diagonal(b1))' + b1 isa BlockDiagonal
            @test b1 + Matrix(Diagonal(b1))' isa BlockDiagonal
        end

        @testset "BlockDiagonal + Diagonal" begin
            D = Diagonal(randn(rng, N))
            D′ = Diagonal(randn(rng, N + N1))

            @test b1 + D isa BlockDiagonal
            @test b1 + D == Matrix(b1) + D
            @test_throws DimensionMismatch b1 + D′

            # Diagonal + BlockDiagonal
            @test D + b1 isa BlockDiagonal
            @test D + b1 == D + Matrix(b1)
            @test_throws DimensionMismatch D′ + b1
        end

        @testset "BlockDiagonal + UniformScaling" begin
            @test b1 + 5I isa BlockDiagonal
            @test b1 + 5I == Matrix(b1) + 5I

            # UniformScaling + BlockDiagonal
            @test 5I + b1 isa BlockDiagonal
            @test 5I + b1 == 5I + Matrix(b1)
        end
    end  # Addition

    @testset "Multiplication" begin
        @testset "BlockDiagonal * BlockDiagonal" begin
            @test b1 * b1 isa BlockDiagonal
            @test Matrix(b1 * b1) ≈ Matrix(b1) * Matrix(b1)
            @test_throws DimensionMismatch b3 * b1
        end

        @testset "BlockDiagonal * Number" begin
            @test b1 * 2 ≈ Matrix(b1) * 2 ≈ 2 * b1
            @test b1 / 2 ≈ Matrix(b1) / 2
            @test b1 * complex(2, 1) ≈ Matrix(b1) * complex(2, 1) ≈ complex(2, 1) * b1
            @test b1 / complex(2, 1) ≈ Matrix(b1) / complex(2, 1)
        end

        @testset "BlockDiagonal * Vector" begin
            @test b1 * a isa Vector
            @test b1 * a ≈ Matrix(b1) * a
            @test_throws DimensionMismatch b1 * b
        end

        @testset "BlockDiagonal * Matrix" begin
            @test b1 * A isa Matrix
            @test b1 * A ≈ Matrix(b1) * A
            @test_throws DimensionMismatch b1 * B

            # Matrix * BlockDiagonal
            @test A′ * b1 isa Matrix
            @test A′ * b1 ≈ A′ * Matrix(b1)
            @test_throws DimensionMismatch A * b1
        end

        @testset "BlockDiagonal * Diagonal" begin
            D = Diagonal(randn(rng, N))
            D′ = Diagonal(randn(rng, N + N1))

            @test b1 * D isa BlockDiagonal
            @test b1 * D ≈ Matrix(b1) * D
            @test_throws DimensionMismatch D′ * b1

            # Diagonal * BlockDiagonal
            @test D * b1 isa BlockDiagonal
            @test D * b1 ≈ D * Matrix(b1)
            @test_throws DimensionMismatch D′ * b1
        end
    end  # Multiplication
end
