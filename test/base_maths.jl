using BlockDiagonals
using LinearAlgebra: Diagonal, I
using Random
using Test

@testset "base_maths.jl" begin
    rng = MersenneTwister(123456)
    blocks1 = [rand(rng, 3, 3), rand(rng, 4, 4)]
    blocks2 = [rand(rng, 3, 3), rand(rng, 5, 5)]

    @testset for V in (Tuple, Vector)
    b1 = BlockDiagonal(V(blocks1))
    b2 = BlockDiagonal(V(blocks2))
    N = size(b1, 1)
    A = rand(rng, N, N + 1)

    @testset "Addition" begin
        @testset "BlockDiagonal + BlockDiagonal" begin
            @test b1 + b1 isa BlockDiagonal
            @test Matrix(b1 + b1) == Matrix(b1) + Matrix(b1)
            @test_throws DimensionMismatch b1 + b2
        end

        @testset "BlockDiagonal + Matrix" begin
            @test b1 + Matrix(b1) isa Matrix
            @test b1 + Matrix(b1) == b1 + b1
            @test_throws DimensionMismatch b1 + Matrix(b2)

            # Matrix + BlockDiagonal
            @test Matrix(b1) + b1 isa Matrix
            @test Matrix(b1) + b1 == b1 + b1
            @test_throws DimensionMismatch Matrix(b1) + b2

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
            D′ = Diagonal(randn(rng, N + 1))

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
            @test_throws DimensionMismatch b2 * b1
        end

        @testset "BlockDiagonal * Number" begin
            @test b1 * 2 ≈ Matrix(b1) * 2 ≈ 2 * b1
            @test b1 / 2 ≈ Matrix(b1) / 2
            @test b1 * complex(2, 1) ≈ Matrix(b1) * complex(2, 1) ≈ complex(2, 1) * b1
            @test b1 / complex(2, 1) ≈ Matrix(b1) / complex(2, 1)
        end

        @testset "BlockDiagonal * Vector" begin
            a = rand(rng, N)
            @test b1 * a isa Vector
            @test b1 * a ≈ Matrix(b1) * a
            b = rand(rng, N + 1)
            @test_throws DimensionMismatch b1 * b
        end
        @testset "Vector^T * BlockDiagonal" begin
            a = rand(rng, N)
            @test a' * b1 isa Adjoint{<:Number, <:Vector}
            @test transpose(a) * b1 isa Transpose{<:Number, <:Vector}
            @test a' * b1 ≈ a' * Matrix(b1)
            @test transpose(a) * b1 ≈ transpose(a) * Matrix(b1)
        end

        @testset "BlockDiagonal * Matrix" begin
            @test b1 * A isa Matrix
            @test b1 * A ≈ Matrix(b1) * A

            B = rand(rng, N + 1, N)
            @test_throws DimensionMismatch b1 * B

            # Matrix * BlockDiagonal
            @test A' * b1 isa Matrix
            @test A' * b1 ≈ A' * Matrix(b1)
            @test_throws DimensionMismatch A * b1

            # degenerate cases
            m = rand(0, 0)
            @test m * BlockDiagonal([m]) == m * m == m
            m = rand(5, 0)
            @test m' * BlockDiagonal([m]) == m' * m == rand(0, 0)
            @test m * BlockDiagonal([m']) == m * m' == zeros(5, 5)
        end

        @testset "BlockDiagonal * Diagonal" begin
            D = Diagonal(randn(rng, N))
            D′ = Diagonal(randn(rng, N + 1))

            @test b1 * D isa BlockDiagonal
            @test b1 * D ≈ Matrix(b1) * D
            @test_throws DimensionMismatch D′ * b1

            # Diagonal * BlockDiagonal
            @test D * b1 isa BlockDiagonal
            @test D * b1 ≈ D * Matrix(b1)
            @test_throws DimensionMismatch D′ * b1
        end

        @testset "Non-Square BlockDiagonal * Non-Square BlockDiagonal" begin
            b4 = BlockDiagonal(V([ones(2, 4), 2 * ones(3, 2)]))
            b5 = BlockDiagonal(V([3 * ones(2, 2), 2 * ones(4, 1)]))

            @test b4 * b5 isa Array
            @test b4 * b5 == [6 * ones(2, 2) 4 * ones(2, 1); zeros(3, 2) 8 * ones(3, 1)]
            # Dimension check
            @test sum(size.(b4.blocks, 1)) == size(b4 * b5, 1)
            @test sum(size.(b5.blocks, 2)) == size(b4 * b5, 2)
        end
    end  # Multiplication
end  # V
end
