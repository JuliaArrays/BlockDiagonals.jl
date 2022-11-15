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

    b64 = BlockDiagonal([rand(rng, 2, 2), rand(rng, 2, 2)])
    b32 = BlockDiagonal([rand(rng, Float32, 2, 2), rand(rng, Float32, 2, 2)])

    bi = BlockDiagonal([zeros(Int, N1, N1), zeros(Int, N2, N2), zeros(Int, N3, N3)])

    bns = BlockDiagonal([rand(rng, N1, N2), rand(rng, N2, N3), rand(rng, N3, N1)])

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

            # Non-square blocks
            @test D + bns isa Matrix
            @test D + bns == D + Matrix(bns)
            @test_throws DimensionMismatch D′ + bns

            @test bns + D isa Matrix
            @test bns + D == D + Matrix(bns)
            @test_throws DimensionMismatch bns + D′
        end

        @testset "BlockDiagonal + UniformScaling" begin
            @test b1 + 5I isa BlockDiagonal
            @test b1 + 5I == Matrix(b1) + 5I

            # UniformScaling + BlockDiagonal
            @test 5I + b1 isa BlockDiagonal
            @test 5I + b1 == 5I + Matrix(b1)
        end
    end  # Addition

    @testset "Subtraction" begin
        @test -b1 isa BlockDiagonal
        @test b1 - b1 isa BlockDiagonal

        @test -b1 == -Matrix(b1)
        @test b1 - b1 == Matrix(b1) - Matrix(b1)
        @test Matrix(b1) - b2 == Matrix(b1) - Matrix(b2)
        @test b1 - Matrix(b2) == Matrix(b1) - Matrix(b2)
    end

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

            # promote_type
            @test b32 * rand(4) isa Vector{Float64}
            @test b64 * rand(Float32, 4) isa Vector{Float64}
        end
        @testset "Vector^T * BlockDiagonal" begin
            @test a' * b1 isa Adjoint{<:Number, <:Vector}
            @test transpose(a) * b1 isa Transpose{<:Number, <:Vector}
            @test a' * b1 ≈ a' * Matrix(b1)
            @test transpose(a) * b1 ≈ transpose(a) * Matrix(b1)
            # Method ambiguity with different eltypes https://github.com/invenia/BlockDiagonals.jl/issues/91
            @test a' * bi isa Adjoint{<:Number, <:Vector}
        end

        @testset "BlockDiagonal * Matrix" begin
            @test b1 * A isa Matrix
            @test b1 * A ≈ Matrix(b1) * A
            @test_throws DimensionMismatch b1 * B

            # Matrix * BlockDiagonal
            @test A′ * b1 isa Matrix
            @test A′ * b1 ≈ A′ * Matrix(b1)
            @test_throws DimensionMismatch A * b1

            # degenerate cases
            m = rand(0, 0)
            @test m * BlockDiagonal([m]) == m * m == m
            m = rand(5, 0)
            @test m' * BlockDiagonal([m]) == m' * m == rand(0, 0)
            @test m * BlockDiagonal([m']) == m * m' == zeros(5, 5)

            # promote_type
            @test b32 * rand(4, 4) isa Matrix{Float64}
            @test rand(4, 4) * b32 isa Matrix{Float64}
            @test b64 * rand(Float32, 4, 4) isa Matrix{Float64}
            @test rand(Float32, 4, 4) * b64 isa Matrix{Float64}
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

            # promote_type
            @test b32 * Diagonal(rand(4)) isa BlockDiagonal{Float64}
            @test Diagonal(rand(4)) * b32 isa BlockDiagonal{Float64}
        end

        @testset "Non-Square BlockDiagonal * Non-Square BlockDiagonal" begin
    	    b4 = BlockDiagonal([ones(2, 4), 2 * ones(3, 2)])
            b5 = BlockDiagonal([3 * ones(2, 2), 2 * ones(4, 1)])

            @test b4 * b5 isa Array
            @test b4 * b5 == [6 * ones(2, 2) 4 * ones(2, 1); zeros(3, 2) 8 * ones(3, 1)]
            # Dimension check
            @test sum(size.(b4.blocks, 1)) == size(b4 * b5, 1)
            @test sum(size.(b5.blocks, 2)) == size(b4 * b5, 2)
        end
    end  # Multiplication
end
