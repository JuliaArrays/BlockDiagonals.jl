using BlockDiagonals
using BlockDiagonals: isequal_blocksizes
using Random
using Test

@testset "blockdiagonal.jl" begin
    rng = MersenneTwister(123456)
    N1, N2, N3 = 3, 4, 5
    N = N1 + N2 + N3
    blocks1 = [rand(rng, N1, N1), rand(rng, N2, N2), rand(rng, N3, N3)]
    blocks2 = [rand(rng, N1, N1), rand(rng, N3, N3), rand(rng, N2, N2)]
    blocks3 = [rand(rng, N1, N1), rand(rng, N2, N2), rand(rng, N2, N2)]

    @testset "$T" for (T, (b1, b2, b3)) in (
        Tuple => (BlockDiagonal(Tuple(blocks1)), BlockDiagonal(Tuple(blocks2)), BlockDiagonal(Tuple(blocks3))),
        Vector => (BlockDiagonal(blocks1), BlockDiagonal(blocks2), BlockDiagonal(blocks3)),
    )
    A = rand(rng, N, N + N1)
    B = rand(rng, N + N1, N + N2)
    A′, B′ = A', B'
    a = rand(rng, N)
    b = rand(rng, N + N1)

    @testset "AbstractArray" begin
        X = rand(2, 2); Y = rand(3, 3)

        @test size(b1) == (N, N)
        @test size(b1, 1) == N && size(b1, 2) == N

        eqs = []
        for i in 1:size(b1, 1)
            for j in 1:size(b1, 2)
                push!(eqs, b1[i, j] ≈ Matrix(b1)[i, j])
            end
        end
        @test all(eqs)

        @inferred Matrix(b1)

        @testset "BlockDiagonal does not copy" begin
            Bxy = BlockDiagonal([X, Y])
            X[1] = 1.1
            @test Bxy[1] == 1.1
            Bxy2 = BlockDiagonal(Bxy)
            Bxy[2] = 2.2
            @test X[2] == Bxy[2] == Bxy2[2] == 2.2
        end

        @testset "parent" begin
            @test parent(b1) isa Union{Tuple,AbstractVector}
            @test eltype(parent(b1)) <: AbstractMatrix
            @test parent(BlockDiagonal([X, Y])) == [X, Y]
            @test parent(BlockDiagonal((X, Y))) == (X, Y)
        end

        @testset "similar" begin
            @test similar(b1) isa BlockDiagonal
            @test size(similar(b1)) == size(b1)
            @test size.(blocks(similar(b1))) == size.(blocks(b1))
        end

        @testset "setindex!" begin
            X = BlockDiagonal([rand(Float32, 5, 5), rand(Float32, 3, 3)])
            X[10] = Int(10)
            @test X[10] === Float32(10.0)
            X[3, 3] = Int(9)
            @test X[3, 3] === Float32(9.0)
            # Should not allow setting value outside on-diagonal blocks to non-zero
            @test_throws ArgumentError X[1, 7] = 1
        end
    end  # AbstractArray

    @testset "isequal_blocksizes" begin
        @test isequal_blocksizes(b1, b1) == true
        @test isequal_blocksizes(b1, similar(b1)) == true
        @test isequal_blocksizes(b1, b2) == false
    end

    @testset "blocks size" begin
        B = BlockDiagonal([rand(3, 3), rand(4, 4)])
        @test nblocks(B) == 2
        @test blocksizes(B) == [(3, 3), (4, 4)]
        @test blocksize(B, 2) == blocksizes(B)[2] == blocksize(B, 2, 2)
    end

    @testset "Equality" begin
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
    end  # Equality

    @testset "Non-Square Matrix" begin
	A1 = ones(2, 4)
	A2 = 2 * ones(3, 2)
	B1 = BlockDiagonal([A1, A2])
        B2 = [A1 zeros(2, 2); zeros(3, 4) A2]

	@test B1 == B2
	# Dimension check
	@test sum(size.(B1.blocks, 1)) == size(B2, 1)
	@test sum(size.(B1.blocks, 2)) == size(B2, 2)
    end  # Non-Square Matrix

    @testset "copy" begin
        bc = similar(b1)

        copy!(bc, b1)
        @test bc == b1

        c = copy(b1)
        @test c == b1

        @test_throws DimensionMismatch copy!(b2, b1)
    end
end
end
