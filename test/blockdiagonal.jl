using BlockDiagonals
using BlockDiagonals: isequal_blocksizes
using StableRNGs
using Test

@testset "blockdiagonal.jl" begin
    rng = StableRNG(123456)
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

        @testset "collect" begin
            B = BlockDiagonal([randn(20, 20) for _ in 1:24])
            collect(B)
            @test @allocated(collect(B)) < 2_000_000
        end

        @testset "BlockDiagonal does not copy" begin
            Bxy = BlockDiagonal([X, Y])
            X[1] = 1.1
            @test Bxy[1] == 1.1
            Bxy2 = BlockDiagonal(Bxy)
            Bxy[2] = 2.2
            @test X[2] == Bxy[2] == Bxy2[2] == 2.2
        end

        @testset "parent" begin
            @test parent(b1) isa Vector{<:AbstractMatrix}
            @test parent(BlockDiagonal([X, Y])) == [X, Y]
        end

        @testset "similar" begin
            @test similar(b1) isa BlockDiagonal
            @test size(similar(b1)) == size(b1)
            @test size.(blocks(similar(b1))) == size.(blocks(b1))

            @test similar(b1, Float32) isa BlockDiagonal{Float32}
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

    @testset "no blocks" begin
        B = BlockDiagonal(Matrix{Float64}[]);
        @test size(B) == (0, 0)
    end

    @testset "getblock" begin
        b = [rand(3, 3), rand(4, 4)]
        B = BlockDiagonal(b)
        @test BlockDiagonals.getblock(B, 1, 1) == b[1]
        @test BlockDiagonals.getblock(B, 2, 2) == b[2]
        @test BlockDiagonals.getblock(B, 1, 2) == zeros(blocksize(B, 1, 2))
        @test_throws BoundsError BlockDiagonals.getblock(B, 1, 3)
    end

    @testset "setblock!" begin
        b = [rand(3, 3), rand(4, 4)]
        r = [rand(3, 3), rand(4, 4)]

        B = BlockDiagonal(b)

        # Vector index
        BlockDiagonals.setblock!(B, r[1], 1)
        @test BlockDiagonals.getblock(B, 1) == r[1]
        @test_throws DimensionMismatch BlockDiagonals.setblock!(B, r[2], 1)
        @test_throws BoundsError BlockDiagonals.setblock!(B, r[2], 3)

        # Cartesian index
        BlockDiagonals.setblock!(B, r[2], 2, 2)
        @test BlockDiagonals.getblock(B, 2, 2) == r[2]
        @test_throws DimensionMismatch BlockDiagonals.setblock!(B, r[1], 2, 2)
        @test_throws ArgumentError BlockDiagonals.setblock!(B, r[1], 1, 2)
        @test_throws BoundsError BlockDiagonals.setblock!(B, r[2], 3, 3)
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

    @testset "getindex bug" begin
        b = BlockDiagonal(AbstractMatrix{Float64}[ones(2, 2)])
        @test b[1] == 1
    end

    @testset "convert(BlockDiagonal{F, T{F}}, block_diagonal)" for T in (
        Symmetric, UpperTriangular, Transpose, Hermitian
    )
        special = T(rand(2, 2))
        b = BlockDiagonal([special])

        convert_first = BlockDiagonal([convert(Matrix, special)])
        convert_last = convert(BlockDiagonal{Float64, Matrix{Float64}}, b)

        @test convert_first == convert_last
    end
end
