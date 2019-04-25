using BlockDiagonals
using BlockDiagonals: blocks, isequal_blocksizes, svd_blockwise
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
    A′, B′ = A', B'
    a = rand(rng, N)
    b = rand(rng, N + N1)

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
    end

    @testset "isequal_blocksizes" begin
        @test isequal_blocksizes(b1, b1) == true
        @test isequal_blocksizes(b1, similar(b1)) == true
        @test isequal_blocksizes(b1, b2) == false
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
    end

    @testset "Unary Linear Algebra" begin

        @testset "$f" for f in (adjoint, det, diag, eigvals, svdvals, transpose, tr)
            @test f(b1) ≈ f(Matrix(b1))
        end

        # Requires a postive semi-definite matrix
        @testset "logdet" begin
            b̂1 = b1 * b1'
            @test logdet(b̂1) ≈ logdet(Matrix(b̂1))
        end
    end

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
    end

    @testset "Multiplication" begin

        @testset "BlockDiagonal * BlockDiagonal" begin
            @test b1 * b1 isa BlockDiagonal
            @test Matrix(b1 * b1) ≈ Matrix(b1) * Matrix(b1)
            @test_throws DimensionMismatch b3 * b1
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
    end

    @testset "Cholesky decomposition" begin
        X = [  4  12 -16
              12  37 -43
             -16 -43  98]
        U = [ 2.0 6.0 -8.0
              0.0 1.0  5.0
              0.0 0.0  3.0]

        B = BlockDiagonal([X, X])
        C = cholesky(B)
        @test C isa Cholesky{Float64, <:BlockDiagonal{Float64}}
        @test C.U ≈ cholesky(Matrix(B)).U
        @test C.U ≈ BlockDiagonal([U, U])
        @test C.L ≈ BlockDiagonal([U', U'])
        @test C.UL ≈ C.U
        @test C.uplo === 'U'
        @test C.info == 0

        M = BlockDiagonal(map(Matrix, blocks(C.L)))
        C = Cholesky(M, 'L', 0)
        @test C.U ≈ cholesky(Matrix(B)).U
        @test C.U ≈ BlockDiagonal([U, U])
        @test C.L ≈ BlockDiagonal([U', U'])
        @test C.UL ≈ C.L
        @test C.uplo === 'L'
        @test C.info == 0
    end

    @testset "Singular Value Decomposition" begin
        X = [  4  12 -16
              12  37 -43
             -16 -43  98]
        B = BlockDiagonal([X, X])

        @testset "full=$full" for full in (true, false)

            @testset "svd_blockwise" begin
                U, S, Vt = svd_blockwise(B; full=full)
                F = SVD(U, S, Vt)
                @test B ≈ F.U * Diagonal(F.S) * F.Vt

                # Matrices should be BlockDiagonal
                @test F isa SVD{Float64, Float64, <:BlockDiagonal{Float64}}
                @test F.U isa BlockDiagonal
                @test F.V isa BlockDiagonal
                @test F.Vt isa BlockDiagonal

                # Should have same values, but not sorted so as to keep BlockDiagonal structure
                F_ = svd(Matrix(B), full=full)
                for fname in fieldnames(SVD)
                    @test sort(vec(getfield(F, fname))) ≈ sort(vec(getfield(F_, fname)))
                end
                # Singular values should be block-wise
                s = svd(X).S
                @test F.S == vcat(s, s)
            end

            @testset "svd" begin
                F = svd(B; full=full)
                F_ = svd(Matrix(B), full=full)

                @test F isa SVD
                @test B ≈ F.U * Diagonal(F.S) * F.Vt

                @test F == F_
                for fname in fieldnames(SVD)
                    @test getfield(F, fname) ≈ getfield(F_, fname)
                end

                # Singular values should be sorted in descending order
                @test F.S == sort(F.S, rev=true)
            end
        end
    end
end  # tests
