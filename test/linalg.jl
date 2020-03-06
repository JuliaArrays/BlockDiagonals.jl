using BlockDiagonals
using BlockDiagonals: svd_blockwise
using LinearAlgebra
using Random
using Test

@testset "linalg.jl" begin
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

    @testset "mul!" begin
        c = similar(b1)
        d = similar(Matrix(b1))
        mul!(c, b1, b1)
        mul!(d, Matrix(b1), Matrix(b1))
        @test c ≈ d
        if VERSION ≥ v"1.3"
            mul!(c, b1, b1, 2., 3.)
            mul!(d, Matrix(b1), Matrix(b1), 2., 3.)
            @test c ≈ d
        end
    end

    @testset "Unary Linear Algebra" begin
        @testset "$f" for f in (adjoint, det, diag, eigvals, inv, pinv, svdvals, transpose, tr)
            @test f(b1) ≈ f(Matrix(b1))
        end

        @testset "permute=$p, scale=$s" for p in (true, false), s in (true, false)
            @testset "$g" for g in (eigmin, eigmax)
                # `b2` has real eigenvals, required for `eigmin`, `b1` has Complex eigenvals
                @test g(b2; permute=p, scale=s) ≈ g(Matrix(b2); permute=p, scale=s)
            end
            @testset "eigvals" begin
                result = eigvals(b2; permute=p, scale=s)
                expected = eigvals(Matrix(b2), permute=p, scale=s)
                @static if VERSION < v"1.2"
                    # the `eigvals` method we hit above did not sort real eigenvalues pre-v1.2
                    # but some `eigvals` methods did, so we always sort real eigenvalues.
                    @test sort!(result) ≈ sort!(expected)
                else
                    @test result ≈ expected
                end
            end
        end

        @testset "eigvals on LinearAlgebra types" begin
            # `eigvals` has different methods for different types, e.g. Hermitian
            b_herm = BlockDiagonal([Hermitian(rand(rng, 3, 3) + I) for _ in 1:3])
            @test eigvals(b_herm) ≈ eigvals(Matrix(b_herm))
            @test eigvals(b_herm, 1.0, 2.0) ≈ eigvals(Hermitian(Matrix(b_herm)), 1.0, 2.0)
        end

        # Requires a postive semi-definite matrix
        @testset "logdet" begin
            b̂1 = b1 * b1'
            @test logdet(b̂1) ≈ logdet(Matrix(b̂1))
        end
    end  # Unary

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
    end  # Cholesky

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
    end  # SVD
end
