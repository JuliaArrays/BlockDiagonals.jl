using BlockDiagonals
using BlockDiagonals: svd_blockwise, eigen_blockwise
using LinearAlgebra
using Random
using Test

# piracy to make SVD approximate comparisons easier
function Base.isapprox(a::SVD, b::SVD)
    return a.U ≈ b.U && a.V ≈ b.V && a.Vt ≈ b.Vt
end

@testset "linalg.jl" begin
    rng = MersenneTwister(123456)
    N1, N2, N3 = 3, 4, 5
    N = N1 + N2 + N3
    b1 = BlockDiagonal([rand(rng, N1, N1), rand(rng, N2, N2), rand(rng, N3, N3)])
    b2 = BlockDiagonal([rand(rng, N1, N1), rand(rng, N3, N3), rand(rng, N2, N2)])
    b3 = BlockDiagonal([rand(rng, N1, N1), rand(rng, N2, N2), rand(rng, N2, N2)])
    b_nonsq = BlockDiagonal([rand(rng, N1, N2), rand(rng, N2, N1)])
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
            mul!(c, b1, b1, 2.0, 3.0)
            mul!(d, Matrix(b1), Matrix(b1), 2.0, 3.0)
            @test c ≈ d
        end
    end

    @testset "Unary Linear Algebra" begin
        nonsquare = (adjoint, diag, pinv, svdvals, transpose)
        @testset "$f" for f in (adjoint, det, diag, eigvals, inv, pinv, svdvals, transpose, tr)
            @test f(b1) ≈ f(Matrix(b1))
            if f in nonsquare
                @test f(b_nonsq) ≈ f(Matrix(b_nonsq))
            end
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

        @testset "Symmetric/Hermitian" begin
            @test Symmetric(b1) == Symmetric(Matrix(b1))
            @test Symmetric(b1, :L) == Symmetric(Matrix(b1), :L)
            @test Hermitian(b1) == Hermitian(Matrix(b1))
            @test Hermitian(b1, :L) == Hermitian(Matrix(b1), :L)
        end

        @testset "Eigen Decomposition" begin

            @testset "eigen $name" for (name, B) in [("", b1), ("symmetric", Symmetric(b1)), ("hermitian", Hermitian(b1))]
                E = eigen(B)
                evals_bd, evecs_bd = E
                evals, evecs = eigen(Matrix(B))

                @test E isa Eigen
                @test Matrix(E) ≈ B 

                # There is no test like @test eigen(B) == eigen(Matrix(B))
                # 1. this fails in the complex case. Probably a convergence thing that leads to ever so slight differences
                # 2. pre version 1.2 this can't be expected to hold at all because the order of eigenvalues was random
                # so I sort the values/vectors (if needed) and then compare them via ≈

                @static if VERSION < v"1.2"
                    # pre-v1.2 we need to sort the eigenvalues consistently
                    # Since eigenvalues may be complex here, I use this function, which works for this test.
                    # This test is already somewhat fragile w. r. t. degenerate eigenvalues 
                    # and this just makes this a little worse.
                    perm_bd = sortperm(real.(evals_bd) + 100*imag.(evals_bd))
                    evals_bd = evals_bd[perm_bd]
                    evecs_bd = evecs_bd[:, perm_bd]

                    perm = sortperm(real.(evals) + 100*imag.(evals))
                    evals = evals[perm]
                    evecs = evecs[:, perm]
                end

                @test evals_bd ≈ evals
                # comparing vectors is more difficult due to a sign ambiguity
                # So we try adding and subtracting the vectors, keeping the smallest magnitude for each entry
                # and compare that with something small.
                # I performed some tests and the largest deviation I found was ~5e-14
                @test all(min.(abs.(evecs_bd - evecs), abs.(evecs_bd + evecs)) .< 1e-13)
            end

            @testset "eigen_blockwise $name" for (name, B) in [("", b1), ("symmetric", Symmetric(b1)), ("hermitian", Hermitian(b1))]
                vals, vecs = eigen_blockwise(B)

                # check types. Eltype varies so not checked here (should be ComplexF64, Float64, Float64)
                @test vecs isa BlockDiagonal
                @test vals isa Vector

                # Note: /(Matrix, BlockDiagonal) fails. I think because we can't do factorize(BlockDiagonal)
                # this is why I convert to Matrix prior to /
                @test B ≈ vecs * Diagonal(vals) / Matrix(vecs)

                # check by block
                cumulative_size = 0
                for (i, block) in enumerate(blocks(B))
                    block_vals = vals[cumulative_size+1:cumulative_size+size(block,1)]
                    cumulative_size += size(block, 1)
                    # adapt eltype to the same to block_vals.
                    # block_vals's eltype is chosen to be compatible across all eigenvalues, thus it might be different
                    block = convert.(eltype(block_vals), block)

                    # from here on the code parallel to the test code above
                    E = Eigen(block_vals, blocks(vecs)[i])
                    evals_bd, evecs_bd = E
                    evals, evecs = eigen(block)
                    
                    @test block ≈ Matrix(E)

                    @static if VERSION < v"1.2"
                        # sorting if needed
                        perm_bd = sortperm(real.(evals_bd) + 100*imag.(evals_bd))
                        evals_bd = evals_bd[perm_bd]
                        evecs_bd = evecs_bd[:, perm_bd]

                        perm = sortperm(real.(evals) + 100*imag.(evals))
                        evals = evals[perm]
                        evecs = evecs[:, perm]
                    end
    
                    @test evals_bd ≈ evals
                    @test all(min.(abs.(evecs_bd - evecs), abs.(evecs_bd + evecs)) .< 1e-13)
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

        BD = BlockDiagonal([X, X])
        C = cholesky(BD)
        @test C isa Cholesky{Float64, <:BlockDiagonal{Float64}}
        @test C.U ≈ cholesky(Matrix(BD)).U
        @test C.U ≈ BlockDiagonal([U, U])
        @test C.L ≈ BlockDiagonal([U', U'])
        @test C.UL ≈ C.U
        @test C.uplo === 'U'
        @test C.info == 0
        @test typeof(C) == Cholesky{Float64, BlockDiagonal{Float64, Matrix{Float64}}}
        @test PDMat(cholesky(BD)) == PDMat(cholesky(Matrix(BD)))

        M = BlockDiagonal(map(Matrix, blocks(C.L)))
        C = Cholesky(M, 'L', 0)
        @test C.U ≈ cholesky(Matrix(BD)).U
        @test C.U ≈ BlockDiagonal([U, U])
        @test C.L ≈ BlockDiagonal([U', U'])
        @test C.UL ≈ C.L
        @test C.uplo === 'L'
        @test C.info == 0
        @test typeof(C) == Cholesky{Float64, BlockDiagonal{Float64, Matrix{Float64}}}
    end  # Cholesky
    @testset "Singular Value Decomposition" begin
        X = [  4  12 -16
              12  37 -43
             -16 -43  98]
        BD = BlockDiagonal([X, X])

        @testset "full=$full" for full in (true, false)

            @testset "svd_blockwise" begin
                U, S, Vt = svd_blockwise(BD; full=full)
                F = SVD(U, S, Vt)
                @test BD ≈ F.U * Diagonal(F.S) * F.Vt

                # Matrices should be BlockDiagonal
                @test F isa SVD{Float64, Float64, <:BlockDiagonal{Float64}}
                @test F.U isa BlockDiagonal
                @test F.V isa BlockDiagonal
                @test F.Vt isa BlockDiagonal

                # Should have same values, but not sorted so as to keep BlockDiagonal structure
                F_ = svd(Matrix(BD), full=full)
                for fname in fieldnames(SVD)
                    @test sort(vec(getfield(F, fname))) ≈ sort(vec(getfield(F_, fname)))
                end
                # Singular values should be block-wise
                s = svd(X).S
                @test F.S == vcat(s, s)
            end

            @testset "svd" begin
                F = svd(BD; full=full)
                F_ = svd(Matrix(BD), full=full)

                @test F isa SVD
                @test BD ≈ F.U * Diagonal(F.S) * F.Vt

                @test F ≈ F_
                for fname in fieldnames(SVD)
                    @test getfield(F, fname) ≈ getfield(F_, fname)
                end

                # Singular values should be sorted in descending order
                @test F.S == sort(F.S, rev=true)
            end
        end
    end  # SVD
    @testset "Left division" begin
        N1 = 20
        N2 = 8
        N3 = 5
        A = BlockDiagonal([rand(rng, N1, N1), rand(rng, N2, N2)])
        B = BlockDiagonal([rand(rng, N1, N2), rand(rng, N3, N1)])
        x = rand(rng, N1 + N2)
        y = rand(rng, N1 + N3)
        @test A \ x ≈ inv(A) * x
        @test B \ y ≈ Matrix(B) \ y
    end
end
