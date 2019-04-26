# Linear algebra functions, in particular extending the `LinearAlgebra` stdlib

for f in (:adjoint, :eigvecs, :inv, :pinv, :transpose)
    @eval LinearAlgebra.$f(B::BlockDiagonal) = BlockDiagonal(map($f, blocks(B)))
end

LinearAlgebra.diag(B::BlockDiagonal) = mapreduce(diag, vcat, blocks(B))
LinearAlgebra.det(B::BlockDiagonal) = prod(det, blocks(B))
LinearAlgebra.logdet(B::BlockDiagonal) = sum(logdet, blocks(B))
LinearAlgebra.tr(B::BlockDiagonal) = sum(tr, blocks(B))

# Real matrices can have Complex eigenvalues; `eigvals` is not type stable.
function LinearAlgebra.eigvals(B::BlockDiagonal; kwargs...)
    # Currently no convention for sorting eigenvalues.
    # This may change in later a Julia version https://github.com/JuliaLang/julia/pull/21598
    return mapreduce(b -> eigvals(b; kwargs...), vcat, blocks(B))
end

# This is copy of the definition for LinearAlgebra.
# Should not be needed once we fix https://github.com/JuliaLang/julia/issues/31843
function LinearAlgebra.eigmax(B::BlockDiagonal; kwargs...)
    v = eigvals(B; kwargs...)
    if eltype(v) <: Complex
        throw(DomainError(A, "`A` cannot have complex eigenvalues."))
    end
    return maximum(v)
end

svdvals_blockwise(B::BlockDiagonal) = mapreduce(svdvals, vcat, blocks(B))
LinearAlgebra.svdvals(B::BlockDiagonal) = sort!(svdvals_blockwise(B); rev=true)

# `B = U * Diagonal(S) * Vt` with `U` and `Vt` `BlockDiagonal` (`S` only sorted block-wise).
function svd_blockwise(B::BlockDiagonal{T}; full::Bool=false) where T
    U = Matrix{float(T)}[]
    S = Vector{float(T)}()
    Vt = Matrix{float(T)}[]
    for b in blocks(B)
        F = svd(b, full=full)
        push!(U, F.U)
        append!(S, F.S)
        push!(Vt, F.Vt)
    end
    return BlockDiagonal(U), S, BlockDiagonal(Vt)
end

function LinearAlgebra.svd(B::BlockDiagonal; full::Bool=false)::SVD
    U, S, Vt = svd_blockwise(B, full=full)
    # Sort singular values in descending order by convention.
    # This means `U` and `Vt` will be `Matrix`s, not `BlockDiagonal`s.
    p = sortperm(S, rev=true)
    return SVD(U[:, p], S[p], Vt[p, :])
end

function LinearAlgebra.cholesky(B::BlockDiagonal)::Cholesky
    C = BlockDiagonal(map(b -> cholesky(b).U, blocks(B)))
    return Cholesky(C, 'U', 0)
end

# Make getproperty on a Cholesky factorized BlockDiagonal return another BlockDiagonal
# where each block is an upper or lower triangular matrix. This ensures that optimizations
# for BlockDiagonal matrices are preserved, though it comes at the cost of reallocating
# a vector of triangular wrappers on each call.
function Base.getproperty(C::Cholesky{T, <:BlockDiagonal{T}}, x::Symbol) where T
    B = getfield(C, :factors)
    uplo = getfield(C, :uplo)
    f = if x === :U
        uplo == 'U' ? UpperTriangular : (X -> UpperTriangular(X'))
    elseif x === :L
        uplo == 'L' ? LowerTriangular : (X -> LowerTriangular(X'))
    elseif x === :UL
        uplo == 'U' ? UpperTriangular : LowerTriangular
    else
        return getfield(C, x)
    end
    return BlockDiagonal(map(f, blocks(B)))
end
