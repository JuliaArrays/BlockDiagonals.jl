# Linear algebra functions, in particular extending the `LinearAlgebra` stdlib

for f in (:adjoint, :eigvecs, :inv, :pinv, :transpose)
    @eval LinearAlgebra.$f(B::BlockDiagonal) = BlockDiagonal(map($f, blocks(B)))
end

LinearAlgebra.diag(B::BlockDiagonal) = mapreduce(diag, vcat, blocks(B))
LinearAlgebra.det(B::BlockDiagonal) = prod(det, blocks(B))
LinearAlgebra.logdet(B::BlockDiagonal) = sum(logdet, blocks(B))
LinearAlgebra.tr(B::BlockDiagonal) = sum(tr, blocks(B))

for f in (:Symmetric, :Hermitian)
    @eval LinearAlgebra.$f(B::BlockDiagonal, uplo::Symbol=:U) = BlockDiagonal([$f(block, uplo) for block in blocks(B)])
end

# Real matrices can have Complex eigenvalues; `eigvals` is not type stable.
@static if VERSION < v"1.2.0-DEV.275"
    # No convention for sorting complex eigenvalues in earlier versions of Julia.
    function LinearAlgebra.eigvals(B::BlockDiagonal, args...; kwargs...)
        vals = mapreduce(b -> eigvals(b, args...; kwargs...), vcat, blocks(B))
        return !isa(vals, Vector{<:Complex}) ? sort(vals) : vals
    end
else
    # Sorting was introduced in Julia v1.2 by https://github.com/JuliaLang/julia/pull/21598
    function LinearAlgebra.eigvals(
        B::BlockDiagonal, args...; sortby::Union{Function, Nothing}=LinearAlgebra.eigsortby, kwargs...
    )
        vals = mapreduce(b -> eigvals(b, args...; kwargs...), vcat, blocks(B))
        return LinearAlgebra.sorteig!(vals, sortby)
    end
end

if VERSION < v"1.3.0-DEV.426"
    # This is copy of the definition for LinearAlgebra, only used to workaround
    # https://github.com/JuliaLang/julia/issues/31843 which was fixed in Julia v1.3
    function LinearAlgebra.eigmax(B::BlockDiagonal; kwargs...)
        v = eigvals(B; kwargs...)
        if eltype(v) <: Complex
            throw(DomainError(A, "`A` cannot have complex eigenvalues."))
        end
        return maximum(v)
    end
end

"""
    blockEigen(B::BlockDiagonal, args...; kwargs...) -> Eigen

Computes the eigen decomposition for each block separately and keeps the block diagonal 
structure in the matrix of eigenvectors. Hence any parameters given are applied to each
eigen decomposition separately, but there is f.e. no global sorting of eigen values.
"""
function blockEigen(B::BlockDiagonal, args...; kwargs...)
    eigens = [eigen(b, args...; kwargs...) for b in blocks(B)]
    #promote to common types
    values = promote([e.values for e in eigens]...)
    vectors = promote([e.vectors for e in eigens]...)
    Eigen(vcat(values...), BlockDiagonal([vectors...]))
end 

## This function never keeps the block diagonal structure for type stability.
function LinearAlgebra.eigen(B::BlockDiagonal, args...; kwargs...)
    values, vectors = blockEigen(B, args...; kwargs...)
    vectors = Matrix(vectors) # always convert to avoid typestability
    @static if VERSION > v"1.2.0-DEV.275"
        if haskey(kwargs, :sortby)
            Eigen(LinearAlgebra.sorteig!(values, vectors,  kwargs[:sortby])...)
        else
            Eigen(LinearAlgebra.sorteig!(values, vectors)...)
        end
    else 
        Eigen(values, vectors) 
    end
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

# 3-Argument mul!
LinearAlgebra.mul!(C::BlockDiagonal, A::BlockDiagonal, B::BlockDiagonal) = _mul!(C, A, B)

if VERSION ≥ v"1.3"
    function LinearAlgebra.mul!(C::BlockDiagonal, A::BlockDiagonal, B::BlockDiagonal, α::Number, β::Number)
        return _mul!(C, A, B, α, β)
    end
end

function _mul!(C::BlockDiagonal, A::BlockDiagonal, B::BlockDiagonal)
    isequal_blocksizes(A, B) || throw(DimensionMismatch("A and B have different block sizes"))
    isequal_blocksizes(C, A) || throw(DimensionMismatch("C has incompatible block sizes"))
    for i in eachindex(blocks(C))
        @inbounds LinearAlgebra.mul!(C.blocks[i], A.blocks[i], B.blocks[i])
    end

    return C
end

function _mul!(C::BlockDiagonal, A::BlockDiagonal, B::BlockDiagonal, α::Number, β::Number)
    isequal_blocksizes(A, B) || throw(DimensionMismatch("A and B have different block sizes"))
    isequal_blocksizes(C, A) || throw(DimensionMismatch("C has incompatible block sizes"))
    for i in eachindex(blocks(C))
        @inbounds LinearAlgebra.mul!(C.blocks[i], A.blocks[i], B.blocks[i], α, β)
    end

    return C
end
