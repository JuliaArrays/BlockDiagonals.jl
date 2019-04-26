module BlockDiagonals

using BlockArrays
using BlockArrays: AbstractBlockSizes, BlockSizes
using FillArrays
using LinearAlgebra

export BlockDiagonal, blocks
# reexport core interfaces from BlockArrays
export Block, BlockSizes, blocksize, blocksizes, nblocks

"""
    BlockDiagonal{T, V<:AbstractMatrix{T}, S<:AbstractBlockSizes} <: AbstractBlockMatrix{T}

A BlockMatrix with square blocks of type `V` on the diagonal, and zeros off the diagonal.
"""
struct BlockDiagonal{T, V<:AbstractMatrix{T}, S<:AbstractBlockSizes} <: AbstractBlockMatrix{T}
    blocks::Vector{V}
    blocksizes::S

    function BlockDiagonal{T, V, S}(blocks::Vector{V}, sizes::S) where {T, V<:AbstractMatrix{T}, S<:AbstractBlockSizes}
        all(is_square, blocks) || throw(ArgumentError("All blocks must be square."))
        return new{T, V, S}(blocks, sizes)
    end
end

function BlockDiagonal(blocks::Vector{V}, sizes::S) where {T, V<:AbstractMatrix{T}, S<:AbstractBlockSizes}
    return BlockDiagonal{T, V, S}(blocks, sizes)
end

function BlockDiagonal(blocks::AbstractVector{<:AbstractMatrix})
    # mapreduce would give Tuples; need Arrays
    sizes = BlockSizes(([size(b, 1), size(b, 2)] for b in blocks)...)
    return BlockDiagonal(blocks, sizes)
end

BlockDiagonal(B::BlockDiagonal) = copy(B)

is_square(A::AbstractMatrix) = size(A, 1) == size(A, 2)

"""
    blocks(B::BlockDiagonal{T, V}) -> Vector{V}

Return the on-diagonal blocks of B.
"""
blocks(B::BlockDiagonal) = B.blocks

## AbstractBlockMatrix interface
BlockArrays.blocksizes(B::BlockDiagonal) = B.blocksizes

# Needs to be `Int` not `Integer` to avoid methods ambiguity. Can be changed after
# BlockArrays v0.9 is released; see https://github.com/JuliaArrays/BlockArrays.jl/issues/82
BlockArrays.blocksize(B::BlockDiagonal, p::Int) = size(blocks(B)[p])
function BlockArrays.blocksize(B::BlockDiagonal, p::Int, q::Int)
    return size(blocks(B)[p], 1), size(blocks(B)[q], 2)
end

# Needs to be `Int` not `Integer` to avoid methods ambiguity. Can be changed after
# BlockArrays v0.9 is released; see https://github.com/JuliaArrays/BlockArrays.jl/issues/82
BlockArrays.nblocks(B::BlockDiagonal, dim::Int) = dim > 2 ? 1 : length(blocks(B))
BlockArrays.nblocks(B::BlockDiagonal) = length(blocks(B)), length(blocks(B))

# Needs to be `Int` not `Integer` to avoid methods ambiguity. Can be changed after
# BlockArrays v0.9 is released; see https://github.com/JuliaArrays/BlockArrays.jl/issues/82
function BlockArrays.getblock(B::BlockDiagonal{T}, p::Int, q::Int) where T
    return p == q ? blocks(B)[p] : Zeros{T}(blocksize(B, p, q))
end

# Allow `B[Block(i)]` as shorthand for i^th diagonal block, i.e. `B[Block(i, i)]`
BlockArrays.getblock(B::BlockDiagonal, p::Integer) = blocks(B)[p]
Base.getindex(B::BlockDiagonal, block::Block{1}) = getblock(B, block.n[1])
Base.setindex!(B::BlockDiagonal, v, block::Block{1}) = setblock!(B, v, block.n[1])

function BlockArrays.setblock!(B::BlockDiagonal{T, V}, v::V, p::Integer) where {T, V}
    if blocksize(B, p) != size(v)
        throw(DimensionMismatch(
            "Cannot set block of size $(blocksize(B, p)) to block of size $(size(v))."
        ))
    end
    return blocks(B)[p] = v
end

# Needs to be `Int` not `Integer` to avoid methods ambiguity. Can be changed after
# BlockArrays v0.9 is released; see https://github.com/JuliaArrays/BlockArrays.jl/issues/82
function BlockArrays.setblock!(B::BlockDiagonal{T, V}, v::V, p::Int, q::Int) where {T, V}
    p == q || throw(ArgumentError("Cannot set off-diagonal block ($p, $q) to non-zero value."))
    return setblock!(B, v, p)
end

## Base
Base.Matrix(B::BlockDiagonal) = cat(blocks(B)...; dims=(1, 2))
Base.size(B::BlockDiagonal) = sum(first∘size, blocks(B)), sum(last∘size, blocks(B))
Base.similar(B::BlockDiagonal) = BlockDiagonal(map(similar, blocks(B)))

function Base.setindex!(B::BlockDiagonal{T}, v, i::Integer, j::Integer) where T
    p, i_, j_ = _block_indices(B, i, j)
    if p > 0
        @inbounds B[Block(p)][i_, end+j_] = v
    elseif !iszero(v)
        throw(ArgumentError(
            "Cannot set entry ($i, $j) in off-diagonal-block to nonzero value $v."
        ))
    end
    return v
end

function Base.getindex(B::BlockDiagonal{T}, i::Integer, j::Integer) where T
    p, i, j = _block_indices(B, i, j)
    # if not in on-diagonal block `p` then value at `i, j` must be zero
    return p > 0 ? B[Block(p)][i, end + j] : zero(T)
end

# Transform indices `i, j` (identifying entry `Matrix(B)[i, j]`) into indices `p, i, j`
# such that the same entry is available via `B[Block(p)][i, end+j]`; `p = -1` if no such `p`.
function _block_indices(B::BlockDiagonal, i::Integer, j::Integer)
    all((0, 0) .< (i, j) .<= size(B)) || throw(BoundsError(B, (i, j)))
    nrows = size.(blocks(B), 1)
    ncols = size.(blocks(B), 2)
    # find the on-diagonal block `p` in column `j`
    p = 0
    while j > 0
        p += 1
        j -= ncols[p]
    end
    i -= sum(nrows[1:(p-1)])
    # if row `i` outside of block `p`, set `p` to place-holder value `-1`
    if i <= 0 || i > nrows[p]
        p = -1
    end
    return p, i, j
end

Base.isapprox(M::AbstractMatrix, B::BlockDiagonal; kwargs...) = isapprox(B, M; kwargs...)
function Base.isapprox(B::BlockDiagonal, M::AbstractMatrix; kwargs...)::Bool
    return isapprox(Matrix(B), Matrix(M); kwargs...)
end

function Base.isapprox(B1::BlockDiagonal, B2::BlockDiagonal; kwargs...)
    return isequal_blocksizes(B1, B2) && all(isapprox.(blocks(B1), blocks(B2); kwargs...))
end

function isequal_blocksizes(B1::BlockDiagonal, B2::BlockDiagonal)::Bool
    return size(B1) == size(B2) && blocksizes(B1) == blocksizes(B2)
end

## Addition
# TODO make type stable, maybe via Broadcasting?
function Base.:+(B1::BlockDiagonal, B2::BlockDiagonal)
    if isequal_blocksizes(B1, B2)
        return BlockDiagonal(blocks(B1) .+ blocks(B2))
    else
        return Matrix(B1) + Matrix(B2)
    end
end

Base.:+(M::AbstractMatrix, B::BlockDiagonal) = B + M
Base.:+(B::BlockDiagonal, M::AbstractMatrix) = Matrix(B) + M

function Base.:+(B::BlockDiagonal, M::StridedMatrix)
    size(B) == size(M) || throw(DimensionMismatch("dimensions must match"))
    A = copy(M)
    row = 1
    for (j, block) in enumerate(blocks(B))
        nrows = size(block, 1)
        rows = row:(row + nrows-1)
        A[rows, rows] .+= block
        row += nrows
    end
    return A
end

Base.:+(M::Diagonal, B::BlockDiagonal) = B + M
function Base.:+(B::BlockDiagonal, M::Diagonal)::BlockDiagonal
    size(B) == size(M) || throw(DimensionMismatch("dimensions must match"))
    A = copy(B)
    d = diag(M)
    row = 1
    for (p, block) in enumerate(blocks(B))
        nrows = size(block, 1)
        rows = row:(row + nrows-1)
        for (i, r) in enumerate(rows)
            A[Block(p)][i, i] += d[r]
        end
        row += nrows
    end
    return A
end

Base.:+(M::UniformScaling, B::BlockDiagonal) = B + M
function Base.:+(B::BlockDiagonal, M::UniformScaling)
    return BlockDiagonal([block + M for block in blocks(B)])
end

## Multiplication
Base.:*(n::Number, B::BlockDiagonal) = B * n
Base.:*(B::BlockDiagonal, n::Number) = BlockDiagonal(n .* blocks(B))

# TODO make type stable, maybe via Broadcasting?
function Base.:*(B1::BlockDiagonal, B2::BlockDiagonal)
    if isequal_blocksizes(B1, B2)
        return BlockDiagonal(blocks(B1) .* blocks(B2))
    else
        return Matrix(B1) * Matrix(B2)
    end
end

function _check_matmul_dims(A::AbstractMatrix, B::AbstractMatrix)
    # match error message from LinearAlgebra
    size(A, 2) == size(B, 1) || throw(DimensionMismatch(
        "A has dimensions $(size(A)) but B has dimensions $(size(B))"
    ))
end

function Base.:*(B::BlockDiagonal, M::AbstractMatrix)
    _check_matmul_dims(B, M)
    ed = 0
    d = map(blocks(B)) do block
        st = ed + 1  # start
        ed += size(block, 2)  # end
        return block * M[st:ed, :]
    end
    return reduce(vcat, d)::Matrix
end

function Base.:*(M::AbstractMatrix, B::BlockDiagonal)
    _check_matmul_dims(M, B)
    ed = 0
    d = map(blocks(B)) do block
        st = ed + 1  # start
        ed += size(block, 1)  # end
        return M[:, st:ed] * block
    end
    return reduce(hcat, d)::Matrix
end

# Diagonal
function Base.:*(B::BlockDiagonal, M::Diagonal)::BlockDiagonal
    _check_matmul_dims(B, M)
    A = copy(B)
    d = diag(M)
    col = 1
    for (p, block) in enumerate(blocks(B))
        ncols = size(block, 2)
        cols = col:(col + ncols-1)
        for (j, c) in enumerate(cols)
            A[Block(p)][:, j] *= d[c]
        end
        col += ncols
    end
    return A
end

function Base.:*(M::Diagonal, B::BlockDiagonal)::BlockDiagonal
    _check_matmul_dims(M, B)
    A = copy(B)
    d = diag(M)
    row = 1
    for (p, block) in enumerate(blocks(B))
        nrows = size(block, 1)
        rows = row:(row + nrows-1)
        for (i, r) in enumerate(rows)
            A[Block(p)][i, :] *= d[r]
        end
        row += nrows
    end
    return A
end

## Division
Base.:/(B::BlockDiagonal, n::Number) = BlockDiagonal(blocks(B) ./ n)

## LinearAlgebra
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

end  # end module
