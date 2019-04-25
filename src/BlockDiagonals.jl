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

A BlockMatrix with square blocks on the diagonal, and zeros off the diagonal.
"""
struct BlockDiagonal{T, V<:AbstractMatrix{T}, S<:AbstractBlockSizes} <: AbstractBlockMatrix{T}
    blocks::Vector{V}
    blocksizes::S

    function BlockDiagonal(blocks::Vector{<:V}, sizes::S) where {T, V<:AbstractMatrix{T}, S<:AbstractBlockSizes}
        all(is_square, blocks) || throw(ArgumentError("All blocks must be square."))
        return new{T, V, S}(blocks, sizes)
    end
end

function BlockDiagonal(blocks::AbstractVector{<:AbstractMatrix})
    # mapreduce would give Tuples; need Arrays
    sizes = BlockSizes(([size(b, 1), size(b, 2)] for b in blocks)...)
    return BlockDiagonal(blocks, sizes)
end

BlockDiagonal(B::BlockDiagonal) = copy(B)

is_square(A::AbstractMatrix) = size(A, 1) === size(A, 2)
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
    blocks(B)[p] = v
end

# Needs to be `Int` not `Integer` to avoid methods ambiguity. Can be changed after
# BlockArrays v0.9 is released; see https://github.com/JuliaArrays/BlockArrays.jl/issues/82
function BlockArrays.setblock!(B::BlockDiagonal{T, V}, v::V, p::Int, q::Int) where {T, V}
    p == q || throw(ArgumentError("Cannot set off-diagonal block ($p, $q) to non-zero value."))
    setblock!(B, v, p)
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
function Base.isapprox(B::BlockDiagonal, M::AbstractMatrix; kwargs...)
    return isapprox(Matrix(B), Matrix(M); kwargs...)
end

function Base.isapprox(B1::BlockDiagonal, B2::BlockDiagonal; kwargs...)
    return isequal_blocksizes(B1, B2) && all(isapprox.(blocks(B1), blocks(B2); kwargs...))
end

function isequal_blocksizes(B1::BlockDiagonal, B2::BlockDiagonal)::Bool
    return size(B1) === size(B2) && blocksizes(B1) == blocksizes(B2)
end

## Addition
function Base.:+(b1::BlockDiagonal, b2::BlockDiagonal)
    if size(b1) == size(b2) && size.(blocks(b1)) == size.(blocks(b2))
        return BlockDiagonal(blocks(b1) .+ blocks(b2))
    else
        return Matrix(b1) + Matrix(b2)
    end
end

Base.:+(m::AbstractMatrix, b::BlockDiagonal) = b + m
function Base.:+(b::BlockDiagonal, m::AbstractMatrix)
    !isdiag(m) && return Matrix(b) + m
    size(b) != size(m) && throw(DimensionMismatch("Can't add matrices of different sizes."))
    d = diag(m)
    si = 1
    sj = 1
    nb = copy(blocks(b))
    for i in 1:length(nb)
        s = size(nb[i])
        nb[i] += @view m[si:s[1] + si - 1, sj:s[2] + sj - 1]
        si += s[1]
        sj += s[2]
    end
    return BlockDiagonal(nb)
end

# function +(A::BlockDiagonal, B::StridedMatrix)::Matrix
#     size(A) != size(B) && throw(DimensionMismatch("dimensions must match"))
#     C = copy(B)
#     row = 1
#     for (j, block) in enumerate(blocks(A))
#         rows = row:row + size(block, 1) - 1
#         C[rows, rows] .+= block
#         row += size(block, 1)
#     end
#     return C
# end

Base.:+(m::Diagonal, b::BlockDiagonal) = b + m
function Base.:+(b::BlockDiagonal, m::Diagonal)
    size(b) != size(m) && throw(DimensionMismatch("Can't add matrices of different sizes."))
    C = similar(b)
    row = 1
    for (j, block) in enumerate(blocks(b))
        rows = row:row + size(block, 1) - 1
        block_C = blocks(C)[j]
        block_C .= block
        for (k, r) in enumerate(rows)
            block_C[k, k] += diag(m)[r]
        end
        row += size(block, 1)
    end
    return C
end

Base.:+(m::UniformScaling, b::BlockDiagonal) = b + m
function Base.:+(b::BlockDiagonal, m::UniformScaling)
    return BlockDiagonal([block + m for block in blocks(b)])
end


## Multiplication
Base.:*(b::BlockDiagonal, n::Real) = BlockDiagonal(n .* blocks(b))
Base.:*(n::Real, b::BlockDiagonal) = b * n

function Base.:*(b::BlockDiagonal, m::BlockDiagonal)
    if size(b) == size(m) && size.(blocks(b)) == size.(blocks(m))
        return BlockDiagonal(blocks(b) .* blocks(m))
    else
        return Matrix(b) * Matrix(m)
    end
end

function Base.:*(b::BlockDiagonal, m::AbstractMatrix)
    if size(b, 2) !== size(m, 1)
        throw(DimensionMismatch("Cannot multiply matrices of size $(size(b)) and $(size(m))."))
    end
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 2) - 1
        push!(d, block * m[st:ed, :])
        st = ed + 1
    end
    return reduce(vcat, d)
end

function Base.:*(m::AbstractMatrix, b::BlockDiagonal)
    if size(b, 1) != size(m, 2)
        throw(DimensionMismatch("Cannot multiply matrices of size $(size(b)) and $(size(m))."))
    end
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 1) - 1
        push!(d, m[:, st:ed] * block)
        st = ed + 1
    end
    return reduce(hcat, d)
end

## Division
Base.:/(b::BlockDiagonal, n::Real) = BlockDiagonal(blocks(b) ./ n)

## LinearAlgebra
for f in (:adjoint, :eigvecs, :inv, :transpose)
    @eval LinearAlgebra.$f(B::BlockDiagonal) = BlockDiagonal(map($f, blocks(B)))
end

LinearAlgebra.diag(B::BlockDiagonal) = mapreduce(diag, vcat, blocks(B))
LinearAlgebra.det(B::BlockDiagonal) = prod(det, blocks(B))
LinearAlgebra.logdet(B::BlockDiagonal) = sum(logdet, blocks(B))
LinearAlgebra.tr(B::BlockDiagonal) = sum(tr, blocks(B))

function LinearAlgebra.eigvals(B::BlockDiagonal)
    eigs = mapreduce(eigvals, vcat, blocks(B))
    eigs isa Vector{<:Complex} && return eigs
    return sort!(eigs)
end

svdvals_blockwise(B::BlockDiagonal) = mapreduce(svdvals, vcat, blocks(B))
LinearAlgebra.svdvals(B::BlockDiagonal) = sort!(svdvals_blockwise(B); rev=true)

# `B = U * Diagonal(S) * Vt` with `U` and `Vt` `BlockDiagonal` (`S` only sorted block-wise).
function svd_blockwise(B::BlockDiagonal; full::Bool=false)
    Fs = svd.(blocks(B); full=full)
    U = BlockDiagonal([F.U for F in Fs])
    S = mapreduce(F -> F.S, vcat, Fs)
    Vt = BlockDiagonal([F.Vt for F in Fs])
    return U, S, Vt
end

function LinearAlgebra.svd(B::BlockDiagonal; full::Bool=false)::SVD
    U, S, Vt = svd_blockwise(B, full=full)
    # Sort singular values in descending order by convention.
    # This means `U` and `Vt` will be `Matrix`s, not `BlockDiagonal`s.
    p = sortperm(S, rev=true)
    return SVD(U[:, p], S[p], Vt[p, :])
end

function LinearAlgebra.cholesky(B::BlockDiagonal)
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
        uplo === 'U' ? UpperTriangular : (X -> UpperTriangular(X'))
    elseif x === :L
        uplo === 'L' ? LowerTriangular : (X -> LowerTriangular(X'))
    elseif x === :UL
        uplo === 'U' ? UpperTriangular : LowerTriangular
    else
        return getfield(C, x)
    end
    return BlockDiagonal(map(f, blocks(B)))
end

end  # end module
