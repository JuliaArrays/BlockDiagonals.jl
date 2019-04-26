# Core functionality for the `BlockDiagonal` type
# including implementing the BlockArray interface and AbstractArray interface

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
