# Core functionality for the `BlockDiagonal` type

"""
    BlockDiagonal{T, V<:AbstractMatrix{T}} <: AbstractMatrix{T}

A matrix with matrices on the diagonal, and zeros off the diagonal.
"""
struct BlockDiagonal{T,V<:AbstractMatrix{T}} <: AbstractMatrix{T}
    blocks::Vector{V}

    function BlockDiagonal{T,V}(blocks::Vector{V}) where {T,V<:AbstractMatrix{T}}
        return new{T,V}(blocks)
    end
end

function BlockDiagonal(blocks::Vector{V}) where {T,V<:AbstractMatrix{T}}
    return BlockDiagonal{T,V}(blocks)
end

BlockDiagonal(B::BlockDiagonal) = B

is_square(A::AbstractMatrix) = size(A, 1) == size(A, 2)

"""
    blocks(B::BlockDiagonal{T, V}) -> Vector{V}

Return the on-diagonal blocks of B.
"""
blocks(B::BlockDiagonal) = B.blocks

# BlockArrays-like functions
"""
    blocksizes(B::BlockDiagonal) -> Vector{Tuple}

Return the size of each on-diagonal block in order.

# Example
```jldoctest; setup = :(using BlockDiagonals)
julia> B = BlockDiagonal([rand(2, 2), rand(3, 3)]);

julia> blocksizes(B)
2-element Vector{Tuple{Int64, Int64}}:
 (2, 2)
 (3, 3)
```
See also [`blocksize`](@ref) for accessing the size of a single block efficiently.
"""
blocksizes(B::BlockDiagonal) = map(size, blocks(B))

"""
    blocksize(B::BlockDiagonal, p::Integer, q::Integer=p) -> Tuple

Return the size of the p^th on-diagonal block. Optionally specify `q` to return the
size of block `p, q`.

# Example
```jldoctest; setup = :(using BlockDiagonals)
julia> X = rand(2, 2); Y = rand(3, 3);

julia> B = BlockDiagonal([X, Y]);

julia> blocksize(B, 1)
(2, 2)

julia> blocksize(B, 1, 2)
(2, 3)
```
See also [`blocksizes`](@ref) for accessing the size of all on-diagonal blocks easily.
"""
blocksize(B::BlockDiagonal, p::Integer) = size(blocks(B)[p])
function blocksize(B::BlockDiagonal, p::Integer, q::Integer)
    return size(blocks(B)[p], 1), size(blocks(B)[q], 2)
end

"""
    nblocks(B::BlockDiagonal[, dim])

Return the number of on-diagonal blocks.

The total number of blocks in the matrix is `nblocks(B)^2`.
"""
nblocks(B::BlockDiagonal) = length(blocks(B))

getblock(B::BlockDiagonal, p::Integer) = blocks(B)[p]
function getblock(B::BlockDiagonal{T}, p::Integer, q::Integer) where {T}
    return p == q ? blocks(B)[p] : Zeros{T}(blocksize(B, p, q))
end

function setblock!(B::BlockDiagonal{T,V}, v::V, p::Integer) where {T,V}
    if blocksize(B, p) != size(v)
        throw(
            DimensionMismatch(
                "Cannot set block of size $(blocksize(B, p)) to block of size $(size(v)).",
            ),
        )
    end
    return blocks(B)[p] = v
end

function setblock!(B::BlockDiagonal{T,V}, v::V, p::Int, q::Int) where {T,V}
    p == q ||
        throw(ArgumentError("Cannot set off-diagonal block ($p, $q) to non-zero value."))
    return setblock!(B, v, p)
end

## Base
function Base.Matrix(B::BlockDiagonal{T}) where {T}
    A = zeros(T, size(B))

    nrows = size.(B.blocks, 1)
    ncols = size.(B.blocks, 2)
    row_idxs = cumsum(nrows) .- nrows .+ 1
    col_idxs = cumsum(ncols) .- ncols .+ 1

    for n in eachindex(blocks(B))
        block_rows = row_idxs[n]:(row_idxs[n]+nrows[n]-1)
        block_cols = col_idxs[n]:(col_idxs[n]+ncols[n]-1)
        A[block_rows, block_cols] .= blocks(B)[n]
    end
    return A
end

Base.collect(B::BlockDiagonal) = Matrix(B)

Base.size(B::BlockDiagonal) =
    sum(first ∘ size, blocks(B); init = 0), sum(last ∘ size, blocks(B); init = 0)

Base.similar(B::BlockDiagonal) = BlockDiagonal(map(similar, blocks(B)))
Base.similar(B::BlockDiagonal, ::Type{T}) where {T} =
    BlockDiagonal(map(b -> similar(b, T), blocks(B)))
Base.parent(B::BlockDiagonal) = B.blocks

@propagate_inbounds function Base.setindex!(B::BlockDiagonal, v, i::Integer, j::Integer)
    p, i_, j_ = _block_indices(B, i, j)
    if p > 0
        @inbounds getblock(B, p)[i_, end+j_] = v
    elseif !iszero(v)
        throw(
            ArgumentError(
                "Cannot set entry ($i, $j) in off-diagonal-block to nonzero value $v.",
            ),
        )
    end
    return v
end

@propagate_inbounds function Base.getindex(
    B::BlockDiagonal{T},
    i::Integer,
    j::Integer,
) where {T}
    p, i, j = _block_indices(B, i, j)
    # if not in on-diagonal block `p` then value at `i, j` must be zero
    @inbounds return p > 0 ? getblock(B, p)[i, end+j] : zero(T)
end

function Base.convert(::Type{BlockDiagonal{T,M}}, b::BlockDiagonal) where {T,M}
    new_blocks = convert.(M, blocks(b))
    return BlockDiagonal(new_blocks)::BlockDiagonal{T,M}
end

# Transform indices `i, j` (identifying entry `Matrix(B)[i, j]`) into indices `p, i, j` such
# that the same entry is available via `getblock(B, p)[i, end+j]`; `p = -1` if no such `p`.
function _block_indices(B::BlockDiagonal, i::Integer, j::Integer)
    all((0, 0) .< (i, j) .<= size(B)) || throw(BoundsError(B, (i, j)))
    # find the on-diagonal block `p` in column `j`
    p = 0
    @inbounds while j > 0
        p += 1
        j -= blocksize(B, p)[2]
    end
    # isempty to avoid reducing over an empty collection
    @views @inbounds i -= isempty(1:(p-1)) ? 0 : sum(size.(blocks(B)[1:(p-1)], 1))
    # if row `i` outside of block `p`, set `p` to place-holder value `-1`
    if i <= 0 || i > blocksize(B, p)[1]
        p = -1
    end
    return p, i, j
end

function Base.copy(b::BlockDiagonal)
    return BlockDiagonal(copy.(blocks(b)))
end

function Base.copy!(dest::BlockDiagonal, src::BlockDiagonal)
    isequal_blocksizes(dest, src) ||
        throw(DimensionMismatch("dest and src have different block sizes"))
    for i in eachindex(blocks(dest))
        @inbounds copyto!(dest.blocks[i], src.blocks[i])
    end
    return dest
end

# Pretty-printing for sparse arrays
function Base.replace_in_print_matrix(B::BlockDiagonal, i::Integer, j::Integer, s::AbstractString)
    p, ip, jp = _block_indices(B, i, j)
    p == -1 && return Base.replace_with_centered_mark(s)
    b = getblock(B, p)
    jp += last(axes(b, 2))
    Base.replace_in_print_matrix(b, ip, jp, s)
end
