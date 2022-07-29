# # constructor
_BlockDiagonal_pullback(Δ::Tangent, nrows, ncols) = (NoTangent(), Δ.blocks)
_BlockDiagonal_pullback(Δ::BlockDiagonal, nrows, ncols) = (NoTangent(), Δ.blocks)
function _BlockDiagonal_pullback(Δ::AbstractThunk, nrows, ncols)
    _BlockDiagonal_pullback(unthunk(Δ), nrows, ncols)
end
function _BlockDiagonal_pullback(Δ::AbstractArray, nrows, ncols)
    row_idxs = cumsum(nrows) .- nrows .+ 1
    col_idxs = cumsum(ncols) .- ncols .+ 1
    Δblocks = map(eachindex(nrows)) do n
        block_rows = row_idxs[n]:(row_idxs[n] + nrows[n] - 1)
        block_cols = col_idxs[n]:(col_idxs[n] + ncols[n] - 1)
        return Δ[block_rows, block_cols]
    end
    return NoTangent(), Δblocks
end

function ChainRulesCore.rrule(::Type{<:BlockDiagonal}, blocks::Vector{V}) where {V}
    nrows = size.(blocks, 1)
    ncols = size.(blocks, 2)
    function BlockDiagonal_pullback(Δ)
        return _BlockDiagonal_pullback(Δ, nrows, ncols)
    end
    return BlockDiagonal(blocks), BlockDiagonal_pullback
end

# densification
function _densification_pullback(Ȳ::AbstractMatrix, T, nrows, ncols)
    row_idxs = cumsum(nrows) .- nrows .+ 1
    col_idxs = cumsum(ncols) .- ncols .+ 1

    Δblocks = map(eachindex(nrows)) do n
        block_rows = row_idxs[n]:(row_idxs[n] + nrows[n] - 1)
        block_cols = col_idxs[n]:(col_idxs[n] + ncols[n] - 1)
        return Ȳ[block_rows, block_cols]
    end
    return (NoTangent(), Tangent{T}(blocks=Δblocks))
end
function _densification_pullback(Ȳ::AbstractThunk, T, nrows, ncols)
    return _densification_pullback(unthunk(Ȳ), T, nrows, ncols)
end
function ChainRulesCore.rrule(::Type{<:Base.Matrix}, B::T) where {T<:BlockDiagonal}
    nrows = size.(B.blocks, 1)
    ncols = size.(B.blocks, 2)
    densification_pullback(ȳ) = _densification_pullback(ȳ, T, nrows, ncols)
    return Matrix(B), densification_pullback
end

# multiplication
function ChainRulesCore.rrule(
        ::typeof(*),
        bm::BlockDiagonal{T, V, S},
        v::StridedVector{T}
    ) where {T<:Union{Real, Complex}, V<:Matrix{T}, S}

    y = bm * v

    # needed for computing Δ * v' blockwise
    nrows = size.(bm.blocks, 1)
    ncols = size.(bm.blocks, 2)
    row_idxs = cumsum(nrows) .- nrows .+ 1
    col_idxs = cumsum(ncols) .- ncols .+ 1

    function bm_vector_mul_pullback(Δy)
        ȳ = unthunk(Δy)
        Δblocks = map(eachindex(nrows)) do i
            block_rows = row_idxs[i]:(row_idxs[i] + nrows[i] - 1)
            block_cols = col_idxs[i]:(col_idxs[i] + ncols[i] - 1)
            return InplaceableThunk(
                X̄ -> mul!(X̄, ȳ[block_rows], v[block_cols]', true, true),
                @thunk(ȳ[block_rows] * v[block_cols]'),
            )
        end

        b̄m = Tangent{BlockDiagonal{T, V, S}}(;blocks=Δblocks)
        v̄ = InplaceableThunk(X̄ -> mul!(X̄, bm', ȳ, true, true), @thunk(bm' * ȳ))
        return NoTangent(), b̄m, v̄
    end
    return y, bm_vector_mul_pullback
end

function ProjectTo(b::BlockDiagonal)
    blocks = map(ProjectTo, b.blocks)
    return ProjectTo{BlockDiagonal}(; blocks=blocks, blocksizes=blocksizes(b))
end

function (project::ProjectTo{BlockDiagonal})(dx::AbstractArray)
    # prepare to index into the dense array
    nrows = first.(project.blocksizes)
    ncols = last.(project.blocksizes)
    row_idxs = cumsum(nrows) .- nrows .+ 1
    col_idxs = cumsum(ncols) .- ncols .+ 1
    # project each block individually
    blocks = map(eachindex(nrows)) do i
        block_rows = row_idxs[i]:(row_idxs[i] + nrows[i] - 1)
        block_cols = col_idxs[i]:(col_idxs[i] + ncols[i] - 1)
        project.blocks[i](dx[block_rows, block_cols])
    end
    return BlockDiagonal(blocks)
end
