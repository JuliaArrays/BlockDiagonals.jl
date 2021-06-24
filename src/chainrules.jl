# constructor
_BlockDiagonal_pullback(Δ::Tangent) = (NoTangent(), Δ.blocks)
_BlockDiagonal_pullback(Δ::AbstractThunk) = _BlockDiagonal_pullback(unthunk(Δ))
function ChainRulesCore.rrule(::Type{<:BlockDiagonal}, blocks::Vector{V}) where {V}
    return BlockDiagonal(blocks), _BlockDiagonal_pullback
end

# densification
function _densification_pullback(Ȳ::Matrix, T, nrows, ncols)
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
        bm::BlockDiagonal{T, V},
        v::StridedVector{T}
    ) where {T<:Union{Real, Complex}, V<:Matrix{T}}

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
                @thunk(ȳ[block_rows] * v[block_cols]'),
                X̄ -> mul!(X̄, ȳ[block_rows], v[block_cols]', true, true)
            )
        end

        b̄m = Tangent{BlockDiagonal{T, V}}(;blocks=Δblocks)
        v̄ = InplaceableThunk(@thunk(bm' * ȳ), X̄ -> mul!(X̄, bm', ȳ, true, true))
        return NoTangent(), b̄m, v̄
    end
    return y, bm_vector_mul_pullback
end
