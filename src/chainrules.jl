# constructor
function ChainRulesCore.rrule(::Type{<:BlockDiagonal}, blocks::Vector{V}) where {V}
    BlockDiagonal_pullback(Δ::Tangent) = (NoTangent(), Δ.blocks)
    return BlockDiagonal(blocks), BlockDiagonal_pullback
end

# densification
function ChainRulesCore.rrule(::Type{<:Base.Matrix}, B::T) where {T<:BlockDiagonal}
    nrows = size.(B.blocks, 1)
    ncols = size.(B.blocks, 2)
    function Matrix_pullback(Δ::Matrix)
        row_idxs = cumsum(nrows) .- nrows .+ 1
        col_idxs = cumsum(ncols) .- ncols .+ 1

        Δblocks = map(eachindex(nrows)) do n
            block_rows = row_idxs[n]:(row_idxs[n] + nrows[n] - 1)
            block_cols = col_idxs[n]:(col_idxs[n] + ncols[n] - 1)
            return Δ[block_rows, block_cols]
        end
        return (NoTangent(), Tangent{T}(blocks=Δblocks))
    end
    return Matrix(B), Matrix_pullback
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

    function bm_vector_mul_pullback(Δ)
        Δblocks = map(eachindex(nrows)) do i
            block_rows = row_idxs[i]:(row_idxs[i] + nrows[i] - 1)
            block_cols = col_idxs[i]:(col_idxs[i] + ncols[i] - 1)
            return InplaceableThunk(
                @thunk(Δ[block_rows] * v[block_cols]'),
                X̄ -> mul!(X̄, Δ[block_rows], v[block_cols]', true, true)
            )
        end
        return (
            NoTangent(),
            Tangent{BlockDiagonal{T, V}}(;blocks=Δblocks),
            InplaceableThunk(
                @thunk(bm' * Δ),
                X̄ -> mul!(X̄, bm', Δ, true, true)
            ),
        )
    end
    return y, bm_vector_mul_pullback
end
