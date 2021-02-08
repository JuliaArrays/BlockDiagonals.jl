# constructor
function ChainRulesCore.rrule(::Type{<:BlockDiagonal}, blocks::Vector{V}) where {V}
    BlockDiagonal_pullback(Δ::Composite) = (NO_FIELDS, Δ.blocks)
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
        return (NO_FIELDS, Composite{T}(blocks=Δblocks))
    end
    return Matrix(B), Matrix_pullback
end

