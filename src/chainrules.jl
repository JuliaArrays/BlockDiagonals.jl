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

# multiplication
function ChainRulesCore.rrule(::typeof(*), bm::BlockDiagonal{T, V}, v::AbstractVector{T}) where {T<:Union{Real, Complex}, V<:AbstractArray{T, 2}}
    y = bm * v

    # needed for computing Δ * v' blockwise
    sizes = size.(bm.blocks)
    high1s = cumsum(first.(sizes))
    low1s = [1, (1 .+ high1s)...]
    high2s = cumsum(last.(sizes))
    low2s = [1, (1 .+ high2s)...]

    function bm_vector_mul_pullback(Δ)
        return (
            NO_FIELDS,
            BlockDiagonal(
                [
                    Δ[low1s[i]:high1s[i]] * v[low2s[i]:high2s[i]]' for i in 1:length(sizes)
                ]
            ),
            bm' * Δ,
        )
    end
    return y, bm_vector_mul_pullback
end
