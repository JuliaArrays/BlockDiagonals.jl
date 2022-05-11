"""
    function Base.:*(A::BlockDiagonal, x::Vector{T}) where {T<:AffExpr}

Multiply a `BlockDiagonal` with a `Vector{AffExpr}` from JuMP so we don't need to convert
the `BlockDiagonal` to a `Matrix` first.
"""
function Base.:*(A::BlockDiagonal, x::Vector{T}) where {T<:AffExpr}
    return mul!(similar(x, T, axes(A, 1)), A, x)
end
