Base.@deprecate(
    BlockDiagonal{T, V}(blocks) where {T, V},
    BlockDiagonal{T, V, all(is_square.(blocks))}(blocks)
)
