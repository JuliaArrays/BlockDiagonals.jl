function FiniteDifferences.to_vec(X::BlockDiagonal)
    x, blocks_from_vec = to_vec(blocks(X))
    BlockDiagonal_from_vec(x_vec) = BlockDiagonal(blocks_from_vec(x_vec))
    return x, BlockDiagonal_from_vec
end

