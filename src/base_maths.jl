# Core maths operations extended for Base

Base.isapprox(M::AbstractMatrix, B::BlockDiagonal; kwargs...) = isapprox(B, M; kwargs...)
function Base.isapprox(B::BlockDiagonal, M::AbstractMatrix; kwargs...)::Bool
    return isapprox(Matrix(B), Matrix(M); kwargs...)
end

function Base.isapprox(B1::BlockDiagonal, B2::BlockDiagonal; kwargs...)
    return isequal_blocksizes(B1, B2) && all(isapprox.(blocks(B1), blocks(B2); kwargs...))
end

function isequal_blocksizes(B1::BlockDiagonal, B2::BlockDiagonal)::Bool
    return size(B1) == size(B2) && blocksizes(B1) == blocksizes(B2)
end

## Addition
# TODO make type stable, maybe via Broadcasting?
function Base.:+(B1::BlockDiagonal, B2::BlockDiagonal)
    if isequal_blocksizes(B1, B2)
        return BlockDiagonal(blocks(B1) .+ blocks(B2))
    else
        return Matrix(B1) + Matrix(B2)
    end
end

Base.:+(M::AbstractMatrix, B::BlockDiagonal) = B + M
Base.:+(B::BlockDiagonal, M::AbstractMatrix) = Matrix(B) + M

function Base.:+(B::BlockDiagonal, M::StridedMatrix)
    size(B) == size(M) || throw(DimensionMismatch("dimensions must match"))
    A = copy(M)
    row = 1
    for (j, block) in enumerate(blocks(B))
        nrows = size(block, 1)
        rows = row:(row + nrows-1)
        A[rows, rows] .+= block
        row += nrows
    end
    return A
end

Base.:+(M::Diagonal, B::BlockDiagonal) = B + M
function Base.:+(B::BlockDiagonal, M::Diagonal)::BlockDiagonal
    size(B) == size(M) || throw(DimensionMismatch("dimensions must match"))
    A = copy(B)
    d = diag(M)
    row = 1
    for (p, block) in enumerate(blocks(B))
        nrows = size(block, 1)
        rows = row:(row + nrows-1)
        for (i, r) in enumerate(rows)
            A[Block(p)][i, i] += d[r]
        end
        row += nrows
    end
    return A
end

Base.:+(M::UniformScaling, B::BlockDiagonal) = B + M
function Base.:+(B::BlockDiagonal, M::UniformScaling)
    return BlockDiagonal([block + M for block in blocks(B)])
end

## Multiplication
Base.:*(n::Number, B::BlockDiagonal) = B * n
Base.:*(B::BlockDiagonal, n::Number) = BlockDiagonal(n .* blocks(B))

# TODO make type stable, maybe via Broadcasting?
function Base.:*(B1::BlockDiagonal, B2::BlockDiagonal)
    if isequal_blocksizes(B1, B2)
        return BlockDiagonal(blocks(B1) .* blocks(B2))
    else
        return Matrix(B1) * Matrix(B2)
    end
end

function _check_matmul_dims(A::AbstractMatrix, B::AbstractMatrix)
    # match error message from LinearAlgebra
    size(A, 2) == size(B, 1) || throw(DimensionMismatch(
        "A has dimensions $(size(A)) but B has dimensions $(size(B))"
    ))
end

function Base.:*(B::BlockDiagonal, M::AbstractMatrix)
    _check_matmul_dims(B, M)
    ed = 0
    d = map(blocks(B)) do block
        st = ed + 1  # start
        ed += size(block, 2)  # end
        return block * M[st:ed, :]
    end
    return reduce(vcat, d)::Matrix
end

function Base.:*(M::AbstractMatrix, B::BlockDiagonal)
    _check_matmul_dims(M, B)
    ed = 0
    d = map(blocks(B)) do block
        st = ed + 1  # start
        ed += size(block, 1)  # end
        return M[:, st:ed] * block
    end
    return reduce(hcat, d)::Matrix
end

# Diagonal
function Base.:*(B::BlockDiagonal, M::Diagonal)::BlockDiagonal
    _check_matmul_dims(B, M)
    A = copy(B)
    d = diag(M)
    col = 1
    for (p, block) in enumerate(blocks(B))
        ncols = size(block, 2)
        cols = col:(col + ncols-1)
        for (j, c) in enumerate(cols)
            A[Block(p)][:, j] *= d[c]
        end
        col += ncols
    end
    return A
end

function Base.:*(M::Diagonal, B::BlockDiagonal)::BlockDiagonal
    _check_matmul_dims(M, B)
    A = copy(B)
    d = diag(M)
    row = 1
    for (p, block) in enumerate(blocks(B))
        nrows = size(block, 1)
        rows = row:(row + nrows-1)
        for (i, r) in enumerate(rows)
            A[Block(p)][i, :] *= d[r]
        end
        row += nrows
    end
    return A
end

## Division
Base.:/(B::BlockDiagonal, n::Number) = BlockDiagonal(blocks(B) ./ n)
