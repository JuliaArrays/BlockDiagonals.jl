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
Base.:+(B::BlockDiagonal, M::AbstractMatrix) = isdiag(M) ? B + Diagonal(M) : Matrix(B) + M

function Base.:+(B::BlockDiagonal, M::StridedMatrix)
    size(B) == size(M) || throw(DimensionMismatch("dimensions must match"))
    if isdiag(M)
        return B + Diagonal(M)
    end
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

function Base.:+(B::BlockDiagonal, M::Diagonal)::BlockDiagonal
    size(B) == size(M) || throw(DimensionMismatch("dimensions must match"))
    A = copy(B)
    d = diag(M)
    row = 1
    for (p, block) in enumerate(blocks(B))
        nrows = size(block, 1)
        rows = row:(row + nrows-1)
        for (i, r) in enumerate(rows)
            getblock(A, p)[i, i] += d[r]
        end
        row += nrows
    end
    return A
end

Base.:+(M::UniformScaling, B::BlockDiagonal) = B + M
function Base.:+(B::BlockDiagonal, M::UniformScaling)
    return BlockDiagonal([block + M for block in blocks(B)])
end

## Subtraction
Base.:-(B::BlockDiagonal) = BlockDiagonal(.-(blocks(B)))
Base.:-(M::AbstractMatrix, B::BlockDiagonal) =  M + -B
Base.:-(B::BlockDiagonal, M::AbstractMatrix) =  -M + B
Base.:-(B::BlockDiagonal, B2::BlockDiagonal) =  B + (-B2)

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

function _check_matmul_dims(A::AbstractMatrix, B::AbstractVecOrMat)
    # match error message from LinearAlgebra
    size(A, 2) == size(B, 1) || throw(DimensionMismatch(
        "A has dimensions $(size(A)) but B has dimensions $(size(B))"
    ))
end

_mulblocksizes(bblocks, ::AbstractVector) = size.(bblocks, 1)
function _mulblocksizes(bblocks, M::AbstractMatrix)
    return zip(size.(bblocks, 1), Base.Iterators.repeated(size(M, 2), length(bblocks)))
end

# avoid ambiguities arising with AbstractVecOrMat
Base.:*(B::BlockDiagonal, x::AbstractVector) = _mul(B, x)
Base.:*(B::BlockDiagonal, X::AbstractMatrix) = _mul(B, X)

function _mul(B::BlockDiagonal{T}, x::AbstractVecOrMat{T2}) where {T, T2}
    _check_matmul_dims(B, x)
    bblocks = blocks(B)
    new_blocksizes = _mulblocksizes(bblocks, x)
    d = similar.(bblocks, promote_type(T, T2), new_blocksizes)
    ed = 0
    @inbounds @views for (p, block) in enumerate(bblocks)
        st = ed + 1  # start
        ed += size(block, 2)  # end
        mul!(d[p], block, selectdim(x, 1, st:ed))
    end
    return reduce(vcat, d)
end

function Base.:*(M::AbstractMatrix{T}, B::BlockDiagonal{T2}) where {T, T2}
    _check_matmul_dims(M, B)
    bblocks = blocks(B)
    new_blocksizes = zip(fill(size(M, 1), length(bblocks)), size.(bblocks, 2))
    d = similar.(bblocks, promote_type(T, T2), new_blocksizes)
    ed = 0
    @inbounds @views for (p, block) in enumerate(bblocks)
        st = ed + 1  # start
        ed += size(block, 1)  # end
        mul!(d[p], M[:, st:ed], block)
    end
    return reduce(hcat, d)
end

# Diagonal
function Base.:*(B::BlockDiagonal{T}, M::Diagonal{T2})::BlockDiagonal where {T, T2}
    _check_matmul_dims(B, M)
    A = similar(B, promote_type(T, T2))
    d = parent(M)
    col = 1
    @inbounds @views for (p, block) in enumerate(blocks(B))
        ncols = size(block, 2)
        cols = col:(col + ncols-1)
        for (j, c) in enumerate(cols)
            mul!(getblock(A, p)[:, j], block[:, j], d[c])
        end
        col += ncols
    end
    return A
end

function Base.:*(M::Diagonal{T}, B::BlockDiagonal{T2})::BlockDiagonal where {T, T2}
    _check_matmul_dims(M, B)
    A = similar(B, promote_type(T, T2))
    d = parent(M)
    row = 1
    @inbounds @views for (p, block) in enumerate(blocks(B))
        nrows = size(block, 1)
        rows = row:(row + nrows-1)
        for (i, r) in enumerate(rows)
            mul!(getblock(A, p)[i, :], block[i, :], d[r])
        end
        row += nrows
    end
    return A
end

function Base.:*(vt::Adjoint{T,<: AbstractVector}, B::BlockDiagonal{T}) where {T}
    return (B' * parent(vt))'
end

function Base.:*(vt::Transpose{T,<: AbstractVector}, B::BlockDiagonal{T}) where {T}
    return transpose(transpose(B) * parent(vt))
end

## Division
Base.:/(B::BlockDiagonal, n::Number) = BlockDiagonal(blocks(B) ./ n)
