module BlockDiagonals

using BlockArrays
using BlockArrays: AbstractBlockSizes, BlockSizes
using FillArrays
using LinearAlgebra

export BlockDiagonal, blocks

"""
    BlockDiagonal{T, V<:AbstractMatrix{T}} <: AbstractBlockMatrix{T}

A BlockMatrix with square blocks on the diagonal, and zeros off the diagonal.
"""
struct BlockDiagonal{T, S<:AbstractBlockSizes} <: AbstractBlockMatrix{T}
    blocks::Vector{<:AbstractMatrix{T}}
    block_sizes::S
    function BlockDiagonal(blocks::Vector{<:AbstractMatrix{T}}, sizes::S) where {T, S<:AbstractBlockSizes}
        all(is_square, blocks) || throw(ArgumentError("All blocks must be square."))
        # mapreduce would give Array of Tuples; want Array of Arrays
        return new{T, S}(blocks, sizes)
    end
end

function BlockDiagonal(blocks::AbstractVector{<:AbstractMatrix})
    # mapreduce would give Array of Tuples; want Array of Arrays
    bs = BlockSizes([[first(size(b)), last(size(b))] for b in blocks]...)
    return BlockDiagonal(blocks, bs)
end

is_square(A::AbstractMatrix) = size(A, 1) === size(A, 2)
blocks(b::BlockDiagonal) = b.blocks

# AbstractBlockMatrix interface
BlockArrays.blocksizes(b::BlockDiagonal) = b.block_sizes
function BlockArrays.blocksize(b::BlockDiagonal, i::Int, j::Int)
    return first(size(blocks(b)[i])), last(size(blocks(b)[j]))
end
BlockArrays.nblocks(b::BlockDiagonal) = length(blocks(b)), length(blocks(b))
BlockArrays.nblocks(b::BlockDiagonal, dims::Integer) = length(blocks(b))
function BlockArrays.getblock(b::BlockDiagonal{T}, i::Int, j::Int) where T
    return i == j ? blocks(b)[i] : Zeros{T}(blocksize(b, i, j))
end
function BlockArrays.setblock!(b::BlockDiagonal{T, V}, v::V, p::Int, q::Int) where {T, V}
    if p != q
        throw(brgumentError("Cannot set off-diagonal block ($p, $q) to a nonzero value."))
    end
    if blocksize(b, p, q) != size(v)
        throw(DimensionMismatch(string(
            "Cannot set block of size $(blocksize(b, p, q)) to block of size $(size(v))"
        )))
    end
    blocks(b)[p] = v
end


Base.Matrix(b::BlockDiagonal) = cat(blocks(b)...; dims=(1, 2))
Base.size(b::BlockDiagonal) = sum(x -> size(x, 1), blocks(b)), sum(x -> size(x, 2), blocks(b))
Base.similar(B::BlockDiagonal) = BlockDiagonal(similar.(blocks(B)))

function Base.getindex(b::BlockDiagonal{T}, i::Int, j::Int) where T
    cols = [size(bb, 2) for bb in blocks(b)]
    rows = [size(bb, 1) for bb in blocks(b)]
    c = 0
    while j > 0
        c += 1
        j -= cols[c]
    end
    i = i - sum(rows[1:(c - 1)])
    (i <= 0 || i > rows[c]) && return zero(T)
    return blocks(b)[c][i, end + j]
end

function Base.isapprox(b1::BlockDiagonal, b2::BlockDiagonal; atol::Real=0)
    return isequal_blocksizes(b1, b2) && all(isapprox.(blocks(b1), blocks(b2); atol=atol))
end

function Base.isapprox(b::BlockDiagonal, m::AbstractMatrix; atol::Real=0)
    return isapprox(Matrix(m), b, atol=atol)
end

function Base.isapprox(m::AbstractMatrix, b::BlockDiagonal; atol::Real=0)
    return isapprox(m, Matrix(b), atol=atol)
end

function isequal_blocksizes(b1::BlockDiagonal, b2::BlockDiagonal)::Bool
    size(b1) === size(b2) || return false
    for m in eachindex(blocks(b1))
        size(blocks(b1)[m]) === size(blocks(b2)[m]) || return false
    end
    return true
end

## Addition
function Base.:+(b1::BlockDiagonal, b2::BlockDiagonal)
    if size(b1) == size(b2) && size.(blocks(b1)) == size.(blocks(b2))
        return BlockDiagonal(blocks(b1) .+ blocks(b2))
    else
        return Matrix(b1) + Matrix(b2)
    end
end

Base.:+(m::AbstractMatrix, b::BlockDiagonal) = b + m
function Base.:+(b::BlockDiagonal, m::AbstractMatrix)
    !isdiag(m) && return Matrix(b) + m
    size(b) != size(m) && throw(DimensionMismatch("Can't add matrices of different sizes."))
    d = diag(m)
    si = 1
    sj = 1
    nb = copy(blocks(b))
    for i in 1:length(nb)
        s = size(nb[i])
        nb[i] += @view m[si:s[1] + si - 1, sj:s[2] + sj - 1]
        si += s[1]
        sj += s[2]
    end
    return BlockDiagonal(nb)
end

# function +(A::BlockDiagonal, B::StridedMatrix)::Matrix
#     size(A) != size(B) && throw(DimensionMismatch("dimensions must match"))
#     C = copy(B)
#     row = 1
#     for (j, block) in enumerate(blocks(A))
#         rows = row:row + size(block, 1) - 1
#         C[rows, rows] .+= block
#         row += size(block, 1)
#     end
#     return C
# end

Base.:+(m::Diagonal, b::BlockDiagonal) = b + m
function Base.:+(b::BlockDiagonal, m::Diagonal)
    size(b) != size(m) && throw(DimensionMismatch("Can't add matrices of different sizes."))
    C = similar(b)
    row = 1
    for (j, block) in enumerate(blocks(b))
        rows = row:row + size(block, 1) - 1
        block_C = blocks(C)[j]
        block_C .= block
        for (k, r) in enumerate(rows)
            block_C[k, k] += diag(m)[r]
        end
        row += size(block, 1)
    end
    return C
end

Base.:+(m::UniformScaling, b::BlockDiagonal) = b + m
function Base.:+(b::BlockDiagonal, m::UniformScaling)
    return BlockDiagonal([block + m for block in blocks(b)])
end


## Multiplication
Base.:*(b::BlockDiagonal, n::Real) = BlockDiagonal(n .* blocks(b))
Base.:*(n::Real, b::BlockDiagonal) = b * n

function Base.:*(b::BlockDiagonal, m::BlockDiagonal)
    if size(b) == size(m) && size.(blocks(b)) == size.(blocks(m))
        return BlockDiagonal(blocks(b) .* blocks(m))
    else
        return Matrix(b) * Matrix(m)
    end
end

function Base.:*(b::BlockDiagonal, m::AbstractMatrix)
    if size(b, 2) !== size(m, 1)
        throw(DimensionMismatch("Cannot multiply matrices of size $(size(b)) and $(size(m))."))
    end
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 2) - 1
        push!(d, block * m[st:ed, :])
        st = ed + 1
    end
    return reduce(vcat, d)
end

function Base.:*(m::AbstractMatrix, b::BlockDiagonal)
    if size(b, 1) != size(m, 2)
        throw(DimensionMismatch("Cannot multiply matrices of size $(size(b)) and $(size(m))."))
    end
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 1) - 1
        push!(d, m[:, st:ed] * block)
        st = ed + 1
    end
    return reduce(hcat, d)
end

## Division
Base.:/(b::BlockDiagonal, n::Real) = BlockDiagonal(blocks(b) ./ n)

## LinearAlgebra methods
function LinearAlgebra.cholesky(B::BlockDiagonal)
    C = BlockDiagonal(map(b -> cholesky(b).U, blocks(B)))
    return Cholesky(C, 'U', 0)
end

function LinearAlgebra.eigvals(b::BlockDiagonal)
    eigs = mapreduce(eigvals, vcat, blocks(b))
    eigs isa Vector{<:Complex} && return eigs
    return sort(eigs)
end


LinearAlgebra.transpose(b::BlockDiagonal) = BlockDiagonal(transpose.(blocks(b)))
LinearAlgebra.adjoint(b::BlockDiagonal) = BlockDiagonal(adjoint.(blocks(b)))

LinearAlgebra.det(b::BlockDiagonal) = prod(det, blocks(b))
LinearAlgebra.logdet(b::BlockDiagonal) = sum(logdet, blocks(b))

LinearAlgebra.tr(b::BlockDiagonal) = sum(tr, blocks(b))
LinearAlgebra.diag(b::BlockDiagonal) = mapreduce(diag, vcat, blocks(b))

# Make getproperty on a Cholesky factorized BlockDiagonal return another BlockDiagonal
# where each block is an upper or lower triangular matrix. This ensures that optimizations
# for BlockDiagonal matrices are preserved, though it comes at the cost of reallocating
# a vector of triangular wrappers on each call.
function Base.getproperty(C::Cholesky{T, BlockDiagonal{T}}, x::Symbol) where T
    B = getfield(C, :factors)
    uplo = getfield(C, :uplo)
    f = if x === :U
        uplo === 'U' ? UpperTriangular : (X -> UpperTriangular(X'))
    elseif x === :L
        uplo === 'L' ? LowerTriangular : (X -> LowerTriangular(X'))
    elseif x === :UL
        uplo === 'U' ? UpperTriangular : LowerTriangular
    else
        return getfield(C, x)
    end
    return BlockDiagonal{T}(map(f, blocks(B)))
end

end  # end module
