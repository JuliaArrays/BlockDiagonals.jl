module BlockDiagonals

export BlockDiagonal, blocks

import Base: size, diag, Matrix, chol, show, display, *, +, /, isapprox, transpose,
    Ac_mul_B, A_mul_Bc, getindex, ctranspose, det, logdet, eigvals, trace, similar

import Base.LinAlg: A_mul_B!

"""
    BlockDiagonal{T, V<:AbstractMatrix{T}} <: AbstractMatrix{T}


"""
struct BlockDiagonal{T, V<:AbstractMatrix{T}} <: AbstractMatrix{T}
    blocks::Vector{V}
    function BlockDiagonal(blocks::Vector{<:V}) where {T, V<:AbstractMatrix{T}}
        @assert all(is_square, blocks)
        return new{T, V}(blocks)
    end
end

@inline blocks(b::BlockDiagonal) = b.blocks
diag(b::BlockDiagonal) = vcat(diag.(blocks(b))...)

function have_equal_block_sizes(A::BlockDiagonal, B::BlockDiagonal)::Bool
    size(A) == size(B) || return false
    for m in eachindex(blocks(A))
        size(blocks(A)[m]) == size(blocks(B)[m]) || return false
    end
    return true
end
@inline is_square(A::AbstractMatrix) = size(A, 1) == size(A, 2)

size(b::BlockDiagonal) = sum(x->size(x, 1), blocks(b)), sum(x->size(x, 2), blocks(b))

similar(B::BlockDiagonal) = BlockDiagonal(similar.(blocks(B)))

Matrix(b::BlockDiagonal) = cat([1, 2], blocks(b)...)

chol(b::BlockDiagonal) = BlockDiagonal(chol.(blocks(b)))
det(b::BlockDiagonal) = prod(det, blocks(b))
logdet(b::BlockDiagonal) = sum(logdet, blocks(b))
function eigvals(b::BlockDiagonal)
    eigs = vcat(eigvals.(blocks(b))...)
    !isa(eigs, Vector{<:Complex}) && return sort(eigs)
    return eigs
end
trace(b::BlockDiagonal) = sum(trace, blocks(b))

transpose(b::BlockDiagonal) = BlockDiagonal(transpose.(blocks(b)))
ctranspose(b::BlockDiagonal) = BlockDiagonal(ctranspose.(blocks(b)))

function getindex(b::BlockDiagonal{T}, i::Int, j::Int) where T
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



########################### Addition #############################

function +(A::BlockDiagonal, B::BlockDiagonal)::Union{Matrix, BlockDiagonal}
    size(A) != size(B) && throw(DimensionMismatch("dimensions must match"))
    return size(blocks(A)) == size(blocks(B)) ?
        BlockDiagonal(blocks(A) .+ blocks(B)) :
        Matrix(A) + Matrix(B)
end

function +(A::BlockDiagonal, B::StridedMatrix)::Matrix
    size(A) != size(B) && throw(DimensionMismatch("dimensions must match"))
    C = copy(B)
    row = 1
    for (j, block) in enumerate(blocks(A))
        rows = row:row + size(block, 1) - 1
        C[rows, rows] .+= block
        row += size(block, 1)
    end
    return C
end
@inline +(A::StridedMatrix, B::BlockDiagonal) = B + A

function +(A::BlockDiagonal, B::Diagonal)
    size(A) != size(B) && throw(DimensionMismatch("dimensions must match"))
    C = similar(A)
    row = 1
    for (j, block) in enumerate(blocks(A))
        rows = row:row + size(block, 1) - 1
        block_C = blocks(C)[j]
        block_C .= block
        for (k, r) in enumerate(rows)
            block_C[k, k] += B.diag[r]
        end
        row += size(block, 1)
    end
    return C
end
@inline +(A::Diagonal, B::BlockDiagonal) = B + A

+(A::BlockDiagonal, B::UniformScaling) = BlockDiagonal([block + B for block in blocks(A)])
+(A::UniformScaling, B::BlockDiagonal) = B + A



################# BlockDiagonal * BlockDiagonal ##################

function A_mul_B!(C::BlockDiagonal, A::BlockDiagonal, B::BlockDiagonal)::BlockDiagonal
    size(A) != size(B) && throw(DimensionMismatch("dimensions must match"))
    size(C) != size(B) && throw(DimensionMismatch("dimensions must match"))
    have_equal_block_sizes(A, B) || throw(DimensionMismatch("block dimensions must match"))
    have_equal_block_sizes(C, A) || throw(DimensionMismatch("block dimensions must match"))
    for m in eachindex(blocks(C))
        A_mul_B!(blocks(C)[m], blocks(A)[m], blocks(B)[m])
    end
    return C
end
function A_mul_B!(C::StridedMatrix, A::BlockDiagonal, B::BlockDiagonal)
    size(A) != size(B) && throw(DimensionMismatch("dimensions must match"))
    size(C) != size(B) && throw(DimensionMismatch("dimensions must match"))
    return A_mul_B!(C, Matrix(A), Matrix(B))
end
function *(A::BlockDiagonal, B::BlockDiagonal)::Union{Matrix, BlockDiagonal}
    return A_mul_B!(have_equal_block_sizes(A, B) ? similar(A) : Matrix(similar(A)), A, B)
end



################### BlockDiagonal * Matrix ####################

function A_mul_B!(C::StridedMatrix, A::BlockDiagonal, B::StridedMatrix)
    size(A, 2) != size(B, 1) && throw(
        DimensionMismatch("A has dimensions $(size(A)) but B has dimensions $(size(B))")
    )
    size(C) != (size(A, 1), size(B, 2)) && throw(
        DimensionMismatch("A has size $(size(A)), B has size $(size(B)), " *
            "C has size $(size(C))")
    )
    row = 1
    for (j, block) in enumerate(blocks(A))
        rows = row:row + size(block, 1) - 1
        A_mul_B!(view(C, rows, :), block, view(B, rows, :))
        row += size(block, 1)
    end
    return C
end
function *(A::BlockDiagonal{T}, B::StridedMatrix{V}) where {T, V}
    return A_mul_B!(Matrix{promote_type(T, V)}(size(A, 1), size(B, 2)), A, B)
end



################### Matrix * BlockDiagonal #####################

function A_mul_B!(C::StridedMatrix, A::StridedMatrix, B::BlockDiagonal)
    size(A, 2) != size(B, 1) && throw(
        DimensionMismatch("A has dimensions $(size(A)) but B has dimensions $(size(B))")
    )
    size(C) != (size(A, 1), size(B, 2)) && throw(
        DimensionMismatch("A has size $(size(A)), B has size $(size(B)), " *
            "C has size $(size(C))")
    )
    col = 1
    for (j, block) in enumerate(blocks(B))
        cols = col:col + size(block, 2) - 1
        A_mul_B!(view(C, :, cols), view(A, :, cols), block)
        col += size(block, 1)
    end
    return C
end

end # end module
