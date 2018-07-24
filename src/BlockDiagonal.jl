module BlockDiagonal

export BlockDiagonal, blocks

import Base: size, diag, Matrix, chol, show, display, *, +, /, isapprox, transpose,
Ac_mul_B, A_mul_Bc, getindex, ctranspose, det, eigvals, trace

type BlockDiagonal{T} <: AbstractMatrix{T}
    blocks::Vector{<:AbstractMatrix{T}}
end

blocks(b::BlockDiagonal) = b.blocks
diag(b::BlockDiagonal) = vcat(diag.(blocks(b))...)

function size(b::BlockDiagonal)
    sizes = size.(blocks(b))
    return sum.(([s[1] for s in sizes], [s[2] for s in sizes]))
end

function isapprox(b1::BlockDiagonal, b2::BlockDiagonal; atol::Real=0)
    size(b1) != size(b2) && return false
    !all(size.(blocks(b1)) == size.(blocks(b2))) && return false
    return all(isapprox.(blocks(b1), blocks(b2), atol=atol))
end
function isapprox(b1::BlockDiagonal, b2::AbstractMatrix; atol::Real=0)
    return isapprox(Matrix(b1), b2, atol=atol)
end
function isapprox(b1::AbstractMatrix, b2::BlockDiagonal; atol::Real=0)
    return isapprox(b1, Matrix(b2), atol=atol)
end

Matrix(b::BlockDiagonal) = cat([1, 2], blocks(b)...)

chol(b::BlockDiagonal) = BlockDiagonal(chol.(blocks(b)))
det(b::BlockDiagonal) = prod(det.(blocks(b)))
function eigvals(b::BlockDiagonal)
    eigs = vcat(eigvals.(blocks(b))...)
    !isa(eigs, Vector{<:Complex}) && return sort(eigs)
    return eigs
end
trace(b::BlockDiagonal) = sum(trace.(blocks(b)))

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

function (*)(b::BlockDiagonal, m::AbstractMatrix)
    size(b, 2) != size(m, 1) && throw(
        DimensionMismatch("A has dimensions $(size(b)) but B has dimensions $(size(m))")
    )
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 2) - 1
        push!(d, block * m[st:ed, :])
        st = ed + 1
    end
    return vcat(d...)
end
function (*)(m::AbstractMatrix, b::BlockDiagonal)
    size(b, 1) != size(m, 2) && throw(
        DimensionMismatch("A has dimensions $(size(b)) but B has dimensions $(size(m))")
    )
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 1) - 1
        push!(d, m[:, st:ed] * block)
        st = ed + 1
    end
    return hcat(d...)
end
function (*)(b::BlockDiagonal, m::BlockDiagonal)
    if size(b) == size(m) && size.(blocks(b)) == size.(blocks(m))
        return BlockDiagonal(blocks(b) .* blocks(m))
    else
        Matrix(b) * Matrix(m)
    end
end

function Ac_mul_B(b::BlockDiagonal, m::BlockDiagonal)
    if size(b) == size(m) && size.(blocks(b)) == size.(blocks(m))
        return BlockDiagonal(Ac_mul_B.(blocks(b), blocks(m)))
    else
        Ac_mul_B(Matrix(b), Matrix(m))
    end
end
function Ac_mul_B(b::BlockDiagonal, m::AbstractMatrix)
    size(b, 1) != size(m, 1) && throw(
        DimensionMismatch("A has dimensions $(size(b)) but B has dimensions $(size(m'))")
    )
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 1) - 1
        push!(d, Ac_mul_B(block, m[st:ed, :]))
        st = ed + 1
    end
    return vcat(d...)
end
function Ac_mul_B(m::AbstractMatrix, b::BlockDiagonal)
    size(b, 1) != size(m, 1) && throw(
        DimensionMismatch("A has dimensions $(size(b)) but B has dimensions $(size(m'))")
    )
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 1) - 1
        push!(d, Ac_mul_B(m[st:ed, :], block))
        st = ed + 1
    end
    return hcat(d...)
end
function A_mul_Bc(b::BlockDiagonal, m::BlockDiagonal)
    if size(b) == size(m) && size.(blocks(b)) == size.(blocks(m))
        return BlockDiagonal(A_mul_Bc.(blocks(b), blocks(m)))
    else
        A_mul_Bc(Matrix(b), Matrix(m))
    end
end
function A_mul_Bc(b::BlockDiagonal, m::AbstractMatrix)
    size(b, 2) != size(m, 2) && throw(
        DimensionMismatch("A has dimensions $(size(b)) but B has dimensions $(size(m'))")
    )
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 2) - 1
        push!(d, A_mul_Bc(block, m[:, st:ed]))
        st = ed + 1
    end
    return vcat(d...)
end
function A_mul_Bc(m::AbstractMatrix, b::BlockDiagonal)
    size(b, 2) != size(m, 2) && throw(
        DimensionMismatch("A has dimensions $(size(b)) but B has dimensions $(size(m'))")
    )
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 2) - 1
        push!(d, A_mul_Bc(m[:, st:ed], block))
        st = ed + 1
    end
    return hcat(d...)
end
(*)(b::BlockDiagonal, n::Real) = BlockDiagonal(n .* blocks(b))
(*)(n::Real, b::BlockDiagonal) = b * n

(/)(b::BlockDiagonal, n::Real) = BlockDiagonal(blocks(b) ./ n)

function (+)(b::BlockDiagonal, m::AbstractMatrix)
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

(+)(m::AbstractMatrix, b::BlockDiagonal) = b + m
function (+)(b::BlockDiagonal, m::BlockDiagonal)
    if size(b) == size(m) && size.(blocks(b)) == size.(blocks(m))
        return BlockDiagonal(blocks(b) .+ blocks(m))
    else
        return Matrix(b) + Matrix(m)
    end
end

end # end module
