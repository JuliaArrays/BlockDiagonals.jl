module BlockDiagonals

using FillArrays: Zeros
using LinearAlgebra

export BlockDiagonal, blocks
export blocksize, blocksizes, nblocks

include("blockdiagonal.jl")
include("base_maths.jl")
include("linalg.jl")

end  # end module
