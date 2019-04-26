module BlockDiagonals

using BlockArrays
using BlockArrays: AbstractBlockSizes, BlockSizes
using FillArrays
using LinearAlgebra

export BlockDiagonal, blocks
# reexport core interfaces from BlockArrays
export Block, BlockSizes, blocksize, blocksizes, nblocks

include("blockdiagonal.jl")
include("base_maths.jl")
include("linalg.jl")

end  # end module
