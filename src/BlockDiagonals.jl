module BlockDiagonals

using Base: @propagate_inbounds
using ChainRulesCore
using FillArrays: Zeros
using FiniteDifferences
using LinearAlgebra

export BlockDiagonal, blocks
export blocksize, blocksizes, nblocks

include("blockdiagonal.jl")
include("base_maths.jl")
include("chainrules.jl")
include("linalg.jl")
include("finitedifferences.jl")

end  # end module
