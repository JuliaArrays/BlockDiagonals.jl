# BlockDiagonals.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://invenia.github.io/BlockDiagonals.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://invenia.github.io/BlockDiagonals.jl/dev)
[![CI](https://github.com/invenia/BlockDiagonals.jl/workflows/CI/badge.svg)](https://github.com/Invenia/BlockDiagonals.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/invenia/BlockDiagonals.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/invenia/BlockDiagonals.jl)
[![code style blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

Functionality for working efficiently with [block diagonal matrices](https://en.wikipedia.org/wiki/Block_matrix#Block_diagonal_matrices).

Construct a `BlockDiagonal` matrix by passing in only the non-zero blocks on the diagonal, and use it as a regular matrix
```julia
julia> using BlockDiagonals

julia> bm = BlockDiagonal([rand(2, 3), ones(3, 2)])
5×5 BlockDiagonal{Float64, Matrix{Float64}}:
 0.289276  0.994487  0.287658  0.0  0.0
 0.659821  0.334724  0.780973  0.0  0.0
 0.0       0.0       0.0       1.0  1.0
 0.0       0.0       0.0       1.0  1.0
 0.0       0.0       0.0       1.0  1.0

julia> v = ones(5);

julia> bm * v
5-element Vector{Float64}:
 1.5714204086879524
 1.7755185907265039
 2.0
 2.0
 2.0

julia> svd(bm)
SVD{Float64, Float64, Matrix{Float64}}
U factor:
5×4 Matrix{Float64}:
  0.0      -0.70666   -0.707553   0.0
  0.0      -0.707553   0.70666    0.0
 -0.57735   0.0        0.0       -0.57735
 -0.57735   0.0        0.0        0.788675
 -0.57735   0.0        0.0       -0.211325
singular values:
4-element Vector{Float64}:
 2.4494897427831783
 1.3801377610748038
 0.6387290946600256
 0.0
Vt factor:
4×5 Matrix{Float64}:
  0.0        0.0        0.0       -0.707107  -0.707107
 -0.486385  -0.680801  -0.547667   0.0        0.0
  0.409549  -0.731322   0.545379   0.0        0.0
  0.0        0.0        0.0       -0.707107   0.707107
```

Additional functionality includes
```julia
julia> nblocks(bm)
2

julia> blocks(bm)
2-element Vector{Matrix{Float64}}:
 [0.2892758623451861 0.9944869494674535 0.2876575968753128; 0.6598212430288488 0.33472423873340906 0.780973108964246]
 [1.0 1.0; 1.0 1.0; 1.0 1.0]

julia> blocksizes(bm)
2-element Vector{Tuple{Int64, Int64}}:
 (2, 3)
 (3, 2)

julia> blocksize(bm, 1)
(2, 3)
```
