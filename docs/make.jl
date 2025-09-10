using Documenter, BlockDiagonals

makedocs(;
    modules = [BlockDiagonals],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = ["Home" => "index.md"],
    repo = "https://github.com/JuliaArrays/BlockDiagonals.jl/blob/{commit}{path}#L{line}",
    sitename = "BlockDiagonals.jl",
)

deploydocs(; repo = "github.com/JuliaArrays/BlockDiagonals.jl")
