using Documenter, BlockDiagonals

makedocs(;
    modules=[BlockDiagonals],
    format=Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/invenia/BlockDiagonals.jl/blob/{commit}{path}#L{line}",
    sitename="BlockDiagonals.jl",
    authors="Invenia Technical Computing",
)

deploydocs(;
    repo="github.com/invenia/BlockDiagonals.jl",
)
