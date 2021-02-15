using BlockDiagonals
using Documenter

makedocs(;
    modules=[BlockDiagonals],
    format=Documenter.HTML(prettyurls=false),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/invenia/BlockDiagonals.jl/blob/{commit}{path}#{line}",
    sitename="BlockDiagonals.jl",
    authors="Invenia Technical Computing",
    strict=true,
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/invenia/BlockDiagonals.jl",
    push_preview=true,
)
