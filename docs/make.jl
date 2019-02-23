using Documenter, nGraph

makedocs(
    modules = [nGraph],
    format = :html,
    checkdocs = :exports,
    sitename = "nGraph.jl",
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/hildebrandmw/nGraph.jl.git",
)
