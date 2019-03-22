using Documenter, nGraph

makedocs(
    modules = [nGraph],
    format = :html,
    checkdocs = :exports,
    html_prettyurls = get(ENV, "CI", nothing) == "true",
    sitename = "nGraph.jl",
    pages = Any[
        "index.md",
        "ngraph" => [
            "ngraph/types.md",
        ],
    ]
)

deploydocs(
    repo = "github.com/hildebrandmw/nGraph.jl.git",
)
