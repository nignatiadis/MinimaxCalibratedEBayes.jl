using Documenter, MinimaxCalibratedEBayes
const MCEB = MinimaxCalibratedEBayes

makedocs(;
    modules=[MinimaxCalibratedEBayes],
    format=  Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/nignatiadis/MinimaxCalibratedEBayes.jl/blob/{commit}{path}#L{line}",
    sitename="MinimaxCalibratedEBayes.jl",
    authors="Nikos Ignatiadis <nikos.ignatiadis01@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/nignatiadis/MinimaxCalibratedEBayes.jl",
)
