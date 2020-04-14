using Documenter, MinimaxCalibratedEBayes
const MCEB = MinimaxCalibratedEBayes

makedocs(;
    modules=[MinimaxCalibratedEBayes],
    format=  Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => Any[
            "man/targets.md",
            "man/prior_classes.md",
            "man/samples.md",
            "man/pilot.md",
            "man/quasiminimax_linear.md",
            "man/inference.md"],
         "Tutorials" => "tutorials.md",
    ],
    repo="https://github.com/nignatiadis/MinimaxCalibratedEBayes.jl/blob/{commit}{path}#L{line}",
    sitename="MinimaxCalibratedEBayes.jl",
    authors="Nikos Ignatiadis <nikos.ignatiadis01@gmail.com>",
    assets=String[],
)

if get(ENV, "CI", nothing) == "true"
    let start_dir = pwd()
        cd(@__DIR__)
        mkpath("build/tutorials")
        tutorial_names = ["linear_estimation.html"; "data_analysis.html"]
        in_path_names = joinpath.("tutorials", tutorial_names)
        out_path_names = joinpath.("build/tutorials",tutorial_names)
        cp.(in_path_names, out_path_names, force=true)
        cd(start_dir)
    end
end
 
deploydocs(;
    repo="github.com/nignatiadis/MinimaxCalibratedEBayes.jl",
)
