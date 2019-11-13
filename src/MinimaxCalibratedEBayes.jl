module MinimaxCalibratedEBayes

using Reexport

@reexport using StatsBase

include("marginal_binning.jl")

export BinnedCalibrator

end # module
