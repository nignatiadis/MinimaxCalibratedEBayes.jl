module MinimaxCalibratedEBayes

using Reexport

@reexport using StatsBase

import StatsBase:Histogram,
                 binindex

import Base:step,
            first,
            last
            
include("marginal_binning.jl")

export MCEBHistogram,
       BinnedCalibrator

end # module
