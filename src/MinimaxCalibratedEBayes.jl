module MinimaxCalibratedEBayes

using Reexport

@reexport using StatsBase,
                Distributions,
                EBayes

import StatsBase:Histogram,
                 binindex

import Statistics:var

import Distributions:cf, pdf

import Base:step,
            first,
            last

import Base.Broadcast: broadcastable

include("marginal_binning.jl")
include("marginalize.jl")

export MCEBHistogram,
       DiscretizedStandardNormalSample,
       BinnedCalibrator,
       marginalize

end # module
