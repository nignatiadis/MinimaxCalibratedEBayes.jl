module MinimaxCalibratedEBayes

using Reexport

@reexport using StatsBase,
                Distributions,
                EBayes

using RecipesBase

import StatsBase:Histogram,
                 binindex,
                 midpoints

import Statistics:var

import Distributions:cf, pdf

import Base:step,
            first,
            last,
            length,
            zero

import Base.Broadcast: broadcastable

include("marginal_binning.jl")
include("marginal_kde.jl")
include("marginalize.jl")

export MCEBHistogram,
       DiscretizedStandardNormalSample,
       DiscretizedAffineEstimator,
       marginalize,
       SincKernel,
       DeLaValleePoussinKernel

end # module
