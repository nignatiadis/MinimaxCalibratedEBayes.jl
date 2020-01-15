module MinimaxCalibratedEBayes

using Reexport

@reexport using StatsBase,
                Distributions,
                EBayes

using RecipesBase
using QuadGK

import StatsBase:Histogram,
                 binindex,
                 midpoints,
                 response

import Statistics:var

import Distributions:cf,
                     pdf,
                     location

import Base:step,
            first,
            last,
            length,
            zero

import Base.Broadcast: broadcastable


include("marginal_binning.jl")
include("marginal_kde.jl")
include("marginalize.jl")
include("inference_targets.jl")
include("butucea_comte.jl")


export MCEBHistogram,
       DiscretizedStandardNormalSample,
       DiscretizedAffineEstimator,
       marginalize,
       SincKernel,
       DeLaValleePoussinKernel,
       EBayesTarget,
       MarginalDensityTarget,
       riesz_representer,
       ButuceaComte

end # module
