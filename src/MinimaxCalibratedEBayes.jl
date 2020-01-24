module MinimaxCalibratedEBayes

using Reexport

@reexport using StatsBase,
                Distributions,
                EBayes

using RecipesBase
using QuadGK
using JuMP
using LinearAlgebra

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
include("inference_targets.jl")
include("normal_rules.jl")
include("butucea_comte.jl")
include("prior_convex_class.jl")


export MCEBHistogram,
       DiscretizedStandardNormalSample,
       DiscretizedAffineEstimator,
       marginalize,
       SincKernel,
       DeLaValleePoussinKernel,
       EBayesTarget,
       MarginalDensityTarget,
       PriorDensityTarget,
       riesz_representer,
       ButuceaComte,
       GaussianMixturePriorClass,
       worst_case_bias,
       SteinMinimaxEstimator

end # module
