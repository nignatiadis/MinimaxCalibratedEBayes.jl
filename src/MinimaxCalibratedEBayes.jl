module MinimaxCalibratedEBayes

using Reexport

@reexport using StatsBase,
                Distributions,
                EBayes

using RecipesBase
using Plots
using QuadGK
using JuMP
using KernelDensity
using LinearAlgebra
using Roots

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

include("bias_adjusted_ci.jl")
include("marginal_binning.jl")
include("marginal_kde.jl")
include("inference_targets.jl")
include("normal_rules.jl")
include("butucea_comte.jl")
include("prior_convex_class.jl")
include("load_datasets.jl")


export MCEBHistogram,
       DiscretizedStandardNormalSample,
       DiscretizedAffineEstimator,
       marginalize,
       SincKernel,
       DeLaValleePoussinKernel,
       certainty_banded_KDE,
       EBayesTarget,
       MarginalDensityTarget,
       PriorDensityTarget,
       riesz_representer,
       PosteriorMeanNumerator,
       LFSRNumerator,
       PosteriorMean,
       LFSR,
       ButuceaComte,
       GaussianMixturePriorClass,
       worst_case_bias,
       SteinMinimaxEstimator,
       steinminimaxplot,
       bias_adjusted_gaussian_ci


end # module
