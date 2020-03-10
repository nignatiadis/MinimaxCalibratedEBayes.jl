module MinimaxCalibratedEBayes

using Reexport

@reexport using StatsBase,
                Distributions,
                EBayes

using DiffResults
using Expectations
using ExponentialFamilies
using ForwardDiff
using JuMP
using KernelDensity
using LaTeXStrings
using LinearAlgebra
using MathOptInterface
using Optim
using OrthogonalPolynomialsQuasi
using Plots
using RecipesBase
using Roots
using Setfield
using QuadGK

import StatsBase:confint,
				 fit,
                 Histogram,
                 binindex,
                 midpoints,
                 response

import Statistics:var

import Distributions:cf,
                     estimate,
                     pdf,
                     location,
					 loglikelihood

import Expectations:expectation

import Base:extrema,
			step,
            first,
            last,
            length,
            zero,
			getindex,
			lastindex


import Base.Broadcast: broadcastable

include("bias_adjusted_ci.jl")
include("marginal_binning.jl")
include("inference_targets.jl")
include("normal_rules.jl")
include("marginal_kde.jl")
include("logspline_g_new.jl")
include("butucea_comte.jl")
include("prior_convex_class.jl")
include("sinkhorn.jl")
include("helper_plots.jl")
include("hermite.jl")


export MCEBHistogram,
	   DiscretizedStandardNormalSamples,
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
       bias_adjusted_gaussian_ci,
	   confint,
	   confintplot,
	   confintplot!,
	   set_neighborhood


end # module
