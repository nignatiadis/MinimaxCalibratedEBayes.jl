module MinimaxCalibratedEBayes


using Reexport

@reexport using Empirikos


#@reexport using StatsBase,
#                Distributions,
#                EBayes

#using ApproxFun:Interval, Fun
#using DiffResults
#using Expectations
#using ExponentialFamilies
#using ForwardDiff
using JuMP
using LinearFractional
using MathOptInterface

using KernelDensity
#using LaTeXStrings
#using LinearAlgebra
#using Optim
#using OrthogonalPolynomialsQuasi
#using OscillatoryIntegrals
#using Parameters
#using Plots
#using RecipesBase
using Roots
using Setfield
#using SpecialFunctions:erfi
#using QuadGK

#import StatsBase:confint,
#				 fit,
#                Histogram,
#                 binindex,
#                 midpoints,
#				 nobs,
#                 response

#import Statistics:var

#import Distributions:cf,
#                     estimate,
#                     pdf,
#                     location,
#					 loglikelihood,
#					 support,
#					 DiscreteNonParametric


#import Expectations:expectation

#import Base:eltype,
#            extrema,
#			step,
#           first,
#            last,
#            length,
#            zero,
#			getindex,
#			lastindex


#import Base.Broadcast: broadcastable


include("load_datasets.jl")
include("target_ci.jl")
include("neighborhood_worst_case.jl")

export NeighborhoodWorstCase

end # module
