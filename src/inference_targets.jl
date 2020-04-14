"""
	EBayesTarget
	
Abstract type that describe Empirical Bayes estimands (which we want to estimate or conduct inference for).
"""
abstract type EBayesTarget end

# maybe have a distinction between 1D EBayes Target and 2D.

# functions so that EBayesTargets can play nicely with vectorization
broadcastable(target::EBayesTarget) = Ref(target)


""" 
	extrema(target::EBayesTarget)
	
Returns a tuple `(a,b)` such that ` a ≦ target ≦ b` is always true. By default this
is ``(-\\infty,+\\infty)``, but e.g., for a `PriorTailProbability` it returns `(0,1)`.
"""	
function extrema(target::EBayesTarget) #fallback
	(-Inf, Inf)
end

function pretty_label(target::EBayesTarget)
	""
end

function (targets::AbstractVector{<:EBayesTarget})(prior)
	[target(prior) for target in targets]
end

"""
	PosteriorEBayesTarget <: EBayesTarget
	
Abstract type for Empirical Bayes estimands that take the form ``E_G[h(\\mu) \\mid Z_i = z]`` for some function ``h``.
"""
abstract type PosteriorEBayesTarget <: EBayesTarget end

"""
	LinearEBayesTarget <: EBayesTarget
	
Abstract type for Empirical Bayes estimands that are linear functionals of the prior ``G``,
i.e., they take the form ``L(G)`` for some function linear functional ``L``.
"""
abstract type LinearEBayesTarget <: EBayesTarget end


"""
	cf(::LinearEBayesTarget, t)
	
The characteristic function of ``L(\\cdot)``, a `LinearEBayesTarget`, which we define as follows:
	
For ``L(\\cdot)`` which may be written as ``L(G) = \\int \\psi(\\mu)dG\\mu`` 
(for a measurable function ``\\psi``) this returns the Fourier transform of ``\\psi``
evaluated at t, i.e., ``\\psi^*(t) = \\int \\exp(it x)\\psi(x)dx``. 


Note that ``\\psi^*(t)`` is such that for distributions ``G`` with density ``g``
(and ``g^*`` the Fourier Transform of ``g``) the following holds:

```math
L(G) = \\frac{1}{2\\pi}\\int g^*(\\mu)\\psi^*(\\mu) d\\mu
```
"""
function cf(::LinearEBayesTarget, t) end

location(target::LinearEBayesTarget) = target.Z

response(target::EBayesTarget) = _response(location(target))

function _response(Z)
	isa(Z, EBayesSample) ? response(Z) : Z
end

# general way of computing these for Mixtures.
function (target::MinimaxCalibratedEBayes.LinearEBayesTarget)(prior::MixtureModel)
    prior_probs = probs(prior)
    componentwise_target =[target(prior_j) for prior_j in components(prior)]
    dot(prior_probs, componentwise_target)
end


#--- MarginalDensityTarget


"""
	MarginalDensityTarget(Z::StandardNormalSample) <: LinearEBayesTarget

## Example call
```julia
MarginalDensityTaget(StandardNormalSample(2.0))
```
## Description 		
Describes the marginal density evaluated at ``Z=z``  (e.g. ``Z=2`` in the example above)
of a sample drawn from the hierarchical model 
```math
\\mu \\sim G, Z \\sim \\mathcal{N}(0,1)
```
In other words, letting ``\\phi`` the Standard Normal pdf
```math
L(G) = \\phi \\star dG(z)
```
Note that `2.0` has to be wrapped inside `StandardNormalSample(2.0)` since this target
depends not only on `G` and the location, but also on the likelihood. Additional
likelihoods will be added in the future.
"""
struct MarginalDensityTarget{NS} <: LinearEBayesTarget
    Z::NS
end

pretty_label(target::MarginalDensityTarget) = L"f(x)"


function cf(target::MarginalDensityTarget{<:StandardNormalSample}, t)
    error_dbn = Normal(response(location(target))) #TODO...
    cf(error_dbn, t)
end

function riesz_representer(target::MarginalDensityTarget{<:StandardNormalSample}, t)
    error_dbn = Normal(response(location(target))) #TODO...
    pdf(error_dbn, t)
end

function extrema(target::MarginalDensityTarget{<:StandardNormalSample})
	(0, 1/sqrt(2π))
end


"""
	PriorDensityTarget(z::Float64) <: LinearEBayesTarget

## Example call
```julia
PriorDensityTarget(2.0)
```
## Description 		
This is the evaluation functional of the density of ``G`` at `z`, i.e.,
``L(G) = G'(z) = g(z)`` or in Julia code `L(G) = pdf(G, z)`.
"""
struct PriorDensityTarget <: LinearEBayesTarget
    x::Float64
end

pretty_label(target::PriorDensityTarget) = L"g(x)"

location(target::PriorDensityTarget) = target.x

function cf(target::PriorDensityTarget, t)
    exp(im*location(target)*t)
end

function (target::PriorDensityTarget)(prior)
    pdf(prior, location(target))
end

# hard coding this since almost everything else mixture related is handled as above.
function (target::PriorDensityTarget)(prior::MixtureModel)
	pdf(prior, location(target))
end

function extrema(target::PriorDensityTarget)
	(0, Inf)
end



#--- PriorTailProbability
"""
	PriorTailProbability(cutoff::Float64) <: LinearEBayesTarget

## Example call
```julia
PriorTailProbability(2.0)
```
## Description 		
This is the evaluation functional of the tail probability of ``G`` at `cutoff`, i.e.,
``L(G) = 1-G(\\text{cutoff})`` or in Julia code `L(G) = ccdf(G, z)`.
"""
Base.@kwdef struct PriorTailProbability <: LinearEBayesTarget
    cutoff::Float64 = 0.0
end

function extrema(target::PriorTailProbability)
	(0, 1)
end

function riesz_representer(target::PriorTailProbability, t)
    one(Float64)*(t >= target.cutoff)
end

function (target::PriorTailProbability)(prior)
	ccdf(prior, target.cutoff)
end


#function integration_domain(t::LinearInferenceTarget)
#end




"""
	LFSRNumerator(Z::StandardNormalSample) <: LinearEBayesTarget

## Example call
```julia
LFSRNumerator(StandardNormalSample(2.0))
```
## Description 		
Describes the linear functional
```math
L(G) =  \\int \\mathbf 1(\\mu \\geq 0) \\phi(z-\\mu) dG(\\mu)
```
This is used as an intermediate step in representing the local false sign rate, c.f. [`PosteriorTarget`](@ref)
and [`LFSR`](@ref).
"""
struct LFSRNumerator{NS <: StandardNormalSample} <: LinearEBayesTarget
    Z::NS
end

function cf(target::LFSRNumerator{<:StandardNormalSample}, t)
    x = response(target)
    exp(im*t*x- t^2/2)*(1+im*erfi((t-im*x)/sqrt(2)))/2
end

function riesz_representer(target::LFSRNumerator{<:StandardNormalSample}, t)
    x = response(target)
    pdf(Normal(), x - t)*(t>=0)
end


#--------- PosteriorMeanNumerator ---------------------------------

"""
	PosteriorMeanNumerator(Z::StandardNormalSample) <: LinearEBayesTarget

## Example call
```julia
PosteriorMeanNumerator(StandardNormalSample(2.0))
```
## Description 		
Describes the linear functional
```math
L(G) =  \\int \\mu \\phi(z-\\mu) dG(\\mu)
```
This is used as an intermediate step in representing the posterior mean, c.f. [`PosteriorTarget`](@ref)
and [`PosteriorMean`](@ref)
"""
struct PosteriorMeanNumerator{NS <: StandardNormalSample} <: LinearEBayesTarget
    Z::NS
end

function cf(target::PosteriorMeanNumerator{<:StandardNormalSample}, t)
    x = response(location(target))
    cf(Normal(x), t)*(x + im*t)
end

function riesz_representer(target::PosteriorMeanNumerator{<:StandardNormalSample}, t)
    x = response(location(target))
    pdf(Normal(), x - t)*t
end


#----------------------- Posterior Targets------------------------------------------
"""
	PosteriorTarget(num_target::LinearEBayesTarget) <: EBayesTarget
	
Type for Empirical Bayes estimands that take the form:
	
```math
E_G[h(\\mu) \\mid Z_i = z] = \\frac{ \\int h(\\mu) \\phi(z-\\mu) dG(\\mu)}{\\int \\phi(z-\\mu) dG(\\mu}
```
`num_target` is a `LinearEBayesTarget` that represents the numerator of the above estimands, i.e., of
``L(G)``.
"""
struct PosteriorTarget{NT <: LinearEBayesTarget} <: PosteriorEBayesTarget
    num_target::NT
end

""" 
	PosteriorMean(Z)

## Example call
```julia
PosteriorMean(StandardNormalSample(2.0))
```
## Description 	
Shortcut for `PosteriorTarget(PosteriorMeanNumerator(Z))`, c.f. [`PosteriorMeanNumerator`](@ref)
"""
PosteriorMean(Z) = PosteriorTarget(PosteriorMeanNumerator(Z))


""" 
	LFSR(Z)

## Example call
```julia
LFSR(StandardNormalSample(2.0))
```
## Description 
Shortcut for `PosteriorTarget(LFSRNumerator(Z))`, c.f. [`LFSRNumerator`](@ref)
"""
LFSR(Z) = PosteriorTarget(LFSRNumerator(Z))

location(target::PosteriorTarget) = location(target.num_target)


function (post_target::PosteriorTarget)(prior)
    num = post_target.num_target(prior)
    denom = MarginalDensityTarget(location(post_target.num_target))(prior)
    num/denom
end

function MarginalDensityTarget(post_target::PosteriorTarget)
	MarginalDensityTarget(location(post_target.num_target))
end 


function extrema(target::PosteriorTarget{LF} where LF<:LFSRNumerator)
	(0.0,1.0)
end

function pretty_label(target::PosteriorTarget{LF} where LF<:LFSRNumerator)
	L"\theta(x) := \Pr[\mu \geq 0 \mid X=x]"
end

function pretty_label(target::PosteriorTarget{PM} where PM<:PosteriorMeanNumerator)
	L"\theta(x) := E[\mu | X=x]"
end


#------------------ Calibration---------------------------------

#---------Should probably just turn this all into one function-----

Base.@kwdef struct CalibratedTarget{T <: PosteriorTarget, S<:Real} <: LinearEBayesTarget
    posterior_target::T
    θ̄::S #Pilot
	denom::S #denominator pilot 
end

function MarginalDensityTarget(calib_target::CalibratedTarget)
	MarginalDensityTarget(calib_target.posterior_target)
end 


# special semantics here.
function (calib_target::CalibratedTarget)(prior)
    numerator_target = calib_target.posterior_target.num_target
    denominator_target = MarginalDensityTarget(calib_target)
	θ̄ = calib_target.θ̄
    numerator_target(prior) - θ̄*denominator_target(prior)
end

# hardcoding
function (calib_target::CalibratedTarget)(prior::MixtureModel)
    numerator_target = calib_target.posterior_target.num_target
    denominator_target = MarginalDensityTarget(calib_target)
	θ̄ = calib_target.θ̄
    numerator_target(prior) - θ̄*denominator_target(prior)
end

#function (target::CalibratedTarget)(prior)
#	target.post.num_target(prior) -
#end


#function riesz_representer(target::CalibratedTarget, t)
#    x = target.num.x
#    θ̄ = target.θ̄
#    riesz_representer(target.num, t) - θ̄*riesz_representer(MarginalDensityTarget(x),t)
#end
#------
