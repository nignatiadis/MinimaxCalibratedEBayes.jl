abstract type EBayesTarget end

# maybe have a distinction between 1D EBayes Target and 2D.

# functions so that EBayesTargets can play nicely with vectorization
broadcastable(target::EBayesTarget) = Ref(target)

function extrema(target::EBayesTarget) #fallback
	(-Inf, Inf)
end

function pretty_label(target::EBayesTarget)
	""
end

function (targets::AbstractVector{<:EBayesTarget})(prior)
	[target(prior) for target in targets]
end


abstract type PosteriorEBayesTarget <: EBayesTarget end
abstract type LinearEBayesTarget <: EBayesTarget end

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

#--- PriorDensityTarget
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



#--- OneSidedPriorTailProbability
Base.@kwdef struct OneSidedPriorTailProbability <: LinearEBayesTarget
    cutoff::Float64 = 0.0
end

function extrema(target::OneSidedPriorTailProbability)
	(0, 1)
end

function riesz_representer(target::OneSidedPriorTailProbability, t)
    one(Float64)*(t >= target.cutoff)
end

function (target::OneSidedPriorTailProbability)(prior)
	ccdf(prior, target.cutoff)
end


#function integration_domain(t::LinearInferenceTarget)
#end




#----------- LFSRNumerator ---------------------------------
# running under assumption X_i >=0...
struct LFSRNumerator{NS <: StandardNormalSample} <: LinearEBayesTarget
    Z::NS
end

function cf(target::LFSRNumerator{<:StandardNormalSample}, t)
    x =
    exp(im*t*x- t^2/2)*(1+im*erfi((t-im*x)/sqrt(2)))/2
end

function riesz_representer(target::LFSRNumerator{<:StandardNormalSample}, t)
    x = response(location(target))
    pdf(Normal(), x - t)*(t>=0)
end


#--------- PosteriorMeanNumerator ---------------------------------

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

struct PosteriorTarget{NT <: LinearEBayesTarget} <: PosteriorEBayesTarget
    num_target::NT
end

PosteriorMean(Z) = PosteriorTarget(PosteriorMeanNumerator(Z))
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
