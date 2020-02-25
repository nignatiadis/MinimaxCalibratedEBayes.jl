abstract type EBayesTarget end

# maybe have a distinction between 1D EBayes Target and 2D.

# functions so that EBayesTargets can play nicely with vectorization
broadcastable(target::EBayesTarget) = Ref(target)

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

location(target::PriorDensityTarget) = target.x

function cf(target::PriorDensityTarget, t)
    exp(im*location(target)*t)
end

function (target::PriorDensityTarget)(prior)
    pdf(prior, location(target))
end

function extrema(target::PriorDensityTarget)
	(0, Inf)
end
#abstract type PosteriorNumeratorTarget <: LinearInferenceTarget end




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

function extrema(target::PosteriorTarget{LF} where LF<:LFSRNumerator)
	(0.0,1.0)
end
