abstract type EBayesTarget end

# maybe have a distinction between 1D EBayes Target and 2D.


abstract type PosteriorEBayesTarget <: EBayesTarget end
abstract type LinearEBayesTarget <: EBayesTarget end



# general way of computing these for Mixtures.
function (target::MinimaxCalibratedEBayes.LinearEBayesTarget)(prior::MixtureModel)
    prior_probs = probs(prior)
    componentwise_target =[target(prior_j) for prior_j in components(prior)]
    dot(prior_probs, componentwise_target)
end



struct MarginalDensityTarget{NS <: StandardNormalSample} <: LinearEBayesTarget
    Z::NS
end

location(target::LinearEBayesTarget) = target.Z

# introduce -> error_distribution();



struct PriorDensityTarget <: LinearEBayesTarget
    x::Float64
end

location(target::PriorDensityTarget) = target.x

function cf(target::PriorDensityTarget, t)
    exp(im*location(target)*t)
end

function (target::PriorDensityTarget)(prior::Distribution)
    pdf(prior, location(target))
end
#abstract type PosteriorNumeratorTarget <: LinearInferenceTarget end




#function integration_domain(t::LinearInferenceTarget)
#end




#----------- LFSRNumerator ---------------------------------
# running under assumption X_i >=0...
struct LFSRNumerator{NS <: StandardNormalSample} <: LinearEBayesTarget
    Z::NS
end

#--------- PosteriorMeanNumerator ---------------------------------

struct PosteriorMeanNumerator{NS <: StandardNormalSample} <: LinearEBayesTarget
    Z::NS
end



#----------------------- Posterior Targets------------------------------------------

struct PosteriorTarget{NT <: LinearEBayesTarget} <: PosteriorEBayesTarget
    num_target::NT
end

PosteriorMean(Z) = PosteriorTarget(PosteriorMeanNumerator(Z))
LFSR(Z) = PosteriorTarget(LFSRNumerator(Z))

function (post_target::PosteriorTarget)(prior)
    num = post_target.num_target(prior)
    denom = MarginalDensityTarget(location(post_target.num_target))(prior)
    num/denom
end
