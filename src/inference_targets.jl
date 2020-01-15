abstract type EBayesTarget end

# maybe have a distinction between 1D EBayes Target and 2D.



abstract type LinearEBayesTarget <: EBayesTarget end

struct MarginalDensityTarget{NS <: StandardNormalSample} <: LinearEBayesTarget
    Z::NS
end

location(target::MarginalDensityTarget) = target.Z

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



#function riesz_representer(target::MarginalDensityTarget, t)
#    pdf(Normal(), target.x - t)
#end,


# TODO: Design this more carefully, maybe add function location() instead of requiring .x access

#function integration_domain(t::LinearInferenceTarget)
#end

#-------- Marginal Density --------------------------------
#struct MarginalDensityTarget <: LinearInferenceTarget
#    x::Float64
#end
