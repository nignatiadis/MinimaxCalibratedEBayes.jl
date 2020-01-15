abstract type EBayesTarget end

abstract type LinearEBayesTarget <: EBayesTarget end

struct MarginalDensityTarget{NS <: StandardNormalSample} <: LinearEBayesTarget
    Z::NS
end

location(target::MarginalDensityTarget) = target.Z

# introduce -> error_distribution();
function cf(target::MarginalDensityTarget{<:StandardNormalSample}, t)
    error_dbn = Normal(response(location(target))) #TODO...
    cf(error_dbn, t)
end

function riesz_representer(target::MarginalDensityTarget{<:StandardNormalSample}, t)
    error_dbn = Normal(response(location(target))) #TODO...
    pdf(error_dbn, t)
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
