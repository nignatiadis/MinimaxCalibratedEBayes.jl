#
# Confidence Interval Tools
#
# LowerUpperConfidenceInterval
Base.@kwdef struct LowerUpperConfidenceInterval{T,M}
    lower::Float64
    upper::Float64
    α::Float64 = 0.05
    estimate::Float64 = (lower + upper)/2
    target::T = nothing
    method::M = nothing
end

function Base.show(io::IO, ci::LowerUpperConfidenceInterval)
    print(io, "lower = ", round(ci.lower,sigdigits=4))
    print(io, ", upper = ", round(ci.upper,sigdigits=4))
    print(io, ", α = ", ci.α)
    print(io, "  (", ci.target,")")
end

@recipe function f(bands::AbstractVector{<:LowerUpperConfidenceInterval})
	x = [Float64(location(band.target)) for band in bands]
    lower = getproperty.(bands, :lower)
    upper = getproperty.(bands, :upper)
    estimate = getproperty.(bands, :estimate)

	background_color_legend --> :transparent
	foreground_color_legend --> :transparent
    grid --> nothing

	cis_ribbon  = estimate .- lower, upper .- estimate
	fillalpha --> 0.36
	seriescolor --> "#018AC4"
	ribbon --> cis_ribbon
    linealpha --> 0
    framestyle --> :box
    legend --> :topleft
	label --> "95\\% CI"
	x, estimate
end


function gaussian_ci(se; maxbias=0.0, α=0.05)
    level = 1 - α
    rel_bias = maxbias/se
    zz = fzero( z-> cdf(Normal(), rel_bias-z) + cdf(Normal(), -rel_bias-z) +  level -1,
        0, rel_bias - quantile(Normal(),(1- level)/2.1))
    zz*se
end


Base.@kwdef struct TargetConfidenceInterval{T,M}
    target::T        = nothing
    method::M        = nothing
    α::Float64       = 0.05
    point::Float64
    se::Float64
    maxbias::Float64 = 0.0
    halflength::Float64 = gaussian_ci(se; maxbias=maxbias, α=α)
    lower::Float64 = point - halflength
    upper::Float64 = point + halflength
end
