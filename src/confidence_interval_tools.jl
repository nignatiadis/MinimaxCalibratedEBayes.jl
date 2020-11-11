#
# Confidence Interval Tools

abstract type ConfindenceInterval end

# LowerUpperConfidenceInterval
Base.@kwdef struct LowerUpperConfidenceInterval <: ConfindenceInterval
    lower::Float64
    upper::Float64
    α::Float64 = 0.05
    estimate::Float64 = (lower + upper)/2
    target = nothing
    method = nothing
end

function Base.show(io::IO, ci::LowerUpperConfidenceInterval)
    print(io, "lower = ", round(ci.lower,sigdigits=4))
    print(io, ", upper = ", round(ci.upper,sigdigits=4))
    print(io, ", α = ", ci.α)
    print(io, "  (", ci.target,")")
end

@recipe function f(bands::AbstractVector{<:ConfindenceInterval})
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


Base.@kwdef struct BiasVarianceConfidenceInterval <: ConfindenceInterval
    target        = nothing
    method        = nothing
    α::Float64       = 0.05
    estimate::Float64
    se::Float64
    maxbias::Float64 = 0.0
    halflength::Float64 = gaussian_ci(se; maxbias=maxbias, α=α)
    lower::Float64 = estimate - halflength
    upper::Float64 = estimate + halflength
end

function Base.show(io::IO, ci::BiasVarianceConfidenceInterval)
    print(io, "lower = ", round(ci.lower,sigdigits=4))
    print(io, ", upper = ", round(ci.upper,sigdigits=4))
    print(io, ", est. = ", round(ci.estimate,sigdigits=4))
    print(io, ", se = ", round(ci.se,sigdigits=4))
    print(io, ", maxbias = ", round(ci.maxbias,sigdigits=4))
    print(io, ", α = ", ci.α)
    print(io, "  (", ci.target,")")
end




Base.@kwdef struct BisectionPair
    c1::Float64
    var1::Float64
    max_bias1::Float64
    estimate1::Float64
    c2::Float64
    var2::Float64
    max_bias2::Float64
    estimate2::Float64
    cov::Float64
end

Base.broadcastable(pair::BisectionPair) = Ref(pair)

function StatsBase.confint(pair::BisectionPair, λ ; α=0.05)
    _se = sqrt( abs2(1-λ)*pair.var1 + abs2(λ)*pair.var2 + 2*λ*(1-λ)*pair.cov)
    _maxbias = (1-λ)*pair.max_bias1 + λ*pair.max_bias2
    _estimate = (1-λ)*pair.estimate1 + λ*pair.estimate2
    bw = gaussian_ci(_se; maxbias=_maxbias, α=α)
    _estimate -bw , _estimate+bw
end
