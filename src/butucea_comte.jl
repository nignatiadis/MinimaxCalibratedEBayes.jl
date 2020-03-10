# algorithm type
struct ButuceaComte end

struct ButuceaComteEstimator{EBT<:LinearEBayesTarget}
    target::EBT
    h::Float64
end

# Make theoretical results of BC practical.
Base.@kwdef struct ButuceaComteNoiseDensityParams{T <: Real}
    γ::T
    α::T
    ρ::T
    κ0upper::T
    κ0lower::T
end

function ButuceaComteNoiseDensityParams(Z::StandardNormalSample)
    ButuceaComteNoiseDensityParams(γ = 0.0, ρ = 2.0, α = 0.5, κ0upper=1.0, κ0lower=1.0)
end

Base.@kwdef struct ButuceaComteSobolevParams{T<:Real}
   b::T
   L::T        # constant
   a::T = 0.0
   r::T = 0.0
end

Base.@kwdef struct ButuceaComteTargetParams{T<:Real}
   B::T        #polynomial decay (1+x^2)^B
   A::T
   R::T
   Cψ::T       #constant
end

function ButuceaComteTargetParams(target::PriorDensityTarget)
    ButuceaComteTargetParams(A=0.0, B=0.0, R=0.0, Cψ=1.0)
end

# only accounting for b,B for now
function butucea_comte_squared_bias_bound(noise::ButuceaComteNoiseDensityParams,
                                  sobolev::ButuceaComteSobolevParams,
                                  target::ButuceaComteTargetParams,
                                  h::Real)
    R = target.R
    r = sobolev.r

    a = target
    B = target.B
    b = sobolev.b

    L = sobolev.L
    Cψ = target.Cψ

    L*Cψ/π/(2b + 2B -1)*h^(1-2b-2B)
end

# only do case PriorDensityTarget for now
function butucea_comte_unit_var_proxy(h::Real)
    1/π^2*exp(h^2/2)/h^2
end



function default_bandwidth(::ButuceaComte,
                           target::MarginalDensityTarget{<:StandardNormalSample},
                           n)
    sqrt(log(n))
end





function default_bandwidth(::ButuceaComte,
                           target::PriorDensityTarget,
                           n)
    # should make this also depend on noise distribution etc ...
    # π*m_n  = (log(n)/(2α + 1))^{1/ρ}
    #gaussian case α=1//2, ρ=2
    sqrt(log(n)/2)
end



function ButuceaComteEstimator(target::EBayesTarget; n::Int64)
    h = default_bandwidth(ButuceaComte, target, n)
    ButuceaComte(target, h)
end

function (cb::ButuceaComteEstimator)(z)
    h = cb.h
    target = cb.target
    f(t) = real(exp(im*t*z)*cf(target,-t)/cf(Normal(),t))
    res, _ = quadgk(f, -h, +h)
    res/2/π
end

function DiscretizedAffineEstimator(mhist, cb::ButuceaComteEstimator)
    DiscretizedAffineEstimator(mhist, z -> cb(z))
end




#function estimate(Xs,cb::ComteButucea)
#    mean(cb.Q.(Xs))
#end

#struct DiscretizedAffineEstimator{MH<:MCEBHistogram}
#    mhist::MH
#    Q::Vector{Float64}
#    Qo::Float64 #offset
#end





#pretty_label(cb::ComteButucea)  = "Butucea-Comte"

#inference_target(cb::ComteButucea) = cb.target

#default_bandwidth(::Type{ComteButucea}, target::LFSRNumerator, n) = sqrt(log(n)*2/3)

#default_bandwidth(::Type{ComteButucea}, n) = sqrt(log(n))
#default_bandwidth(::Type{ComteButucea}, target, n) = default_bandwidth(ComteButucea, n)
