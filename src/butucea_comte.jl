# algorithm type
struct ButuceaComte end


"""
    ButuceaComteOptions(; bandwidth = :auto)
    
The Butucea Comte estimator of linear functional in the deconvolution model. Given a `LinearEBayesTarget`
with characteristic function ``\\psi^*`` and samples convolved with Gaussian noise (with ``\\phi`` Standard Gaussian pdf),
it is defined as follows
 
```math
\\hat{L}_{\\text{BC},h_m} = \\frac{1}{2 \\pi m}\\sum_{i=1}^m \\int_{-1/h_m}^{1/h_m} \\exp(it Z_k) \\frac{\\psi^*(-t)}{\\varphi^*(t)}dt
```
``h`` is the bandwidth, the option `:auto` will pick it automatically.
 
## Reference:
    >Butucea, C. and Comte, F., 2009. 
    >Adaptive estimation of linear functionals in the convolution model and applications.
    >Bernoulli, 15(1), pp.69-98.
"""
Base.@kwdef struct ButuceaComteOptions{S}
    bandwidth::S = :auto
end

fit(bcopt::ButuceaComteOptions, Zs) = bcopt #do nothing

struct ButuceaComteEstimator{EBT<:LinearEBayesTarget}
    target::EBT
    h::Float64
    f_real #ApproxFun based 
    f_imag #ApproxFun based
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



function default_bandwidth(::Type{ButuceaComteEstimator},
                           target::MarginalDensityTarget{<:StandardNormalSample},
                           n)
    sqrt(log(n))
end

default_bandwidth(::Type{ButuceaComteEstimator}, target::LFSRNumerator, n) = sqrt(log(n)*2/3)

default_bandwidth(::Type{ButuceaComteEstimator}, target::PosteriorMeanNumerator, n) = sqrt(log(n))



function default_bandwidth(::Type{ButuceaComteEstimator},
                           target::PriorDensityTarget,
                           n)
    # should make this also depend on noise distribution etc ...
    # π*m_n  = (log(n)/(2α + 1))^{1/ρ}
    #gaussian case α=1//2, ρ=2
    sqrt(log(n)/2)
end



function ButuceaComteEstimator(target::EBayesTarget; n::Int64)
    h = default_bandwidth(ButuceaComteEstimator, target, n)
    ButuceaComteEstimator(target, h)
end

function ButuceaComteEstimator(target::EBayesTarget, h::Float64)
    #todo: remove hardcoding of Normal
    _f(t) = cf(target,-t)/cf(Normal(),t) 
    _set = Interval(-h,h)
    _f_real = Fun(real ∘ _f, _set)
    _f_imag = Fun(imag ∘ _f, _set)
    ButuceaComteEstimator(target, h, _f_real, _f_imag)
end

function (cb::ButuceaComteEstimator)(z)
    real_part = fourier(cb.f_real, z)
    imag_part = fourier(cb.f_imag, z)

    real(real_part + im*imag_part)/2/π
    #h = cb.h
    ##target = cb.target
    #f(t) = real(exp(im*t*z)*cf(target,-t)/cf(Normal(),t))
    #res, _ = quadgk(f, -h, +h) res/2π
end

function DiscretizedAffineEstimator(mhist, cb::ButuceaComteEstimator)
    DiscretizedAffineEstimator(mhist, z -> cb(z))
end


""" 
    estimate(target::LinearEBayesTarget, bcopt::ButuceaComteOptions, Zs)
                         
bla
"""
function estimate(target::LinearEBayesTarget, bcopt::ButuceaComteOptions, 
                  Zs::AbstractVector{<:StandardNormalSample})
    (bcopt.bandwidth == :auto) || error("only :auto bandwidth for now")
    bcest = ButuceaComteEstimator(target;  n=nobs(Zs))
    mean(bcest.(response.(Zs)))
end 



#function estimate(Xs,cb::ComteButucea)
#    mean(cb.Q.(Xs))
#end

#struct DiscretizedAffineEstimator{MH<:MCEBHistogram}
#    mhist::MH
#    Q::Vector{Float64}
#    Qo::Float64 #offset
#end






#default_bandwidth(::Type{ComteButucea}, n) = sqrt(log(n))
#default_bandwidth(::Type{ComteButucea}, target, n) = default_bandwidth(ComteButucea, n)
