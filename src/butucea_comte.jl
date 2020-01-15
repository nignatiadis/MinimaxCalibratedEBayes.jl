struct ButuceaComte{T<:EBayesTarget}
    target::EBayesTarget
    h::Float64 # not really bandwidth -> truncation limit...
end

function default_bandwidth(::Type{ButuceaComte},
                           target::MarginalDensityTarget{<:StandardNormalSample},
                           n)
    sqrt(log(n))
end

function default_bandwidth(::Type{ButuceaComte},
                           target::PriorDensityTarget,
                           n)
    # should make this also depend on noise distribution etc ...
    # π*m_n  = (log(n)/(2α + 1))^{1/ρ}
    #gaussian case α=1//2, ρ=2
    sqrt(log(n)/2)
end


function ButuceaComte(target::EBayesTarget, h::Float64)
    ButuceaComte{typeof(target)}(target, h)
end

function ButuceaComte(target::EBayesTarget, n::Int64)
    h = default_bandwidth(ButuceaComte, target, n)
    ButuceaComte(target, h)
end

function (cb::ButuceaComte)(z)
    h = cb.h
    target = cb.target
    f(t) = real(exp(im*t*z)*cf(target,-t)/cf(Normal(),t))
    res, _ = quadgk(f, -h, +h)
    res/2/π
end

function DiscretizedAffineEstimator(mhist, cb::ButuceaComte)
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
