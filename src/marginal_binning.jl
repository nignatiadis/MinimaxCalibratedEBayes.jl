# helper functions and types to deal with the fact that we are discretizing the output variables.
struct MCEBHistogram{S<:StepRangeLen, H<:Histogram}
    grid::S
    hist::H
end

function MCEBHistogram(grid::StepRangeLen)
    xs = [-Inf; collect(grid); +Inf]
    ws = zeros(Float64, length(grid)+1)
    hist = Histogram(xs, ws, :right, false)
    MCEBHistogram(grid, hist)
end

StatsBase.binindex(mh::MCEBHistogram, x) = StatsBase.binindex(mh.hist, x)

Base.first(mh::MCEBHistogram) = Base.first(mh.grid)
Base.last(mh::MCEBHistogram) = Base.last(mh.grid)
Base.step(mh::MCEBHistogram) = Base.step(mh.grid)
broadcastable(mh::MCEBHistogram) = Ref(mh)

pdf(mh::MCEBHistogram) = mh.hist.weights

struct DiscretizedStandardNormalSample{T <: Number, MH<:MCEBHistogram} <: EBayesSample{T}
    Z::T
    mhist::MH
end

response(s::DiscretizedStandardNormalSample) = s.Z
var(s::DiscretizedStandardNormalSample) = 1

eltype(s::DiscretizedStandardNormalSample{T}) where T = T
zero(s::DiscretizedStandardNormalSample{T}) where T = zero(T)


struct BinnedCalibrator{MH<:MCEBHistogram}
    mhist::MH
    Q::Vector{Float64}
    Qo::Float64 #offset
end

BinnedCalibrator(mhist, Q) = BinnedCalibrator(mhist, Q, zero(Float64))

function (calib::BinnedCalibrator)(x)
    idxs = StatsBase.binindex(calib.mhist, x)
    calib.Q[idxs] + calib.Qo
end
