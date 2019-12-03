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
midpoints(mh::MCEBHistogram) = midpoints(mh.grid)

struct DiscretizedStandardNormalSample{T <: Number, MH<:MCEBHistogram} <: EBayesSample{T}
    Z::T
    mhist::MH
end

response(s::DiscretizedStandardNormalSample) = s.Z
var(s::DiscretizedStandardNormalSample) = 1

eltype(s::DiscretizedStandardNormalSample{T}) where T = T
zero(s::DiscretizedStandardNormalSample{T}) where T = zero(T)


struct DiscretizedAffineEstimator{MH<:MCEBHistogram}
    mhist::MH
    Q::Vector{Float64}
    Qo::Float64 #offset
end

DiscretizedAffineEstimator(mhist, Q) = DiscretizedAffineEstimator(mhist, Q, Base.zero(Float64))

function DiscretizedAffineEstimator(mhist::MCEBHistogram, F::Function)
    Q = zeros(Float64, length(mhist.grid) + 1)
    Q[2:(end-1)] = F.(midpoints(mhist))
    DiscretizedAffineEstimator(mhist, Q)
end

function (calib::DiscretizedAffineEstimator)(x)
    idxs = StatsBase.binindex(calib.mhist, x)
    calib.Q[idxs] + calib.Qo
end



function _get_plot_x(affine::DiscretizedAffineEstimator)
    x = midpoints(affine.mhist.grid)
end

function _get_plot_y(affine::DiscretizedAffineEstimator)
    y = affine.Q[2:(end-1)] .+ affine.Qo
end


@recipe function f(affine::DiscretizedAffineEstimator)
    seriestype  -->  :path
    _get_plot_x(affine), _get_plot_y(affine)
end

@recipe function f(affines::Vector{<:DiscretizedAffineEstimator})
    seriestype  -->  :path
    # TODO: Check this is the same for all of them.
    x = _get_plot_x(affines[1])
    y = hcat(_get_plot_y.(affines)...)
    x,y
end