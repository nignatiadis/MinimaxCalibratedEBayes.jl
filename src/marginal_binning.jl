# helper functions and types to deal with the fact that we are discretizing the output variables.

struct MCEBHistogram{S<:StepRangeLen, H<:Histogram, T}
    grid::S
    hist::H
    infty_bound::T
    max_bias::T
    η_infl::T
end

function MCEBHistogram(grid, hist;
                       infty_bound = Inf,
                       max_bias = 0.0,
                       η_infl =  0.01)
    MCEBHistogram(grid, hist, infty_bound, max_bias, η_infl)
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

DiscretizedStandardNormalSample(mhist) = DiscretizedStandardNormalSample(NaN, mhist)
function DiscretizedStandardNormalSample(a::Number, grid::StepRangeLen)
    DiscretizedStandardNormalSample(a, MCEBHistogram(grid))
end

response(s::DiscretizedStandardNormalSample) = s.Z
var(s::DiscretizedStandardNormalSample) = 1

eltype(s::DiscretizedStandardNormalSample{T}) where T = T
zero(s::DiscretizedStandardNormalSample{T}) where T = zero(T)

pdf(s::DiscretizedStandardNormalSample) = pdf(s.mhist)


struct DiscretizedAffineEstimator{MH<:MCEBHistogram}
    mhist::MH
    Q::Vector{Float64}
    Qo::Float64 #offset
end


DiscretizedAffineEstimator(Z::DiscretizedStandardNormalSample, args...) = DiscretizedAffineEstimator(Z.mhist, args...)

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



function _get_plot_x(mhist::MCEBHistogram)
    x = midpoints(mhist.grid)
end

function _get_plot_y(mhist::MCEBHistogram)
    y = pdf(mhist)[2:(end-1)]
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


@recipe function f(mhist::MCEBHistogram;
                        show_bands=true,
                        as_density=false)
    x = MinimaxCalibratedEBayes._get_plot_x(mhist)
    y = MinimaxCalibratedEBayes._get_plot_y(mhist)
    infty_bound = mhist.infty_bound
    h = step(mhist)

    if as_density
        x = x
        y = y ./ h #well... somewhat hacky.
        infty_bound = infty_bound ./ h
        ylab --> "Density"
    else
        ylab --> "Probability"
    end

    seriestype  -->  :path

    if show_bands
        lower_lims = y .- max.(y .- infty_bound, 0)
        upper_lims = infty_bound
        ribbons --> (lower_lims, upper_lims)
        fillalpha --> 0.2
    end

    xlab --> "x"
    x,y
end
