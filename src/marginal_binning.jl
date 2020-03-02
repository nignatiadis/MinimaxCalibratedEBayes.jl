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

function MCEBHistogram(grid::StepRangeLen, Zs::AbstractArray=[])
    xs = [-Inf; collect(grid); +Inf]
    hist = fit(Histogram, Zs, xs; closed=:right)
    MCEBHistogram(grid, hist)
end

StatsBase.binindex(mh::MCEBHistogram, x) = StatsBase.binindex(mh.hist, x)

Base.first(mh::MCEBHistogram) = Base.first(mh.grid)
Base.last(mh::MCEBHistogram) = Base.last(mh.grid)
Base.step(mh::MCEBHistogram) = Base.step(mh.grid)
broadcastable(mh::MCEBHistogram) = Ref(mh)

getindex(mhist::MCEBHistogram, k) = getindex(mhist.hist.weights, k)
lastindex(mhist::MCEBHistogram) = lastindex(mhist.hist.weights)


pdf(mh::MCEBHistogram) = mh.hist.weights ./ sum(mh.hist.weights)
midpoints(mh::MCEBHistogram) = midpoints(mh.grid)


# DiscretizedStandardNormalSamples

struct DiscretizedStandardNormalSamples{MH<:MCEBHistogram,
                                       IX<:Union{Nothing, Vector{Int}},
                                       Tvar,
                                       Tmin,
                                       Tmax}
    mhist::MH                                   
    idx::IX
    var_proxy::Tvar
    f_min::Tmin
    f_max::Tmax
end

function DiscretizedStandardNormalSamples(Zs::AbstractVector{<:StandardNormalSample},
                                          grid::AbstractVector;
                                          kwargs...)
    mhist = MCEBHistogram(grid, response.(Zs))
    DiscretizedStandardNormalSamples(Zs, mhist; kwargs...)
end


# point is ... can think of above as Int{K} in MLJ-lang
function DiscretizedStandardNormalSamples(Zs::AbstractVector{<:StandardNormalSample},
                                          mhist::MCEBHistogram;
                                          keep_idx=true,
                                          var_proxy=1/sqrt(2*pi),
                                          f_min=nothing,
                                          f_max=nothing)
    Zs = response.(Zs) # rethink.
    if keep_idx
        idx = StatsBase.binindex.(mhist, Zs)
    else
        idx = nothing
    end
    DiscretizedStandardNormalSamples(mhist, idx, var_proxy, f_min, f_max)
end

response(Z_discr::DiscretizedStandardNormalSamples) = Z_discr.mhist.hist # what I want to say.

length(Z_discr::DiscretizedStandardNormalSamples) = sum(Z_discr.mhist.hist.weights)

struct DiscretizedStandardNormalSample{DISC<:DiscretizedStandardNormalSamples}
    samples::DISC
    bin::Int
end



# (i) indexing -> give
function (Zs_disc::DiscretizedStandardNormalSamples)(i::Int)
    DiscretizedStandardNormalSample(Zs_disc, i)
end




# AffineEstimator
struct DiscretizedAffineEstimator{MH<:MCEBHistogram}
    mhist::MH
    Q::Vector{Float64}
    Qo::Float64 #offset
end


DiscretizedAffineEstimator(Z::DiscretizedStandardNormalSamples, args...) = DiscretizedAffineEstimator(Z.mhist, args...)

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



#TODO IDEA

# AbstractSummarizedEBayesSamoles

# dispatch on the below....
# HistogramSamples {BaseSamples + MCEBHistogram + optional idx mapping everyone to their bin}
# through kwarg, retain idx
# NeighborhoodHistogramSamples {BaseSamples + MCEBHistogram}
