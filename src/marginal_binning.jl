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

# point is ... can think of above as Int{K} in MLJ-lang
function DiscretizedStandardNormalSamples(mhist::MCEBHistogram;
                                          var_proxy=1/sqrt(2*pi),
                                          f_min=nothing,
                                          f_max=nothing)
    DiscretizedStandardNormalSamples(mhist, nothing, var_proxy, f_min, f_max)
end

function DiscretizedStandardNormalSamples(marginal_grid::AbstractVector)
	DiscretizedStandardNormalSamples(MCEBHistogram(marginal_grid))
end

response(Z_discr::DiscretizedStandardNormalSamples) = Z_discr.mhist.hist # what I want to say.
nobs(Z_discr::DiscretizedStandardNormalSamples) = sum(Z_discr.mhist.hist.weights)
nbins(Z_discr::DiscretizedStandardNormalSamples) = length(Z_discr.mhist.grid) + 1
#eltype(Z_discr::DiscretizedStandardNormalSamples) = eltype(Z_discr.mhist.grid)
#length(Z_discr::DiscretizedStandardNormalSamples) = sum(Z_discr.mhist.hist.weights)

function support(Z_discr::DiscretizedStandardNormalSamples)
    mhist = Z_discr.mhist
    supp = zeros(eltype(mhist.grid), nbins(Z_discr))
    supp[2:(end-1)] = StatsBase.midpoints(mhist.grid)

    # replace by gap just adjacent
    h = step(Z_discr.mhist.grid)
    supp[1] = mhist.grid[1] - h/2
    supp[end] =  mhist.grid[end] + h./2
    supp
end



# TODO: a lot of overlap with code for set_neighborhood based on KDE
function set_neighborhood(Zs_discr::DiscretizedStandardNormalSamples,
                          prior::Distribution;
						  C∞ = Inf,
						  varproxy_lower_bound = false)
    
    grid = Zs_discr.mhist.grid

	f_marginal = pdf(marginalize(prior, Zs_discr))

	
    f_max = zeros(length(f_marginal))
    f_min = zeros(length(f_marginal))
	var_proxy = zeros(length(f_marginal))
    # fill in intermediate points first
    for i in 2:length(grid)
        x_left = grid[i-1]
        x_right = grid[i]
        h = x_right - x_left  # step(grid)...
        f_max[i] = f_marginal[i] + C∞*h
        f_min[i] = max(f_marginal[i] - C∞*h, 0.0)
        var_proxy[i] = varproxy_lower_bound ? max(f_marginal[i], C∞*h) : f_marginal[i]
    end

    f_max[1] = Inf
    f_min[1] = 0.0
    var_proxy[1] = f_marginal[1]
    f_max[end] = Inf
    f_min[end] = 0.0
    var_proxy[end] = f_marginal[end]

    Zs_discr = @set Zs_discr.var_proxy = var_proxy
    Zs_discr = @set Zs_discr.f_min = f_min
    Zs_discr = @set Zs_discr.f_max = f_max
    Zs_discr
end




struct DiscretizedStandardNormalSample{DISC<:DiscretizedStandardNormalSamples}
    samples::DISC
    bin::Int
end



# (i) indexing -> give
function (Zs_disc::DiscretizedStandardNormalSamples)(i::Int)
    DiscretizedStandardNormalSample(Zs_disc, i)
end


# FIXME: this will have to go once implement iterator protocol for
# DiscretizedStandardNormalSample
broadcastable(Zs::DiscretizedStandardNormalSamples) = Ref(Zs)

# convert to discretenonparametric dbn
function DiscreteNonParametric(Zs_discr::DiscretizedStandardNormalSamples)
    supp = support(Zs_discr)
    probs = pdf(Zs_discr.mhist)
    DiscreteNonParametric(supp, probs)
end

function DiscreteNonParametric(Zs_discr::MCEBHistogram)
   DiscreteNonParametric(DiscretizedStandardNormalSamples(Zs_discr))
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

function (Q::DiscretizedAffineEstimator)(Zs::DiscretizedStandardNormalSamples)
	# TODO: what if idx is nothing?
	Q.Q[Zs.idx] .+ Q.Qo 
end

function (Q::DiscretizedAffineEstimator)(x::StandardNormalSample)
    Q(response(x))
end

#  function (calib::DiscretizedAffineEstimator)(x)
#    idxs = StatsBase.binindex(calib.mhist, x)
#   calib.Q[idxs] + calib.Qo
#end

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
