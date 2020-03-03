function DiscretizedAffineEstimator(mhist::MCEBHistogram, kernel::ContinuousUnivariateDistribution)
    DiscretizedAffineEstimator(mhist, x->pdf(kernel, x))
end



struct SincKernel <: ContinuousUnivariateDistribution
   h::Float64 #Bandwidth
end

default_bandwidth(::Type{SincKernel}, m) = 1/sqrt(log(m))
SincKernel(; m = 1_000) = SincKernel(default_bandwidth(SincKernel, m))

cf(a::SincKernel, t) = one(Float64)*(-1/a.h <= t <= 1/a.h)

function pdf(a::SincKernel, t)
   if t==zero(Float64)
       return(one(Float64)/pi/a.h)
   else
       return(sin(t/a.h)/pi/t)
   end
end

struct DeLaValleePoussinKernel <: ContinuousUnivariateDistribution
   h::Float64 #Bandwidth
end


default_bandwidth(a::Type{DeLaValleePoussinKernel}, m) = 1.5/sqrt(log(m))

function DeLaValleePoussinKernel(; m = 1_000)
     DeLaValleePoussinKernel(default_bandwidth(DeLaValleePoussinKernel, m))
end

function cf(a::DeLaValleePoussinKernel, t)
   if abs(t * a.h) <= 1
       return(one(Float64))
   elseif abs(t * a.h) <= 2
       return(2*one(Float64) - abs(t * a.h))
   else
       return(zero(Float64))
   end
end

function pdf(a::DeLaValleePoussinKernel, t)
   if t==zero(Float64)
       return(3*one(Float64)/2/pi/a.h)
   else
       return(a.h*(cos(t/a.h)-cos(2*t/a.h))/pi/t^2)
   end
end


Base.@kwdef struct KDEInfinityBand{T<:Real}
    C∞::T
    a_min::T
    a_max::T
    fitted_kde
    interp_kde
    kernel
end


function certainty_banded_KDE(Xs, a_min, a_max;
                        npoints = 4096,
                        kernel=DeLaValleePoussinKernel,
                        bandwidth = default_bandwidth(kernel, length(Xs)),
                        nboot = 1_000)

    kernel = kernel(bandwidth)
    h = bandwidth
    m = length(Xs)

    lo, hi = extrema(Xs)
    lo_kde, hi_kde = min(a_min, lo - 6*h), max(a_max, hi + 6*h)
    midpts = range(lo_kde, hi_kde; length = npoints)

    fitted_kde = kde(Xs, KernelDensity.UniformWeights(m), midpts, kernel)
    interp_kde = InterpKDE(fitted_kde)
    midpts_idx = findall( (midpts .>= a_min) .& (midpts .<= a_max) )
    C∞_boot = Vector{Float64}(undef, nboot)



    for k =1:nboot
        # Poisson bootstrap to estimate certainty band
        Z_pois = rand(Poisson(1), m)
        ws =  Weights(Z_pois/sum(Z_pois))
        f_kde_pois =  kde(Xs, ws, midpts, kernel)
        C∞_boot[k] = maximum(abs.(fitted_kde.density[midpts_idx] .- f_kde_pois.density[midpts_idx]))
    end

    C∞ = median(C∞_boot)

    KDEInfinityBand(C∞=C∞,
                    a_min=a_min,
                    a_max=a_max,
                    fitted_kde=fitted_kde,
                    interp_kde=interp_kde,
                    kernel=kernel)
end

function certainty_banded_KDE(Xs, Zs_discr::DiscretizedStandardNormalSamples; kwargs...)
    a_min, a_max = extrema(Zs_discr.mhist.grid)
    certainty_banded_KDE(Xs, a_min, a_max; kwargs...)
end

function set_neighborhood(Zs_discr::DiscretizedStandardNormalSamples,
                                fkde::MinimaxCalibratedEBayes.KDEInfinityBand;
                                n_interp = 25,
                                η_infl = 1.1) # todo add bias adjustment too
    grid = Zs_discr.mhist.grid
    fkde_interp = fkde.interp_kde.itp
    n = length(Zs_discr)

    f_max = zeros(length(grid) + 1)
    f_min = zeros(length(grid) + 1)
    var_proxy =  zeros(length(grid) + 1)

    C∞ = η_infl * fkde.C∞
    # fill in intermediate points first
    for i in 2:length(grid)
        x_left = grid[i-1]
        x_right = grid[i]
        h = x_right - x_left  # step(grid)...
        interp_res = fkde_interp(range(x_left, x_right, length=n_interp))
        f_max[i] = maximum(interp_res)*h + C∞
        f_min[i] = max( minimum(interp_res)*h - C∞, 0.0)
        var_proxy[i] = max(maximum(interp_res)*h, C∞)
    end

    #fill in first and last element

    # DKW constant with log(n) instead of log(2/alpha)
    C_dkw = sqrt(log(n)/2n)
    f_max[1] = pdf(Zs_discr.mhist)[1] + C_dkw
    f_min[1] = max(pdf(Zs_discr.mhist)[1] - C_dkw, 0.0)
    var_proxy[1] = f_max[1]
    f_max[end] = pdf(Zs_discr.mhist)[end] + C_dkw
    f_min[end] = max(pdf(Zs_discr.mhist)[end] - C_dkw, 0.0)
    var_proxy[end] = f_max[end]

    Zs_discr = @set Zs_discr.var_proxy = var_proxy
    Zs_discr = @set Zs_discr.f_min = f_min
    Zs_discr = @set Zs_discr.f_max = f_max
    Zs_discr
end


function set_neighborhood(Zs::DiscretizedStandardNormalSamples,
                          prior::Distribution;
                          C∞_density = Inf,
                          min_var_proxy = (C∞_density < Inf) ? C∞_density : 0.0)

   f = pdf(marginalize(prior, Zs))
   Zs = @set Zs.var_proxy = f
   C∞ = C∞_density
   if C∞ < Inf
       Zs = @set Zs.f_max = f .+ C∞
       Zs = @set Zs.f_min = max.( f .- C∞, 0.0)
       Zs = @set Zs.var_proxy = max.(f, C∞)
   end

   Zs
end
