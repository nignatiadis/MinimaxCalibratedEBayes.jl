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



function certainty_banded_KDE(Xs, a_min, a_max;
                        npoints = 4096,
                        kernel=DeLaValleePoussinKernel,
                        bandwidth = default_bandwidth(kernel, length(Xs)),
                        nboot = 1000)

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

    (C∞=C∞, fitted_kde=fitted_kde, interp_kde=interp_kde, kernel=kernel)
end
