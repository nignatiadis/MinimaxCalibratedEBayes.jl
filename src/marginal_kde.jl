function DiscretizedAffineEstimator(mhist::MCEBHistogram, kernel::ContinuousUnivariateDistribution)
    DiscretizedAffineEstimator(mhist, x->pdf(kernel, x))
end

"""
    SincKernel(h)
    
Implements the `SincKernel` with bandwidth `h` to be used for kernel density estimation
through the `KernelDensity.jl` package. The sinc kernel is defined as follows: 
```math
K_{\\text{sinc}}(x) = \\frac{\\sin(x)}{\\pi x} 
```
It is not typically used for kernel density estimation, because this kernel is not
a density itself. However, it is particularly well suited to deconvolution problems
and estimation of very smooth densities because its Fourier transform is the following:
```math
K^*_{\\text{sinc}}(t) = \\mathbf 1( t \\in [-1,1])
```
"""
struct SincKernel <: ContinuousUnivariateDistribution
   h::Float64 #Bandwidth
end

_default_bandwidth(::Type{SincKernel}, m::Integer) = 1/sqrt(log(m))
SincKernel(; m = 1_000) = SincKernel(default_bandwidth(SincKernel, m))

cf(a::SincKernel, t) = one(Float64)*(-1/a.h <= t <= 1/a.h)

function pdf(a::SincKernel, t)
   if t==zero(Float64)
       return(one(Float64)/pi/a.h)
   else
       return(sin(t/a.h)/pi/t)
   end
end




"""
    DeLaValleePoussinKernel(h)
    
Implements the `DeLaValleePoussinKernel` with bandwidth `h` to be used for kernel density estimation
through the `KernelDensity.jl` package. The De La Vallée-Poussin kernel is defined as follows: 
```math
K_V(x) = \\frac{\\cos(x)-\\cos(2x)}{\\pi x^2}
```
Its use case is similar to the [`SincKernel`](@ref), however it has the advantage of being integrable
(in the Lebesgue sense). Its Fourier transform is the following:
```math
K^*_V(t) = \\begin{cases} 
 1, & \\text{ if } x\\in[-1,1] \\\\ 
 0, &\\text{ if } |t| \\geq 2 \\\\ 
 2-|t|,& \\text{ if } |t| \\in [1,2]
 \\end{cases}
```
"""
struct DeLaValleePoussinKernel <: ContinuousUnivariateDistribution
   h::Float64 #Bandwidth
end


_default_bandwidth(a::Type{DeLaValleePoussinKernel}, m::Integer) = 1.5/sqrt(log(m))

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

"""
    KDEInfinityBandOptions(; kernel=DeLaValleePoussinKernel
                             bandwidth = nothing,
                             a_min, 
                             a_max,
                             nboot = 1000)

This struct contains hyperparameters that will be used for constructing a neighborhood
of the marginal density. The steps of the method (and corresponding hyperparameter meanings)
are as follows
* First a kernel density estimate ``\\bar{f}`` of the data is fit with `kernel` as the
kernel and `bandwidth` (the default `bandwidth = nothing` corresponds to automatic
bandwidth selection).
* Second, a Poisson bootstrap with `nboot` replication will be used to estimate a ``L_{\\infty}``
neighborhood ``c_m`` of the true density ``f`` which is such that with probability tending to 1:
```math
\\sup_{x \\in [a_{\\text{min}} , a_{\\text{max}}]} | \\bar{f}(x) - f(x)| \\leq c_m
```
Note that the bound is valid from `a_min` to `a_max`. 

## Reference:
  > Paul Deheuvels and Gérard Derzko. Asymptotic certainty bands for kernel density 
  > estimators based upon a bootstrap resampling scheme. In Statistical models and methods 
  > for biomedical and technical systems, pages 171–186. Springer, 2008
"""
Base.@kwdef struct KDEInfinityBandOptions{T<:Real, 
                                         S<:Union{T, Nothing}}
   a_min::T
   a_max::T
   npoints::Integer = 4096
   bandwidth::S = nothing
   kernel = DeLaValleePoussinKernel
   nboot::Integer = 1000
   η_infl::T = 1.1
   n_interp::Integer = 25
end

"""
    KDEInfinityBand
    
The result of running `fit()     
"""
Base.@kwdef struct KDEInfinityBand{T<:Real}
    C∞::T
    a_min::T
    a_max::T
    fitted_kde
    interp_kde
    kernel
    η_infl::T = 1.1
    n_interp::Integer = 25
end

function _default_bandwidth(kernel, Xs::AbstractVector, ::Nothing)
    _default_bandwidth(kernel, Xs)
end 

function _default_bandwidth(kernel, Xs::AbstractVector)
    _default_bandwidth(kernel, length(Xs))
end 

function _default_bandwidth(kernel, Xs::AbstractVector, h::Number)
    h
end 

function StatsBase.fit(opt::KDEInfinityBandOptions, Xs)
    a_min = opt.a_min
    a_max = opt.a_max
    npoints = opt.npoints
    kernel = opt.kernel
    bandwidth = _default_bandwidth(kernel, Xs, opt.bandwidth)
    nboot = opt.nboot
    tmp_res = certainty_banded_KDE(Xs, a_min, a_max; npoints = npoints, 
                         kernel = kernel, bandwidth = bandwidth,
                         nboot = nboot)
    tmp_res = @set tmp_res.η_infl = opt.η_infl
    tmp_res = @set tmp_res.n_interp = opt.n_interp
    tmp_res
end

function certainty_banded_KDE(Xs, a_min, a_max;
                        npoints = 4096,
                        kernel=DeLaValleePoussinKernel,
                        bandwidth = _default_bandwidth(kernel, length(Xs)),
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



# plotting of infinite band




@recipe function f(ctband::KDEInfinityBand;
                        show_bands=true)
    y = ctband.fitted_kde.density
    x = ctband.fitted_kde.x
    xlim --> (ctband.a_min, ctband.a_max)
    ylab --> "Density"

    seriestype  -->  :path
    linewidth --> 2
    color --> :purple
    
    infty_bound = ctband.C∞ * ctband.η_infl
    
    if show_bands
        _ys_lower, _ys_upper = density_bands_to_ribbons(y, infty_bound)
        ribbons --> (_ys_lower, _ys_lower)
        fillalpha --> 0.2
    end

    xguide --> L"x"
    x,y
end


function set_neighborhood(Zs_discr::DiscretizedStandardNormalSamples,
                          fkde::MinimaxCalibratedEBayes.KDEInfinityBand)
    
    @unpack n_interp, η_infl = fkde 
    grid = Zs_discr.mhist.grid
    fkde_interp = fkde.interp_kde.itp
    n = nobs(Zs_discr)

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
        f_max[i] = maximum(interp_res)*h + C∞*h
        f_min[i] = max( minimum(interp_res)*h - C∞*h, 0.0)
        var_proxy[i] = max(maximum(interp_res)*h, C∞*h)
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
