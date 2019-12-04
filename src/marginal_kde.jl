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


default_bandwidth(a::Type{DeLaValleePoussinKernel}, m) = 1/sqrt(log(m))

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