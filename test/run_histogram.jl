using MinimaxCalibratedEBayes
using Test
using Plots
pgfplotsx()


xs = range(-3,3;step=0.5)
ws = Float64.(collect(1:14))
tmp_hist = MCEBHistogram(xs)

@test StatsBase.midpoints(tmp_hist.grid) == StatsBase.midpoints(tmp_hist)


calib_normal = DiscretizedAffineEstimator(tmp_hist, Normal(0, 0.5))
calib_sin = DiscretizedAffineEstimator(tmp_hist, sin)

calib_sinc = DiscretizedAffineEstimator(tmp_hist, SincKernel(m=10_000))

calib_array = [calib_sin, calib_normal, calib_sinc]
@test isa(calib_array, AbstractVector{<:DiscretizedAffineEstimator})

plot(calib_normal)
plot(calib_sin)

plot(calib_array, label=["bla" "blabla" "sinc"])

z = DiscretizedStandardNormalSample(0.5, tmp_hist)
zs = DiscretizedStandardNormalSample.(randn(100), tmp_hist)


@test var(marginalize(Normal(0,0), StandardNormalSample(1.0))) == 1.0
@test var(marginalize(Normal(0,1), StandardNormalSample(1.0))) ≈ 2.0

tmp_marginalize = marginalize(Normal(), z)
marginalize_probs = pdf(tmp_marginalize)
@test sum(pdf(tmp_marginalize)) == 1

midpoints(marginalize_probs)

plot(marginalize_probs)

first(tmp_hist)

zs[1].mhist
tmp_calib = DiscretizedAffineEstimator(tmp_hist, ws, 1.0)

@test tmp_calib(-4) == 2.00
@test tmp_calib.([-4, 4]) == [2.0, 15.0]
@test tmp_calib.([-4, -3, -2.9, 4]) == [2.0, 2.0, 3.0, 15.0]

@test step(tmp_hist) == step(xs)
@test first(tmp_hist) == first(xs)
@test last(tmp_hist) == last(xs)
