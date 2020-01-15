using MinimaxCalibratedEBayes
using Test
using Plots

mhist =  MCEBHistogram(-5:0.02:5)
marginal_target = MarginalDensityTarget(StandardNormalSample(0.0))

cb_disc = DiscretizedAffineEstimator(mhist, ButuceaComte(marginal_target, 10_000))

pgfplotsx()
plot(cb_disc)
