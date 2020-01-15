using MinimaxCalibratedEBayes
using Test
using Plots

mhist =  MCEBHistogram(-6:0.02:6)
marginal_target = MarginalDensityTarget(StandardNormalSample(0.0))

marginal_target_cb = DiscretizedAffineEstimator(mhist, ButuceaComte(marginal_target, 10_000))

#test for symmetry
@test marginal_target_cb(-5.0001) == marginal_target_cb(+5.0001)


pgfplotsx()
plot(cb_disc)

prior_target = PriorDensityTarget(0.0)

prior_target_cb = DiscretizedAffineEstimator(mhist, ButuceaComte(prior_target, 10_000))

@test prior_target_cb(-5.0001) == prior_target_cb(+5.0001)

plot(prior_target_cb)


gmix_tmp = GaussianMixturePriorClass(0.1, [0.0], Gurobi.Optimizer)

Z = DiscretizedStandardNormalSample(1.0, mhist)

res = worst_case_bias(marginal_target_cb,
                 Z,
                 gmix_tmp,
                 marginal_target)

tmp = JuMP.value.(res[1][:πs])

findmax(tmp)

res2 = worst_case_bias(prior_target_cb,
                 Z,
                 gmix_tmp,
                 prior_target) #point is, I can check bias...

sum(prior_target_cb.Q .* prob_mass) - pdf(Normal(0.0, 0.1), 0.0)
prob_mass = pdf(marginalize(Normal(0.0, 0.1),Z))

sum(prob_mass)

integrate()

# check bias now.
