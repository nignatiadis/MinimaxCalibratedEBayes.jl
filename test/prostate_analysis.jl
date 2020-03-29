using MinimaxCalibratedEBayes
using RCall
using Plots
using StatsPlots
using LaTeXStrings
using ExponentialFamilies
using MosekTools
using Random

const MCEB = MinimaxCalibratedEBayes

pgfplotsx()


# # Prostate dáta analysis

# ## Preliminaries
 
# Let us make some analysis choices from the beginning 

# ### Empirical Bayes Targets
# Our goal is to estimate the following two targets:

target_grid = -3:0.1:3
post_means = PosteriorMean.(StandardNormalSample.(target_grid))
lfsrs =  LFSR.(StandardNormalSample.(target_grid))

# ### Empirical Bayes Prior Class
# For the prior class we consider a mixture of Gaussians
# that have standard deviation 0.2 and with mixing measure
# supported on ... Note that we also specify the solver 
# to be used for solving the convex programming problems;
# here we use Mosek.
gcal = GaussianMixturePriorClass(0.2, -3:0.01:3, Mosek.Optimizer)

# ### Discretization of marginal observations 
marginal_grid = -6:0.05:6



# ## Load the data
prostz_file = MinimaxCalibratedEBayes.prostate_data_file
R"load($prostz_file)"
@rget prostz;
Zs = StandardNormalSample.(prostz)

# We will implement the preprocessing steps manually here.
# Afterwards, we will use wrapper that automate all of these steps.

# First we split our data into two folds
Random.seed!(200)
Zs_train, Zs_test, _, _ = MCEB._split_data(Zs)

# let us quickly check the range of values in the folds
extrema(response(Zs_train)), extrema(response(Zs_test))

# Now we use the first fold to estimate an $L_\infty$ neighborhood
# of the marginal density. We also fit a logspline model of the
# prior density to get our pilot estimators.

# ### Neighborhood construction:

infinity_band_options = KDEInfinityBandOptions(a_min=-6.0, a_max=6.0)
fkde_train = fit(infinity_band_options, response.(Zs_train))

# Let us visualize the result by showing a histogram of the datapoints
# the estimated density function and the $L_\infty$ bands.
histogram(response.(Zs_train), bins=50, normalize=true,
          label="", alpha=0.4)
plot!(fkde_train, label="Kernel estimate",
      legend=:topright)


# ### G-model fitting


# We choose the following class 
exp_family = ContinuousExponentialFamilyModel(Uniform(-3.6,3.6),
                         collect(-3.6:0.1:3.6); df=5, scale=true)	 
exp_family_solver = MCEB.ExponentialFamilyDeconvolutionMLE(cefm = exp_family, c0=0.001)

# Let us fit to the training data		 
exp_family_fit = fit(exp_family, Zs_train)

# Let us look at the pilot estimates we got
lfsr_logspline_estimate = estimate.(lfsrs , exp_family_fit)
post_mean_logspline_estimate = estimate.(post_means , exp_family_fit)

plot(plot(target_grid, lfsr_logspline_estimate),
     plot(target_grid, post_mean_logspline_estimate))


# discretize our test samples to implement minimax calibration methodology
Zs_test_discr = DiscretizedStandardNormalSamples(Zs_test, marginal_grid)
Zs_test_discr = set_neighborhood(Zs_test_discr, fkde_train)


# add all of the above in one struct
mceb_setup = MinimaxCalibratorSetup(Zs_train = Zs_train,
                                    Zs_test = Zs_test,
						            prior_class = gcal,
						            fkde_train = fkde_train,
						            Zs_test_discr = Zs_test_discr,
						            pilot_method = exp_family_fit)



MCEB.default_δ_min(length(Zs_test), fkde_train.C∞)

calib_fit.δ

post_target1 = LFSR(StandardNormalSample(2.0))
numerator_target = post_target1.num_target
denominator_target = MarginalDensityTarget(post_target1)

estimate(numerator_target, exp_family_fit)



estimate(targ, )
pilot_est = estimate(post_target1, exp_family_fit)
num_est = estimate(numerator_target, exp_family_fit)
denom_est = estimate(denominator_target, exp_family_fit)
num_est/denom_est #oops de einai to idio akribws... -> prosoxh.


calib_target = MCEB.CalibratedTarget(posterior_target = post_target1, 
                                     θ̄ = pilot_est,
									 denom = denom_est)


calib_fit = fit(mceb_setup, calib_target)

mypostfit = fit(mceb_setup, post_target1)


steinminimaxplot(calib_fit)

estimate(calib_fit, calib_target, Zs_test)
target_bias_std(calib_fit, calib_target, Zs_test)




confint(post_target1, mypostfit, Zs_test)

# katse de 8a eprepe na einai 0? :P 
calib_target(exp_family_fit.cef)

denom_est = estimate(, exp_family_fit)
Calib


 



mceb_setup!(fit )
# Let us first try to estimate a linear target
#prior_density_target = LFSRNumerator(StandardNormalSample(1.0))
#prior_target_fit = fit(mceb_setup, prior_density_target)						  
#steinminimaxplot(prior_target_fit)
#estimate(prior_density_target, prior_target_fit, Zs_test)
# 0.211 .177, 0.246
#confint(prior_density_target, prior_target_fit, Zs_test)

SteinMinimaxEstimator
#mystein

using Parameters


# mycols =["#424395" "#EB549A" "#5EC2DA" "#EBC915" "#018AC4"  "#550133"]




#function StatsBase.fit(setup::SteinMinimaxSetup, target::MCEB.LinearEBayesTarget)
#    SteinMinimaxEstimator(setup.Zs_test_discr, setup.prior_class, target, setup.delta_tuner)
#end

#stein1 = StatsBase.fit(linear_setup, prior_target)

Base.@kwdef struct MinimaxCalibratorOptions{SPL,
                                     GCAL <: MCEB.ConvexPriorClass,
									 GR <: AbstractVector,
									 INF,
									 PS,
									 OPT,
									 D <: MCEB.DeltaTuner}
   split::SPL = :random
   prior_class::GCAL
   marginal_grid::GR
   infinity_band_options::OPT = KDEInfinityBandOptions(a_min = minimum(marginal_grid),
                                              a_max = maximum(marginal_grid))
   pilot_options::PS
   tuner::D = MCEB.RMSE(5_000, 0.0)
end




tmp = MinimaxCalibratorOptions2(prior_class = gcal, 
                               marginal_grid = marginal_grid,
							   pilot_options = exp_family_solver)

import StatsBase:fit
using Parameters


function fit(mceb_opt::MinimaxCalibratedEBayesOptions, Zs::AbstractVector{<:StandardNormalSample})
	@unpack split, prior_class, marginal_grid,
	        infinity_band_options, pilot_options, tuner = mceb_opt
	Zs_train, Zs_test, _, _ = MCEB._split_data(Zs, split)

end





