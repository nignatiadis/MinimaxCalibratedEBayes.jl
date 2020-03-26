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


# # Prostate data analysis

# ## Preliminaries
 
# Let us make some analysis choices from the beginning 

# ### Empirical Bayes Targets
# Our goal is to estimate the following two targets:

target_grid = -3:0.05:3
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
marginal_grid = -6:0.02:6



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

exp_family = ContinuousExponentialFamilyModel(Uniform(-3.6,3.6),
                         collect(-3.6:0.1:3.6); df=5, scale=true)
						 
	
using Optim
using Expectations					 
exp_family_fit = fit(exp_family,
	         Zs_train,
			 integrator = expectation(exp_family.base_measure; n=50),
			 c0 = 0.01,
			 solver = NewtonTrustRegion(),
			 optim_options = Optim.Options(show_trace=true, show_every=1, g_tol=1e-6),
			 initializer = LindseyMethod(500))


lfsr_logspline_estimate = estimate.(lfsrs , exp_family_fit)
post_mean_logspline_estimate = estimate.(post_means , exp_family_fit)

plot(target_grid, lfsr_logspline_estimate)
plot(target_grid, post_mean_logspline_estimate)


# discretize our test samples to implement minimax calibration methodology
Zs_test_discr = DiscretizedStandardNormalSamples(Zs_test, marginal_grid)
Zs_test_discr = set_neighborhood(Zs_test_discr, fkde_train)




# mycols =["#424395" "#EB549A" "#5EC2DA" "#EBC915" "#018AC4"  "#550133"]




#function StatsBase.fit(setup::SteinMinimaxSetup, target::MCEB.LinearEBayesTarget)
#    SteinMinimaxEstimator(setup.Zs_test_discr, setup.prior_class, target, setup.delta_tuner)
#end

#stein1 = StatsBase.fit(linear_setup, prior_target)








