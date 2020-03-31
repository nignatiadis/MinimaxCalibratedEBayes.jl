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

target_grid = -3:0.2:3
post_means = PosteriorMean.(StandardNormalSample.(target_grid))
lfsrs =  LFSR.(StandardNormalSample.(target_grid))

# ### Empirical Bayes Prior Class
# For the prior class we consider a mixture of Gaussians
# that have standard deviation 0.2 and with mixing measure
# supported on ... Note that we also specify the solver 
# to be used for solving the convex programming problems;
# here we use Mosek.
gcal = GaussianMixturePriorClass(0.2, -3:0.02:3, Mosek.Optimizer)

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
prostate_marginal_plot = histogram(response.(Zs_train), bins=50, normalize=true,
          label="", alpha=0.6, fillcolor=:lightgray)
plot!(fkde_train, label="Kernel estimate \n  (and bands)",
      legend=:topright)


# ### G-model fitting


# We choose the following class 
exp_family = ContinuousExponentialFamilyModel(Uniform(-3.6,3.6),
                         collect(-3.6:0.1:3.6); df=5, scale=true)	 
exp_family_solver = MCEB.ExponentialFamilyDeconvolutionMLE(cefm = exp_family, c0=0.001)

# Let us fit to the training data		 
exp_family_fit = fit(exp_family_solver, Zs_train)

# Let us look at the pilot estimates we got
lfsr_logspline_estimate = estimate.(lfsrs , exp_family_fit)
post_mean_logspline_estimate = estimate.(post_means , exp_family_fit)

expfamily_fits_plot = plot(plot(target_grid, lfsr_logspline_estimate),
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
						            pilot_method = exp_family_fit,
									delta_tuner =  MCEB.HalfCIWidth)#MCEB.RMSE)

post_means_fits = fit.(mceb_setup, post_means)
lfsrs_fits = fit.(mceb_setup, lfsrs)

gr()
prostate_postmeans_plot = plot(post_means_fits, ylims=(-2.5,2.5))
prostate_lfsr_plot = plot(lfsrs_fits, label=nothing)
plot(prostate_lfsr_plot, prostate_postmeans_plot, layout=(1,2))


# # The impact of Neighborhoods: Moving to opportunity

# We next seek to repeat the analysis in the second dataset. Here we will also
# see how most of the steps manually conducted for the prostate dataset
# can be automated.

# ## Loading the dataset
# Let us start by loading the dataset and filtering out missing observations.

using CSV
nbhood_file = MinimaxCalibratedEBayes.chetty_hendren_file
nbhoods_df = CSV.read(nbhood_file, types=Dict(3=>String));
missing_dx = [!ismissing(x) for x in nbhoods_df[!,:p25_coef]]
β_hat = [nbhoods_df[i,:p25_coef] for i in findall(missing_dx)]
se_hat = [nbhoods_df[i,:p25_se] for i in findall(missing_dx)]
nbhood_Zs = StandardNormalSample.(β_hat./se_hat);
length(nbhood_Zs)


# ## Setup the minimax calibrator
# The above steps repeat all the setup we conducted for the prostate data
# to compute the `mceb_setup` object.
Random.seed!(34)
mceb_options = MinimaxCalibratorOptions(prior_class = gcal, 
                               marginal_grid = marginal_grid,
							   pilot_options = exp_family_solver)
nbhood_mceb_setup = fit(mceb_options, nbhood_Zs)


extrema(response.(nbhood_mceb_setup.Zs_train)),extrema(response.(nbhood_mceb_setup.Zs_test))

GR.settextfontprec(232, 3)
gr()
nbhood_post_means_fits = fit.(nbhood_mceb_setup, post_means)
nbhood_lfsrs_fits = fit.(nbhood_mceb_setup, lfsrs)

prostate_postmeans_plot = plot(post_means_fits, ylims=(-3.2,3.2), 
                            fg_legend=:transparent, bg_legend=:transparent,
							legend = (0.35,0.84))
prostate_lfsrs_plot = plot(lfsrs_fits, label=nothing)

nbhood_postmeans_plot = plot(nbhood_post_means_fits,  ylims=(-3.2,3.2), label=nothing)
nbhood_lfsrs_plot = plot(nbhood_lfsrs_fits,  label=nothing)


prostate_marginal_plot = histogram(response.(Zs_train), bins=20, normalize=true,
          label="", alpha=0.4, fillcolor=:lightgray)
plot!(prostate_marginal_plot, fkde_train, fillalpha=0.45, #L"\bar{f}(x)\pm c_m",
      linewidth=1, label=nothing, #L"\bar{f}(x)\pm c_m",
	  legend=:topleft,  bg_legend=:transparent, 
	  fg_legend=:transparent)
	  
nbhood_marginal_plot = histogram(response.(nbhood_mceb_setup.Zs_train), bins=20, normalize=true,
	            label=nothing, alpha=0.4, fillcolor=:lightgray)
plot!(nbhood_marginal_plot, nbhood_mceb_setup.fkde_train, fillalpha=0.45, 
                  label=nothing, linewidth=1)
	  	  	  
plot()
using Plots.Measures
dataset_plot = plot(prostate_marginal_plot, prostate_postmeans_plot, prostate_lfsrs_plot,
	 nbhood_marginal_plot, nbhood_postmeans_plot, nbhood_lfsrs_plot, 
	 title = ["a) Prostate" "b)" "c)" " d) Neighborhoods" "e)" "f)"],
	 layout= (2,3), size=(750,500))
dataset_plot

savefig(dataset_plot, "dataset_plots.pdf")
#savefig(dataset_plot, "mceb_datasets.pdf")
	
	
