# Linear estimation 

using MinimaxCalibratedEBayes
using MosekTools
using Plots
using Setfield
using LaTeXStrings
using LaTeXTabulars

const MCEB = MinimaxCalibratedEBayes


# Define the prior (effect size) distribution we will use throughout
prior_dbn = MixtureModel(Normal, [(-2,0.2), (2, 0.2)])
marginal_grid = -6:0.01:6

# # Marginal Density Target, Gaussian Mixture prior

n_marginal = 10_000


gmix = GaussianMixturePriorClass(0.2, -3:0.01:3, Mosek.Optimizer, (QUIET=true,))


C∞ = 0.02


Zs_big = DiscretizedStandardNormalSamples(-10:0.005:10)
Zs_big_var = @set Zs_big.var_proxy = pdf(marginalize(prior_dbn, Zs_big))
Zs_big_nbhood = set_neighborhood(Zs_big, prior_dbn; C∞ = C∞)
#│ `with_optimizer(Ipopt.Optimizer, max_cpu_time=60.0)` becomes `optimizer_with_attributes(Ipopt.Optimizer, "max_cpu_time" => 60.0)`.
Zs_discr = DiscretizedStandardNormalSamples(marginal_grid)
Zs_discr_var = @set Zs_discr.var_proxy = pdf(marginalize(prior_dbn, Zs_discr))
Zs_discr_nbhood = set_neighborhood(Zs_discr, prior_dbn; C∞ = C∞)

marginal_target = MarginalDensityTarget(StandardNormalSample(0.0))

marginal_fit = SteinMinimaxEstimator(Zs_discr_var, gmix,
                                     marginal_target, MCEB.RMSE(n_marginal, 0.0))
marginal_fit_nbhood = SteinMinimaxEstimator(Zs_discr_nbhood, gmix,
									 marginal_target, MCEB.RMSE(n_marginal, 0.0))

bc_marginal = MCEB.ButuceaComteEstimator(marginal_target;  n=n_marginal)
bc_marginal_affine = DiscretizedAffineEstimator(Zs_big_var.mhist, bc_marginal)




function se_est(estimator, Zs, n)
	first_moment = sum(estimator.Q .* Zs.var_proxy)
	second_moment = sum(estimator.Q.^2 .* Zs.var_proxy)
	sqrt(second_moment - first_moment^2)/sqrt(n)
end 

function latex_minimax_tbl(tbl_name, estimator_sets, prior_class, target, n; rounding_digits=4)
	line1=["", L"\se[G]{\hat{L}}",L"\sup_{G \in \mathcal{G}}\lvert\Bias[G]{\hat{L},L}\rvert", L"\sup_{G \in \mathcal{G}_m}\lvert\Bias[G]{\hat{L},L}\rvert"]
	lines= [line1, Rule(),]
	for (est_name, estimator_set) in estimator_sets
		estimator = estimator_set[1]
		Zs = estimator_set[2]
		Zs_nbhood = estimator_set[3]
		bias_nonbhood = worst_case_bias(estimator, Zs, prior_class, target).max_abs_bias
		bias_nbhood = worst_case_bias(estimator, Zs_nbhood, prior_class, target).max_abs_bias
		se_calc = se_est(estimator, Zs, n)
		se_calc, bias_nonbhood, bias_nbhood = round.( (se_calc, bias_nonbhood, bias_nbhood), digits=rounding_digits)
		push!(lines, [est_name, se_calc, bias_nonbhood, bias_nbhood])
	end
	latex_tabular(tbl_name, Tabular("l|ccc"), lines) 
end 




latex_minimax_tbl("marginal_density_affine.tex",
                  ["Minimax" => (marginal_fit.Q,Zs_discr_var, Zs_discr_nbhood),
                   L"Minimax-$\infty$" => (marginal_fit_nbhood.Q, Zs_discr_var, Zs_discr_nbhood),
				   "Butucea-Comte" => (bc_marginal_affine, Zs_big_var, Zs_big_nbhood)],  
                   gmix, marginal_target, n_marginal)


using PGFPlotsX
pgfplotsx()
def_size = (850,440)
third_col = RGB(68/255, 69/255, 145/255)
import Plots:pgfx_sanitize_string
pgfx_sanitize_string(s::AbstractString) = s


marginal_density_affine = steinminimaxplot(marginal_fit, marginal_fit_nbhood; size=def_size,
									                   ylim_relative_offset=1.4)

plot!(marginal_density_affine[1], marginal_grid, bc_marginal.(marginal_grid), 
      linestyle=:dot, linecolor = third_col, label="Butucea-Comte")

savefig(marginal_density_affine, "marginal_density_affine.pdf")




n_prior_cdf = 200
prior_cdf_target = PriorTailProbability(0.0)


prior_cdf_fit = SteinMinimaxEstimator(Zs_discr_var, gmix,
                                      prior_cdf_target, MCEB.RMSE(n_prior_cdf, 0.0))

prior_cdf_fit_nbhood = SteinMinimaxEstimator(Zs_discr_nbhood, gmix,
									  prior_cdf_target, MCEB.RMSE(n_prior_cdf, 0.0))


									  
latex_minimax_tbl("prior_cdf_affine.tex",
                ["Minimax" => (prior_cdf_fit.Q,Zs_discr_var, Zs_discr_nbhood),
                 L"Minimax-$\infty$" => (prior_cdf_fit_nbhood.Q, Zs_discr_var, Zs_discr_nbhood)],  
                 gmix, prior_cdf_target, n_prior_cdf)

prior_cdf_affine = steinminimaxplot(prior_cdf_fit, prior_cdf_fit_nbhood; size=def_size,
									  					 ylim_relative_offset=1.4)

savefig(prior_cdf_affine, "prior_cdf_affine.pdf") 

n_prior_density = 200
prior_density_target = PriorDensityTarget(0.0)


prior_dbn_normal = Normal(0, 2)

using QuadGK
myf = quadgk(t->abs2(cf(prior_dbn,t))*(t^2+1)^2, -Inf, +Inf)[1]/(2*π)
#│ `with_optimizer(Ipopt.Optimizer, max_cpu_time=60.0)` becomes `optimizer_with_attributes(Ipopt.Optimizer, "max_cpu_time" => 60.0)`.
Zs_discr_var_normal = @set Zs_discr.var_proxy = pdf(marginalize(prior_dbn_normal, Zs_discr))
Zs_discr_nbhood_normal = set_neighborhood(Zs_discr, prior_dbn_normal; C∞ = C∞)

Zs_big = DiscretizedStandardNormalSamples(-15:0.01:15)
Zs_big_var_normal = @set Zs_big.var_proxy = pdf(marginalize(prior_dbn_normal, Zs_big))
Zs_big_nbhood_normal = set_neighborhood(Zs_big, prior_dbn_normal; C∞ = C∞)


hmclass = MCEB.HermitePriorClass(qmax=90, sobolev_order=2, sobolev_bound=0.5,
                                 solver=Mosek.Optimizer, solver_params = (QUIET=true,))

prior_density_fit = SteinMinimaxEstimator(Zs_discr_var_normal, hmclass,
								          prior_density_target, MCEB.RMSE(n_prior_density,0.0))

prior_density_fit_nbhood = SteinMinimaxEstimator(Zs_discr_nbhood_normal, hmclass,
								   prior_density_target, MCEB.RMSE(n_prior_density,0.0))

bc_prior = MCEB.ButuceaComteEstimator(prior_density_target;  n=n_prior_density)
bc_prior_affine_small = DiscretizedAffineEstimator(Zs_discr_var_normal.mhist, bc_prior)

bc_prior_affine = DiscretizedAffineEstimator(Zs_big_var_normal.mhist, bc_prior)

#0.0856
tmp1 = worst_case_bias(bc_prior_affine, Zs_big_var_normal, hmclass, prior_density_target)
tmp2 = worst_case_bias(bc_prior_affine_small, Zs_discr_var_normal, hmclass, prior_density_target)


tmp4 = worst_case_bias(bc_prior_affine, Zs_big_nbhood_normal, hmclass, prior_density_target)




plot(bc_prior_affine)
ys = pdf.(tmp4.min_g, -15:0.01:15)

plot(-15:0.01:15, ys)

latex_minimax_tbl("prior_density_affine.tex",
                  ["Minimax" => (prior_density_fit.Q,Zs_discr_var_normal, Zs_discr_nbhood_normal),
                   L"Minimax-$\infty$" => (prior_density_fit_nbhood.Q, Zs_discr_var_normal, Zs_discr_nbhood_normal),
				   "Butucea-Comte" => (bc_prior_affine, Zs_big_var_normal, Zs_big_nbhood_normal)],  
                   hmclass, prior_density_target, n_prior_density)




prior_density_affine = steinminimaxplot(prior_density_fit, prior_density_fit_nbhood; size=def_size,
									     ylim_relative_offset=1.3, ylim_panel_e= (0,0.33))

plot!(prior_density_affine[1], marginal_grid, bc_prior.(marginal_grid), 
      linestyle=:dot, linecolor = third_col, label="Butucea-Comte")

savefig(prior_density_affine, "prior_density_affine.pdf")



using LaTeXTabulars
using LaTeXStrings




latex_tabular("tmp.tex", Tabular("l|ccc"), 
                      [line1,
					   Rule(),
					   line2,
					   line3,
					   line4]) 
					  



