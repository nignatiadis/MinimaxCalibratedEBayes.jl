# compare output from KernelDensity.jl with sinc kernel
# to what Comte-Butucea yields. (should be the same)
# also look at how DeLaValleePoussin performs

using MinimaxCalibratedEBayes
using Test
using Distributions
using Random


prior_dist = MixtureModel(Normal, [(-0.3,0.5), (1.05,0.5)])
marginal_dist = marginalize(prior_dist, StandardNormalSample(NaN))

n = 100_000
Random.seed!(1)
Xs = rand(marginal_dist, n)
extrema(Xs)



certainty_kde_res_dlv = certainty_banded_KDE(Xs, -4, 4; kernel=DeLaValleePoussinKernel)
certainty_kde_res_sinc = certainty_banded_KDE(Xs, -4, 4; kernel=SincKernel)

fitted_kde = certainty_kde_res[:fitted_kde]

f_true = pdf.(marginal_dist, fitted_kde.x)
C∞ = certainty_kde_res[:C∞]*1.01

true_C = maximum(abs.(f_true .- fitted_kde.density))


using MosekTools

xgrid = -3:0.01:3
σ = 0.5
gmix_tmp = GaussianMixturePriorClass(σ, xgrid, Mosek.Optimizer)

Z1 = DiscretizedStandardNormalSample(-6:0.01:6)
LinCalibSinc = DiscretizedAffineEstimator(Z1, certainty_kde_res_sinc[:kernel])
LinCalibDLV = DiscretizedAffineEstimator(Z1, certainty_kde_res_sinc[:kernel])


max_bias
plot(LinCalib)

max_bias_res = worst_case_bias(LinCalib,
                               Z1,
                               gmix_tmp,
                               MarginalDensityTarget(StandardNormalSample(0.0));
                               maximization=true)
max_bias_res[:max_bias]

using Plots
pgfplotsx()

y = fitted_kde.density
infty_bound = C∞
lower_lims = y .- max.(y .- infty_bound, 0)
upper_lims = fill(infty_bound, length(lower_lims))

plot(fitted_kde.x, fitted_kde.density,
       linestyle=:dash,
       ribbon = (lower_lims, upper_lims),
       fillcolor="red", fillalpha=0.3)
plot!(fitted_kde.x, f_true, color="black", linestyle=:solid)
#----------------------------------------
#---- Test with actual Sinc Kernel-------
#----------------------------------------
f_sinc = sinc_kde(Xs, marginal_grid, SincKernel)


x_tst_1 = f_sinc.x[700]
f_tst_1 = f_sinc.density[700]

# Let us compare to the result from comte_butucea
f_cbt_1 = estimate(Xs, ComteButucea,
  MarginalDensityTarget(x_tst_1), marginal_grid)

@test f_cbt_1 ≈ f_tst_1 atol=0.001

#----- Check default dispatch ---------------------------------
dv_kde = sinc_kde(Xs, marginal_grid, DeLaValleePoussinKernel)
dv_kde_auto_dispatch = sinc_kde(Xs, marginal_grid)
@test dv_kde.density == dv_kde_auto_dispatch.density

#--------------------------------------------------------------
#---- Tests with both Sinc and DeLaValleePoussin Kernels-------
#--- Are our bands longer than true bands?
#--------------------------------------------------------------

f_true = pdf.(Ref(d_true), marginal_grid)

for T in [SincKernel, DeLaValleePoussinKernel]
  f_kde= sinc_kde(Xs, marginal_grid, T)
  true_C = maximum(abs.(f_true .- f_kde.density))

  f_nb = fit(BinnedMarginalDensityNeighborhood, Xs, marginal_grid, T)
  f_nb.C_std

  f_nb_ds = fit(BinnedMarginalDensityNeighborhood, Xs, ds, T)
  @show true_C, f_nb_ds.C_std, f_nb_ds.C_bias, f_nb_ds.C_std + f_nb_ds.C_bias
  @test true_C <= f_nb_ds.C_std + f_nb_ds.C_bias
end
