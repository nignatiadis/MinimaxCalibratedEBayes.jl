using Empirikos

using RCall
using Plots
using StatsPlots
using LaTeXStrings
using MosekTools
using JuMP
using MinimaxCalibratedEBayes
using Setfield

quiet_mosek = optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)

theme(:default;
      background_color_legend = :transparent,
      foreground_color_legend = :transparent,
      grid=nothing,
      frame=:box,
      thickness_scaling=1.3)


pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\PP}[2][]{\mathbb{P}_{#1}\left[#2\right]}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\EE}[2][]{\mathbb{E}_{#1}\left[#2\right]}")

prostz_file = MinimaxCalibratedEBayes.prostate_file
R"load($prostz_file)"
@rget prostz


Zs = StandardNormalSample.(prostz)

gcal_scalemix = Empirikos.set_defaults(GaussianScaleMixtureClass(), Zs; hints = Dict(:grid_scaling => 1.1))
gcal_locmix = MixturePriorClass(Normal.(-3:0.05:3, 0.2))


dkw_nbhood = DvoretzkyKieferWolfowitz(0.05)

fitted_dkw = fit(dkw_nbhood, Zs)
x_dkw = collect(keys(fitted_dkw.summary))
F_dkw = collect(values(fitted_dkw.summary))
plot(response.(x_dkw), F_dkw, ribbon = fitted_dkw.band, label=nothing)


infty_nbhood = Empirikos.InfinityNormDensityBand(a_min=-4.0,a_max=4.0);
fitted_infty_nbhood = fit(infty_nbhood, Zs)
fitted_infty_nbhood.C∞

prostate_marginal_plot = histogram(response.(Zs), bins=50, normalize=true,
          label="", alpha=0.4, fillcolor=:lightgray,
          grid=nothing, size=(500,350), xlim=(-4,4))

plot!(response.(fitted_infty_nbhood.midpoints), fitted_infty_nbhood.estimated_density, ribbon=fitted_infty_nbhood.C∞)


nbhood_method_dkw_scalemix = NeighborhoodWorstCase(neighborhood = fitted_dkw,
                               convexclass = gcal_scalemix, solver = quiet_mosek)

nbhood_method_kde_scalemix = NeighborhoodWorstCase(neighborhood = fitted_infty_nbhood,
                               convexclass = gcal_scalemix, solver = quiet_mosek)


nbhood_method_dkw_locmix = NeighborhoodWorstCase(neighborhood = fitted_dkw,
                               convexclass = gcal_locmix, solver = quiet_mosek)

nbhood_method_kde_locmix = NeighborhoodWorstCase(neighborhood = fitted_infty_nbhood,
                               convexclass = gcal_locmix, solver = quiet_mosek)


discr = Empirikos.Discretizer(-4:0.01:4)

lam_kde_scalemix = Empirikos.LocalizedAffineMinimax(neighborhood = (@set infty_nbhood.α=0.01),
                            discretizer=discr,
                            solver=quiet_mosek, convexclass=gcal_scalemix)

lam_kde_locmix = Empirikos.LocalizedAffineMinimax(neighborhood = (@set infty_nbhood.α=0.01),
                            discretizer=discr,
                            solver=quiet_mosek, convexclass=gcal_locmix)


ts= -3:0.2:3
lfsrs = Empirikos.PosteriorProbability.(StandardNormalSample.(ts), Interval(0,nothing))
postmean_targets = Empirikos.PosteriorMean.(StandardNormalSample.(ts))


postmean_ci_dkw_scalemix = confint.(nbhood_method_dkw_scalemix, postmean_targets, Zs)
postmean_ci_kde_scalemix = confint.(nbhood_method_kde_scalemix, postmean_targets, Zs)
postmean_ci_lam_scalemix = confint.(lam_kde_scalemix, postmean_targets, Zs)

plot(ts, postmean_ci_kde_scalemix, label="KDE-Loc", fillcolor=:darkorange, fillalpha=0.5, ylim=(-2,2))
plot!(ts, postmean_ci_dkw_scalemix, label="DKW-Loc",show_ribbon=false, alpha=0.9, color=:black)
plot!(ts, postmean_ci_lam_scalemix, label="Affine minimax",show_ribbon=true, fillcolor=:blue, fillalpha=0.4)
plot!([-3.0;3.0], [-3.0; 3.0], seriestype=:line, linestyle=:dot, label=nothing, color=:lightgrey)
plot!(xlabel = L"x", ylabel=L"\EE{\mu \mid X=x}", size=(380,250))

savefig("prostate_scalemix_postmean.tikz")

plot!(ts, postmean_targets.(prior))

postmean_ci_dkw_locmix = confint.(nbhood_method_dkw_locmix, postmean_targets, Zs)
postmean_ci_kde_locmix = confint.(nbhood_method_kde_locmix, postmean_targets, Zs)

postmean_ci_lam_locmix = confint.(lam_kde_locmix, postmean_targets, Zs)

confint(nbhood_method_kde_locmix, Empirikos.PosteriorMean.(StandardNormalSample.(0.0)), Zs)



low = getproperty.(postmean_ci_dkw_scalemix, :lower)
upp = getproperty.(postmean_ci_dkw_scalemix, :upper)

low2 = getproperty.(postmean_ci_dkw_locmix, :lower)
upp2 = getproperty.(postmean_ci_dkw_locmix, :upper)

plot(ts, [low upp], xlim=(-3,3))
plot(1, ts, postmean_ci_dkw_scalemix, label="DKW scale")
plot!(1, ts, postmean_ci_kde_scalemix, label="KDE scale", fillcolor=:purple)

plot(ts, [low2 upp2], xlim=(-3,3))

pgfplotsx()
plot(ts, postmean_ci_kde_locmix, label="KDE loc", fillcolor=:blue)
plot(ts, postmean_ci_lam_locmix, label="KEK",show_ribbon=false, linestyle=:solid, xlim=(-3,3))
plot!(ts, postmean_ci_lam_scalemix, label="SCALE",show_ribbon=true, xlim=(-3,3))


hline!([0.0])
plot!([-3.0; 3.0], [-3.0;3.0])
