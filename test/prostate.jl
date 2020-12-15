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
gcal_locmix = MixturePriorClass(Normal.(-3:0.05:3, 0.25))


dkw_nbhood = DvoretzkyKieferWolfowitz(0.05)

fitted_dkw = fit(dkw_nbhood, Zs)
plot(fitted_dkw, subsample=300, label="DKW band",
     xlab=L"x", ylab=L"\widehat{F}_n(x)",  size=(380,280))
savefig("prostate_dkw_band.tikz")

infty_nbhood = Empirikos.InfinityNormDensityBand(a_min=-3.0,a_max=3.0);
fitted_infty_nbhood = fit(infty_nbhood, Zs)
fitted_infty_nbhood.C∞



prostate_marginal_plot = histogram(response.(Zs), bins=50, normalize=true,
          label="Histogram", fillalpha=0.4, linealpha=0.4, fillcolor=:lightgray,
          size=(380,280), xlims=(-4.5,4.5))
plot!(prostate_marginal_plot, fitted_infty_nbhood,
      label="KDE band", xlims=(-4.5,4.5),
      yguide=L"\widehat{f}_n(x)", xguide=L"x")
plot!([-3.0;3.0], seriestype=:vline, linestyle=:dot, label=nothing, color=:lightgrey)
savefig("prostate_kde_band.tikz")

nbhood_method_dkw_scalemix = NeighborhoodWorstCase(neighborhood = fitted_dkw,
                               convexclass = gcal_scalemix, solver = quiet_mosek)

nbhood_method_kde_scalemix = NeighborhoodWorstCase(neighborhood = fitted_infty_nbhood,
                               convexclass = gcal_scalemix, solver = quiet_mosek)


nbhood_method_dkw_locmix = NeighborhoodWorstCase(neighborhood = fitted_dkw,
                               convexclass = gcal_locmix, solver = quiet_mosek)

nbhood_method_kde_locmix = NeighborhoodWorstCase(neighborhood = fitted_infty_nbhood,
                               convexclass = gcal_locmix, solver = quiet_mosek)


discr = Empirikos.Discretizer(-3.0:0.005:3.0)

lam_kde_scalemix = Empirikos.LocalizedAffineMinimax(neighborhood = (@set infty_nbhood.α=0.01),
                            discretizer=discr,
                            solver=quiet_mosek, convexclass=gcal_scalemix)

lam_kde_locmix = Empirikos.LocalizedAffineMinimax(neighborhood = (@set infty_nbhood.α=0.01),
                            discretizer=discr,
                            solver=quiet_mosek, convexclass=gcal_locmix)


ts= -3:0.2:3
lfsrs = Empirikos.PosteriorProbability.(StandardNormalSample.(ts), Interval(0,nothing))
postmean_targets = Empirikos.PosteriorMean.(StandardNormalSample.(ts))


# Posterior Mean

postmean_ci_dkw_scalemix = confint.(nbhood_method_dkw_scalemix, postmean_targets, Zs)
postmean_ci_kde_scalemix = confint.(nbhood_method_kde_scalemix, postmean_targets, Zs)
postmean_ci_lam_scalemix = confint.(lam_kde_scalemix, postmean_targets, Zs)

plot(ts, postmean_ci_kde_scalemix, label="KDE-Loc", fillcolor=:darkorange, fillalpha=0.5, ylim=(-2.55,2.55))
plot!(ts, postmean_ci_dkw_scalemix, label="DKW-Loc",show_ribbon=false, alpha=0.9, color=:black)
plot!(ts, postmean_ci_lam_scalemix, label="Affine minimax",show_ribbon=true, fillcolor=:blue, fillalpha=0.4)
plot!([-3.0;3.0], [-3.0; 3.0], seriestype=:line, linestyle=:dot, label=nothing, color=:lightgrey)
plot!(xlabel = L"x", ylabel=L"\EE{\mu \mid X=x}", size=(380,280))

savefig("prostate_scalemix_postmean.tikz")


postmean_ci_dkw_locmix = confint.(nbhood_method_dkw_locmix, postmean_targets, Zs)
postmean_ci_kde_locmix = confint.(nbhood_method_kde_locmix, postmean_targets, Zs)
postmean_ci_lam_locmix = confint.(lam_kde_locmix, postmean_targets, Zs)

plot(ts, postmean_ci_kde_locmix, label="KDE-Loc", fillcolor=:darkorange, fillalpha=0.5, ylim=(-2.55,2.55))
plot!(ts, postmean_ci_dkw_locmix, label="DKW-Loc",show_ribbon=false, alpha=0.9, color=:black)
plot!(ts, postmean_ci_lam_locmix, label="Affine minimax",show_ribbon=true, fillcolor=:blue, fillalpha=0.4)
plot!([-3.0;3.0], [-3.0; 3.0], seriestype=:line, linestyle=:dot, label=nothing, color=:lightgrey)
plot!(xlabel = L"x", ylabel=L"\EE{\mu \mid X=x}", size=(380,280))

savefig("prostate_locmix_postmean.tikz")

# Local false sign rate

lfsr_ci_dkw_scalemix = confint.(nbhood_method_dkw_scalemix, lfsrs, Zs)
lfsr_ci_kde_scalemix = confint.(nbhood_method_kde_scalemix, lfsrs, Zs)
lfsr_ci_lam_scalemix = confint.(lam_kde_scalemix, lfsrs, Zs)

plot(ts, lfsr_ci_kde_scalemix, label="KDE-Loc", fillcolor=:darkorange, fillalpha=0.5, ylim=(0,1))
plot!(ts, lfsr_ci_dkw_scalemix, label="DKW-Loc",show_ribbon=false, alpha=0.9, color=:black)
plot!(ts, lfsr_ci_lam_scalemix, label="Affine minimax",show_ribbon=true, fillcolor=:blue, fillalpha=0.4)
plot!([-3;3], [0.5; 0.5], seriestype=:line, linestyle=:dot, label=nothing, color=:lightgrey)
plot!(xlabel = L"x", ylabel=L"\PP{\mu \geq 0 \mid X=x}", size=(380,280))

savefig("prostate_scalemix_lfsr.tikz")


lfsr_ci_dkw_locmix = confint.(nbhood_method_dkw_locmix, lfsrs, Zs)
lfsr_ci_kde_locmix = confint.(nbhood_method_kde_locmix, lfsrs, Zs)
lfsr_ci_lam_locmix = confint.(lam_kde_locmix, lfsrs, Zs)

plot([-3;3], [0.5; 0.5], seriestype=:line, linestyle=:dot, label=nothing, color=:lightgrey)
plot!(ts, lfsr_ci_dkw_locmix, label="KDE-Loc", fillcolor=:darkorange, fillalpha=0.5, ylim=(0,1))
plot!(ts, lfsr_ci_dkw_locmix, label="DKW-Loc",show_ribbon=false, alpha=0.9, color=:black)
plot!(ts, lfsr_ci_lam_locmix, label="Affine minimax",show_ribbon=true, fillcolor=:blue, fillalpha=0.4)
plot!(xlabel = L"x", ylabel=L"\PP{\mu \geq 0 \mid X=x}", size=(380,280))

savefig("prostate_locmix_lfsr.tikz")
