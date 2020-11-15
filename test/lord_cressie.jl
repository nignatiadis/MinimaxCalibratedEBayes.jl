using MinimaxCalibratedEBayes

using CSV
using DataFrames
using Plots
using LaTeXStrings
using MosekTools
using JuMP


pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\PP}[2][]{\mathbb{P}_{#1}\left[#2\right]}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\EE}[2][]{\mathbb{E}_{#1}\left[#2\right]}")


lord_cressie = CSV.File(MinimaxCalibratedEBayes.lord_cressie_file) |> DataFrame

Zs = Empirikos.MultinomialSummary(BinomialSample.(lord_cressie.x, 20),
                                  lord_cressie.N2)

Zs_full = Empirikos.MultinomialSummary(BinomialSample.(lord_cressie.x, 20),
                                  lord_cressie.N1)




quiet_mosek = optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)

empirical_probs_subset = StatsBase.weights(Zs)/nobs(Zs)
empirical_probs = StatsBase.weights(Zs_full)/nobs(Zs_full)

plot(0:20, empirical_probs, seriestype=:sticks, frame=:box,
            grid=nothing, color=:purple,
            xguide=L"x",yguide=L"\widehat{f}_n(x)",thickness_scaling=2, label=nothing)
#plot!((0:20) .+ 0.2, empirical_probs, seriestype=:sticks, frame=:box,
#            grid=nothing, color=:red, label=nothing, linestyle=:dash)


gcal = DiscretePriorClass(range(0,stop=1.0,length=300))
postmean_targets = Empirikos.PosteriorMean.(BinomialSample.(0:20,20))


chisq_nbhood = Empirikos.ChiSquaredNeighborhood(0.05)
fitted_chisq_nbhood = StatsBase.fit(chisq_nbhood, Zs_full)

nbhood_method_chisq = NeighborhoodWorstCase(neighborhood = fitted_chisq_nbhood,
                                       convexclass= gcal, solver=Mosek.Optimizer)

chisq_cis = confint.(nbhood_method_chisq, postmean_targets, Zs_full)

lower_chisq_ci = getproperty.(chisq_cis, :lower)
upper_chisq_ci = getproperty.(chisq_cis, :upper)


fitted_dkw = fit(DvoretzkyKieferWolfowitz(0.05), Zs_full)
nbhood_method_dkw = NeighborhoodWorstCase(neighborhood = fitted_dkw,
                                       convexclass= gcal, solver=Mosek.Optimizer)

dkw_cis = confint.(nbhood_method_dkw, postmean_targets, Zs_full)

lower_dkw_ci = getproperty.(dkw_cis, :lower)
upper_dkw_ci = getproperty.(dkw_cis, :upper)


plot(0:20, upper_chisq_ci, fillrange=lower_chisq_ci ,seriestype=:sticks,
            frame=:box,
            grid=nothing,
            xguide = L"x",
            yguide = L"\EE{\mu \mid X=x}",
            legend = :topleft,
            linewidth=2,
            background_color_legend = :transparent,
            foreground_color_legend = :transparent, ylim=(-0.05,1), thickness_scaling=1.3,
            label=L"\chi^2 \textrm{-Loc}")
plot!(lord_cressie.x, [lord_cressie.Lower1 lord_cressie.Upper1], seriestype=:scatter,  markershape=:xcross,
             label=["Cressie" nothing])
plot!(0:20, [lower_dkw_ci upper_dkw_ci], seriestype=:scatter,  markershape=:utriangle,
             label=["DKW-Loc" nothing], color=:blue)
#plot( (0:20), upper_ci_lam, fillrange=lower_ci_lam ,seriestype=:sticks, label="Minimax")





lam_chisq = MinimaxCalibratedEBayes.LocalizedAffineMinimax(neighborhood = chisq_nbhood,
                            solver=quiet_mosek, convexclass=gcal, discretizer=nothing,
                            delta_objective = MinimaxCalibratedEBayes.HalfCIWidth(),
                            delta_grid = 5:0.5:15,
                            plugin_G = Empirikos.KolmogorovSmirnovMinimumDistance(gcal, quiet_mosek, nothing))


tmp = fit(lam_chisq, Empirikos.PosteriorTargetNumerator(postmean_targets[7]), Zs_full)


plot(tmp.δs, tmp.δs_objective)
#postmean_ci_lam_dkw = [confint(lam_dkw, target, Zs_full) for target in postmean_targets[7:7]]
