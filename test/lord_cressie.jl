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

Zs_full = Empirikos.MultinomialSummary(BinomialSample.(lord_cressie.x, 20),
                                  lord_cressie.N2)

Zs = Empirikos.MultinomialSummary(BinomialSample.(lord_cressie.x, 20),
                                  lord_cressie.N1)


Zs_full = deepcopy(Zs)
n0 = pop!(Zs_full.store, BinomialSample(0, 20))
n1 = pop!(Zs_full.store, BinomialSample(1, 20))
updated_keys =  [BinomialSample(Interval(0,1), 20); collect(keys(Zs_full))]
updated_values = [n0+n1; collect(values(Zs_full))]
Zs_full = Empirikos.MultinomialSummary(updated_keys, updated_values)




quiet_mosek = optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)

empirical_probs = StatsBase.weights(Zs)/nobs(Zs)

plot(0:20, sqrt.(empirical_probs), seriestype=:sticks, frame=:box,
            grid=nothing, color=:grey, markershape=:circle,
            markerstrokealpha = 0, ylim=(-0.001,sqrt(0.15)),
            xguide=L"x",yguide=L"\sqrt{\widehat{f}_n(x)}",thickness_scaling=1.3,
            label=nothing, size=(500,350))

savefig("lord_cressie_pdf.tikz")


gcal = DiscretePriorClass(range(0.0,stop=1.0,length=300))
postmean_targets = Empirikos.PosteriorMean.(BinomialSample.(0:20,20))


chisq_nbhood = Empirikos.ChiSquaredNeighborhood(0.05)
fitted_chisq_nbhood = StatsBase.fit(chisq_nbhood, Zs_full)

nbhood_method_chisq = NeighborhoodWorstCase(neighborhood = chisq_nbhood,
                                       convexclass= gcal, solver=Mosek.Optimizer)

chisq_cis = confint.(nbhood_method_chisq, postmean_targets, Zs_full)

lower_chisq_ci = getproperty.(chisq_cis, :lower)
upper_chisq_ci = getproperty.(chisq_cis, :upper)


fitted_dkw = fit(DvoretzkyKieferWolfowitz(0.05), Zs_full)

nbhood_method_dkw = NeighborhoodWorstCase(neighborhood = DvoretzkyKieferWolfowitz(0.05),
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
plot!(lord_cressie.x, [lord_cressie.Lower2 lord_cressie.Upper2], seriestype=:scatter,  markershape=:xcross,
            label=["Cressie" nothing])
plot!(lord_cressie.x, [lord_cressie.Lower1 lord_cressie.Upper1], seriestype=:scatter,  markershape=:xcross,
             label=["Cressie" nothing])
plot!(0:20, [lower_dkw_ci upper_dkw_ci], seriestype=:scatter,  markershape=:utriangle,
             label=["DKW-Loc" nothing], color=:blue)
#plot( (0:20), upper_ci_lam, fillrange=lower_ci_lam ,seriestype=:sticks, label="Minimax")


plot(0:20, upper_lam_ci, fillrange=lower_lam_ci ,seriestype=:sticks,
            frame=:box,
            grid=nothing,
            xguide = L"x",
            yguide = L"\EE{\mu \mid X=x}",
            legend = :topleft,
            linewidth=2,
            linecolor=:blue,
            alpha = 0.4,
            background_color_legend = :transparent,
            foreground_color_legend = :transparent, ylim=(-0.01,1.01), thickness_scaling=1.3,
            label="Affine Minimax",
            size=(500,350))

plot!(0:20, [lower_chisq_ci upper_chisq_ci], seriestype=:scatter,  markershape=:hline,
            label=[L"\chi^2\textrm{-Loc}" nothing], markerstrokecolor= :darkorange, markersize=4.5)

plot!(0:20, [lower_dkw_ci upper_dkw_ci], seriestype=:scatter,  markershape=:circle,
             label=["DKW-Loc" nothing], color=:black, alpha=0.9, markersize=2.0, markerstrokealpha=0)

savefig("lord_cressie_posterior_mean.tikz")




lam_chisq = MinimaxCalibratedEBayes.LocalizedAffineMinimax(
                            neighborhood = fit(Empirikos.ChiSquaredNeighborhood(0.01), Zs_full),
                            solver=quiet_mosek, convexclass=gcal, discretizer=nothing,
                            delta_objective = MinimaxCalibratedEBayes.HalfCIWidth(),
                            plugin_G = Empirikos.KolmogorovSmirnovMinimumDistance(gcal, quiet_mosek, nothing))

lam_dkw = MinimaxCalibratedEBayes.LocalizedAffineMinimax(neighborhood = Empirikos.DvoretzkyKieferWolfowitz(0.01),
                            solver=quiet_mosek, convexclass=gcal, discretizer=nothing,
                            delta_objective = MinimaxCalibratedEBayes.HalfCIWidth(),
                            plugin_G = Empirikos.KolmogorovSmirnovMinimumDistance(gcal, quiet_mosek, nothing))


tmp2 = confint(lam_chisq, numerator(postmean_targets[1]), Zs_full)

tmp = fit(lam_chisq, numerator(postmean_targets[1]), Zs)

plot(1:20, collect(values(tmp.Q)), seriestype=:scatter)
plot(support(tmp.g1), probs(tmp.g1), seriestype=:sticks)
plot!(support(tmp.g2), probs(tmp.g2), seriestype=:sticks)

plot(tmp.δs, tmp.δs_objective)
postmean_ci_lam = confint.(lam_chisq, postmean_targets, Zs_full)
postmean_ci_lam_dkw = confint.(lam_dkw, postmean_targets, Zs_full)

lower_lam_ci = getproperty.(postmean_ci_lam, :lower)
upper_lam_ci = getproperty.(postmean_ci_lam, :upper)

lower_lam_dkw_ci = getproperty.(postmean_ci_lam_dkw, :lower)
upper_lam_dkw_ci = getproperty.(postmean_ci_lam_dkw, :upper)
