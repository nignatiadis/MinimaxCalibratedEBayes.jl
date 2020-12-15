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



# χ^2 neighborhood fit
chisq_nbhood = Empirikos.ChiSquaredNeighborhood(0.05)
nbhood_method_chisq = NeighborhoodWorstCase(neighborhood = chisq_nbhood,
                                       convexclass= gcal, solver=Mosek.Optimizer)

chisq_cis = confint.(nbhood_method_chisq, postmean_targets, Zs_full)

lower_chisq_ci = getproperty.(chisq_cis, :lower)
upper_chisq_ci = getproperty.(chisq_cis, :upper)

# DKW neighborhood fit

nbhood_method_dkw = NeighborhoodWorstCase(neighborhood = DvoretzkyKieferWolfowitz(0.05),
                                       convexclass= gcal, solver=Mosek.Optimizer)

dkw_cis = confint.(nbhood_method_dkw, postmean_targets, Zs_full)

lower_dkw_ci = getproperty.(dkw_cis, :lower)
upper_dkw_ci = getproperty.(dkw_cis, :upper)

# Affine Minimax
lam_chisq = MinimaxCalibratedEBayes.LocalizedAffineMinimax(
                            neighborhood = fit(Empirikos.ChiSquaredNeighborhood(0.01), Zs_full),
                            solver=quiet_mosek, convexclass=gcal, discretizer=nothing,
                            delta_objective = MinimaxCalibratedEBayes.HalfCIWidth(),
                            plugin_G = Empirikos.KolmogorovSmirnovMinimumDistance(gcal, quiet_mosek, nothing))



postmean_ci_lam = confint.(lam_chisq, postmean_targets, Zs)
lower_lam_ci = getproperty.(postmean_ci_lam, :lower)
upper_lam_ci = getproperty.(postmean_ci_lam, :upper)

# Plots

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

plot!([0;20], [0.0; 1.0], seriestype=:line, linestyle=:dot, label=nothing, color=:lightgrey)

savefig("lord_cressie_posterior_mean.tikz")
