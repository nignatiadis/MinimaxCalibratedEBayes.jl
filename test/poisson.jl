using Empirikos
using MosekTools
using JuMP
using LaTeXTabulars
using LaTeXStrings
quiet_mosek = optimizer_with_attributes(Mosek.Optimizer,
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => 10^(-15))

zs = 0:7
Ns = [7840; 1317; 239; 42; 14; 4; 4; 1]
Zs = Empirikos.MultinomialSummary(PoissonSample.(zs), Ns)


zs = 0:6
Ns = [103704; 14075; 1766; 255; 45; 6; 2]
Zs = Empirikos.MultinomialSummary(PoissonSample.(zs), Ns)

tmp_keys = [PoissonSample.(0:4); PoissonSample(Interval(5,nothing))];
Zs_full = Empirikos.MultinomialSummary(tmp_keys, [Ns[1:5]; 8]);
# code to do below in nicer way?
empirical_probs = StatsBase.weights(Zs)/nobs(Zs)

robbins = empirical_probs[2:end] ./ empirical_probs[1:end-1] .* (1:6)
robbins = [robbins; 0.0]


postmean_targets = PosteriorMean.(keys(Zs))


gcal = DiscretePriorClass(0.0:0.02:8.0)#Empirikos.set_defaults(DiscretePriorClass(), Zs)

npmle_fit = fit(NPMLE(gcal, quiet_mosek), Zs)

npmle_postmeans = postmean_targets.(npmle_fit)


[robbins npmle_postmeans]

fitted_dkw = fit(DvoretzkyKieferWolfowitz(0.05), Zs)

nbhood_method_dkw = NeighborhoodWorstCase(neighborhood = fitted_dkw,
                                          convexclass= gcal, solver=quiet_mosek)


dkw_cis = confint.(nbhood_method_dkw, postmean_targets, Zs)

lower_dkw_ci = getproperty.(dkw_cis, :lower)
upper_dkw_ci = getproperty.(dkw_cis, :upper)

# χ^2 neighborhood fit
chisq_nbhood = Empirikos.ChiSquaredNeighborhood(0.05)
nbhood_method_chisq = NeighborhoodWorstCase(neighborhood = chisq_nbhood,
                                       convexclass= gcal, solver=quiet_mosek)

chisq_cis = confint.(nbhood_method_chisq, postmean_targets, Zs_full)


# Affine Minimax
lam_chisq = Empirikos.LocalizedAffineMinimax(
    neighborhood = fit(Empirikos.ChiSquaredNeighborhood(0.01), Zs_full),
    solver=quiet_mosek, convexclass=gcal, discretizer=nothing)



postmean_ci_lam = confint.(lam_chisq, postmean_targets, Zs_full)
lower_lam_ci = getproperty.(postmean_ci_lam, :lower)
upper_lam_ci = getproperty.(postmean_ci_lam, :upper)

γ_hat_mom_textbook =  1.001
rate_hat_textbook = 6.458
gamma_hat = Gamma(γ_hat_mom_textbook, 1/rate_hat_textbook)


theta_hat_parametric = postmean_targets.(gamma_hat)


pdf.(marginalize(PoissonSample(), gamma_hat), 0)*nobs(Zs_full)
pdf.(marginalize(PoissonSample(), gamma_hat), 1)*nobs(Zs_full)
pdf.(marginalize(PoissonSample(), gamma_hat), 3)*nobs(Zs_full)
pdf.(marginalize(PoissonSample(), gamma_hat), 4)*nobs(Zs_full)
pdf.(marginalize(PoissonSample(), gamma_hat), 5)*nobs(Zs_full)
pdf.(marginalize(PoissonSample(), gamma_hat), 6)*nobs(Zs_full)


tbl_spec = Tabular("lllllll")
line1 = ["z",
         L"\# Z_i =z",
         L"\widehat{\theta}_{\Gamma}(z)",
         L"\widehat{\theta}_{\text{Robbins}}(z)",
         L"\widehat{\theta}_{\text{NPMLE}}(z)",
         "Simultaneous interval",
         "Pointwise interval"]
lines  = [line1, Rule()]
for (i,z) in enumerate(zs)
    chisq_low = round(chisq_cis[i].lower, digits=2)
    chisq_hi  = round(chisq_cis[i].upper, digits=2)
    chisq_ci = "$(chisq_low)-$(chisq_hi)"

    lam_low = round(postmean_ci_lam[i].lower, digits=2)
    lam_hi  = round(postmean_ci_lam[i].upper, digits=2)
    lam_ci = "$(lam_low)-$(lam_hi)"
    line = Any[z, Ns[i], round(theta_hat_parametric[i], digits=2),
                round(robbins[i], digits=2),
                round(npmle_postmeans[i], digits=2),
                chisq_ci, lam_ci]
    push!(lines, line)
end
latex_tabular("bichsel.tex", tbl_spec, lines)
