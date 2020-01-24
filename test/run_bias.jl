using JuMP
using Gurobi
using MinimaxCalibratedEBayes
using Test
using Plots
pgfplotsx()
gr()
grid = -3:0.06:3
length(grid)
σ = 0.2
gmix_tmp = GaussianMixturePriorClass(σ, grid, Gurobi.Optimizer)

true_dist = MixtureModel(Normal, [(-2,.2), (+2,.2)])

tupl = ( BarConvTol = 1e-12, BarQCPConvTol = 1e-12, FeasibilityTol = 1e-9, OptimalityTol = 1e-9)

gmix_accurate =  GaussianMixturePriorClass(σ, grid,
                                           Gurobi.Optimizer,
                                           tupl)

Z1 = DiscretizedStandardNormalSample(-6:1.0:6)

mhist_mhist = marginalize(true_dist, Z1)
pdf(mhist_mhist)[1:end] =  pdf(mhist_mhist) .+ quantile(pdf(mhist_mhist), 0.1)
pdf(mhist_mhist) .+ 0.01
Z2 = DiscretizedStandardNormalSample(0.0, mhist_mhist)

plot(Z2)

marginal_target = MarginalDensityTarget(StandardNormalSample(0.0))
prior_target = PriorDensityTarget(0.0)

δs = exp10.(range(-3.0, stop=-1.0, length=70))
δs = range(0.001, 0.2, length=50)

δ = 0.001
mm_1 = MinimaxCalibratedEBayes.modulus_problem(Z2, gmix_tmp, prior_target, δ)


mm_2 = MinimaxCalibratedEBayes.modulus_problem(Z2, gmix_accurate, prior_target, δ)

mms = [MinimaxCalibratedEBayes.modulus_problem(Z2, gmix_accurate, marginal_target, δ) for δ in δs]

mms[1]
var_mms = sqrt.(var.(mms))
max_bias_squared_mms = worst_case_bias.(mms)
mse_mms = sqrt.( var_mms.^2 .+ max_bias_squared_mms.^2)

using LaTeXStrings
using Plots
gr()
plot(sqrt.(δs), [var_mms max_bias_squared_mms mse_mms],
                     seriestype = :path,
                     xlabel = L"\delta",
                     linewidth=4.0,
                     color=["orange" "purple" "black"],
                     label=["variance" "max_bias_squared" "mse"])

pl = steinminimaxplot(mms[3])
pl1!



δ = 0.0001
δp = 0.021
mm = MinimaxCalibratedEBayes.modulus_problem(Z2, gmix_tmp, prior_target, δ)


plot(mm.Q)

var(mm)
worst_case_bias(mm)^2




mycols=["#424395" "#018AC4" "#5EC2DA" "#EBC915" "#EB549A" "#550133"]



plot(marginalize(mms[1].g1, mms[1].Z))

pyplot()
pl =steinminimaxplot(mms[30])
plot!(pl[2])
pl

mm_δp = MinimaxCalibratedEBayes.modulus_problem(Z2, gmix_tmp, prior_target, δp)

g1 = mm2[:g1]
g2 = mm2[:g2]

plot(-3:0.01:3, x->pdf(mm2[:g1],x))
plot!(-3:0.01:3, x->pdf(mm2[:g2],x))

#f1_cont = marginalize(g1, StandardNormalSample(0.0))


f1 = marginalize(g1, Z2)
f2 = marginalize(g2, Z2)

plot(f1)
plot!(f2, color="red")
plot!(Z2.mhist, color="black")

ω_δ_prime = -JuMP.dual(mm2[:model][:bound_delta])
ω_δ = mm2[:obj_val]


-JuMP.dual(mm_δp[:model][:bound_delta])

mm2[:obj_val] .- mm_δp[:obj_val]

(mm2[:obj_val] .- mm_δp[:obj_val])/(ω_δ_prime*(δ - δp))


L1 = mm2[:L1]
L2 = mm2[:L2]

Q = ω_δ_prime/δ*(pdf(f2) .- pdf(f1))./pdf(Z2)
Q_0  = (L1+L2)/2 - ω_δ_prime/(2*δ)*sum( (pdf(f2) .- pdf(f1)).* (pdf(f2) .+ pdf(f1)) ./ pdf(Z2))
Q_0_prime  = (L1+L2)/2 - ω_δ_prime/2/δ*sum( (pdf(f2).^2 .- pdf(f1).^2) ./ pdf(Z2))

stein_estimator = DiscretizedAffineEstimator(Z2, Q, Q_0)

max_bias = (ω_δ - δ*ω_δ_prime)/2
lp_worst_bias = worst_case_bias(stein_estimator, Z2, gmix_tmp, prior_target)
max_bias/lp_worst_bias[:max_bias]

plot(Q)

plot(tmp, xlim=(-3.5,3.5))
plot(grid(Z2), Q .+ Q_0)

pdf(f2)[1:5]


plot(pdf(f1))
plot!(pdf(f2), color="red")
plot!(pdf(Z2), color="black")


L1 = mm2[:L1]
L2 = mm2[:L2]

marginal_target(Normal(0,1))
L_mid = (L1 + L2)/2

@test mm2[:obj_val] ≈ L2-L1



gr()
plot(f2)
plot!(f1, color="red")




mm2[:obj_val]

mm2_model = mm2[:model]

normalized_rhs(mm2_model[:bound_delta])
set_normalized_rhs(mm2_model[:bound_delta], 0.1)
optimize!(mm2_model)

mm = MinimaxCalibratedEBayes.modulus_problem(Z2, gmix_tmp, marginal_target, 0.1)
m1 = mm[:model]
mmc = m1[:bound_delta]
dual(mmc)
normalized_rhs(mmc)
obj_1 = mm[:obj_val]

mm_plus_delta = MinimaxCalibratedEBayes.modulus_problem(Z2, gmix_tmp, marginal_target, 0.1001)
obj_2 = mm_plus_delta[:obj_val]

obj_2 - obj_1
dual(mmc)*(0.1001 - 0.1)


mmc = m1[:pseudo_chisq_constraint]


dual(m1, ""

mm[:g1]
plot(-3:0.01:3, x->pdf(mm[:g1],x))
plot!(-3:0.01:3, x->pdf(mm[:g2],x), color="red")

plot(-5:0.01:5, x->pdf(mm2[:g1],x))
plot!(-5:0.01:5, x->pdf(mm2[:g2],x), color="red")




# In fact all computations with marginalized object.

# param -> A*x

# finite-dim representation, here: probs
# want to map: probs ->
