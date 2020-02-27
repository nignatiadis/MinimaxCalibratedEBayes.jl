using MinimaxCalibratedEBayes
using EBayes
using ExponentialFamilies
using DataFrames

const MCEB = MinimaxCalibratedEBayes

taus = range( -4,  6, step=0.2)
taus_dense = range(-4,6, step=0.05)
twintower_expfamily = ContinuousExponentialFamilyModel(Uniform(-4, 6), collect(taus); df=8, scale=true)
α_example =  [10; 0.0; 6.0; 9.0; -3.0; -2.0; -8.0; -1.0]
cef_example = twintower_expfamily(α_example)

alpha_level = 0.9
# index settings by...
targets = [LFSR.(StandardNormalSample.(xs));
           PosteriorMean.(StandardNormalSample.(xs))]
nreps = 200

res_tuples = []
for i=1:nreps
  μs = rand(cef_example, n) #
  Zs = StandardNormalSample.(μs .+ randn(n))
  Zs_disc = DiscretizedStandardNormalSamples(Zs, -5:0.05:6)
  fcef = fit(twintower_expfamily, Zs_disc; c0=0.01)
  for t in targets
      true_target = t(cef_example)
      result_nt = MCEB.target_bias_std(t, fcef)
      ci_nt = confint(t, fcef; level=alpha_level, worst_case_bias_adjusted=true)
      res_nt = (target = t, true_dist = cef_example, iteration = i,
                method = "logspline",
                target_value = true_target, result_nt..., lower_ci = ci_nt[1], upper_ci = ci_nt[2])

      push!(res_tuples, res_nt)
  end
end


μs = rand(cef_example, n) #
Zs = StandardNormalSample.(μs .+ randn(n))
Zs_disc = DiscretizedStandardNormalSamples(Zs, -5:0.05:6)
fcef = fit(twintower_expfamily, Zs_disc; c0=10.0)

t = targets[20]
t(cef_example)
MCEB.target_bias_std(t, fcef)
MCEB.target_bias_std(t, fcef; bias_corrected=false)

MCEB.confint(t, fcef)
MCEB.confint(t, fcef; worst_case_bias_adjusted = false)


using DataFrames
tmp_df = DataFrame(res_tuples)

t = targets[10]

using Query

res_tuples_process = @from i in res_tuples begin
       @select  {i..., realized_bias = i.target_value - i.estimated_target,
                       covers = i.lower_ci <= i.target_value <= i.upper_ci,
                       location = response(i.target)}
       @collect DataFrame
end

mydf = @from i in res_tuples_process begin
       @group i by {i.target, i.true_dist, i.method, i.target_value} into g
       @select {key(g)...,
                estimated_target = mean(g.estimated_target),
                coverage = mean(g.covers),
								estimated_bias  = mean(g.estimated_bias),
								estimated_std  = mean(g.estimated_std),
								target_string = MCEB.pretty_label(key(g).target),
								target_location = response(key(g).target)}
       @collect DataFrame
end

using LaTeXStrings
mydf_lfsr = @filter(mydf, _.target_string == 	L"\theta(x) = E[\mu | X=x]") |> DataFrame

plot(mydf_lfsr[!,:target_location], mydf_lfsr[!,:coverage])
plot(mydf_lfsr[!,:target_location], mydf_lfsr[!,:estimated_bias])
plot!(mydf_lfsr[!,:target_location], mydf_lfsr[!,:estimated_std])


function target_text(target::PosteriorTarget{LF} where LF<:LFSRNumerator)
	"LFSR"
end

function pretty_label(target::PosteriorTarget{PM} where PM<:PosteriorMeanNumerator)
	L"\theta(x) = E[\mu | X=x]"
end

describe(mydf[!,:coverage])

mydf[:coverage]
