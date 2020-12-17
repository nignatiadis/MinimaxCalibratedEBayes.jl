using Empirikos
using JLD2
using DataFrames
using StatsPlots

pgfplotsx()

ts= -3:0.2:3

targets = Empirikos.PosteriorMean.(StandardNormalSample.(ts))
prior = Empirikos.AshPriors[:Spiky]

JLD2.@load "spiky_postmean.jld2"
JLD2.@load "spiky_lfsr.jld2"

methods = [:dkw_scalemix;
           :kde_scalemix;
           :lam_scalemix;
           :dkw_locmix;
           :kde_locmix;
           :lam_locmix]

_df = DataFrame(target = Any[],
    t = Float64[],
    method = Symbol[],
    cover = Bool[],
    ground_truth = Float64[],
    lower = Float64[],
    upper = Float64[],
    ci_length = Float64[],
    id = Int64[]
)

for (i, ci) in enumerate(ci_list)
    for method in methods
        ci_method = ci[method]
        if !isa(ci_method, Exception)
            lower = getproperty.(ci_method, :lower)
            upper = getproperty.(ci_method, :upper)
            _targets = getproperty.(ci_method, :target)
            ground_truth = _targets.(prior)
            cover = lower .<= ground_truth .<= upper
            ci_length = upper .- lower
            append!(_df, DataFrame(cover=cover,lower=lower,
                      upper=upper, ci_length = ci_length,
                      ground_truth=ground_truth,
                      t=ts, target=_targets, id=i, method=method))
        end
    end
end

gdf = groupby(_df, [:target; :t; :method])

combined_gdf = combine(gdf, valuecols(gdf) .=> median, nrow)

failure_examples = filter(:nrow =>  <(200), combined_gdf )

@df combined_gdf plot(:t, :ci_length_median, group=:method, ylim=(0,1))


tmp_df = filter(:t => ==(2.2), combined_gdf)
ggdf = groupby(combined_gdf, :method)
ggdf[(method=methods[1],)]
