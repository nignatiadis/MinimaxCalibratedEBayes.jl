# Pilot estimators

## Butucea-Comte 

```@docs
MCEB.ButuceaComteOptions
```

A `target::LinearEBayesTarget`  can be estiamted based on Standard Normal samples `Zs`
with the Butucea-Comte method `bcopt::ButuceaComteOptions` as follows

```julia
estimate(target, bcop, Zs)
```

```@docs
Distributions.estimate(::MCEB.LinearEBayesTarget, ::MCEB.ButuceaComteOptions, ::AbstractVector{<:StandardNormalSample})
``` 





