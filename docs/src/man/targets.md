# Empirical Bayes estimands  

In this section we introduce the interface for describing the inferential targets
of the empirical Bayes analysis.

```@setup targets
using MinimaxCalibratedEBayes
``` 

```@docs
MCEB.EBayesTarget
```

Any `target::EBayesTarget` may be used as a function on distributions, e.g., for a [`PriorDensityTarget`](@ref)
```@example targets
target = PriorDensityTarget(2.0)
target(Normal(0,1)) == pdf(Normal(0,1), 2.0)
```

Similarly for a [`MarginalDensityTarget`](@ref)
```@example targets
target = MarginalDensityTarget(StandardNormalSample(1.0))
target(Normal(0,1)) == pdf(Normal(0,sqrt(2)), 1.0)
```

Other interface functions implemented for all `EBayesTarget`s include:
```@docs
extrema(::EBayesTarget)
```

## Linear Functionals
```@docs
MCEB.LinearEBayesTarget
```
### Interface
```@docs
cf(::MCEB.LinearEBayesTarget, t)
```

### Implemented linear targets

```@docs
MarginalDensityTarget
PriorDensityTarget
PriorTailProbability
```

## Posterior Estimands 

```@docs
PosteriorTarget
```

### Posterior mean 

```@docs
PosteriorMean
PosteriorMeanNumerator
```

### Local false sign rate

```@docs 
LFSR
LFSRNumerator
```