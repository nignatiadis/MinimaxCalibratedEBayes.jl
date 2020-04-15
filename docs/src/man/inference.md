# Confidence intervals  

The basic function to construct confidence intervals is the following
```@docs
StatsBase.confint(::EBayesTarget, ::Any, ::Any)
```
Note that when the estimator is a [`SteinMinimaxEstimator`](@ref), it is typically
assumed that `Zs` is derived from data that is separate from the bata used in the
construction of the quasi-minimax estimator.

## Typical workflow
A typical workflow looks as follows (see data analysis tutorial for two examples):

First, set up a [`MinimaxCalibratorOptions`](@ref) instance:

```julia
mceb_options = MinimaxCalibratorOptions(prior_class = ..., marginal_grid = ..., pilot_options = ...)
```

Next fit `mceb_options` to the data and thereby construct a [`MinimaxCalibratorSetup`](@ref) object.
This contains for example the fitted [`KDEInfinityBand`](@ref).

```julia							   
mceb_setup = fit(mceb_options, nbhood_Zs)
```

Next for a given empirical Bayes target, we compute its pilot estimator, as well as the
minimax linear estimator for the calibrated target:

```julia
target = PosteriorMean(StandardNormalSample(1.0))
calibrated_estimator_fit = fit(mceb_setup, target)
```

Finally we can compute confidence interval as

```julia
confint(target, calibrated_estimator_fit)
```

## Reference 

```@docs
MinimaxCalibratorOptions
MinimaxCalibratorSetup
```


