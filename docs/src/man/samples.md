# Discretization and Localization

## Discretized Empirical Bayes samples

We represent Standard Normal samples that have been discretized through the following type:

```@docs
DiscretizedStandardNormalSamples
```




## Localization
The methodology is most effective when the marginal distribution of the $Z_i$ is constrained. 
This is achieved through the `set_neighborhood` function:
 

```@docs
set_neighborhood(::DiscretizedStandardNormalSamples,::KDEInfinityBand)
set_neighborhood(::DiscretizedStandardNormalSamples,::Distribution)
```
## Neighborhood estimation

Here we explain how to construct a neighborhood band which can be used with `set_neighborhood` function described above, based on the data. 

We currently provide the Poisson bootstrap of Deheuvels and Derzko, which can be instantiated
with the following options:

```@docs
KDEInfinityBandOptions
```

The following kernels can be used:

```@docs
DeLaValleePoussinKernel
SincKernel
```

The output of fitting a `KDEInfinityBandOptions` object to data is a `KDEInfinityBand` object.

```@docs
KDEInfinityBand
```