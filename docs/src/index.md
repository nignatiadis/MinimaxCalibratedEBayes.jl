# MinimaxCalibratedEBayes.jl


```math
\mu \sim G, \ \ Z \sim \mathcal{N}(\mu, \, 1)
```

The ingredients of the approach are the following.



  >Ignatiadis, Nikolaos, and Stefan Wager. "Bias-Aware Confidence Intervals for Empirical Bayes Analysis." [arXiv:1902.02774](https://arxiv.org/abs/1902.02774) (2019)



## Installation

The package is not available in the Julia registry yet. It may be installed from Github as follows
```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/JuliaApproximation/OscillatoryIntegrals.jl"))
Pkg.add(PackageSpec(url="https://github.com/nignatiadis/Splines2.jl"))
Pkg.add(PackageSpec(url="https://github.com/nignatiadis/ExponentialFamilies.jl"))
Pkg.add(PackageSpec(url="https://github.com/nignatiadis/MinimaxCalibratedEBayes.jl"))
```





