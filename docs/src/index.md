# MinimaxCalibratedEBayes.jl


```math
\mu \sim G, \ \ Z \sim \mathcal{N}(\mu, \, 1)
```
The ingredients of the approach are the following. #
* The class of potential effect size distributions $\mathcal{G}$.
* The estimator of the marginal density $\bar{f}$ and neighborhood radius $c_m$.
* The choice of $\delta_m$.
* The pilot estimators for. 

## Reference

This package implements the method described in the following paper

  >Ignatiadis, Nikolaos, and Stefan Wager. "Bias-Aware Confidence Intervals for Empirical Bayes Analysis." [arXiv:1902.02774](https://arxiv.org/abs/1902.02774) (2019)

  
!!! note "A remark on notation"
      See the paper for details about the method. Note that the paper uses the notation
      $X_i$ for the Standard Normal Samples, while the documentation and the package
      here use the notation $Z_i$.

The paper provides a general framework for estimation of empirical Bayes estimands and linear
functionals that provides substantial flexibility. The package here has been designed around
this framework with the goal of modularity. Please open an issue in the Github repository if there is 
a combination of estimators/targets/effect size distributions/likelihoods that you would like to use,
which have not been implemented.

## Installation

The package is not available in the Julia registry yet. It may be installed from Github as follows
```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/JuliaApproximation/OscillatoryIntegrals.jl"))
Pkg.add(PackageSpec(url="https://github.com/nignatiadis/Splines2.jl"))
Pkg.add(PackageSpec(url="https://github.com/nignatiadis/ExponentialFamilies.jl"))
Pkg.add(PackageSpec(url="https://github.com/nignatiadis/MinimaxCalibratedEBayes.jl"))
```






