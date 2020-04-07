var documenterSearchIndex = {"docs":
[{"location":"man/targets/#Empirical-Bayes-estimands-1","page":"Empirical Bayes estimands","title":"Empirical Bayes estimands","text":"","category":"section"},{"location":"man/targets/#","page":"Empirical Bayes estimands","title":"Empirical Bayes estimands","text":"In this section we introduce the interface for describing the inferential targets of the empirical Bayes analysis.","category":"page"},{"location":"man/targets/#","page":"Empirical Bayes estimands","title":"Empirical Bayes estimands","text":"using MinimaxCalibratedEBayes","category":"page"},{"location":"man/targets/#","page":"Empirical Bayes estimands","title":"Empirical Bayes estimands","text":"MCEB.EBayesTarget","category":"page"},{"location":"man/targets/#MinimaxCalibratedEBayes.EBayesTarget","page":"Empirical Bayes estimands","title":"MinimaxCalibratedEBayes.EBayesTarget","text":"EBayesTarget\n\nAbstract type that describe Empirical Bayes estimands (which we want to estimate or conduct inference for).\n\n\n\n\n\n","category":"type"},{"location":"man/targets/#","page":"Empirical Bayes estimands","title":"Empirical Bayes estimands","text":"Any target::EBayesTarget may be used as a function on distributions, e.g., for a PriorDensityTarget","category":"page"},{"location":"man/targets/#","page":"Empirical Bayes estimands","title":"Empirical Bayes estimands","text":"target = PriorDensityTarget(2.0)\ntarget(Normal(0,1)) == pdf(Normal(0,1), 2.0)","category":"page"},{"location":"man/targets/#","page":"Empirical Bayes estimands","title":"Empirical Bayes estimands","text":"Similarly for a MarginalDensityTarget","category":"page"},{"location":"man/targets/#","page":"Empirical Bayes estimands","title":"Empirical Bayes estimands","text":"target = MarginalDensityTarget(StandardNormalSample(1.0))\ntarget(Normal(0,1)) == pdf(Normal(0,sqrt(2)), 1.0)","category":"page"},{"location":"man/targets/#","page":"Empirical Bayes estimands","title":"Empirical Bayes estimands","text":"Other interface functions implemented for all EBayesTargets include:","category":"page"},{"location":"man/targets/#","page":"Empirical Bayes estimands","title":"Empirical Bayes estimands","text":"extrema(::EBayesTarget)","category":"page"},{"location":"man/targets/#Base.extrema-Tuple{EBayesTarget}","page":"Empirical Bayes estimands","title":"Base.extrema","text":"extrema(target::EBayesTarget)\n\nReturns a tuple (a,b) such that a ≦ target ≦ b is always true. By default this is (-infty+infty), but e.g., for a PriorTailProbability it returns (0,1).\n\n\n\n\n\n","category":"method"},{"location":"man/targets/#Linear-Functionals-1","page":"Empirical Bayes estimands","title":"Linear Functionals","text":"","category":"section"},{"location":"man/targets/#","page":"Empirical Bayes estimands","title":"Empirical Bayes estimands","text":"MCEB.LinearEBayesTarget","category":"page"},{"location":"man/targets/#MinimaxCalibratedEBayes.LinearEBayesTarget","page":"Empirical Bayes estimands","title":"MinimaxCalibratedEBayes.LinearEBayesTarget","text":"LinearEBayesTarget <: EBayesTarget\n\nAbstract type for Empirical Bayes estimands that are linear functionals of the prior G, i.e., they take the form L(G) for some function linear functional L.\n\n\n\n\n\n","category":"type"},{"location":"man/targets/#Interface-1","page":"Empirical Bayes estimands","title":"Interface","text":"","category":"section"},{"location":"man/targets/#","page":"Empirical Bayes estimands","title":"Empirical Bayes estimands","text":"cf(::MCEB.LinearEBayesTarget, t)","category":"page"},{"location":"man/targets/#Distributions.cf-Tuple{MinimaxCalibratedEBayes.LinearEBayesTarget,Any}","page":"Empirical Bayes estimands","title":"Distributions.cf","text":"cf(::LinearEBayesTarget, t)\n\nThe characteristic function of L(cdot), a LinearEBayesTarget, which we define as follows:\n\nFor L(cdot) which may be written as L(G) = int psi(mu)dGmu  (for a measurable function psi) this returns the Fourier transform of psi evaluated at t, i.e., psi^*(t) = int exp(it x)psi(x)dx. \n\nNote that psi^*(t) is such that for distributions G with density g (and g^* the Fourier Transform of g) the following holds:\n\nL(G) = frac12piint g^*(mu)psi^*(mu) dmu\n\n\n\n\n\n","category":"method"},{"location":"man/targets/#Implemented-linear-targets-1","page":"Empirical Bayes estimands","title":"Implemented linear targets","text":"","category":"section"},{"location":"man/targets/#","page":"Empirical Bayes estimands","title":"Empirical Bayes estimands","text":"MarginalDensityTarget\nPriorDensityTarget\nPriorTailProbability","category":"page"},{"location":"man/targets/#MinimaxCalibratedEBayes.MarginalDensityTarget","page":"Empirical Bayes estimands","title":"MinimaxCalibratedEBayes.MarginalDensityTarget","text":"MarginalDensityTarget(Z::StandardNormalSample) <: LinearEBayesTarget\n\nExample call\n\nMarginalDensityTaget(StandardNormalSample(2.0))\n\nDescription\n\nDescribes the marginal density evaluated at Z=z  (e.g. Z=2 in the example above) of a sample drawn from the hierarchical model \n\nmu sim G Z sim mathcalN(01)\n\nIn other words, letting phi the Standard Normal pdf\n\nL(G) = phi star dG(z)\n\nNote that 2.0 has to be wrapped inside StandardNormalSample(2.0) since this target depends not only on G and the location, but also on the likelihood. Additional likelihoods will be added in the future.\n\n\n\n\n\n","category":"type"},{"location":"man/targets/#MinimaxCalibratedEBayes.PriorDensityTarget","page":"Empirical Bayes estimands","title":"MinimaxCalibratedEBayes.PriorDensityTarget","text":"PriorDensityTarget(z::Float64) <: LinearEBayesTarget\n\nExample call\n\nPriorDensityTarget(2.0)\n\nDescription\n\nThis is the evaluation functional of the density of G at z, i.e., L(G) = G(z) = g(z) or in Julia code L(G) = pdf(G, z).\n\n\n\n\n\n","category":"type"},{"location":"man/targets/#MinimaxCalibratedEBayes.PriorTailProbability","page":"Empirical Bayes estimands","title":"MinimaxCalibratedEBayes.PriorTailProbability","text":"PriorTailProbability(cutoff::Float64) <: LinearEBayesTarget\n\nExample call\n\nPriorTailProbability(2.0)\n\nDescription\n\nThis is the evaluation functional of the tail probability of G at cutoff, i.e., L(G) = 1-G(textcutoff) or in Julia code L(G) = ccdf(G, z).\n\n\n\n\n\n","category":"type"},{"location":"man/targets/#Posterior-Estimands-1","page":"Empirical Bayes estimands","title":"Posterior Estimands","text":"","category":"section"},{"location":"man/prior_classes/#Convex-classes-of-priors-1","page":"Convex classes of priors","title":"Convex classes of priors","text":"","category":"section"},{"location":"man/prior_classes/#","page":"Convex classes of priors","title":"Convex classes of priors","text":"MCEB.ConvexPriorClass\nGaussianMixturePriorClass\nHermitePriorClass","category":"page"},{"location":"man/prior_classes/#MinimaxCalibratedEBayes.ConvexPriorClass","page":"Convex classes of priors","title":"MinimaxCalibratedEBayes.ConvexPriorClass","text":"ConvexPriorClass\n\nAbstract type representing convex classes of probability distributions mathcalG.\n\n\n\n\n\n","category":"type"},{"location":"man/prior_classes/#MinimaxCalibratedEBayes.GaussianMixturePriorClass","page":"Convex classes of priors","title":"MinimaxCalibratedEBayes.GaussianMixturePriorClass","text":"GaussianMixturePriorClass(σ_prior, grid, solver)\n\nClass of distributions G in mathcalG that can be written as the convolution of a Normal  distribution with standard deviation σ_prior and a discrete distribution pi supported on grid.\n\nG = mathcalN(0sigma_textprior^2) star pi\n\nThe solver object is a MathOptInterface compatible optimizer such as Mosek.Optimizer that will be  used to solve the convex modulus of continuity (or worst-case-bias estimation) problem.\n\n\n\n\n\n","category":"type"},{"location":"man/prior_classes/#MinimaxCalibratedEBayes.HermitePriorClass","page":"Convex classes of priors","title":"MinimaxCalibratedEBayes.HermitePriorClass","text":"HermitePriorClass(qmax, sobolev_order, sobolev_bound, solver)\n\nClass of densities g in mathcalG that can be written as g(mu) = sum_j=0^qmax h_j(mu), where h_j is the j-th Hermite function and further satisfy the Sobolev constraint (with sobolev_order corresponding to b and sobolev_bound to C): \n\nint_-infty^infty g^*(t)^2(t^2+1)^b dt leq 2pi C\n\nThe solver object is a MathOptInterface compatible optimizer such as Mosek.Optimizer that will be  used to solve the convex modulus of continuity (or worst-case-bias estimation) problem.\n\n\n\n\n\n","category":"type"},{"location":"man/pilot/#Pilot-estimators-1","page":"Pilot estimators","title":"Pilot estimators","text":"","category":"section"},{"location":"man/pilot/#Butucea-Comte-1","page":"Pilot estimators","title":"Butucea-Comte","text":"","category":"section"},{"location":"man/pilot/#","page":"Pilot estimators","title":"Pilot estimators","text":"MCEB.ButuceaComteOptions","category":"page"},{"location":"man/pilot/#MinimaxCalibratedEBayes.ButuceaComteOptions","page":"Pilot estimators","title":"MinimaxCalibratedEBayes.ButuceaComteOptions","text":"ButuceaComteOptions(; bandwidth = :auto)\n\nThe Butucea Comte estimator of linear functional in the deconvolution model. Given a LinearEBayesTarget with characteristic function psi^* and samples convolved with Gaussian noise (with phi Standard Gaussian pdf), it is defined as follows\n\nhatL_textBCh_m = frac12 pi msum_i=1^m int_-1h_m^1h_m exp(it Z_k) fracpsi^*(-t)varphi^*(t)dt\n\nh is the bandwidth, the option :auto will pick it automatically.\n\nReference:\n\n>Butucea, C. and Comte, F., 2009. \n>Adaptive estimation of linear functionals in the convolution model and applications.\n>Bernoulli, 15(1), pp.69-98.\n\n\n\n\n\n","category":"type"},{"location":"man/pilot/#","page":"Pilot estimators","title":"Pilot estimators","text":"A target::LinearEBayesTarget  can be estiamted based on Standard Normal samples Zs with the Butucea-Comte method bcopt::ButuceaComteOptions as follows","category":"page"},{"location":"man/pilot/#","page":"Pilot estimators","title":"Pilot estimators","text":"estimate(target, bcop, Zs)","category":"page"},{"location":"man/pilot/#","page":"Pilot estimators","title":"Pilot estimators","text":"Distributions.estimate(::MCEB.LinearEBayesTarget, ::MCEB.ButuceaComteOptions, ::AbstractVector{<:StandardNormalSample})","category":"page"},{"location":"man/pilot/#Distributions.estimate-Tuple{MinimaxCalibratedEBayes.LinearEBayesTarget,MinimaxCalibratedEBayes.ButuceaComteOptions,AbstractArray{#s181,1} where #s181<:StandardNormalSample}","page":"Pilot estimators","title":"Distributions.estimate","text":"estimate(target::LinearEBayesTarget, bcopt::ButuceaComteOptions, Zs)\n\nbla\n\n\n\n\n\n","category":"method"},{"location":"man/pilot/#","page":"Pilot estimators","title":"Pilot estimators","text":"Note that the Butucea-Comte estimator is a linear estimator, that is it takes the form:","category":"page"},{"location":"man/pilot/#Logspline-G-modeling-1","page":"Pilot estimators","title":"Logspline G-modeling","text":"","category":"section"},{"location":"man/samples/#Discretization-and-Localization-1","page":"Discretization and Localization","title":"Discretization and Localization","text":"","category":"section"},{"location":"man/samples/#Empirical-Bayes-samples-1","page":"Discretization and Localization","title":"Empirical Bayes samples","text":"","category":"section"},{"location":"man/samples/#Discretization-1","page":"Discretization and Localization","title":"Discretization","text":"","category":"section"},{"location":"man/samples/#","page":"Discretization and Localization","title":"Discretization and Localization","text":"DiscretizedStandardNormalSamples","category":"page"},{"location":"man/samples/#MinimaxCalibratedEBayes.DiscretizedStandardNormalSamples","page":"Discretization and Localization","title":"MinimaxCalibratedEBayes.DiscretizedStandardNormalSamples","text":"DiscretizedStandardNormalSamples(marginal_grid)\n\nThis type represents discretization of Standard Normal Samples to the grid marginal_grid. Let t_1  dotsc  t_K the elements of marginal_grid, then this type implies that we map each sample Z sim mathcalN(01) to one of the intervals t_j t_j+1)j=0dotscK, with t_0 = -infty and t_k+1 =  +infty.  Our inference will depend only on the discretized samples.\n\nDiscretizedStandardNormalSamples(Zs::AbstractVector{<:StandardNormalSample}, marginal_grid)\n\nIf we also provide the constructor with a vector of StandardNormalSample's, then it also stores the Histogram of the samples (according to the binning defined above) in the mhist slot.  This is the multinomial sufficient statistics of the discretized data.\n\nThe DiscretizedStandardNormalSamples type also contains three more slots:\n\nf_min,f_max: These can be either nothing or vectors of length equal to the number of \n\nhistogram bins. They contain bounds so that:\n\nf_textmink leq mathbb P_G Z in textbin k leq f_textmaxk \n\nvar_proxy: This is typically a vector of point estimates of mathbb P_G Z in textbin k.\n\n\n\n\n\n","category":"type"},{"location":"man/samples/#Localization-1","page":"Discretization and Localization","title":"Localization","text":"","category":"section"},{"location":"man/samples/#","page":"Discretization and Localization","title":"Discretization and Localization","text":"Neighborhood construction","category":"page"},{"location":"man/samples/#","page":"Discretization and Localization","title":"Discretization and Localization","text":"KDEInfinityBandOptions","category":"page"},{"location":"man/samples/#MinimaxCalibratedEBayes.KDEInfinityBandOptions","page":"Discretization and Localization","title":"MinimaxCalibratedEBayes.KDEInfinityBandOptions","text":"KDEInfinityBandOptions(; kernel=DeLaValleePoussinKernel\n                         bandwidth = nothing,\n                         a_min, \n                         a_max,\n                         nboot = 1000)\n\nThis struct contains hyperparameters that will be used for constructing a neighborhood of the marginal density. The steps of the method (and corresponding hyperparameter meanings) are as follows\n\nFirst a kernel density estimate barf of the data is fit with kernel as the\n\nkernel and bandwidth (the default bandwidth = nothing corresponds to automatic bandwidth selection).\n\nSecond, a Poisson bootstrap with nboot replication will be used to estimate a L_infty\n\nneighborhood c_m of the true density f which is such that with probability tending to 1:\n\nsup_x in a_textmin  a_textmax  barf(x) - f(x) leq c_m\n\nNote that the bound is valid from a_min to a_max. \n\nReference:\n\nPaul Deheuvels and Gérard Derzko. Asymptotic certainty bands for kernel density  estimators based upon a bootstrap resampling scheme. In Statistical models and methods  for biomedical and technical systems, pages 171–186. Springer, 2008\n\n\n\n\n\n","category":"type"},{"location":"man/samples/#","page":"Discretization and Localization","title":"Discretization and Localization","text":"Kernels that can be used:","category":"page"},{"location":"man/samples/#","page":"Discretization and Localization","title":"Discretization and Localization","text":"DeLaValleePoussinKernel\nSincKernel","category":"page"},{"location":"man/samples/#MinimaxCalibratedEBayes.DeLaValleePoussinKernel","page":"Discretization and Localization","title":"MinimaxCalibratedEBayes.DeLaValleePoussinKernel","text":"DeLaValleePoussinKernel(h)\n\nImplements the DeLaValleePoussinKernel with bandwidth h to be used for kernel density estimation through the KernelDensity.jl package. The De La Vallée-Poussin kernel is defined as follows: \n\nK_V(x) = fraccos(x)-cos(2x)pi x^2\n\nIts use case is similar to the SincKernel, however it has the advantage of being integrable (in the Lebesgue sense). Its Fourier transform is the following:\n\nK^*_V(t) = begincases \n 1  text if  xin-11  \n 0 text if  t geq 2  \n 2-t text if  t in 12\n endcases\n\n\n\n\n\n","category":"type"},{"location":"man/samples/#MinimaxCalibratedEBayes.SincKernel","page":"Discretization and Localization","title":"MinimaxCalibratedEBayes.SincKernel","text":"SincKernel(h)\n\nImplements the SincKernel with bandwidth h to be used for kernel density estimation through the KernelDensity.jl package. The sinc kernel is defined as follows: \n\nK_textsinc(x) = fracsin(x)pi x \n\nIt is not typically used for kernel density estimation, because this kernel is not a density itself. However, it is particularly well suited to deconvolution problems and estimation of very smooth densities because its Fourier transform is the following:\n\nK^*_textsinc(t) = mathbf 1( t in -11)\n\n\n\n\n\n","category":"type"},{"location":"#MinimaxCalibratedEBayes.jl-1","page":"Home","title":"MinimaxCalibratedEBayes.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"mu sim G   Z sim mathcalN(mu  1)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The ingredients of the approach are the following. #","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The class of potential effect size distributions mathcalG.\nThe estimator of the marginal density barf and neighborhood radius c_m.\nThe choice of delta_m.\nThe pilot estimators for. ","category":"page"},{"location":"#Reference-1","page":"Home","title":"Reference","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"This package implements the method described in the following paper","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Ignatiadis, Nikolaos, and Stefan Wager. \"Bias-Aware Confidence Intervals for Empirical Bayes Analysis.\" arXiv:1902.02774 (2019)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"note: A remark on notation\nSee the paper for details about the method. Note that the paper uses the notation   X_i for the Standard Normal Samples, while the documentation and the package   here use the notation Z_i.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The paper provides a general framework for estimation of empirical Bayes estimands and linear functionals that provides substantial flexibility. The package here has been designed around this framework with the goal of modularity. Please open an issue in the Github repository if there is  a combination of estimators/targets/effect size distributions/likelihoods that you would like to use, which have not been implemented.","category":"page"},{"location":"#Installation-1","page":"Home","title":"Installation","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"The package is not available in the Julia registry yet. It may be installed from Github as follows","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using Pkg\nPkg.add(PackageSpec(url=\"https://github.com/JuliaApproximation/OscillatoryIntegrals.jl\"))\nPkg.add(PackageSpec(url=\"https://github.com/nignatiadis/Splines2.jl\"))\nPkg.add(PackageSpec(url=\"https://github.com/nignatiadis/ExponentialFamilies.jl\"))\nPkg.add(PackageSpec(url=\"https://github.com/nignatiadis/MinimaxCalibratedEBayes.jl\"))","category":"page"}]
}
