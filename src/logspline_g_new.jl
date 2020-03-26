# Hard code for exponential families; temporary...






####### The above two are identical! Clean up.
function (target::MarginalDensityTarget{<:StandardNormalSample})(prior::ContinuousExponentialFamily;
	                                                             ngrid=100,
																 integrator = expectation(prior.base_measure; n = ngrid))
	function integrand(μ)
		riesz_representer(target,μ)*pdf(prior, μ; include_base_measure=false)
	end
    integrator(integrand)
end

function (target::PosteriorMeanNumerator{<:StandardNormalSample})(prior::ContinuousExponentialFamily;
	                                                             ngrid=100,
																 integrator = expectation(prior.base_measure; n = ngrid))
	function integrand(μ)
		riesz_representer(target,μ)*pdf(prior, μ; include_base_measure=false)
	end
    integrator(integrand)
end


#### LFSFNumerator requires different treatment.

function (target::LFSRNumerator{<:StandardNormalSample})(prior::ContinuousExponentialFamily;
	                                                             ngrid=500)
    b = maximum(prior) #TODO CHECK b>0
	integrator = expectation(Uniform(0, b); n=ngrid)
	function integrand(μ)
		riesz_representer(target,μ)*pdf(prior, μ; include_base_measure=true)*b
	end
    integrator(integrand)
end

#----------------- Under discretization
function (target::MarginalDensityTarget{<:DiscretizedStandardNormalSample})(prior::ContinuousExponentialFamily;
                                                                            integrator=expectation(prior.base_measure; n=500))
    Z_disc = location(target)
    grid = Z_disc.samples.mhist.grid
    i = Z_disc.bin
    if i == 1
        integrand = μ -> cdf(Normal(μ), first(grid))*pdf(prior, μ; include_base_measure=false)
    elseif i == lastindex(Z_disc.samples.mhist)
        integrand = μ -> ccdf(Normal(μ), last(grid))*pdf(prior, μ; include_base_measure=false)
    else
        integrand = μ -> (cdf(Normal(μ), grid[i]) -  cdf(Normal(μ), grid[i-1]))*pdf(prior, μ; include_base_measure=false)
    end
    # refactor into riesz representer..
    integrator(integrand)
end


#----------- loglikelihood


function loglikelihood(prior::ContinuousExponentialFamily,
                       Zs_discr::DiscretizedStandardNormalSamples;
                       integrator = expectation(prior.base_measure; n=200),
					   normalize=true)
    mhist = Zs_discr.mhist # need to be a bit careful...
    n_weights =  lastindex(mhist)
    ll = 0
    for i=1:n_weights
        wt = mhist[i]
        if wt > 0
            target = MarginalDensityTarget(Zs_discr(i))
            ll += log(target(prior; integrator=integrator))*wt/n
        end
    end
    ll
end


function loglikelihood(prior::ContinuousExponentialFamily,
                       Zs::AbstractVector{<:StandardNormalSample};
                       integrator = expectation(prior.base_measure; n=50),
					   normalize=true)
	n = normalize ? length(Zs) : 1
	sum([log(MarginalDensityTarget(Z)(prior; integrator=integrator))/n for Z in Zs])
end



### Fitting Routine through (penalized MLE)

Base.@kwdef mutable struct FittedContinuousExponentialFamilyModel{CEFM<:ContinuousExponentialFamilyModel,
	                                          CEF<:ContinuousExponentialFamily,
											  T<:Real,
											  VT<:AbstractVector{T}}
	cefm::CEFM
	α_opt::VT
	α_bias = zero(α_opt)
	cef::CEF = cefm(α_opt)
	α_covmat = nothing
	nll_hessian = nothing
	nll_gradient = nothing
	regularizer_hessian = nothing
	regularizer_gradient = nothing
	fitter = nothing
end

Base.broadcastable(fcef::FittedContinuousExponentialFamilyModel) = Ref(fcef)


@with_kw struct ExponentialFamilyDeconvolutionMLE{CEFM<:ContinuousExponentialFamilyModel,
	                                           T <: Real}
	cefm::CEFM
	c0::T = 0.01
	integrator = expectation(cefm.base_measure; n=50)
	solver = NewtonTrustRegion()
	optim_options = Optim.Options(show_trace=true, show_every=1, g_tol=1e-6)
	initializer = LindseyMethod(500)
end


function fit(cefm::ContinuousExponentialFamilyModel,
	         Zs::Union{EBayesSamples, DiscretizedStandardNormalSamples};
			 kwargs...)
    deconv = ExponentialFamilyDeconvolutionMLE(cefm=cefm, kwargs...)
	fit(deconv, Zs)		 
end			 

function fit(deconv::ExponentialFamilyDeconvolutionMLE,
	         Zs::Union{EBayesSamples, DiscretizedStandardNormalSamples})
			 
			 cefm = deconv.cefm
			 @unpack c0, integrator, solver, optim_options, initializer = deconv

			 n = nobs(Zs)
			 # initialize method through Lindsey's method
			 # this mostly makes sense for normal Zs
			 if isa(initializer, LindseyMethod)
			 	fit_init = fit(cefm, response(Zs), initializer)
			 	α_init = fit_init.α
			 else
				α_init = initializer
			 end
 		 	# set up objective
			 function _nll(α)
				 exp_family = cefm(α; integrator=integrator)
				 -loglikelihood(exp_family, Zs; integrator=integrator, normalize=true)
			 end
			 _s(α) = c0*norm(α)/n #sum(abs2, α)/n# c0*norm(α)/n #allow other choices
 			 _penalized_nll(α) = _nll(α) + _s(α)

			 # ready to descend
			 optim_res = optimize(_penalized_nll, α_init,
			                          solver, optim_options;
			 						 autodiff = :forward)

			α_opt = Optim.minimizer(optim_res)
			cef = cefm(α_opt)

			hessian_storage_nll = DiffResults.HessianResult(α_opt)
			hessian_storage_nll = ForwardDiff.hessian!(hessian_storage_nll, _nll, α_opt);

			nll_hessian = DiffResults.hessian(hessian_storage_nll)
			nll_gradient = DiffResults.gradient(hessian_storage_nll)


			# project onto psd cone
			nll_hessian_eigen = eigen(Symmetric(nll_hessian))
			nll_hessian_eigen.values .= max.(nll_hessian_eigen.values, 0.0)
			nll_hessian = Matrix(nll_hessian_eigen)


			hessian_storage_s = DiffResults.HessianResult(α_opt)
			hessian_storage_s = ForwardDiff.hessian!(hessian_storage_s, _s, α_opt);

			regularizer_hessian = DiffResults.hessian(hessian_storage_s)
			regularizer_gradient = DiffResults.gradient(hessian_storage_s)

			inv_mat = inv( (nll_hessian + regularizer_hessian).*sqrt(n))
			α_covmat = inv_mat*nll_hessian*inv_mat

			α_bias = -inv_mat*regularizer_gradient .* sqrt(n)

			FittedContinuousExponentialFamilyModel(;cefm=cefm,
			                                       α_opt=α_opt,
												   α_covmat=α_covmat,
												   α_bias = α_bias,
												   nll_hessian=nll_hessian,
												   nll_gradient=nll_gradient,
												   regularizer_hessian=regularizer_hessian,
												   regularizer_gradient=regularizer_gradient,
												   fitter=optim_res)
end


function target_bias_std(target::EBayesTarget,
	                     fcef::MinimaxCalibratedEBayes.FittedContinuousExponentialFamilyModel;
	                     bias_corrected=true,
						 clip=true)
	_fun(α) = target(fcef.cefm(α))
	α_opt = fcef.α_opt
	target_gradient = ForwardDiff.gradient(_fun, α_opt);
	target_bias = LinearAlgebra.dot(target_gradient, fcef.α_bias)
	target_variance = LinearAlgebra.dot(target_gradient, fcef.α_covmat*target_gradient)
	target_value = bias_corrected ? _fun(α_opt) - target_bias : _fun(α_opt)

	if clip
		target_value = clamp(target_value, extrema(target)...)
	end

	(estimated_target = target_value,
	 estimated_bias = target_bias,
	 estimated_std = sqrt(target_variance))
end

function Distributions.estimate(target::EBayesTarget, fcef::MinimaxCalibratedEBayes.FittedContinuousExponentialFamilyModel; kwargs...)
	target_bias_std(target, fcef; kwargs...)[:estimated_target]
end

function StatsBase.confint(target::EBayesTarget,
	                       fcef::MinimaxCalibratedEBayes.FittedContinuousExponentialFamilyModel;
						   level::Real = 0.9,
						   worst_case_bias_adjusted = false,
						   clip = true)
    if worst_case_bias_adjusted
		res = target_bias_std(target, fcef;
		                      clip = false, #we will clip the CIs themselves instead
							  bias_corrected = false) # we will account for worst case bias instead
		maxbias = abs(res[:estimated_bias])
	else
		res =  target_bias_std(target, fcef; clip=false, bias_corrected = true)
		maxbias = 0.0
	end

	q_mult = bias_adjusted_gaussian_ci(res[:estimated_std], maxbias=maxbias , level=level)

    L,U = res[:estimated_target] .+ (-1,1).*q_mult

	if clip
		L, U = clamp.((L,U), extrema(target)... )
	end

	L,U
end
