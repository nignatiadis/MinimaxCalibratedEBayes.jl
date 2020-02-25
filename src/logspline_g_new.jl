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
function (target::MarginalDensityTarget{<:DiscretizedStandardNormalSample2})(prior::ContinuousExponentialFamily;
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
	n = normalize ? length(Zs_discr) : 1
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
	cef::CEF = cefm(α_opt .- α_bias)
	α_covmat = nothing
	nll_hessian = nothing
	nll_gradient = nothing
	regularizer_hessian = nothing
	regularizer_gradient = nothing
	fitter = nothing
end

Base.broadcastable(fcef::FittedContinuousExponentialFamilyModel) = Ref(fcef)


function fit(cefm::ContinuousExponentialFamilyModel,
	         Zs::Union{EBayesSamples, DiscretizedStandardNormalSamples};
			 integrator = expectation(cefm.base_measure; n=50),
			 c0 = 1e-6,
			 optim_options = Optim.Options(show_trace=true, show_every=1, g_tol=1e-4)) # to stabilize optimization

			 n = length(Zs)
			 # initialize method through Lindsey's method
			 # this mostly makes sense for normal Zs
			 fit_lindsey = fit(cefm, response(Zs), LindseyMethod(500))
 		 	# set up objective
			 function _nll(α)
				 exp_family = cefm(α; integrator=integrator)
				 -loglikelihood(exp_family, Zs; integrator=integrator, normalize=true)
			 end
			 _s(α) = c0*norm(α)/n #allow other choices
 			 _penalized_nll(α) = _nll(α) + _s(α)

			 # ready to descend
			 optim_res = optimize(_penalized_nll, fit_lindsey.α,
			                          Newton(), optim_options;
			 						 autodiff = :forward)

			α_opt = Optim.minimizer(optim_res)
			cef = cefm(α_opt)

			hessian_storage_nll = DiffResults.HessianResult(α_opt)
			hessian_storage_nll = ForwardDiff.hessian!(hessian_storage_nll, _nll, α_opt);

			nll_hessian = DiffResults.hessian(hessian_storage_nll)
			nll_gradient = DiffResults.gradient(hessian_storage_nll)

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


function _target_bias_std(target::EBayesTarget, fcef::MinimaxCalibratedEBayes.FittedContinuousExponentialFamilyModel; bias_corrected=true)
	_fun(α) = target(fcef.cefm(α))
	α_opt = fcef.α_opt
	target_gradient = ForwardDiff.gradient(_fun, α_opt);
	target_bias = LinearAlgebra.dot(target_gradient, fcef.α_bias)
	target_variance = LinearAlgebra.dot(target_gradient, fcef.α_covmat*target_gradient)
	target_value = bias_corrected ? _fun(α_opt) - target_bias : _fun(α_opt)
	(target = target_value,
	 bias = target_bias,
	 std = sqrt(target_variance))
end

function StatsBase.confint(target::EBayesTarget,
	                       fcef::MinimaxCalibratedEBayes.FittedContinuousExponentialFamilyModel;
						   level::Real = 0.9, kwargs...)
	res = _target_bias_std(target, fcef; kwargs...)
	q = quantile(Normal(), (1+level)/2)
    res[:target] .+ (-1,1).*q.*res[:std]
end
