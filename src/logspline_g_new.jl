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


function fit(cefm::ContinuousExponentialFamilyModel,
	         Zs::Union{EBayesSamples, DiscretizedStandardNormalSamples};
			 integrator = expectation(cefm.base_measure; n=50),
			 c0 = 0.001,
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

			hessian_storage_s = DiffResults.HessianResult(α_opt)
			hessian_storage_s = ForwardDiff.hessian!(hessian_storage_s, _s, α_opt);

			(cef=cef, α=α_opt, nll=hessian_storage_nll, s=hessian_storage_s)
end
