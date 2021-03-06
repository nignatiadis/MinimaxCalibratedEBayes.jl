# Main entry point to the function

# MCEB +-

# mceb(DiscretizedNormalSamples, Target;
#        PriorFamily = ,
#        alpha=0.1,
#        nbhood=nothing)



# -






"""
    _split_data(n_total::Integer)
    _split_data(Zs::AbstractVector)
	_split_data(Zs::AbstractVector, ::Symbol)

Helper functions to partition all observations
into two disjoint folds. 
"""
function _split_data(n_total::Integer)
    n_half = ceil(Int, n_total/2)
    idx_test = sample(1:n_total, n_half, replace=false)
    idx_train = setdiff(1:n_total, idx_test)
    idx_train, idx_test
end

function _split_data(Zs::AbstractVector)
    n = length(Zs)
    idx_train, idx_test = _split_data(n)
    Zs[idx_train], Zs[idx_test], idx_train, idx_test
end

function _split_data(Zs::AbstractVector, sym::Symbol)
    if sym==:random
    	_split_data(Zs)
	else 
		throw(DomainError("Only :random implemented."))
	end
end



function _default_tuner(delta_tuner::DeltaTuner, Zs_test_discr, fkde_train)
	delta_tuner
end 

function _default_tuner(delta_tuner::Type{<:DeltaTuner}, Zs_test_discr, fkde_train::KDEInfinityBand)
	n = nobs(Zs_test_discr)
	δ_min = default_δ_min(n, fkde_train.C∞)
	delta_tuner(n, δ_min)
end 

function _default_tuner(delta_tuner::Type{<:DeltaTuner}, Zs_test_discr, ::Nothing)
	n = nobs(Zs_test_discr)
	δ_min = 0.0
	delta_tuner(n, δ_min)
end 

function _default_tuner(Zs_test_discr, fkde_train)
	_default_tuner(RMSE, Zs_test_discr, fkde_train)
end

# train => first fold
# test -> second fold


""" 
	MinimaxCalibratorSetup(;Zs_test, 
                           Zs_train,
                           fkde_train,
                           prior_class,
                           Zs_test_discr,
                           delta_tuner,
                           pilot_method,
                           cache_target)

## Description:							
This creates the object that holds all preliminary calculations necessary to conduct
inference for empirical Bayes estimands. It can be either constructed manually, or more typically
through `fit(opt::MinimaxCalibratorOptions, Zs)` where `Zs` comprises all the data and `opt`
is a [`MinimaxCalibratorOptions`](@ref) object. See `data_analysis.html` for examples of
both approaches. 

## Arguments:
* `Zs_test`: Samples in the second data fold.
* `Zs_train`: Samples in the first data fold.
* `fkde_train`: A [`KDEInfinityBand`](@ref) instance computed based on `Zs_train` or external data (but not `Zs_test`).	
* `Zs_test_discr`: A discretization of `Zs_test` into [`DiscretizedStandardNormalSamples`](@ref).
* `delta_tuner`: Specifies how the bias-variance tradeoff (i.e., ``\\delta``) will be navigated, for example [`RMSE`](@ref) or [`HalfCIWidth`](@ref).
* `pilot_method`: Method for computing the pilot estimator for the numerator and denominator of the empirical Bayes estimand. Should support the interface `estimate(target::LinearEBayesTarget, pilot_method, Zs_train)`.
* `cache_target`: Defaults to `false`. Setting it to `true` will speed up computations when multiple similar targets are being estimated (e.g., PosteriorMeans along a grid).						 							
"""
mutable struct MinimaxCalibratorSetup{DS <: DiscretizedStandardNormalSamples,
                                           IDX,
										   ESTR,
                                           ESTE,
                                           Gcal <: ConvexPriorClass,
                                           DT <: DeltaTuner,
                                           P,
                                           NB}
    Zs_train::ESTR
    Zs_test::ESTE
    idx_train::IDX 
    idx_test::IDX 
    prior_class::Gcal
    fkde_train::NB 
    Zs_test_discr::DS
    delta_tuner::DT 
    pilot_method::P
	cache_target::Bool
	previous_target
	delta_cached::Float64
end


function MinimaxCalibratorSetup(;Zs_test, 
	                             prior_class,
								 Zs_test_discr,
								 delta_tuner = _default_tuner(Zs_test_discr, fkde_train),
								 Zs_train = nothing,
								 idx_train = nothing,
								 idx_test = nothing,
								 fkde_train = nothing,
								 pilot_method = nothing,
								 cache_target = false,
								 previous_target = nothing,
								 delta_cached = NaN) #default_pilot?


	delta_tuner = _default_tuner(delta_tuner, Zs_test_discr, fkde_train)
								 
	MinimaxCalibratorSetup(Zs_train, Zs_test, idx_train, idx_test, 
	                       prior_class, fkde_train, Zs_test_discr,
						   delta_tuner, pilot_method,
						   cache_target, previous_target, delta_cached)

end

#update_delta_tuner()

broadcastable(setup::MinimaxCalibratorSetup) = Ref(setup)



function StatsBase.fit(mceb_setup::MinimaxCalibratorSetup, target::LinearEBayesTarget)
	@unpack Zs_test_discr, prior_class, delta_tuner = mceb_setup
	model = initialize_modulus_problem(Zs_test_discr, prior_class, target)
	opt_δ = optimal_δ(model, delta_tuner)
	modulus_at_δ!(model, opt_δ)
	sm  = SteinMinimaxEstimator(Zs_test_discr,
	                            prior_class,
	                            target,
	                            model)
end


""" 
	CalibratedMinimaxEstimator(target::PosteriorTarget, sme::SteinMinimaxEstimator,
	                           pilot, pilot_num pilot_denom)

Object that stores a `pilot` estimator (including both the numerator `pilot_num`
and the denominator `pilot_denom`, where `pilot = pilot_num/pilot_denom`), as well
the quasi-minimax estimator for the calibrated target.	
"""
Base.@kwdef struct CalibratedMinimaxEstimator{T<:PosteriorTarget,
	                                          SME<:SteinMinimaxEstimator,
											  S<:Real}
	target::T
	sme::SME
	pilot::S
	pilot_num::S
	pilot_denom::S
end

function _update_setup(mceb_setup::MinimaxCalibratorSetup, target::EBayesTarget)
	@unpack cache_target, previous_target, delta_cached = mceb_setup
	
	if (cache_target) && (typeof(previous_target) == typeof(target))
		updated_setup = @set mceb_setup.delta_tuner = FixedDelta(delta_cached)
	else 
		updated_setup = mceb_setup
	end
	updated_setup
end

function StatsBase.fit(mceb_setup::MinimaxCalibratorSetup, target::PosteriorTarget)
	@unpack Zs_train, Zs_test_discr, 
	        prior_class, delta_tuner, pilot_method,
			cache_target = mceb_setup
	
	numerator_target = target.num_target 
	denominator_target = MarginalDensityTarget(location(target.num_target)) #	perhaps expose this as MarginalDensityTarget(target)

	num_hat = estimate(numerator_target, pilot_method, Zs_train)
	denom_hat = estimate(denominator_target, pilot_method, Zs_train)
	theta_hat = num_hat/denom_hat 
	
	
	calib_target = CalibratedTarget(posterior_target = target, 
	                                θ̄ = theta_hat,
									denom = denom_hat)
									
	tmp_setup = _update_setup(mceb_setup, target)
    calib_fit = fit(tmp_setup, calib_target)
	
	if cache_target
		mceb_setup.delta_cached = calib_fit.δ
		mceb_setup.previous_target = target
	end
	
	CalibratedMinimaxEstimator(target=target,
                               sme=calib_fit,
							   pilot=theta_hat,
							   pilot_num=num_hat,
							   pilot_denom=denom_hat)
	
end

function target_bias_std(target::PosteriorTarget, 
	                     calib::CalibratedMinimaxEstimator,
						 Zs=calib.sme.Z; kwargs...)
			
	calib_target = calib.target		
	cres = target_bias_std(calib_target, calib.sme, Zs; kwargs...)
	
	ctarget = cres.estimated_target
	cbias = cres.estimated_bias
	cstd = cres.estimated_std
	
	pilot_denom = calib.pilot_denom 
	# TODO: sanity check this is the same as target of denominator.
	pilot = calib.pilot 
	
	(estimated_target = pilot + ctarget/pilot_denom,
	estimated_bias = cbias/pilot_denom,
	estimated_std = cstd/pilot_denom)
end


""" 
	MinimaxCalibratorOptions(split = :random
	                         prior_class::ConvexPriorClass,
	                         marginal_grid,
	                         infinity_band_options,
	                         pilot_options,
	                         tuner = HalfCIWidth
	                         cache_target = false)
							 
Struct that wraps all options required to automatically
create a [`MinimaxCalibratorSetup`](@ref) object. The options are as follows

* `split`: Defaults to `:random`, i.e., a random 50-50 split of the data.
* `prior_class`: A [`ConvexPriorClass`](@ref) in which the true ``G`` is believed to lie.
* `marginal_grid`: Marginal domain discretization grid.
* `infinity_band_options`: Options for constructing the localization, cf. [`KDKDEInfinityBandOptions`](@ref).
* `pilot_options`: 	Options that describe how to derive the pilot estimators, for example [`ButuceaComteOptions`](@ref).
* `tuner`: Specifies how the bias-variance tradeoff (i.e., ``\\delta``) will be navigated, for example [`RMSE`](@ref) or [`HalfCIWidth`](@ref).
* `cache_target`: Defaults to `false`. Setting it to `true` will speed up computations when multiple similar targets are being estimated (e.g., PosteriorMeans along a grid).						 							
"""
Base.@kwdef struct MinimaxCalibratorOptions{SPL,
                                     GCAL <: ConvexPriorClass,
									 GR <: AbstractVector,
									 PS,
									 OPT,
									 D}
   split::SPL = :random
   prior_class::GCAL
   marginal_grid::GR
   infinity_band_options::OPT = KDEInfinityBandOptions(a_min = minimum(marginal_grid),
                                              a_max = maximum(marginal_grid))
   pilot_options::PS
   tuner::D = HalfCIWidth
   cache_target::Bool = false
end


function fit(mceb_opt::MinimaxCalibratorOptions, Zs::AbstractVector{<:StandardNormalSample})
	@unpack split, prior_class, marginal_grid,
	        infinity_band_options, pilot_options, tuner, cache_target = mceb_opt
	Zs_train, Zs_test, idx_train, idx_test = _split_data(Zs, split)
	
	a_min, a_max = extrema(marginal_grid)
	infinity_band_options = KDEInfinityBandOptions(a_min=a_min, a_max=a_max)
	# TODO: let fkde_train be better at getting dispatched
	fkde_train = fit(infinity_band_options, response.(Zs_train))

	pilot_fit = fit(pilot_options, Zs_train)

	Zs_test_discr = DiscretizedStandardNormalSamples(Zs_test, marginal_grid)
	Zs_test_discr = set_neighborhood(Zs_test_discr, fkde_train)

	mceb_setup = MinimaxCalibratorSetup(Zs_train = Zs_train,
	                                    Zs_test = Zs_test,
										idx_train = idx_train,
										idx_test = idx_test,
							            prior_class = prior_class,
							            fkde_train = fkde_train,
							            Zs_test_discr = Zs_test_discr,
							            pilot_method = pilot_fit,
										delta_tuner = tuner,
										cache_target= cache_target)
end