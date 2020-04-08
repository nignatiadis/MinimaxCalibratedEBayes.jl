using MinimaxCalibratedEBayes
using ECOS

const MCEB = MinimaxCalibratedEBayes


@testset "Check Proposition 1" begin
	
prior_dbn = MixtureModel(Normal, [(-2,0.4), (2, 0.4)])
target = PriorDensityTarget(1.0)
prior_class = GaussianMixturePriorClass(0.4, -3:0.1:3, ECOS.Optimizer)
Zs_discr = DiscretizedStandardNormalSamples(-4:0.1:4)
C∞ = 0.02
Zs_discr_nbhood = set_neighborhood(Zs_discr, prior_dbn; C∞ = C∞)
delta_tuner= MCEB.FixedDelta(0.001)


steinminimax_fit = SteinMinimaxEstimator(Zs_discr, prior_class,
								         target, delta_tuner)
							
steinminimax_fit_nbhood = SteinMinimaxEstimator(Zs_discr_nbhood, prior_class,
								         target, delta_tuner)

# Check if Delta is correct
@test steinminimax_fit_nbhood.δ == delta_tuner.δ

@test MCEB.get_δ(steinminimax_fit_nbhood.model, recalculate_δ=false) ≈ delta_tuner.δ atol=1e-6
@test MCEB.get_δ(steinminimax_fit_nbhood.model; recalculate_δ=true ) ≈ delta_tuner.δ atol=1e-6
	

# Check statements about bias
bias_at_g2 = worst_case_bias(steinminimax_fit_nbhood.Q, Zs_discr_nbhood,
                             steinminimax_fit_nbhood.g2, target)[:bias]

bias_at_g1 = worst_case_bias(steinminimax_fit_nbhood.Q, Zs_discr_nbhood,
							steinminimax_fit_nbhood.g1, target)[:bias]
				
worst_case_bias_stein = worst_case_bias(steinminimax_fit_nbhood.Q, Zs_discr_nbhood, 
                                        prior_class, target)
	
	
										
@test 	worst_case_bias_stein.max_bias ≈ -worst_case_bias_stein.min_bias atol=0.0001	
@test 	worst_case_bias_stein.max_bias ≈ worst_case_bias_stein.max_abs_bias atol=0.0001										 														 
@test 	abs2(worst_case_bias_stein.max_bias) ≈ worst_case_bias_stein.max_squared_bias atol=0.0001										 														 
				 														 
@test 	worst_case_bias_stein.max_bias ≈ bias_at_g1 atol=0.0001
@test 	worst_case_bias_stein.min_bias ≈ bias_at_g2 atol=0.0001											 
@test   worst_case_bias_stein.max_bias ≈ steinminimax_fit_nbhood.max_bias atol=0.0001
@test   2*worst_case_bias_stein.max_bias ≈ steinminimax_fit_nbhood.ω_δ - steinminimax_fit_nbhood.δ*steinminimax_fit_nbhood.ω_δ_prime atol=0.0001

end
