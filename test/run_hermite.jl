using MinimaxCalibratedEBayes
using QuadGK
using OffsetArrays

const MCEB = MinimaxCalibratedEBayes

using ECOS


@testset "Check Sobolev to coefficients expansion" begin

target = PriorDensityTarget(0.0)
marginal_grid = -3:0.02:3
Zs_discr = DiscretizedStandardNormalSamples(marginal_grid)

bc = MCEB.ButuceaComteEstimator(target;  n=1000)
bc_marginal_affine = DiscretizedAffineEstimator(Zs_discr.mhist, bc)

herm_class = HermitePriorClass(qmax=30, sobolev_order=1.0, sobolev_bound=0.2,
                               solver=ECOS.Optimizer)

herm_worst_case_bias = worst_case_bias(bc_marginal_affine, 
                                       Zs_discr, herm_class, target,
						               boundary_mass=0.1)


g_min = herm_worst_case_bias.min_g

@test quadgk(x->pdf(g_min, x), -Inf, Inf)[1] ≈ 1
@test cf(g_min, 0.0) ≈ 1 + 0im 
@test cf(g_min, 1.0) ≈ quadgk(x->exp(im*x)*pdf(g_min,x), -Inf, Inf)[1]
@test cf(g_min, -1.0) ≈ quadgk(x->exp(-im*x)*pdf(g_min,x), -Inf, Inf)[1]
@test herm_class.sobolev_bound ≈ quadgk(x -> abs2(cf(g_min, x))*(x^2+1), -Inf, Inf)[1]/(2π)

herm_class.sobolev_bound 
#function _coefs_to_sobolev_norm(αs)
#	q = length(αs)
#	αstilde = zeros(q + 1)
#	αs_o = OffsetArray(αs, 0:(q-1))
#	αstilde_o = OffsetArray(αstilde, 0:q)
#	αstilde_o[0] = αs_o[1]/sqrt(2)
#	for j=1:q		
#	end
#	αstilde_o
#end
function _coefs_to_sobolev_norm(αs)
	q = length(αs)
	αs_o = OffsetArray([αs;0], 0:q)
	quad_term = abs2(αs_o[1])/2 + sum(abs2, αs_o)
	sob_vec = [(sqrt(j)*αs_o[j-1]-sqrt(j+1)*αs_o[j+1]) for j in 1:(q-1)]
	sob_term = sum(abs2, sob_vec)/2
	quad_term + sob_term
end 

@test _coefs_to_sobolev_norm(g_min.coefs) ≈ herm_class.sobolev_bound atol=0.00001	

end
