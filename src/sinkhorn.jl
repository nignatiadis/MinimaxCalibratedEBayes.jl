

abstract type AbstractKullbackLeiblerProx end

struct FixedMarginal{D<:DiscreteNonParametric} <: AbstractKullbackLeiblerProx
    d::D
end


import Base:length

function length(fm::FixedMarginal)
    length(support(fm.d))
end

# u is prob vector
(fm::FixedMarginal)(u) = probs(fm.d)

function FixedMarginal(v::AbstractVector)
    n = length(v)
    FixedMarginal(DiscreteNonParametric(sort(v), fill(1/n, n)))
end


function sinkhorn(dist_mat, kl_prox1, kl_prox2;
                   ϵ = 1.0,
                   rounds=2,
                   u = fill(1.0, length(kl_prox1)),
                   v = fill(1.0, length(kl_prox2)))

     K = exp.(-dist_mat / ϵ)
     Kv = K*v
     Ktu = K'*u

     for iter=1:rounds
             u = kl_prox1(Kv) ./ Kv
             Ktu = K'*u
             v = kl_prox2(Ktu) ./ Ktu
             Kv = K*v
     end
     (u=u, v=v, plan=u .* K .* v')
end


Base.@kwdef mutable struct FixedMarginalConvolution{ZT <: DiscretizedStandardNormalSamples,
                                            GCAL <: ConvexPriorClass}
    Zs_discr::ZT
    prior_class::GCAL
    model = nothing
end

length(fmc::FixedMarginalConvolution) = nbins(fmc.Zs_discr)



function initialize_dkl_projection(prior_class::ConvexPriorClass,
	                            Zs_discr::DiscretizedStandardNormalSamples)

								mproj = Model(prior_class.solver)

								π = add_prior_variables!(mproj, prior_class)
								f = marginalize(prior_class, Zs_discr, π)

								@variable(mproj, entropy_upper_bd_vec[1:length(f)])
								@variable(mproj, prob_vec_dummy[1:nbins(Zs_discr)])


								for i=1:length(f)
								         @constraint(mproj,
								         [entropy_upper_bd_vec[i]; f[i]; prob_vec_dummy[i]] in MathOptInterface.ExponentialCone())
								end

								@objective(mproj, Min, -sum(entropy_upper_bd_vec ))
								mproj
end




function (fmc::FixedMarginalConvolution)(a)
	fmc(a, fmc.model)
end

# mutating and not clear from type...
function (fmc::FixedMarginalConvolution)(a, ::Nothing)
	mproj = initialize_dkl_projection(fmc.prior_class, fmc.Zs_discr)
	fmc.model = mproj
	fmc(a, fmc.model)
end

function (fmc::FixedMarginalConvolution)(a, mproj::JuMP.Model)
	@show sum(a)
	fix.(mproj[:prob_vec_dummy], a)
	optimize!(mproj)
	prior_vals = JuMP.value.(mproj[:πs])
	@show sum(prior_vals)
	if isapprox(sum(prior_vals) , 1.0; atol=0.001)
		prior_vals = max.(prior_vals, 0.0)
		prior_vals = prior_vals/sum(prior_vals)
	end
	prior_res = fmc.prior_class(prior_vals)
	pdf( marginalize(prior_res, fmc.Zs_discr))
end
