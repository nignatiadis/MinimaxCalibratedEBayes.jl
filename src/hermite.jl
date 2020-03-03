const _hermite = Hermite()
const DEFAULT_HERMITE_INTEGRATOR = expectation(Normal(); n=100)

# physicist's hermite polynomials


function hermite_fun_noexp(xs, order)
	n = order - 1
	_hermite[xs, order]./sqrt(2^n*factorial(n)*sqrt(pi))
end

function hermite_fun(xs, order)
	hermite_fun_noexp(xs, order).*exp.(-xs.^2/2)
end

hermite_cf_coefficient(order) = im^(order-1)*sqrt(2*π)

#fourier_transform(t, order) = hermite_fun(t, order)*cf_coefficient(order)

struct HermiteBasisFunction
	q::Integer
end


function (target::MarginalDensityTarget{<:DiscretizedStandardNormalSample})(prior::HermiteBasisFunction;
                                                                            integrator=DEFAULT_HERMITE_INTEGRATOR)
    Z_disc = location(target)
    grid = Z_disc.samples.mhist.grid
    i = Z_disc.bin
    if i == 1
        base_integrand = μ -> cdf(Normal(μ), first(grid))
    elseif i == lastindex(Z_disc.samples.mhist)
        base_integrand = μ -> ccdf(Normal(μ), last(grid))
    else
        base_integrand = μ -> (cdf(Normal(μ), grid[i]) -  cdf(Normal(μ), grid[i-1]))
    end
	integrand = μ -> base_integrand(μ)*hermite_fun_noexp(μ, prior.q)
    # refactor into riesz representer..
    integrator(integrand)
end

Base.@kwdef mutable struct HermitePriorClass <: ConvexPriorClass
	q::Integer
	sobolev_order::Integer #not necessarily but should suffice for now
	sobolev_bound::Float64
	integrator = E_hermite
	solver = nothing
	solver_params = ()
	cached_integrals = nothing
end

function add_prior_variables!(model::JuMP.Model, hermite_class::HermitePriorClass; var_name = "πs") # adds constraints
    q = hermite_class.q
    tmp_vars = @variable(model, [i=1:q])
    model[Symbol(var_name)] = tmp_vars
	cf_coefs = cf_coefficient.(1:q)
	con = @constraint(model, dot(tmp_vars, real.(cf_coefs)) == 1.0)
	con2 = @constraint(model, dot(tmp_vars, imag.(cf_coefs)) == 0.0)
    tmp_vars
end

function marginalize(hermite_class::HermitePriorClass,
                     Zs_discr::DiscretizedStandardNormalSamples,
                     param_vec) #-> A*param_vec
    q_max = hermite_class.q
	qdtr =  hermite_class.integrator
	Zs_discr_length = length(Zs_discr.mhist.grid) + 1
	A_mat = [ MarginalDensityTarget(Zs_discr(i))(HermiteBasisFunction(q);
	                                             integrator = qdtr) for i=1:Zs_discr_length, q=1:q_max]
    A_mat*param_vec
end

function linear_functional(hermite_class::HermitePriorClass,
                           target::PriorDensityTarget, #let's hardcode to this for now.
                           param_vec)
	x = location(target)
	q_max = hermite_class.q
    v = hermite_fun.(x, 1:q_max)
    dot(v, param_vec)
end

struct HermiteQuasiDistribution<: Distribution{Univariate, Continuous}
	q::Integer
	coefs::Vector{Float64}
end

function pdf(herm_dbn::HermiteQuasiDistribution, x)
	 dot(hermite_fun.(x, 1:herm_dbn.q), herm_dbn.coefs)
end



function marginalize(hermite_dbn::HermiteQuasiDistribution,
                     Zs_discr::DiscretizedStandardNormalSamples)
   ws = marginalize(hm_prior_class, Zs_discr, hermite_dbn.coefs)
   mhist = Zs_discr.mhist
   mhist = @set mhist.hist.weights = ws
end

function (hermite_class::HermitePriorClass)(param_vec)
    q = hermite_class.q
  	HermiteQuasiDistribution(q, param_vec)
end
