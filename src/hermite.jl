const _hermite = OrthogonalPolynomialsQuasi.Hermite()
# perhaps call fastGaussQuadrature directly
const DEFAULT_HERMITE_INTEGRATOR = sqrt(2π)*expectation(Normal(); n=100)
const DEFAULT_SQUARED_HERMITE_INTEGRATOR = sqrt(π)*expectation(Normal(0, 1/sqrt(2)); n=101)

# physicist's hermite polynomials


function hermite_fun_noexp(xs, n)
	xs = big.(xs)
	n = big.(n )
	Float64.(_hermite[xs, n+1]./sqrt(2^n*factorial(n)*sqrt(pi)))
end

function hermite_fun(xs, n)
	hermite_fun_noexp(xs, n).*exp.(-xs.^2/2)
end

hermite_cf_coefficient(n) = im^n*sqrt(2*π)

#fourier_transform(t, order) = hermite_fun(t, order)*hermite_cf_coefficient(order)

# Hermite = Mixture of Hermite Basis Functions
struct HermiteBasisFunction
	q::Integer
end

(herm::HermiteBasisFunction)(x) = MCEB.hermite_fun(x, herm.q)

function expectation(herm::HermiteBasisFunction,
	                 base_integrator::Expectation = DEFAULT_HERMITE_INTEGRATOR)
	hermite_integrator = deepcopy(base_integrator)
	wts = hermite_fun_noexp.(nodes(hermite_integrator), herm.q)
	Expectations.weights(hermite_integrator) .*= wts
	hermite_integrator
end

# wow this should really be dispatching on integrator
function (target::MarginalDensityTarget{<:DiscretizedStandardNormalSample})(integrator::Expectation)
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
    # refactor into riesz representer.. -> Oh already thought about it...
    integrator(base_integrand)
end

Base.@kwdef mutable struct HermitePriorClass{T<:Real} <: ConvexPriorClass
	qmax::Integer
	sobolev_order::T  #not necessarily but should suffice for now
	sobolev_bound::Float64
	integrators = expectation.(HermiteBasisFunction.(0:qmax), Ref(DEFAULT_HERMITE_INTEGRATOR))
	squared_integrator = DEFAULT_SQUARED_HERMITE_INTEGRATOR
	solver = nothing
	solver_params = ()
end

function add_prior_variables!(model::JuMP.Model, hermite_class::HermitePriorClass; var_name = "πs") # adds constraints
    qmax = hermite_class.qmax
	nparams = qmax + 1
    tmp_vars = @variable(model, [i=1:nparams])
    model[Symbol(var_name)] = tmp_vars
	cf_coefs = hermite_cf_coefficient.(0:qmax) .* hermite_fun_noexp.(0.0, 0:qmax)
	con = @constraint(model, dot(tmp_vars, real.(cf_coefs)) == 1.0)
	con2 = @constraint(model, dot(tmp_vars, imag.(cf_coefs)) == 0.0)
	#tmp_grid = -7:0.02:4
	tmp_grid = sort([nodes(hermite_class.squared_integrator); nodes(hermite_class.integrators[1])])
	# rewrite below
	A_mat_prior = [hermite_fun(x, qidx)  for x in tmp_grid, qidx=0:qmax]
	nonneg_constraint = @constraint(model, A_mat_prior*tmp_vars .>= 0)

	integrator2 = hermite_class.squared_integrator
	b = hermite_class.sobolev_order
	sobolev_bound = sqrt(hermite_class.sobolev_bound*2π)

	ft_mat = [hermite_fun_noexp(x, qidx)*hermite_cf_coefficient(qidx)  for x in integrator2.nodes, qidx=0:qmax]
	real_ft = (real(ft_mat) * tmp_vars) .* sqrt.(integrator2.weights .* (integrator2.nodes.^2 .+ 1).^b)
	imag_ft = imag(ft_mat) * tmp_vars .* sqrt.(integrator2.weights .* (integrator2.nodes.^2 .+ 1).^b)

	@constraint(model, [sobolev_bound; real_ft; imag_ft] in SecondOrderCone())

	#@constraint(model, sum( real_ft.^2 .* integrator2.weights) + sum( imag_ft.^2 .* integrator2.weights) <= sobolev_bound)

    tmp_vars
end

function marginalize(hermite_class::HermitePriorClass,
                     Zs_discr::DiscretizedStandardNormalSamples,
                     param_vec) #-> A*param_vec
	qdtrs =  hermite_class.integrators
	Zs_discr_length = length(Zs_discr.mhist.grid) + 1
	A_mat = [ MarginalDensityTarget(Zs_discr(i))(qdtr) for i=1:Zs_discr_length, qdtr in qdtrs]
    A_mat*param_vec
end

function linear_functional(hermite_class::HermitePriorClass,
                           target::PriorDensityTarget, #let's hardcode to this for now.
                           param_vec)
	x = location(target)
	qmax = hermite_class.qmax
    v = hermite_fun.(x, 0:qmax)
    dot(v, param_vec)
end

struct HermiteQuasiDistribution<: Distribution{Univariate, Continuous}
	q::Integer
	coefs::Vector{Float64}
end

function pdf(herm_dbn::HermiteQuasiDistribution, x)
	 dot(hermite_fun.(x, 0:herm_dbn.q), herm_dbn.coefs)
end



function marginalize(hermite_dbn::HermiteQuasiDistribution,
                     Zs_discr::DiscretizedStandardNormalSamples)
	#temporary hack
   hm_prior_class = HermitePriorClass(qmax=hermite_dbn.q,
                    sobolev_order =2 , sobolev_bound = 10.0)
   ws = marginalize(hm_prior_class, Zs_discr, hermite_dbn.coefs)
   mhist = Zs_discr.mhist
   mhist = @set mhist.hist.weights = ws
end

function (hermite_class::HermitePriorClass)(param_vec)
    q = hermite_class.qmax
  	HermiteQuasiDistribution(q, param_vec)
end
