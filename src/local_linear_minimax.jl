#----------------------------------
# Structure
#----------------------------------
# ModulusModel contains all JuMP related information
# LocalizedAffineMinimax contains optimization information
# SteinMinimaxEstimator contains the fit result
#





Base.@kwdef struct ModulusModel
    method
    model
    g1
    g2
    f1
    f2
    f_sqrt
    Δf
    δ_max
    δ_up
    bound_delta
    target
end

function get_δ(Δf)
    norm(JuMP.value.(Δf))
end

function get_δ(model::ModulusModel; recalculate_δ = false)
    if recalculate_δ
        get_δ(model.Δf)
    else
      JuMP.value(model.δ_up)
    end
end

"""
    DeltaTuner

Abstract type used to represent ways of picking
``\\delta`` at which to solve the modulus problem, cf.
Manuscript. Different choices of ``\\delta`` correspond
to different choices of the Bias-Variance tradeoff with
every choice leading to Pareto-optimal tradeoff.
"""
abstract type DeltaTuner end

abstract type BiasVarAggregate <: DeltaTuner end

function get_bias_var(modulus_model::ModulusModel)
    @unpack model = modulus_model
    δ = get_δ(modulus_model)
    ω_δ = objective_value(model)
    ω_δ_prime = -JuMP.dual(modulus_model.bound_delta)
    max_bias = (ω_δ - δ*ω_δ_prime)/2
    unit_var_proxy = ω_δ_prime^2
    max_bias, unit_var_proxy
end

function (bv::BiasVarAggregate)(modulus_model::ModulusModel)
    bv(get_bias_var(modulus_model)...)
end

"""
    RMSE(n::Integer, δ_min::Float64) <: DeltaTuner

A `DeltaTuner` to optimizes
the worst-case (root) mean squared error.  Here `n` is the sample
size used for estimation.
"""
struct RMSE{N} <: BiasVarAggregate
    n::N
end

(rmse::RMSE)(bias, unit_var_proxy) =  sqrt(bias^2 + unit_var_proxy/rmse.n)

function Empirikos._set_defaults(rmse::RMSE, Zs; hints)
    RMSE(length(Zs)) #nobs?
end

Base.@kwdef struct LocalizedAffineMinimax{N, G, M}
    convexclass::G
    neighborhood::N
    solver
    discretizer
    plugin_G
    data_split = :none
    delta_grid = 0.2:0.3:5
    delta_objective = RMSE(DataBasedDefault())
    modulus_model::M = nothing
    n = nothing
end

function initialize_modulus_model(method::LocalizedAffineMinimax, target, δ)

    estimated_marginal_density = method.discretizer
    neighborhood = method.neighborhood
    gcal = method.convexclass

    solver = method.solver

    model = Model(solver)

    g1 = Empirikos.prior_variable!(model, gcal)
    g2 = Empirikos.prior_variable!(model, gcal)

    Empirikos.neighborhood_constraint!(model, neighborhood, g1)
    Empirikos.neighborhood_constraint!(model, neighborhood, g2)

    Zs = collect(keys(estimated_marginal_density))
    f_sqrt = sqrt.(collect(values(estimated_marginal_density)))

    f1 = pdf.(g1, Zs)
    f2 = pdf.(g2, Zs)

    @variable(model, δ_up)

    Δf = @expression(model, (f1 - f2)./f_sqrt)

    @constraint(model, pseudo_chisq_constraint,
           [δ_up; Δf] in SecondOrderCone())

    @objective(model, Max, target(g2) - target(g1))

    @constraint(model, bound_delta, δ_up <= δ)

    ModulusModel(method=method, model=model, g1=g1, g2=g2, f1=f1, f2=f2,
        f_sqrt=f_sqrt, Δf=Δf, δ_max=Inf, δ_up=δ_up,
        bound_delta=bound_delta, target=target)
end


function set_δ!(modulus_model, δ)
    set_normalized_rhs(modulus_model.bound_delta, δ)
    optimize!(modulus_model.model)
    modulus_model
end

function set_target!(modulus_model, target)
    if modulus_model.target == target
        return modulus_model
    end
    @unpack model, g1, g2 = modulus_model
    @objective(model, Max, target(g2) - target(g1))
    modulus_model = @set modulus_model.target = target
    optimize!(modulus_model.model)
    modulus_model
end


function StatsBase.fit(method::LocalizedAffineMinimax, target, Zs; kwargs...)
    method = Empirikos.set_defaults(method, Zs; kwargs...)
    fitted_nbhood = StatsBase.fit(method.neighborhood, Zs; kwargs...)
    fitted_plugin_G = StatsBase.fit(method.plugin_G, Zs; kwargs...) #TODO SPECIAL CASE for ::Distribution
    discr = method.discretizer

    # todo: fix this
    _ints = StandardNormalSample.(discr.sorted_intervals)
    fitted_density = Empirikos.DiscretizedDictFunction(discr,
                                              DictFunction(_ints, pdf.(fitted_plugin_G.prior, _ints)))

    method = @set method.neighborhood = fitted_nbhood
    method = @set method.plugin_G = fitted_plugin_G
    method = @set method.discretizer = fitted_density

    n = nobs(Zs) #TODO: length or nobs?
    method = @set method.n = n

    @unpack delta_grid = method

    δ1 = delta_grid[1]
    modulus_model = initialize_modulus_model(method, target, δ1)
    method = @set method.modulus_model = modulus_model

    _fit_initialized(method::LocalizedAffineMinimax, target, Zs; kwargs...)
end




Base.@kwdef mutable struct SteinMinimaxEstimator{M, T, D, S}
    target::T
    δ::Float64
    ω_δ::Float64
    ω_δ_prime::Float64
    g1
    g2
    Q::D
    max_bias::Float64
    unit_var_proxy::Float64 #n::Int64
    modulus_model::M #    δ_tuner::DeltaTuner
    δs::S = zeros(Float64,5)
    δs_objective::S = zeros(Float64,5)
end

var(sme::SteinMinimaxEstimator) = sme.unit_var_proxy/sme.n


function worst_case_bias(sme::SteinMinimaxEstimator)
    sme.max_bias
end

#"""
#SteinMinimaxEstimator(Zs_discr::DiscretizedStandardNormalSamples,
#              prior_class::ConvexPriorClass,
##              target::LinearEBayesTarget,
#              δ_tuner::DeltaTuner
#
#Computes a linear estimator optimizing a worst-case bias-variance tradeoff (specified by `δ_tuner`)
#for estimating a linear `target` over `prior_class` based on [`DiscretizedStandardNormalSamples`](@ref)
#`Zs_discr`.
#"""
#function SteinMinimaxEstimator(Zs_discr::DiscretizedStandardNormalSamples,
#                   gmix::ConvexPriorClass,
#                   target::LinearEBayesTarget,
#                   δ_tuner::DeltaTuner)

#modulus_problem = initialize_modulus_problem(Zs_discr, gmix, target)
#δ_opt = optimal_δ(modulus_problem, δ_tuner)
#modulus_problem = modulus_at_δ!(modulus_problem, δ_opt)
#SteinMinimaxEstimator(Zs_discr, gmix, target, modulus_problem;
#             δ = δ_opt, δ_tuner = δ_tuner)
#end


function SteinMinimaxEstimator(modulus_model::ModulusModel)
    @unpack model, method, target = modulus_model
    @unpack convexclass, discretizer = method

    estimated_marginal_density = discretizer

    δ = get_δ(modulus_model)
    ω_δ = objective_value(model)
    ω_δ_prime = -JuMP.dual(modulus_model.bound_delta)

    g1 = convexclass(JuMP.value.(modulus_model.g1.finite_param))
    g2 = convexclass(JuMP.value.(modulus_model.g2.finite_param))

    L1 = target(g1)
    L2 = target(g2)

    Zs = collect(keys(estimated_marginal_density))

    f1 = pdf.(g1, Zs)
    f2 = pdf.(g2, Zs)

    f̄s = collect(values(estimated_marginal_density))

    Q = ω_δ_prime/δ*(f2 .- f1)./f̄s
    Q_0  = (L1+L2)/2 -
        ω_δ_prime/(2*δ)*sum( (f2 .- f1).* (f2 .+ f1) ./ f̄s)

    Q = Empirikos.DiscretizedDictFunction(estimated_marginal_density.discretizer, DictFunction(Zs, Q .+ Q_0))

    max_bias = (ω_δ - δ*ω_δ_prime)/2
    unit_var_proxy = ω_δ_prime^2


SteinMinimaxEstimator(
              target=target,
              δ=δ,
              ω_δ=ω_δ,
              ω_δ_prime=ω_δ_prime,
              g1=g1,
              g2=g2,
              Q=Q,
              max_bias=max_bias,
              unit_var_proxy=unit_var_proxy,
              modulus_model=modulus_model)
end


function _fit_initialized(method::LocalizedAffineMinimax, target, Zs; kwargs...)
    @unpack modulus_model, delta_grid, delta_objective,n = method
    set_target!(modulus_model, target)

    δs_objective = zeros(Float64, length(delta_grid))
    for (index, δ) in enumerate(delta_grid)
        set_δ!(modulus_model, δ/sqrt(n)) #TODO: sanity checks
        δs_objective[index] = delta_objective(modulus_model)
    end
    idx_best = argmin(δs_objective)
    set_δ!(modulus_model, delta_grid[idx_best]/sqrt(n)) #TODO: sanity checks
    sm = SteinMinimaxEstimator(modulus_model)
    sm = @set sm.δs = collect(delta_grid)
    sm = @set sm.δs_objective = δs_objective
    sm
end


function StatsBase.confint(Q::SteinMinimaxEstimator, target, Zs; α=0.05)
    _bias = Q.max_bias
    _se = std(Q.Q.(Zs))/sqrt(nobs(Zs))
    point_estimate = mean(Q.Q.(Zs))
    halfwidth = MinimaxCalibratedEBayes.gaussian_ci(_se; maxbias=_bias, α=α)
    (lower = point_estimate - halfwidth, upper = point_estimate + halfwidth)
end
