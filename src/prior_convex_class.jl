# Object to store representation of prior

abstract type ConvexPriorClass end



mutable struct GaussianMixturePriorClass{T<:Real,
                                         VT<:AbstractVector{T}} <: ConvexPriorClass
    σ_prior::Float64
    grid::VT
    solver
    solver_params
end

GaussianMixturePriorClass(σ_prior, grid, solver) = GaussianMixturePriorClass(σ_prior, grid, solver, ())
GaussianMixturePriorClass(σ_prior, grid) = GaussianMixturePriorClass(σ_prior, grid, Gurobi.Optimizer)

length(gmix_class::GaussianMixturePriorClass) = length(gmix_class.grid)
location(gmix_class::GaussianMixturePriorClass) = gmix_class.grid



function add_prior_variables!(model, gmix_class::GaussianMixturePriorClass; var_name = "πs") # adds constraints
    n_priors = length(gmix_class)
    tmp_vars = @variable(model, [i=1:n_priors])
    model[Symbol(var_name)] = tmp_vars
    set_lower_bound.(tmp_vars, 0.0)
    con = @constraint(model, sum(tmp_vars) == 1.0)
    tmp_vars
end

function marginalize(gmix_class::GaussianMixturePriorClass,
                     z::DiscretizedStandardNormalSample,
                     param_vec) #-> A*param_vec
    σ_prior = gmix_class.σ_prior
    grid = gmix_class.grid
    A_mat = hcat([ pdf(marginalize(Normal(μ, σ_prior), z)) for μ in grid]...)
    A_mat*param_vec
end

function linear_functional(gmix_class::GaussianMixturePriorClass,
                           target::EBayesTarget,
                           param_vec)
    σ_prior = gmix_class.σ_prior
    grid = gmix_class.grid
    v = [target(Normal(μ, σ_prior)) for μ in grid]
    dot(v, param_vec)
end

function (gmix_class::GaussianMixturePriorClass)(param_vec)
    σ_prior = gmix_class.σ_prior
    grid = gmix_class.grid
    #TODO: check param_vec is probability vector
    MixtureModel(Normal, [(μ, σ_prior) for μ in grid], param_vec)
end


# In fact all computations with marginalized object.

# param -> A*x

# finite-dim representation, here: probs
# want to map: probs ->



function worst_case_bias(Q::DiscretizedAffineEstimator,
                  Z::DiscretizedStandardNormalSample,
                  gmix_class::GaussianMixturePriorClass,
                  target::EBayesTarget;
                  maximization=true)

    model = Model(with_optimizer(gmix_class.solver; gmix_class.solver_params...))
    πs = add_prior_variables!(model, gmix_class; var_name = "πs")
    fs = marginalize(gmix_class, Z, πs)
    L = linear_functional(gmix_class, target, πs)
    #if (C < Inf)
    #    @constraint(jm, f3 .- f_marginal .<= C*h_marginal_grid)
    #    @constraint(jm, f3 .- f_marginal .>= -C*h_marginal_grid)
    #end

    if maximization
        @objective(model, Max, Q.Qo + dot(Q.Q,fs) - L)
    else
        @objective(model, Min, Q.Qo + dot(Q.Q,fs) - L)
    end

    optimize!(model)
    obj_val = objective_value(model)

    (max_worst_g = gmix_class(JuMP.value.(πs)),
     max_bias = obj_val)
end


#function initialize_modulus_problem( )

#nd

#function optimize!(modulus_problem, Z, gmix_class, target, δ)

#end
Base.@kwdef mutable struct SteinMinimaxEstimator
                Z::DiscretizedStandardNormalSample
                prior_class::ConvexPriorClass
                target::EBayesTarget
                δ::Float64
                ω_δ::Float64
                ω_δ_prime::Float64
                g1
                g2
                Q::DiscretizedAffineEstimator
                max_bias::Float64
                unit_var_proxy::Float64
                n::Int
                model
end

function initialize_modulus_problem(Z::DiscretizedStandardNormalSample,
                                    prior_class::GaussianMixturePriorClass,
                                    target::EBayesTarget)

    model = Model(with_optimizer(prior_class.solver; prior_class.solver_params...))
    πs1 = add_prior_variables!(model, prior_class; var_name = "πs1")
    πs2 = add_prior_variables!(model, prior_class; var_name = "πs2")

    fs1 = marginalize(prior_class, Z, πs1)
    fs2 = marginalize(prior_class, Z, πs2)

    model[:fs1] = fs1
    model[:fs2] = fs2


    #if (C < Inf)
    #    @constraint(jm, f3 .- f_marginal .<= C*h_marginal_grid)
    #    @constraint(jm, f3 .- f_marginal .>= -C*h_marginal_grid)
    #end

    L1 = linear_functional(prior_class, target, πs1)
    L2 = linear_functional(prior_class, target, πs2)

    model[:L1] = L1
    model[:L2] = L2

    f̄s_sqrt = sqrt.(pdf(Z))
    model[:f̄s_sqrt] = sqrt.(pdf(Z))
                    #pseudo_chisq_dist = sum( (fs1 .- fs2).^2)
                                        #@constraint(model, pseudo_chisq_dist <= δ)
    @variable(model, δ_up)
    model[:δ_up] = δ_up

    Δf = (fs1 .- fs2)./f̄s_sqrt
    model[:Δf] = Δf
    @constraint(model, pseudo_chisq_constraint,
               [δ_up; Δf] in SecondOrderCone())


               function get_δ(model; recalculate_δ = false)
                   if recalculate_δ
                       norm(JuMP.value.(model[:Δf]))
                   else
                     JuMP.value(model[:δ_up])
                   end
               end

    @objective(model, Max, L2 -L1)
    optimize!(model)
    δ = get_δ(model; recalculate_δ = true)
    @constraint(model, bound_delta, δ_up == δ)
    model
end






abstract type DeltaTuner end

struct FixedDelta <: DeltaTuner
    δ::Float64
end

#abstract type


abstract type BiasVarAggregate end

function (bv::BiasVarAggregate)(model)
    bv()
end

struct RMSE <: BiasVarAggregate end
(::RMSE)(bias, se) = sqrt(bias^2 + se^2)
const rmse = RMSE()

struct HalfCIWidth <: BiasVarAggregate
    α::Float64
end
HalfCIWidth() = HalfCIWidth(0.9)
function (half_ci::HalfCIWidth)(bias, se)
    bias_adjusted_gaussian_ci(se, maxbias=bias, α=half_ci.α)
end

function subfamily_mse(model, n)
    ω_δ = objective_value(model)
    ω_δ_prime = -JuMP.dual(model[:bound_delta])
    δ = JuMP.normalized_rhs(bound_delta)
    max_bias = (ω_δ - δ*ω_δ_prime)/2
    unit_var_proxy = ω_δ_prime^2

    -ω_δ^2/(4 + n*δ^2)
end



struct ObjectiveDelta <: DeltaTuner
    mse
end



function modulus_problem(Z::DiscretizedStandardNormalSample,
                        prior_class::GaussianMixturePriorClass,
                        target::EBayesTarget,
                        δ::Float64;
                        n=10_000)



    @objective(model, Max, L2 - L1)

    optimize!(model)
    ω_δ = objective_value(model)
    ω_δ_prime = -JuMP.dual(model[:bound_delta])

    g1 = prior_class(JuMP.value.(πs1))
    g2 = prior_class(JuMP.value.(πs2))
    L1 = JuMP.value(L1)
    L2 = JuMP.value(L2)

    #construct estimator
    f1 = marginalize(g1, Z)
    f2 = marginalize(g2, Z)

    Q = ω_δ_prime/δ*(pdf(f2) .- pdf(f1))./pdf(Z)
    Q_0  = (L1+L2)/2 -
           ω_δ_prime/(2*δ)*sum( (pdf(f2) .- pdf(f1)).* (pdf(f2) .+ pdf(f1)) ./ pdf(Z))

    stein = DiscretizedAffineEstimator(Z, Q, Q_0)
    max_bias = (ω_δ - δ*ω_δ_prime)/2
    unit_var_proxy = ω_δ_prime^2

    SteinMinimaxEstimator(Z=Z,
                          prior_class=prior_class,
                          target=target,
                          δ=δ,
                          ω_δ=ω_δ,
                          ω_δ_prime=ω_δ_prime,
                          g1=g1,
                          g2=g2,
                          Q=stein,
                          max_bias=max_bias,
                          unit_var_proxy=unit_var_proxy,
                          n=n,
                          model=model)
end

var(sme::SteinMinimaxEstimator) = sme.unit_var_proxy/sme.n
function worst_case_bias(sme::SteinMinimaxEstimator)
    sme.max_bias
end


@userplot SteinMinimaxPlot

@recipe function f(h::SteinMinimaxPlot; x_grid = -5:0.01:5)

    sme = first(h.args)
    layout := @layout [panel1 panel2 panel3]

    # main histogram2d
    @series begin
        seriestype := :path
        subplot := 1
        #xlim := extrema(x_grid)
        sme.Q
    end

    # upper histogram
    g1 = sme.g1
    g2 = sme.g2
    g1_xs = pdf.(g1, x_grid)
    g2_xs = pdf.(g2, x_grid)
    @series begin
        seriestype := :path
        subplot := 2
        linecolor --> [:pink :purple]
        label := ["g1" "g2"]
        x_grid, [g1_xs g2_xs]
    end


    Z=sme.Z
    Z_continuous = StandardNormalSample(0.0)

    f1=marginalize(g1, Z_continuous)
    f2=marginalize(g2, Z_continuous)

    f1_xs=pdf.(f1, x_grid)
    f2_xs=pdf.(f2, x_grid)

    @series begin
        seriestype := :path
        subplot := 3
        linecolor --> [:orange :brown]
        x_grid, [f1_xs f2_xs]
    end
end







##marginal_pdf(gm_class, params) ->  #


#struct LipschitzPriorClass <: ConvexPriorClass
#end

# parametrize by slopes? easier constraints but harder to ....






#function check_bias(ma::MinimaxCalibrator; maximization=true)
#    ds = ma.ds
#    f = ma.f
#    C = ma.C
#    ε = ma.ε_reg
#    m = ma.m
#    target = ma.target

#    Q = ma.Q

#    check_bias(Q, ds, f, m, target; C=C, maximization=maximization)
#end
