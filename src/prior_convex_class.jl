# Object to store representation of prior

abstract type ConvexPriorClass end


mutable struct GaussianMixturePriorClass{T<:Real,
                                         VT<:AbstractVector{T}} <: ConvexPriorClass
    σ_prior::Float64
    grid::VT
    solver
end


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
    v = [ target(Normal(μ, σ_prior)) for μ in grid]
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

    model = Model(with_optimizer(gmix_class.solver))
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



function modulus_problem(Z::DiscretizedStandardNormalSample,
                        gmix_class::GaussianMixturePriorClass,
                        target::EBayesTarget,
                        δ::Float64)

    model = Model(with_optimizer(gmix_class.solver))
    πs1 = add_prior_variables!(model, gmix_class; var_name = "πs1")
    πs2 = add_prior_variables!(model, gmix_class; var_name = "πs2")

    fs1 = marginalize(gmix_class, Z, πs1)
    fs2 = marginalize(gmix_class, Z, πs2)

    f̄s = pdf(Z)# pdf(Z) #

    L1 = linear_functional(gmix_class, target, πs1)
    L2 = linear_functional(gmix_class, target, πs2)

    #pseudo_chisq_dist = sum( (fs1 .- fs2).^2)
    #@constraint(model, pseudo_chisq_dist <= δ)
    @constraint(model, blabla, [δ; fs1-fs2] in SecondOrderCone())

    #if (C < Inf)
    #    @constraint(jm, f3 .- f_marginal .<= C*h_marginal_grid)
    #    @constraint(jm, f3 .- f_marginal .>= -C*h_marginal_grid)
    #end

    @objective(model, Max, L2 - L1)

    optimize!(model)
    obj_val = objective_value(model)

    (g1 = gmix_class(JuMP.value.(πs1)),
     g2 = gmix_class(JuMP.value.(πs2)),
     obj_val = obj_val)
end



# way to define optim representation
#

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
