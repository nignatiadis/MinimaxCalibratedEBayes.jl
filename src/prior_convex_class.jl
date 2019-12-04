# Object to store representation of prior

abstract type ConvexPriorClass end


mutable struct GaussianMixturePriorClass{T<:Real,
                                         VT<:AbstractVector{T}} <: ConvexPriorClass
    σ_prior::Float64
    grid::VT
    solver
end

GaussianMixturePriorClass(σ_prior, grid) = GaussianMixturePriorClass(σ_prior, grid, Gurobi.Optimizer)
#function GaussianMixturePriorClass(σ_prior, grid)
#    GaussianMixturePriorClass(σ_prior, grid, [], Matrix{eltype(σ_prior)}(undef, 0, 0))
#end
length(gmix_class::GaussianMixturePriorClass) = length(gmix_class.grid)



function add_prior_variables!(model, gmix_class::GaussianMixturePriorClass; var_name = "πs") # adds constraints
    n_priors = length(gmix_class)
    tmp_vars = @variable(model, [i=1:n])
    model[Symbol(var_name)] = tmp_vars
    set_lower_bound.(tmp_vars, 0.0)
    con = @constraint(model, sum(tmp_vars) == 1.0)
    model
end

function marginalize(cvx_prior_class, mhist, param_vec) #-> A*param_vec

end


# In fact all computations with marginalized object.

# param -> A*x

# finite-dim representation, here: probs
# want to map: probs ->



function worst_case_bias(Q::BinnedCalibrator,
                  Z::DiscretizedStandardNormalSample,
                  gmix_class::GaussianMixturePriorClass,
                  target = LFSRNumerator(2.0);
                  maximization=true)

    model = Model(with_optimizer(gmix_class.solver))
    add_prior_variables!(model, gmix_class; var_name = "πs")

    #if (C < Inf)
    #    @constraint(jm, f3 .- f_marginal .<= C*h_marginal_grid)
    #    @constraint(jm, f3 .- f_marginal .>= -C*h_marginal_grid)
    #end

    if maximization
        @objective(jm, Max, Q.Qo + dot(Q.Q,f3)-dot(L,π3))
    else
        @objective(jm, Min, Q.Qo + dot(Q.Q,f3)-dot(L,π3))
    end

    status = solve(jm)

    t_curr = getobjectivevalue(jm)
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

