# Object to store representation of prior
"""
    ConvexPriorClass 
    
Abstract type representing convex classes of probability distributions ``\\mathcal{G}``.
"""
abstract type ConvexPriorClass end


"""
	GaussianMixturePriorClass(σ_prior, grid, solver)
	
Class of distributions ``G \\in \\mathcal{G}`` that can be written as the convolution of a Normal 
distribution with standard deviation `σ_prior` and a discrete distribution ``\\pi``
supported on `grid`.

```math
G = \\mathcal{N}(0,\\sigma_{\\text{prior}}^2) \\star \\pi
``` 

The `solver` object is a `MathOptInterface` compatible optimizer such as `Mosek.Optimizer` that will be 
used to solve the convex modulus of continuity (or worst-case-bias estimation) problem.
"""
mutable struct GaussianMixturePriorClass{T<:Real,
                                         VT<:AbstractVector{T}} <: ConvexPriorClass
    σ_prior::Float64
    grid::VT
    solver
    solver_params
end

GaussianMixturePriorClass(σ_prior, grid, solver) = GaussianMixturePriorClass(σ_prior, grid, solver, [])
#GaussianMixturePriorClass(σ_prior, grid) = GaussianMixturePriorClass(σ_prior, grid, Gurobi.Optimizer)

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
                     z::DiscretizedStandardNormalSamples,
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

# TODO: more rigorous projection onto simplex?
function fix_πs(gmix_class::GaussianMixturePriorClass, πs)
    πs = max.(πs, 0.0)
    πs = πs ./ sum(πs)
end

fix_πs(::ConvexPriorClass, πs) = πs


# In fact all computations with marginalized object.

# param -> A*x

# finite-dim representation, here: probs
# want to map: probs ->


function worst_case_bias(general_Q,
                  Z::DiscretizedStandardNormalSamples,
                  gmix_class::ConvexPriorClass,
                  target::LinearEBayesTarget;
                  boundary_mass = Inf)
     grid = Z.mhist.grid
     tmpcalib = DiscretizedAffineEstimator(MCEBHistogram(grid), general_Q)
     worst_case_bias(tmpcalib, Z, gmix_class, target; boundary_mass=boundary_mass)
end

function worst_case_bias(Q::DiscretizedAffineEstimator,
                  Z::DiscretizedStandardNormalSamples,
                  prior_class::ConvexPriorClass,
                  target::LinearEBayesTarget;
                  boundary_mass = Inf)

    model = Model(optimizer_with_attributes(prior_class.solver, prior_class.solver_params...))
    πs = add_prior_variables!(model, prior_class; var_name = "πs")
    fs = marginalize(prior_class, Z, πs)
    L = linear_functional(prior_class, target, πs)


    if (!isnothing(Z.f_max))
        idx_upper = idx_enforce_upper_bound(prior_class, Z)#
        @constraint(model, f1_upper, fs[idx_upper] .<= Z.f_max[idx_upper])
    end

    if (!isnothing(Z.f_min))
        idx_nonzero = idx_enforce_lower_bound(prior_class, Z) #
        @constraint(model, f1_lower, fs[idx_nonzero] .>= Z.f_min[idx_nonzero])
    end

    if boundary_mass < Inf
        @constraint(model, fs[1] <= boundary_mass)
        @constraint(model, fs[end] <= boundary_mass)
    end
    #    @constraint(jm, f3 .- f_marginal .<= C*h_marginal_grid)
    #    @constraint(jm, f3 .- f_marginal .>= -C*h_marginal_grid)
    #end


    @objective(model, Max, Q.Qo + dot(Q.Q,fs) - L)
    optimize!(model)
    max_bias = objective_value(model)
    
    # TODO let prior_class function do this by itself
    max_g = prior_class(fix_πs(prior_class, JuMP.value.(πs))) 


    @objective(model, Min, Q.Qo + dot(Q.Q,fs) - L)
    optimize!(model)
    min_bias = objective_value(model)
    min_g = prior_class(fix_πs(prior_class, JuMP.value.(πs)))

    max_abs_bias = max(abs(max_bias), abs(min_bias))

    (max_abs_bias = max_abs_bias,
     max_squared_bias = max_abs_bias^2,
     max_bias = max_bias, min_bias = min_bias,
     max_g = max_g, min_g = min_g,
     model = model)
end


#function initialize_modulus_problem( )

#nd

#function optimize!(modulus_problem, Z, gmix_class, target, δ)

#end


# helper functions for working with model.
function get_δ(model; recalculate_δ = false)
    if recalculate_δ
        norm(JuMP.value.(model[:Δf]))
    else
      JuMP.value(model[:δ_up])
    end
end

# TODO: figure out why this errors
function idx_enforce_lower_bound(prior_class::ConvexPriorClass, Z::DiscretizedStandardNormalSamples)
    1:length(Z.f_min)#1:length(Z.f_min)
end 

function idx_enforce_lower_bound(prior_class::GaussianMixturePriorClass, Z::DiscretizedStandardNormalSamples)
    findall( Z.f_min .> 0.0)
end 

# TODO: figure out why this errors
function idx_enforce_upper_bound(prior_class::ConvexPriorClass, Z::DiscretizedStandardNormalSamples)
    findall( Z.f_max .< 1/sqrt(2π))#1:length(Z.f_max)
end 

function idx_enforce_upper_bound(prior_class::GaussianMixturePriorClass, Z::DiscretizedStandardNormalSamples)
    findall( Z.f_max .< 1/sqrt(2π))
end 

function initialize_modulus_problem(Z::DiscretizedStandardNormalSamples,
                                    prior_class::ConvexPriorClass,
                                    target::EBayesTarget)

    model = Model(optimizer_with_attributes(prior_class.solver, prior_class.solver_params...))
    πs1 = add_prior_variables!(model, prior_class; var_name = "πs1")
    πs2 = add_prior_variables!(model, prior_class; var_name = "πs2")

    fs1 = marginalize(prior_class, Z, πs1)
    fs2 = marginalize(prior_class, Z, πs2)

    model[:fs1] = fs1
    model[:fs2] = fs2

    if (!isnothing(Z.f_max))
        idx_upper = idx_enforce_upper_bound(prior_class, Z)#
        @constraint(model, f1_upper, fs1[idx_upper] .<= Z.f_max[idx_upper])
        @constraint(model, f2_upper, fs2[idx_upper] .<= Z.f_max[idx_upper])
    end

    if (!isnothing(Z.f_min))
        idx_nonzero = idx_enforce_lower_bound(prior_class, Z) #
        @constraint(model, f1_lower, fs1[idx_nonzero] .>= Z.f_min[idx_nonzero])
        @constraint(model, f2_lower, fs2[idx_nonzero] .>= Z.f_min[idx_nonzero])
    end

    L1 = linear_functional(prior_class, target, πs1)
    L2 = linear_functional(prior_class, target, πs2)

    model[:L1] = L1
    model[:L2] = L2

    f̄s_sqrt = sqrt.(Z.var_proxy)
    model[:f̄s_sqrt] = f̄s_sqrt
                   
    @variable(model, δ_up)
    model[:δ_up] = δ_up

    Δf = (fs1 .- fs2)./f̄s_sqrt
    model[:Δf] = Δf
    @constraint(model, pseudo_chisq_constraint,
               [δ_up; Δf] in SecondOrderCone())

    @objective(model, Max, L2 -L1)
    optimize!(model)
    δ = get_δ(model; recalculate_δ = true)
    model[:δ_max] = δ
    @constraint(model, bound_delta, δ_up == δ)
    model
end

function modulus_at_δ!(model, δ)
    set_normalized_rhs(model[:bound_delta], δ)
    optimize!(model)
    model
end



function δ_path(model, δs)
    max_biases = Vector{Float64}(undef, length(δs))
    unit_variances   = Vector{Float64}(undef, length(δs))
    for (i,δ) in enumerate(δs)
        model = modulus_at_δ!(model, δ)
        tmp_bias, tmp_var = get_bias_var(model)
        max_biases[i] = tmp_bias
        unit_variances[i] = tmp_var
    end
    max_biases, unit_variances
end




default_δ_min(n, C∞_density) = sqrt( log(n)/n)*C∞_density


#function solve_modulus_problem




abstract type DeltaTuner end

struct FixedDelta <: DeltaTuner
    δ::Float64
end

function optimal_δ(model::JuMP.Model, objective::FixedDelta)
    objective.δ
end

#struct VarProxyUpperBound <: DeltaTuner
#
#end


abstract type BiasVarAggregate <:DeltaTuner end

function get_bias_var(model::JuMP.Model)
    δ = get_δ(model)
    ω_δ = objective_value(model)
    ω_δ_prime = -JuMP.dual(model[:bound_delta])
    max_bias = (ω_δ - δ*ω_δ_prime)/2
    unit_var_proxy = ω_δ_prime^2
    max_bias, unit_var_proxy
end

function (bv::BiasVarAggregate)(model::JuMP.Model)
    bv(get_bias_var(model)...)
end

struct RMSE <: BiasVarAggregate
    n::Integer
    δ_min::Float64
end

(rmse::RMSE)(bias, unit_var_proxy) =  sqrt(bias^2 + unit_var_proxy/rmse.n)

struct HalfCIWidth <: BiasVarAggregate
    n::Integer
    α::Float64
    δ_min::Float64
end


HalfCIWidth(n::Integer, δ_min::Float64) = HalfCIWidth(n, 0.9, δ_min)

function (half_ci::HalfCIWidth)(bias, unit_var_proxy)
    se = sqrt(unit_var_proxy/half_ci.n)
    bias_adjusted_gaussian_ci(se, maxbias=bias, level=half_ci.α)
end


function optimal_δ(model::JuMP.Model, objective::BiasVarAggregate)
    f = δ -> objective(modulus_at_δ!(model, δ))
    δ_max = model[:δ_max]
    δ_min = objective.δ_min
    Optim.optimize(f, δ_min, δ_max; rel_tol=1e-3).minimizer
end



function subfamily_mse(model, n)
    ω_δ = objective_value(model)
    ω_δ_prime = -JuMP.dual(model[:bound_delta])
    δ = JuMP.normalized_rhs(bound_delta)
    max_bias = (ω_δ - δ*ω_δ_prime)/2
    unit_var_proxy = ω_δ_prime^2

    -ω_δ^2/(4 + n*δ^2)
end





Base.@kwdef mutable struct SteinMinimaxEstimator
                Z::DiscretizedStandardNormalSamples
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
                δ_tuner::DeltaTuner
                δ_opt::Float64
                model
end

var(sme::SteinMinimaxEstimator) = sme.unit_var_proxy/sme.n


function worst_case_bias(sme::SteinMinimaxEstimator)
    sme.max_bias
end


function SteinMinimaxEstimator(Zs_discr::DiscretizedStandardNormalSamples,
                               gmix::ConvexPriorClass,
                               target::LinearEBayesTarget,
                               δ_tuner::DeltaTuner)

   modulus_problem = initialize_modulus_problem(Zs_discr, gmix, target)
   δ_opt = optimal_δ(modulus_problem, δ_tuner)
   modulus_problem = modulus_at_δ!(modulus_problem, δ_opt)
   SteinMinimaxEstimator(Zs_discr, gmix, target, modulus_problem;
                         δ = δ_opt, δ_tuner = δ_tuner)
end


function SteinMinimaxEstimator(Z::DiscretizedStandardNormalSamples,
                               prior_class::ConvexPriorClass,
                               target::LinearEBayesTarget,
                               model::JuMP.Model;
                               δ = get_δ(model),
                               δ_tuner = FixedDelta(δ)
                              )


    ω_δ = objective_value(model)
    ω_δ_prime = -JuMP.dual(model[:bound_delta])

    πs1 = JuMP.value.(model[:πs1])
    πs2 = JuMP.value.(model[:πs2])

    πs1 = fix_πs(prior_class, πs1)
    πs2 = fix_πs(prior_class, πs2)

    
    g1 = prior_class(πs1)
    g2 = prior_class(πs2)

    L1 = JuMP.value(model[:L1])
    L2 = JuMP.value(model[:L2])
    #construct estimator
    f1 = marginalize(g1, Z)
    f2 = marginalize(g2, Z)

    f̄s = Z.var_proxy
    Q = ω_δ_prime/δ*(pdf(f2) .- pdf(f1))./Z.var_proxy
    Q_0  = (L1+L2)/2 -
           ω_δ_prime/(2*δ)*sum( (pdf(f2) .- pdf(f1)).* (pdf(f2) .+ pdf(f1)) ./ f̄s)

    stein = DiscretizedAffineEstimator(Z, Q, Q_0)

    max_bias = (ω_δ - δ*ω_δ_prime)/2
    unit_var_proxy = ω_δ_prime^2

    n = nobs(Z)
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
                          δ_opt = δ,
                          δ_tuner = δ_tuner,
                          model=model)
end



target_bias_std(target::EBayesTarget, model, Zs; kwargs...) = target_bias_std(target, model; kwargs...)
confint(target::EBayesTarget, model, Zs; kwargs...) = confint(target, model; kwargs...)

function target_bias_std(target::EBayesTarget, 
	                     sme::SteinMinimaxEstimator,
						 Zs=sme.Z;
                         std_type = :empirical)
	Qs = sme.(Zs) # assume Gaussian samples?	
    #@show length(Qs)				
	estimated_target = mean(Qs)
    if std_type == :empirical
	    estimated_std = std(Qs)/sqrt(nobs(Zs))
    elseif std_type == :var_proxy
        estimated_std = sqrt(sme.unit_var_proxy / nobs(Zs))
    else
        error("only :empirical and :var_proxy sdt_type defined")
    end
         
	estimated_bias = worst_case_bias(sme)
	(estimated_target = estimated_target,
	 estimated_bias = estimated_bias,
	 estimated_std = estimated_std)
end 

function (sme::SteinMinimaxEstimator)(x)
	sme.Q(x)
end

function Distributions.estimate(target::EBayesTarget, fitted_model, args...; kwargs...)
	target_bias_std(target, fitted_model, args...; kwargs...)[:estimated_target]
end
#		
#	end


function StatsBase.confint(target::EBayesTarget,
	                       sme, #SteinMinimaxEstimator #default fallback
                           args...; 
						   level = 0.9,
						   clip = true,
                           kwargs...)
	res = target_bias_std(target, sme, args...; kwargs...)
	maxbias = abs(res[:estimated_bias])
	q_mult = bias_adjusted_gaussian_ci(res[:estimated_std], maxbias=maxbias , level=level)
    L,U = res[:estimated_target] .+ (-1,1).*q_mult
	L, U = clip ? clamp.((L,U), extrema(target)... ) : (L,U)
    L,U
end



@userplot SteinMinimaxPlot

@recipe function f(h::SteinMinimaxPlot; x_grid = -5:0.01:5,
	                                          ylim_calibrator = nothing,
											  ylim_relative_offset = 1.25,
                                              ylim_panel_e = nothing)
	arg_length = length(h.args)
	arg_length <= 2 || error("at most two plots allowed")
    sme = first(h.args)

	if arg_length == 2
		sme_band = last(h.args)
		layout := @layout [a{0.45w} grid(2,2)]
	else 
		layout := @layout [panel1 panel2 panel3]
	end
	
	
	def_size = (850,260)

	col_g1 = RGB(77/255,137/255,193/255)
	col_gm1 = RGB(73/255,2/255,53/255)
	col_band = "#F6CD7F"#RGB(243/255,232/255,176/255)
	fg_legend --> :transparent
	bg_legend --> :transparent

    @series begin
        seriestype := :path
        subplot := 1
		linestyle := :dash
		if !isnothing(ylim_calibrator)
			ylim := ylim_calibrator
		end
        xlim := extrema(x_grid)
	    color := RGB(205/255,84/255,150/255)	# RGB(68/255,60/255,145/255)
		label := "Minimax"
        sme.Q
    end
	
	@series begin
		seriestype := :path
		subplot := 1
		linestyle := :solid
		color := RGB(132/255,193/255,216/255)	
		xlim := extrema(x_grid)
	    label := L"Minimax-$\infty$"
        title := "a)"
		legend := :topright
        xguide := L"x"
		sme_band.Q
	end
	

    # upper histogram
    g1 = sme.g1
    g2 = sme.g2
    g1_xs = pdf.(g1, x_grid)
    g2_xs = pdf.(g2, x_grid)
	ylim_prior = maximum([g1_xs;g2_xs])*ylim_relative_offset
    @series begin
        seriestype := :path
        subplot := 2
        linecolor := [col_g1 col_gm1]
        label :=  [L"g_1" L"g_{-1}"]
        xlim := extrema(x_grid)
		ylim := (0, ylim_prior)
        title := "b)"
        xguide := L"\mu"
		legend := :topright
        x_grid, [g1_xs g2_xs]
    end
	
	if (arg_length == 2)
		g1_band = sme_band.g1
		g2_band = sme_band.g2
		g1_xs = pdf.(g1_band, x_grid)
		g2_xs = pdf.(g2_band, x_grid)
		@series begin
			seriestype := :path
			subplot := 4
			linecolor := [col_g1 col_gm1]
			label :=  [L"g_1" L"g_{-1}"]
            xlim := extrema(x_grid)
			ylim := (0, maximum([g1_xs;g2_xs])*ylim_relative_offset)
            title := "d)"
			legend := :topright
            xguide := L"\mu"
			x_grid, [g1_xs g2_xs]
		end
	end

	# hardcoding for now to only work with StandardNormal
    Z=sme.Z
    Z_continuous = StandardNormalSample(0.0)

    f1=marginalize(g1, Z_continuous)
    f2=marginalize(g2, Z_continuous)

    f1_xs=pdf.(f1, x_grid)
    f2_xs=pdf.(f2, x_grid)

	ylim_marginal = maximum([f1_xs;f2_xs])*ylim_relative_offset
    @series begin
        seriestype := :path
        subplot := 3
        xlim := extrema(x_grid)
		label :=  [L"f_{G_{1}}" L"f_{G_{-1}}"]#[L"\varphi  \star g_1" L"\varphi \star g_{-1} "]
        linecolor --> [col_g1 col_gm1]
		ylim --> (0, ylim_marginal)
        title := "c)"
		legend := :topright
        xguide := L"x"
        x_grid, [f1_xs f2_xs]
    end
	
	if (arg_length == 2)
		f1_band=marginalize(g1_band, Z_continuous)
		f2_band=marginalize(g2_band, Z_continuous)

		f1_xs=pdf.(f1_band, x_grid)
		f2_xs=pdf.(f2_band, x_grid)

		@series begin
			seriestype := :path
			subplot := 5
		    label :=  [L"f_{G_{1}}" L"f_{G_{-1}}"]#[L"\varphi  \star g_1" L"\varphi \star g_{-1} "]
			linecolor --> [col_g1 col_gm1]
			x_grid, [f1_xs f2_xs]
		end	
        
        @series begin
            subplot := 5
            xs = midpoints(sme_band.Z.mhist.grid)
            h = step(sme_band.Z.mhist.grid)
            ys_lower = sme_band.Z.f_min[2:(end-1)] ./ h
            ys_upper = sme_band.Z.f_max[2:(end-1)] ./ h
            # hack for now 
            fillrange := ys_upper
            color := col_band
            label := L"$\infty$-band"
            xs, ys_lower
        end	
        
        @series begin
            seriestype := :path
            subplot := 5
            label :=  nothing#[L"\varphi  \star g_1" L"\varphi \star g_{-1} "]
            linecolor --> [col_g1 col_gm1]
            if isnothing(ylim_panel_e)
                ylim := (0, ylim_marginal)
            else 
                ylim := ylim_panel_e
            end
            xlim := extrema(x_grid)
            title := "e)"
            legend := :topright
            xguide := L"x"
            x_grid, [f1_xs f2_xs]
        end	
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
