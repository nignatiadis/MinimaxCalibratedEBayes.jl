struct NeighborhoodWorstCase{N,G}
    neighborhood::N
    convexclass::G
    solver
end

function Empirikos.nominal_alpha(nbhood::NeighborhoodWorstCase)
    Empirikos.nominal_alpha(nbhood.neighborhood)
end


Base.@kwdef struct FittedNeighborhoodWorstCase{T, NW<:NeighborhoodWorstCase, M, P, V}
    method::NW
    target::T = nothing
    model::M
    gmodel::P
    g1::V = nothing
    g2::V = nothing
    lower::Float64 = -Inf
    upper::Float64 = +Inf
end



function StatsBase.fit(method::NeighborhoodWorstCase, target, Zs; kwargs...)
    Zs = Empirikos.summarize_by_default(Zs) ? summarize(Zs) : Zs
    method = Empirikos.set_defaults(method, Zs; kwargs...)

    fitted_nbhood = StatsBase.fit(method.neighborhood, Zs)
    method = @set method.neighborhood = fitted_nbhood

    StatsBase.fit(method, target)
end


function StatsBase.fit(method::NeighborhoodWorstCase{<:Empirikos.FittedEBayesNeighborhood},
    target::Empirikos.AbstractPosteriorTarget)

    lfp = LinearFractionalModel(method.solver)
    g = Empirikos.prior_variable!(lfp, method.convexclass)

    Empirikos.neighborhood_constraint!(lfp, method.neighborhood, g)

    fitted_worst_case = FittedNeighborhoodWorstCase(method=method,
        model=lfp,
        gmodel=g,
        target=target)

    StatsBase.fit(fitted_worst_case, target)
end

function StatsBase.fit(method::FittedNeighborhoodWorstCase{<:Empirikos.AbstractPosteriorTarget},
                       target::Empirikos.AbstractPosteriorTarget)

    g = method.gmodel
    lfp = method.model

    target_numerator = numerator(target)
    target_numerator_g = target_numerator(g)
    target_denominator = denominator(target)
    target_denominator_g = target_denominator(g)


    set_objective(lfp, JuMP.MOI.MIN_SENSE, target_numerator_g, target_denominator_g)
    optimize!(lfp)
    _min = objective_value(lfp)
    g1 = g(JuMP.value.(g.finite_param))
    set_objective(lfp, JuMP.MOI.MAX_SENSE, target_numerator_g, target_denominator_g)
    optimize!(lfp)
    _max = objective_value(lfp)
    g2 = g(JuMP.value.(g.finite_param))

    FittedNeighborhoodWorstCase(method=method.method,
        target=target,
        model=lfp,
        gmodel=g,
        g1=g1,
        g2=g2,
        lower=_min,
        upper=_max)
end


function StatsBase.fit(method::NeighborhoodWorstCase{<:Empirikos.FittedEBayesNeighborhood},
    target::Empirikos.LinearEBayesTarget)

    lp = Model(method.solver)
    g = Empirikos.prior_variable!(lp, method.convexclass)

    Empirikos.neighborhood_constraint!(lp, method.neighborhood, g)

    fitted_worst_case = FittedNeighborhoodWorstCase(method=method,
        model=lp,
        gmodel=g,
        target=target)

    StatsBase.fit(fitted_worst_case, target)
end

function StatsBase.fit(method::FittedNeighborhoodWorstCase{<:Empirikos.LinearEBayesTarget},
                       target::Empirikos.LinearEBayesTarget)

    g = method.gmodel
    lp = method.model

    target_g = target(g)

    @objective(lp, Min, target_g)
    optimize!(lp)
    _min = objective_value(lp)
    g1 = g(JuMP.value.(g.finite_param))

    @objective(lp, Max, target_g)
    optimize!(lp)
    _max = objective_value(lp)
    g2 = g(JuMP.value.(g.finite_param))

    FittedNeighborhoodWorstCase(method=method.method,
        target=target,
        model=lp,
        gmodel=g,
        g1=g1,
        g2=g2,
        lower=_min,
        upper=_max)
end





function StatsBase.confint(fitted_worst_case::FittedNeighborhoodWorstCase)
    @unpack target, method, lower, upper = fitted_worst_case
    α = nominal_alpha(method.neighborhood)
    LowerUpperConfidenceInterval(α=α, target=target, method=method, lower=lower,
                                 upper=upper)
end

function StatsBase.confint(nbhood::Union{NeighborhoodWorstCase,FittedNeighborhoodWorstCase}, target, args...)
    _fit = StatsBase.fit(nbhood, target, args...)
    StatsBase.confint(_fit)
end


function Base.broadcasted(::typeof(StatsBase.confint), nbhood::NeighborhoodWorstCase, targets, args...)
    _fit = StatsBase.fit(nbhood, targets[1], args...)
    confint_vec = fill(confint(_fit), length(targets))
    for (index, target) in enumerate(targets[2:end])
        confint_vec[index+1] = StatsBase.confint(StatsBase.fit(_fit, target))
    end
    confint_vec
end
