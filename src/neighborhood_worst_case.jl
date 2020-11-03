struct NeighborhoodWorstCase{N,G}
    neighborhood::N
    convexclass::G
    solver
end



Base.@kwdef struct LowerUpperConfidenceInterval{T,M}
    lower::Float64
    upper::Float64
    α::Float64 = 0.05
    target::T = nothing
    method::M = nothing
end

function StatsBase.fit(method::NeighborhoodWorstCase{<:Empirikos.EBayesNeighborhood}, Zs, target)
    fitted_nbhood = StatsBase.fit(method.neighborhood, Zs)
    method = @set method.neighborhood = fitted_nbhood
    StatsBase.fit(method, Zs, target)
end


function StatsBase.fit(method::NeighborhoodWorstCase{<:Empirikos.FittedEBayesNeighborhood},
    Zs,
    targets::Union{<:Empirikos.AbstractPosteriorTarget, AbstractVector{<:Empirikos.AbstractPosteriorTarget}})

    lfp = LinearFractionalModel(method.solver)
    g = Empirikos.prior_variable!(lfp, method.convexclass)

    Empirikos.neighborhood_constraint!(lfp, method.neighborhood, g)

    confints = Vector{LowerUpperConfidenceInterval}(undef, length(targets))

    for (index,target) in enumerate(targets)
        target_numerator = numerator(target)
        target_numerator_g = target_numerator(g)
        target_denominator = denominator(target)
        target_denominator_g = target_denominator(g)

        set_objective(lfp, JuMP.MOI.MIN_SENSE, target_numerator_g, target_denominator_g)
        optimize!(lfp)
        _min = objective_value(lfp)
        set_objective(lfp, JuMP.MOI.MAX_SENSE, target_numerator_g, target_denominator_g)
        optimize!(lfp)
        _max = objective_value(lfp)

        confints[index] = LowerUpperConfidenceInterval(target=target, lower=_min, upper=_max)
    end
    confints
end


function StatsBase.fit(method::NeighborhoodWorstCase{<:Empirikos.FittedEBayesNeighborhood},
    Zs,
    targets::Union{<:Empirikos.AbstractPosteriorTarget, AbstractVector{<:Empirikos.LinearEBayesTarget}})

    lp = Model(method.solver)
    g = Empirikos.prior_variable!(lp, method.convexclass)

    Empirikos.neighborhood_constraint!(lp, method.neighborhood, g)

    confints = Vector{LowerUpperConfidenceInterval}(undef, length(targets))

    for (index,target) in enumerate(targets)
        target_g = target(g)

        @objective(lp, Min, target_g)
        optimize!(lp)
        _min = objective_value(lp)
        @objective(lp, Max, target_g)
        optimize!(lp)
        _max = objective_value(lp)

        confints[index] = LowerUpperConfidenceInterval(target=target, lower=_min, upper=_max)
    end
    confints
end
