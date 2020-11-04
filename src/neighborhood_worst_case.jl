struct NeighborhoodWorstCase{N,G}
    neighborhood::N
    convexclass::G
    solver
end

function Empirikos.nominal_alpha(nbhood::NeighborhoodWorstCase)
    Empirikos.nominal_alpha(nbhood.neighborhood)
end

function Empirikos.set_nominal_alpha(nbhood::NeighborhoodWorstCase; kwargs...)
    @set nbhood.neighborhood =  Empirikos.set_nominal_alpha(nbhood.neighborhood; kwargs...)
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

#
# Confidence Interval Tools
#
# LowerUpperConfidenceInterval
Base.@kwdef struct LowerUpperConfidenceInterval{T,M}
    lower::Float64
    upper::Float64
    α::Float64 = 0.05
    estimate::Float64 = (lower + upper)/2
    target::T = nothing
    method::M = nothing
end

function Base.show(io::IO, ci::LowerUpperConfidenceInterval)
    print(io, "lower = ", round(ci.lower,sigdigits=4))
    print(io, ", upper = ", round(ci.upper,sigdigits=4))
    print(io, ", α = ", ci.α)
    print(io, "  (", ci.target,")")
end

@recipe function f(bands=AbstractVector{<:LowerUpperConfidenceInterval})
	x = [Float64(location(band.target)) for band in bands]
    lower = getproperty.(bands, :lower)
    upper = getproperty.(bands, :upper)
    estimate = getproperty.(bands, :estimate)

	background_color_legend --> :transparent
	foreground_color_legend --> :transparent
    grid --> nothing

	cis_ribbon  = estimate .- lower, upper .- estimate
	fillalpha --> 0.36
	seriescolor --> "#018AC4"
	ribbon --> cis_ribbon
    linealpha --> 0
    framestyle --> :box
    legend --> :topleft
	label --> "95\\% CI"
	x, estimate
end

function StatsBase.fit(method::NeighborhoodWorstCase{<:Empirikos.EBayesNeighborhood}, target, Zs)
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
