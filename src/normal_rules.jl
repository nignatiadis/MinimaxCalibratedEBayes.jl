
# marginalize:
## Normal (Distribution), NormalSample -> Normal distribution
## Normal (Distribution), DiscretizedNormalSample -> MCEBHistogram ...(?)
## PriorConvexClass, DiscretizedNormalSample -> ...



function marginalize(prior::Normal, Z::EBayes.AbstractNormalSample)
    prior_var = var(prior)
    prior_μ = mean(prior)
    likelihood_var = var(Z)
    marginal_σ = sqrt(likelihood_var + prior_var)
    Normal(prior_μ, marginal_σ)
end

function marginalize(prior::MixtureModel, Z::EBayes.AbstractNormalSample)
    prior_probs = probs(prior)
    marginal_components = marginalize.(components(prior), Ref(Z))
    MixtureModel(marginal_components, prior_probs)
end

const NormalOrNormalMixture = Union{Normal,
    MixtureModel{Univariate, Continuous, Normal}}

function marginalize(prior::NormalOrNormalMixture, Z::DiscretizedStandardNormalSamples)
    marginal_normal = marginalize(prior, StandardNormalSample(NaN))

    grid = Z.mhist.grid
    hist = deepcopy(Z.mhist.hist)
    hist = @set hist.weights = zeros(Float64, length(hist.weights))

    hist.weights[1] = cdf(marginal_normal,  first(grid))
    hist.weights[end] = ccdf(marginal_normal, last(grid))

    for i=2:length(grid)
        hist.weights[i] = cdf(marginal_normal, grid[i]) -  cdf(marginal_normal, grid[i-1])
        #TODO: add sanity check that this is approx. h*density
    end

    MCEBHistogram(grid, hist)
end



# Normal: MarginalDensityTarget

function (target::MarginalDensityTarget{<:StandardNormalSample})(prior::Normal)
    x = response(location(target)) #ok this notation is not nice...
    pdf(marginalize(prior, StandardNormalSample(0.0)), x)
end



## Posterior of Normal DBN

function _normal_normal_posterior(prior::Normal, Z::EBayes.AbstractNormalSample)
    x = response(Z)
    sigma_squared = var(Z)
    prior_mu = mean(prior)
    prior_A = var(prior)

    post_mean = x*(prior_A)/(prior_A + sigma_squared) + sigma_squared/(prior_A + sigma_squared)*prior_mu
    post_var = prior_A * sigma_squared / (prior_A + sigma_squared)
    Normal(post_mean, sqrt(post_var))
end


# Normal: PosteriorMeanTarget

function (target::MinimaxCalibratedEBayes.PosteriorMeanNumerator{<:StandardNormalSample})(prior::Normal)
     Z = location(target)
     _post = _normal_normal_posterior(prior, Z)
     ratio = mean(_post)
     denom = MarginalDensityTarget(Z)(prior)
     ratio*denom
end

# Normal: LocalFalseSignRate


function (target::MinimaxCalibratedEBayes.LFSRNumerator{<:StandardNormalSample})(prior::Normal)
     Z = location(target)
     _post = _normal_normal_posterior(prior, Z)
     ratio = ccdf(_post, 0)
     denom = MarginalDensityTarget(Z)(prior)
     ratio*denom
end


#function (target::PosteriorMeanNumerator{<:StandardNormalSample})(prior::Normal)
#    x = response(location(target)) #ok this notation is not nice...
#    pdf(marginalize(prior, StandardNormalSample(0.0)), x)
#end

#function cf(target::MarginalDensityTarget{<:StandardNormalSample}, t)
#    error_dbn = Normal(response(location(target))) #TODO...
#    cf(error_dbn, t)
#end

#function riesz_representer(target::MarginalDensityTarget{<:StandardNormalSample}, t)
#    error_dbn = Normal(response(location(target))) #TODO...
#    pdf(error_dbn, t)
#end




# Normal: LocalFalseSignRate




#σ = std(dist)
#μ = mean(dist)
#(isfinite(σ) && isfinite(μ)) || throw(ArgumentError("Parameters σ, μ must be finite."))
#gh = gausshermite(n)
#nodes = gh[1].*(sqrt(2)*(σ)) .+ μ
#weights = gh[2]./sqrt(pi)

# TODO:

 # Type MarginalDistribution()...
