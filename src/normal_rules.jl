
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


function marginalize(prior::Normal, Z::DiscretizedStandardNormalSample)
    marginal_normal = marginalize(prior, StandardNormalSample(response(Z)))

    grid = Z.mhist.grid
    hist = deepcopy(Z.mhist.hist)

    hist.weights[1] = cdf(marginal_normal,  first(grid))
    hist.weights[end] = ccdf(marginal_normal, last(grid))

    for i=2:length(grid)
        hist.weights[i] = cdf(marginal_normal, grid[i]) -  cdf(marginal_normal, grid[i-1])
        #TODO: add sanity check that this is approx. h*density
    end

    MCEBHistogram(grid, hist, infty_bound = 0.0)
end

function (target::MarginalDensityTarget{<:StandardNormalSample})(prior::Normal)
    x = response(location(target)) #ok this notation is not nice...
    pdf(marginalize(prior, StandardNormalSample(0.0)), x)
end
