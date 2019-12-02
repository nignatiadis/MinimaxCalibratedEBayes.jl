function marginalize(prior::Normal, Z::EBayes.AbstractNormalSample)
    prior_var = var(prior)
    prior_μ = mean(prior)
    likelihood_var = var(Z)
    marginal_σ = sqrt(likelihood_var + prior_var)
    Normal(prior_μ, marginal_σ)
end


function marginalize(prior::Normal, Z::DiscretizedStandardNormalSample)
    marginal_normal = marginalize(prior, StandardNormalSample(response(Z)))

    mhist = deepcopy(Z.mhist)
    hist = mhist.hist

    hist.weights[1] = cdf(marginal_normal,  first(mhist))
    hist.weights[end] = ccdf(marginal_normal, last(mhist))

    for i=2:length(mhist.grid)
        hist.weights[i] = cdf(marginal_normal, mhist.grid[i]) -  cdf(marginal_normal, mhist.grid[i-1])
    end
    mhist
end