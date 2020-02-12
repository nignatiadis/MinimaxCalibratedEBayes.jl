# return +-

function bias_adjusted_gaussian_ci(se, maxbias=0.0, α=0.9)
    rel_bias = maxbias/se
    zz = fzero( z-> cdf(Normal(), rel_bias-z) + cdf(Normal(), -rel_bias-z) +  α -1,
        0, rel_bias - quantile(Normal(),(1- α)/2.1))
    zz*se
end
