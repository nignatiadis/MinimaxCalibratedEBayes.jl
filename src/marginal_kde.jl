function DiscretizedAffineEstimator(mhist::MCEBHistogram, kernel::ContinuousUnivariateDistribution)
    DiscretizedAffineEstimator(mhist, x->pdf(kernel, x))
end
