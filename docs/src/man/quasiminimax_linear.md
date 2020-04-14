# Quasi-minimax linear estimators

Here we seek to demonstrate how we can solve optimization problems of the following form: Given a partition of the real line $(-\infty, \infty)$ into bins $I_{k}$, find the constant $Q_0$ and function $Q(\cdot)$ which optimize the following optimization problem:

$$\min_{Q_0,Q} \left\{ \sup_{ G \in \mathcal{G}} \{ (Q_0 + E_{G}[Q(Z_i)] - L(G))^2\} + \lambda(\delta)\frac{1}{n}\int Q^2(z) \bar{f}(z) dz \right\}$$

We solve this over all functions $Q(\cdot)$ that are piecewise constant on $I_k$. This corresponds to finding the linear estimator:

$$\hat{L} = Q_0 + \frac{1}{n} \sum_{i=1}^n Q(Z_i)$$

that solves a worst-case bias-variance problem, where the exact trade-off is parametrized
by $\lambda(\delta)$.

Specification of the optimization problem consists of specifying the following:

* The convex class of priors (effect size distributions) $\mathcal{G}$.
* The linear functional$L(G)$ operating on $G \in \mathcal{G}$.
* A pilot estimator for the marginal density of the observations $\bar{f}(\cdot)$.
* A way of navigating the bias-variance tradeoff, i.e. of picking $\delta$.

```@docs
SteinMinimaxEstimator
```

## Bias-variance tradeoff 

The following options are available for navigating the bias-variance tradeoff:
```@docs
MCEB.FixedDelta
MCEB.RMSE
MCEB.HalfCIWidth
```


## Worst case bias

The worst-case bias of an arbitrary linear estimator can be computed with the following function:
```@docs
worst_case_bias
```