# 01 — Probability for world modeling: from zero to useful

## Why point estimates are often insufficient

Consider two identical-looking closed boxes. One contains a heavy object; one is empty. A photo cannot resolve which box you have. If your model returns just 'weight = 5 kg', it invents certainty. A distribution such as `P(weight=10)=0.5, P(weight=0)=0.5` represents ambiguity.

A **random variable** is a quantity whose value is uncertain. For discrete `X`, probabilities `P(X=x)` sum to 1. For continuous `X`, a **density** `p(x)` integrates to 1; `p(x)` itself is not the probability of exactly `x`.

## Joint, conditional, marginal

- Joint: `P(X=x,Y=y)` describes both together.
- Marginal: `P(X=x)=Σ_y P(X=x,Y=y)` (integral for continuous Y).
- Conditional: `P(X=x|Y=y)=P(X=x,Y=y)/P(Y=y)` if denominator is positive.
- Chain rule: `p(x,y)=p(x)p(y|x)=p(y)p(x|y)`.
- Independence: `p(x,y)=p(x)p(y)`; conditional independence given `z` means `p(x,y|z)=p(x|z)p(y|z)`.

**Bayes' rule** is not a new learning algorithm; it's the chain rule rearranged:

\[p(s\mid o)=\frac{p(o\mid s)p(s)}{p(o)}.\]

Prior = beliefs before evidence; likelihood = how likely the observation is under a hypothesis; posterior = belief after evidence; evidence/normalizer = sum or integral of numerator.

### Numeric example: noisy alarm

`P(intrusion)=0.01`, `P(alarm|intrusion)=0.9`, `P(alarm|no intrusion)=0.1`. Then:

\[P(\text{intrusion}\mid\text{alarm})=\frac{0.9(0.01)}{0.9(0.01)+0.1(0.99)}\approx0.0833.\]

Even after a ringing alarm, intrusion is ~8.3% probable because the base rate is low. A forecasting system must respect this; a high anomaly score is not automatically a calibrated incident probability.

## Expectation, variance, covariance

\[\mathbb E[X]=\sum_x xP(x),\qquad \mathrm{Var}(X)=\mathbb E[(X-\mu)^2].\]

For a random vector, the covariance matrix `Σ_ij=E[(X_i−μ_i)(X_j−μ_j)]` captures linear co-variation. Uncertainty may be correlated across dimensions (position and velocity, two moving objects).

The Gaussian `N(μ,σ²)` is specified by its mean and variance. A multidimensional normal `N(μ,Σ)` uses covariance `Σ`. Diagonal Gaussians assume conditionally independent dimensions given inputs; that assumption may be convenient but false.

## Expected loss vs likelihood

A deterministic regression model trained with mean squared error predicts the conditional **mean** under standard squared loss. If the future is equally likely left or right, the mean may be an impossible 'middle' outcome. Under a Gaussian conditional with fixed variance, minimizing negative log-likelihood is (up to constants and scale) minimizing MSE. To represent multimodal futures, we may use categorical distributions, mixture density networks, ensembles, autoregressive models, or diffusion.

For discrete outcomes, negative log likelihood is `−log p_θ(y|x)` and reduces to cross entropy on a one-hot label. Likelihood measures fit of observed transitions but alone doesn't guarantee that long rollouts or plans are good.

## Entropy, KL and why direction matters

Entropy: `H(P)=−Σ_x P(x)log P(x)` measures average surprise under `P`.

\[D_{KL}(q\|p)=\mathbb E_q\left[\log\frac{q(z)}{p(z)}\right]\ge0.\]

KL = 0 only when the distributions agree (almost everywhere). It's asymmetric: `KL(q||p) ≠ KL(p||q)`. If `q` puts mass where `p` is zero, `KL(q||p)` can be infinite. In an RSSM loss, we usually want the **prior dynamics prediction** to agree with the **posterior using real data**; regularization makes imagination states resemble inference states.

**Check yourself:** Why does predicting the average of two equally likely velocities misrepresent the world, even when the model has low expected squared error?
