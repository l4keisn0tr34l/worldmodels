# 19 — Representation collapse, VICReg, EMA, stop-gradient, SIGReg

## Define the exact failure

Let two correlated observations x and y have latent outputs `z=f(x)` and `z'=g(y)`. Loss `E||z−z'||²` is minimized if both equal a fixed constant for **all** inputs. Low loss therefore does not imply that the representation distinguishes objects, motion or anything. Constant collapse is the extreme case; **dimensional collapse** occurs when only a few independent directions carry information.

## Why stop-gradient and EMA help, but don't prove anything

A stop-gradient `sg(y)` freezes the backward path to one side for that loss. An EMA target `θ_target ← m θ_target+(1−m) θ_online` changes slowly. This breaks symmetry, helps stabilize targets, and can empirically prevent collapse **in particular training recipes**. But a constant target encoder and predictor is still a fixed point for naïve objectives. Stability/anti-collapse arises from the overall method, not one magical operator.

## VICReg: three terms with different jobs

Given batch matrices `Z,Z'∈R^{B×D}`:

1. **Invariance:** `s(Z,Z')=(1/B)Σ_i||z_i−z_i'||²` aligns compatible pairs.
2. **Variance:** each feature dimension should have nontrivial spread across the batch. Define `std_j(Z)=sqrt(Var(Z_:j)+ε)`, then

\[v(Z)=\frac1D\sum_{j=1}^D\max(0,\gamma-\mathrm{std}_j(Z)).\]

3. **Covariance:** avoid redundant dimensions. Center `Z` per dimension; covariance `C=Z_c^T Z_c/(B−1)`. Penalize its off-diagonal entries:

\[c(Z)=\frac1D\sum_{i\ne j} C_{ij}^2.\]

Total:

\[L=\lambda s(Z,Z')+\mu[v(Z)+v(Z')]+\nu[c(Z)+c(Z')].\]

The variance term directly punishes collapsed dimensions; covariance encourages decorrelation. But **zero covariance is not full statistical independence**, and preserving spread isn't proof of semantic usefulness. Strong regularization can also destroy meaningful structure.

## Tiny example

If every batch embedding is `[2,2,2]`, invariance between matching pairs may be perfect; per-dimension std=0 and variance term strongly penalizes it. If embeddings are `[u,u,u]` with u varying by sample, variance is nonzero, but all dimensions duplicate the same signal. The covariance term penalizes that redundancy.

## Barlow Twins (related, not identical)

Barlow Twins aligns corresponding features by encouraging a standardized cross-correlation matrix between two views to approximate identity. VICReg acts with explicit per-branch variance + covariance terms. Both relate to preventing redundant representations while aligning views.

## SIGReg and LeJEPA

[LeJEPA paper](https://arxiv.org/abs/2511.08544) proposes **SIGReg** regularization targeting an approximately isotropic Gaussian embedding distribution, using random projected views of embedding distributions and distributional matching. Intuition: rather than constrain only mean/covariance, regularize more of distribution shape. This permits predictive embeddings trained with alternative/no teacher-student EMA heuristics in the studied setting. The theoretical minimax motivation requires assumptions about downstream tasks; do not assume every world state should literally have a Gaussian physical distribution.

## A debugging checklist

- Embedding standard deviation per dimension across examples.
- Singular values / covariance eigenspectrum (effective rank).
- Off-diagonal covariance and near-duplicate channels.
- Pairwise cosine similarities for unrelated inputs.
- Linear probes predicting known variables and held-out downstream task performance.
- Sensitivity to action and temporal offset.
- If training loss drops while all embeddings become identical, stop: you found collapse.

**Check yourself:** Explain *constant collapse* vs *dimensional collapse*, and why variance and covariance need separate penalties. Can random images with high-dimensional noise have excellent feature variance but poor semantic representation?
