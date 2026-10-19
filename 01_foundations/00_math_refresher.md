# 00 — Math refresher: exactly what the later equations need

This is intentionally not a whole linear algebra course. Read whenever a symbol or derivation blocks your progress.

## Scalars, vectors, matrices, tensors

A **scalar** is one number. A **vector** `z∈R^D` is D numbers. A **matrix** `W∈R^(M×D)` maps a D-vector to an M-vector: `y=Wz+b` with `y∈R^M`. A **tensor** generalizes an array to arbitrary axes. `obs∈R^[B,T,C,H,W]` is a 5-axis tensor: batch, time, channel, height, width. Axes are not inherently semantic; you assign meaning and must preserve it.

A dot product `x·y=Σ_i x_i y_i`; Euclidean norm `||x||₂=sqrt(Σ_i x_i²)`; squared error `||x−y||₂²` is the sum of squared component-wise differences. Cosine similarity `x·y/(||x||||y||)` compares directions, not magnitude; beware zero norms.

## Transpose and covariance

`X∈R^[B,D]` holds B samples, D features. Center columns `Xc=X−mean(X,axis=0)`. Sample covariance is `C=XcᵀXc/(B−1)` in `R^[D,D]`. Diagonal entries measure each feature's spread; off-diagonals measure co-variation. Singular values indicate how many effective independent directions features span. All-constant embeddings have near-zero singular values.

## Calculus and backprop

A derivative `df/dx` quantifies local response; the **gradient** `∇_x L` collects partial derivatives for a scalar loss. A **Jacobian** `J_ij=∂f_i/∂x_j` handles vector-to-vector functions. For a composition `y=f(g(x))`, gradients multiply using chain rule. Through time in an RNN:

\[\frac{\partial h_T}{\partial h_t}=\prod_{k=t+1}^T\frac{\partial h_k}{\partial h_{k-1}}.\]

If matrix products repeatedly contract directions, gradients vanish; if they repeatedly expand, gradients explode. The RSSM's recurrent memory and optimization are still subject to these issues. Gradient clipping bounds global gradient norm; it does not fix all unstable dynamics or model bias.

## Expected value and optimization

To optimize expected return `E_{a∼π_θ}[R(a)]`, you need to know how gradients can pass through sampling. If `a=μ_θ+σ_θ ε`, differentiable reparameterization permits pathwise gradients. If actions are discrete categories, naive differentiation through a sample index does not work; policy-gradient or straight-through estimators provide alternatives with different bias/variance trade-offs.

A learned dynamics model can be differentiated through when transitions/actions are differentiable, or used in derivative-free methods like CEM. 'Model-based' does **not** mean end-to-end differentiable control.

## Logs, likelihoods, and cross entropy

Log turns products into sums: `log Π_t p_t = Σ_t log p_t`. Because long sequences have tiny joint probabilities, training often maximizes **log likelihood** or minimizes negative log likelihood. For a categorical model with true class y and predicted probabilities p, loss is `−log p_y`. Cross entropy for soft targets q is `−Σ_y q_y log p_y`. It is different from CEM, the cross-entropy **method** planner.

## Stable numeric practices

- Softmax should subtract max logits for stability.
- Standard deviations are positive; predict `log σ` and exponentiate with reasonable clipping.
- `KL(q||p)` often sums over latent dimensions but averages over batch/time; write reduction explicitly.
- Terminal masks prevent continuing from ended episodes.
- Standardize input state features using training-set statistics only; use exactly the same transform at test time.

**Check yourself:** If `W` is 128×32 and `z` is 32-dimensional, what shape is `Wz`? Why is `Σ_t log p_t` more stable than directly multiplying `p_t`?
