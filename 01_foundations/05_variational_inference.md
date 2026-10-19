# 05 — VAEs, latent variables, the ELBO, and KL: fully unpacked

You need this chapter to read PlaNet, RSSM, Dreamer, and probabilistic JEPAs. Start with a model with latent causes `z` and observed data `x`:

\[p_\theta(x,z)=p(z)p_\theta(x\mid z).\]

`p(z)` is a prior (e.g., standard Gaussian). `p_θ(x|z)` is the decoder/observation model. To generate: sample z, then sample x. To infer the cause of an observed x, we want posterior `p_θ(z|x)`, often computationally intractable due to the integral `p_θ(x)=∫p(x,z)dz`.

## Approximate inference

Introduce encoder `q_φ(z|x)` to approximate the true posterior. It outputs a distribution, not simply a vector. For a diagonal Gaussian, it predicts `μ(x)` and `log σ²(x)`. Draw a sample via reparameterization:

\[z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\quad \epsilon\sim\mathcal N(0,I).\]

This shifts sampling noise into ε, letting gradients flow through μ and σ. The decoder receives sampled z and predicts a distribution over x.

## Derive the evidence lower bound (ELBO)

Start with `log p(x)`, insert `q(z|x)/q(z|x)` inside the marginal, and apply Jensen's inequality (`log E[U] ≥ E[log U]`):

\[\begin{aligned}
\log p_\theta(x)&=\log\mathbb E_{q_\phi(z\mid x)}\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]\\
&\ge\mathbb E_q[\log p_\theta(x,z)-\log q_\phi(z\mid x)]\\
&=\mathbb E_q[\log p_\theta(x\mid z)]-D_{KL}(q_\phi(z\mid x)\|p(z)).
\end{aligned}\]

Hence **negative ELBO**, minimized during training:

\[\mathcal L=-\mathbb E_q\log p_\theta(x\mid z)+\beta D_{KL}(q_\phi(z\mid x)\|p(z)).\]

For `β=1` this is the ordinary VAE ELBO. `β≠1` is a modified objective, not strictly the same lower bound to `log p(x)`.

Why two terms?

- **Reconstruction / likelihood:** force z to explain x.
- **KL to prior:** make encoded samples resemble what the generator can sample at test time, while limiting posterior information flow.

If KL is overweighted, `q(z|x)` can collapse toward the prior and the decoder ignores z (**posterior collapse**). If KL is too weak, each input can get an idiosyncratic code that cannot be sampled from the prior. This is distinct from JEPA's 'all embeddings constant' representational collapse (though concepts relate).

## Exact KL for diagonal Gaussians

If `q=N(μ,diag(σ²))`, `p=N(0,I)` and z is D-dimensional:

\[D_{KL}(q\|p)=\frac12\sum_{j=1}^D(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2).\]

Example: `μ=0,σ=1` makes KL=0. `μ=2,σ=1` makes KL=2 in one dimension. This is the price of deviating from the prior.

## Sequential ELBO intuition

In dynamics models, prior is no longer static `N(0,I)`: it is conditional **on history and action**, `p_θ(z_t|h_t)`; posterior additionally sees current observation `q_φ(z_t|h_t,o_t)`. The KL term measures mismatch between the prior's prediction and the encoder's evidence-adjusted belief.

\[\mathcal L_t= -\mathbb E_q\log p_\theta(o_t\mid h_t,z_t) + \beta D_{KL}(q_\phi(z_t\mid h_t,o_t)\|p_\theta(z_t\mid h_t)).\]

Rewards and termination predictions are often added. The **posterior** is used during training from real trajectories; the **prior** must work for imagination without the unseen future observation.

**Check yourself:** Derive the KL term in sequential training and explain why the agent cannot sample the next posterior while imagining a future before observing it.
