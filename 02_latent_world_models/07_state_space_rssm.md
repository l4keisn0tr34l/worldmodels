# 07 — RSSMs, dissected from the GRU you already know

## Why not just use a GRU?

A GRU over observation features, `h_t=GRU(h_{t-1},e(o_t))`, can summarize history, but we want two distinct abilities:

1. Maintain temporally persistent information, like velocity and object identity.
2. Represent uncertainty and uncertainty-resolving evidence, like whether an obscured ball is still moving left.

A **Recurrent State-Space Model (RSSM)** combines deterministic recurrent memory `h_t` with stochastic state `z_t`. A standard Dreamer/PlaNet-style variant can be written (one consistent indexing convention):

\[h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1})\]
\[p_\theta(z_t\mid h_t)\quad\text{(prior / dynamics prediction)}\]
\[q_\phi(z_t\mid h_t,e_\phi(o_t))\quad\text{(posterior / observation update)}.\]

The **representation of state** fed to decoder, reward and actor is often `(h_t,z_t)` (or a concatenated vector). Neither component is an ordinary physical state label; they are jointly learned predictive features.

### Link to known RNN operations

- The `GRU` weights are shared across time. Its input is **previous** stochastic latent and action, often concatenated.
- The **prior head** transforms `h_t` into distribution parameters (Gaussian mean/std or categorical logits).
- The **posterior head** transforms `(h_t, encoder(o_t))` into distribution parameters incorporating evidence.
- The **image decoder** predicts current observation from `(h_t,z_t)`; **reward head** estimates reward; **continuation head** often estimates probability the episode survives.

## A time-step walkthrough

At the end of time `t−1` you have `h_{t−1}`, `z_{t−1}` and chosen `a_{t−1}`:

```text
h_(t−1), z_(t−1), a_(t−1)
             │
             ▼
       GRU transition ───────> h_t
                                │
                 ┌──────────────┴───────────────┐
                 ▼                              ▼
          prior p(z_t|h_t)        observation o_t -> encoder
                 │                              │
                 │                              ▼
                 │                    posterior q(z_t|h_t,o_t)
                 │                              │
                 │                       sample z_t
                 │                              │
                 │                    decoder/reward/continue
                 │
      in imagination use prior z_t
```

**Training / observation time:** observe `o_t` → sample posterior → reconstruct observation and predict reward → compute KL against prior. **Imagination / future time:** observation does not exist yet → sample prior → roll GRU → compute imagined reward/value. Using `q(z_t|o_t)` in imagined futures would require peeking at future observations.

## Why two parts?

A recurrent vector `h_t` makes deterministic memory convenient: persistent information about motion and history can flow without being resampled. Stochastic `z_t` permits a range of plausible hidden states given the same history and observations; it makes sampling possible for imagined futures. This factorization is an architectural bias, not a theorem that `h_t` necessarily encodes memory and `z_t` necessarily encodes uncertainty cleanly.

`h_t` shape might be `[batch, 256]`; `z_t` shape `[batch,32]` if Gaussian, or `[batch,32,32]` if 32 categorical variables with 32 classes each. These are illustrative, **not universal hyperparameters**. A latent categorical sample can be represented as one-hot tokens; straight-through gradients enable training through the discrete sample approximately.

## RSSM generative factorization

For actions and initial state given, a simplified generative model is

\[p_\theta(o_{1:T},z_{1:T}\mid a_{0:T-1})=\prod_{t=1}^T p_\theta(z_t\mid h_t)\,p_\theta(o_t\mid h_t,z_t),\]

with `h_t=f(h_{t−1},z_{t−1},a_{t−1})`. Initial-state terms and reward/termination factors are omitted for clarity. Approximate inference uses `q(z_t|h_t,o_t)` along the observed sequence.

## A training objective

\[\mathcal L_{\rm model}=\sum_t \left[\underbrace{-\log p_\theta(o_t\mid h_t,z_t)}_{\rm observation}+\underbrace{-\log p_\theta(r_t\mid h_t,z_t)}_{\rm reward}+\beta\underbrace{D_{KL}(q_\phi(z_t\mid h_t,o_t)\|p_\theta(z_t\mid h_t))}_{\rm dynamics\ consistency}\right].\]

Some variants use an additional termination/continuation loss, KL balancing with different gradient routing, free bits, alternative reconstruction loss, and multi-step overshooting. Don't assume all Dreamer variants use the same objective.

### KL balancing intuition

When the posterior and prior disagree, there are two knobs: make posterior easier for the prior to match, or improve prior prediction. Balancing routes different proportions of KL gradients to different networks, e.g., `KL(sg(q)||p)` mostly trains the prior, `KL(q||sg(p))` mostly trains the posterior. These are different *gradient objectives* though their forward KL values coincide (because stop-gradient changes derivatives only).

## Posterior and prior are not interchangeable

Observation sequence used to build accurate past state estimates: posterior. Hypothetical trajectory under candidate future actions: prior. In Dreamer, actor/critic are trained from imagined prior rollouts seeded from real posterior states. In PlaNet, the CEM planner scores candidate prior rollouts.

**Check yourself:** Explain the role of `a_(t-1)` in the GRU update, `o_t` in the posterior update, and why the prior must support a no-observation rollout. Then sketch tensor shapes for a 16-example batch.
