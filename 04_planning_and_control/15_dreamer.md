# 15 — Dreamer: learning a policy from imagined trajectories

## The leap from PlaNet

PlaNet searches over candidate actions **each time** it needs a decision. Dreamer instead trains an **actor** and **critic** using rollouts of the learned latent dynamics model. It then executes the actor, commonly without expensive online search.

[Dream to Control](https://arxiv.org/abs/1912.01603) established the central recipe. Later DreamerV2 uses discrete stochastic latents and other improvements; DreamerV3 further updates stabilization and training choices. Do not assume that a sketch of DreamerV1 exactly describes V2/V3.

## Three coupled learners

1. **World model:** RSSM prior/posterior + observation/reward/continuation predictions from real experience.
2. **Actor** `π_ψ(a|h,z)`: choose actions in imagined latent trajectories.
3. **Critic** `V_η(h,z)`: estimate expected future return in latent state space.

## A full training iteration

```text
collect real episodes under current policy
              ↓
add (observation, action, reward, done) to replay
              ↓
train RSSM: encoder, GRU, prior/posterior, decoder, reward, continue
              ↓
sample posterior seed states from replay sequences
              ↓
imagine H-step trajectories via prior + actor (NO future images)
              ↓
compute imagined reward, continuation, and lambda returns
              ↓
update critic to fit these returns; update actor to improve them
              ↓
repeat environment interaction
```

### Imagined trajectory

Starting `(h_t,z_t)` from a real posterior, draw `a_t∼π(a|h_t,z_t)`; update GRU and draw `z_(t+1)∼p(z|h_(t+1))`; get `r̂` and continuation `ĉ`; repeat. The imagined rewards and learned values determine actor/critic training targets.

## Lambda returns: not merely adding rewards

Define one-step bootstrap target

\[G_t^{(1)}=r_{t+1}+\gamma c_{t+1}V(s_{t+1}).\]

Longer n-step return:

\[G_t^{(n)}=\sum_{k=1}^{n}\gamma^{k-1}\Big(\prod_{j=1}^{k-1}c_{t+j}\Big)r_{t+k}+\gamma^n\Big(\prod_{j=1}^{n}c_{t+j}\Big)V(s_{t+n}).\]

The λ-return weights multi-step returns: conceptually `(1−λ)Σ_{n=1}^{H−1}λ^{n−1}G^(n) + λ^{H−1}G^(H)` for finite H. Here `c` (0/1 or probability of continuation) prevents bootstrapping past episode ends. λ near 0 emphasizes one-step bootstrapping (lower variance, more critic bias); λ near 1 uses long rewards (less bootstrap bias, more rollout uncertainty). Specific Dreamer versions can implement the recursion and losses differently.

## How does the actor get gradients?

In continuous differentiable systems, gradient can travel from imagined return through predicted states and policy choices. For stochastic/discrete variables, straight-through estimators, score-function gradients or mixed estimators are used depending on model/version. A policy gradient need not be identical to naïvely backpropagating through sampled categorical actions.

Actor objectives also need entropy regularization/exploration control. Critic targets are often stop-gradient so the critic cannot cheat by changing target values. Target critics or slow-moving value networks reduce bootstrap instability.

## DreamerV2 and discrete latent variables

DreamerV2 uses categorical stochastic latent features (multiple categorical variables); sample with straight-through-style estimators while retaining distributions for KL-based dynamics learning. Why not continuous only? Categorical states can represent distinct outcomes/discrete modes and have different optimization properties. They do not magically eliminate error or represent semantic object identities.

## A complete mental model

- **Observe reality** to learn the world model.
- **Dream possible experience** cheaply inside the learned model.
- **Learn a policy/value function in those dreams**.
- **Act in reality and correct the model**.

That loop is powerful but hazardous: a critic and actor optimized in a flawed dream may exploit systematic model errors. Always evaluate the actor in the actual held-out environment.

**Check yourself:** During imagined actor training, does the imagined future use posterior `q(z_(t+1)|o_(t+1))` or prior `p(z_(t+1)|h_(t+1))`? Explain without appealing to memorized terminology.
