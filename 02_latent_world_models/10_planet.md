# 10 — PlaNet: learning latent dynamics for planning from pixels

[PlaNet](https://arxiv.org/abs/1811.04551) is a key bridge from visual state estimation to online control. It learns a latent model from images + interactions, then uses **model predictive control (MPC)** with a sampling-based planner to choose actions.

## Architecture

Use an image encoder, RSSM deterministic memory and stochastic state, observation decoder, and reward predictor. Training includes sequence-level variational objectives. At decision time, plan in latent space, not pixels.

Simplified sequence:

```text
image -> posterior z_t plus h_t
                 |
   candidates a_t:t+H-1
                 |
      RSSM prior rollout
                 |
    predicted reward path
                 |
    rank action sequences
                 |
 execute first action; observe again
```

## Why MPC?

Imagine you're balancing a pole. Predict 15 steps under candidate pushes, choose best sequence, execute **one** push, receive a fresh frame, and replan. You never have to trust the 15-step prediction all the way to completion. This sharply reduces the consequences of some long-rollout drift.

## Cross-entropy method in brief

Sample a batch of continuous action sequences from time-indexed Gaussian distributions. Roll all of them through RSSM, compute predicted total reward, retain a high-scoring subset (`elites`), refit the means and stds to the elite sequences, repeat, and execute first action from best/final sequence. Full derivation and runnable implementation are in `../04_planning_and_control/14_mpc_cem_pets.md` and `../07_labs/toy_cem_planner.py`.

## Latent overshooting

If the one-step prior agrees with the posterior but long rollouts fail, impose a training signal comparing multi-step prior predictions to posterior states further ahead. Conceptually:

\[\mathcal L_{overshoot} \propto \sum_{t,d>1} D_{KL}\!\left(q(z_{t+d}\mid\text{observed sequence})\ \big\|\ p_\theta^{(d)}(z_{t+d}\mid z_t,a_{t:t+d-1})\right).\]

This is schematic: the actual PlaNet latent-overshooting objective handles multi-step distributions and gradient structure carefully. The central reason for the term is that training should reflect **what the model will do when rolled forward**, not only corrected one-step predictions.

## PlaNet vs Dreamer

| Design | PlaNet | Dreamer |
|---|---|---|
| Learns latent dynamics | Yes | Yes |
| Selects action by | Online trajectory search (CEM) | Actor learned from imagination |
| Uses real observation again after acting | Yes | Yes |
| Inference planning compute | Larger search-time cost | Often cheap policy pass |
| Main risk | Planning exploits model error | Actor/critic exploit model error |

This is a structural comparison; actual versions vary.

**Check yourself:** If PlaNet can replan every step, why does its model still need accurate multi-step dynamics?
