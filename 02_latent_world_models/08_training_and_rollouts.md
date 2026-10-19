# 08 — Training world models: objectives, implementation, and failures

## Dataset contract

Given trajectories with observations `o_t`, actions `a_t`, rewards `r_{t+1}`, and termination flags `done_{t+1}`, verify alignment: the action at `t` must be the action that precedes `o_{t+1}`. Misalignment can produce a model that appears to learn but cannot respond causally to actions.

A typical replay-buffer minibatch has `[B,T,...]` observations, `[B,T−1,A]` actions, `[B,T−1]` rewards, and masks for padding/termination. When starting mid-episode, decide how to initialize recurrent memory (burn-in, cached state, or zeros) and avoid passing hidden states across unrelated episodes.

## Model choices and loss terms

| Predicted signal | Example head | Standard loss | Subtlety |
|---|---|---|---|
| Next state | MLP / RNN / Transformer | MSE or NLL | States may be hidden or non-identifiable |
| Observation | CNN decoder / video model | Negative log likelihood, denoise loss | Pixel detail can dominate |
| Reward | scalar/discrete distribution | MSE, classification or NLL | Rare rewards and skewed scales |
| Continuation | Bernoulli head | Binary cross entropy | Do not bootstrap after true terminal states |
| Prior vs posterior | probability heads | KL(q||p) | Excess KL can erase useful info |
| Multi-step state | unrolled dynamics | rollout error / KL | Need proper stop-gradient and masking |

**Loss weighting matters.** The system can reconstruct backgrounds perfectly while failing to track the tiny object determining reward. A high-quality world model must retain decision-relevant information. Many image-based agents include reward/value gradients or prediction heads to encourage this.

## Teacher-forced versus imagined rollouts

During supervised training, let input state be inferred from the real observation each step. During planning, after `t=0` each successor input is from the model's predicted state, not an actual observation. That is the classic **exposure bias / compounding error** distinction.

Track these separately:

1. **Teacher-forced one-step MSE / NLL.** Useful optimization diagnostic, not sufficient.
2. **Open-loop k-step error.** Seed from real observations, then no correction for k steps.
3. **Closed-loop task success.** Replan at each real observation; does the model improve control?

Also compare baselines: constant velocity, last-frame copy, random actions, and known simple physics whenever appropriate. If the neural model can't beat them, architecture sophistication isn't the fix.

## Practical pseudocode for RSSM training

```python
# Conceptual pseudocode, not drop-in library-specific code
h, z = model.initial(batch_size)
loss = 0
for t in range(sequence_length):
    # actions[t-1] causes transition to current o[t]
    h = model.gru(h, z, actions[t-1]) if t > 0 else h
    prior = model.prior(h)
    e = model.encoder(obs[t])
    posterior = model.posterior(h, e)
    z = posterior.rsample()  # Gaussians; discrete needs another estimator
    features = concat(h, z)
    obs_nll = -model.obs_dist(features).log_prob(obs[t])
    reward_nll = -model.reward_dist(features).log_prob(reward_at_t[t])
    kl = kl_divergence(posterior, prior)
    loss += (obs_nll + reward_nll + beta * kl).mean()
loss.backward()
optimizer.step()
```

## Four pitfalls with gradients

- **Posterior collapse:** decoder ignores the stochastic state. Test latent ablations / mutual dependence, not just KL magnitude.
- **Naive target prediction collapse:** simultaneously training predictor and encoder with `||f(o_t)−f(o_(t+1))||²` can produce constant features; see JEPA chapters.
- **Excessive backprop through long rollouts:** can explode/vanish or exhaust memory. Truncated BPTT, gradient clipping, and model unroll choices help.
- **Model exploitation:** the controller searches huge action space and finds optimistic hallucinations. Constrain actions, use uncertainty estimates, shorten horizon, validate with real interactions.

## Evaluation design

Split **by trajectories / scenarios**, not adjacent frames randomly. Adjacent frames share nearly identical content; random frame splits overestimate generalization. Report one-step and multi-step losses by horizon, uncertainty calibration where applicable, OOD scenarios, confidence intervals across seeds, closed-loop return, control latency, real interaction budget. Beware selective reporting of a pretty reconstructed rollout.

**Check yourself:** Which train/test split is more convincing—random frames of the same run or complete unseen episodes—and why?
