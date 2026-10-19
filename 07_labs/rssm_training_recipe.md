# RSSM in PyTorch: a build recipe, not a fake turnkey implementation

This specifies operations, shapes, losses, and tests without pretending to have implemented an end-to-end Dreamer. Use it to build a training script on simple generated sequences.

## Tensors

Assume `B=16,T=32`, image `o∈R^[B,T,3,64,64]`, continuous actions `a∈R^[B,T−1,2]`, deterministic `h∈R^[B,256]`, stochastic Gaussian `z∈R^[B,32]`, observation encoder `e(o)∈R^[B,128]`.

```python
# Pseudocode; instantiate encoder/prior/posterior/reward/decoder/GRU yourself.
h = torch.zeros(B, 256, device=device)
z = torch.zeros(B, 32, device=device)
loss = 0.0
for t in range(T):
    if t > 0:
        h = gru(torch.cat([z, actions[:, t-1]], -1), h)
    mu_p, logstd_p = prior_head(h).chunk(2, dim=-1)
    dist_p = torch.distributions.Normal(mu_p, logstd_p.clamp(-5, 2).exp())
    e = image_encoder(obs[:, t])
    mu_q, logstd_q = posterior_head(torch.cat([h, e], -1)).chunk(2, dim=-1)
    dist_q = torch.distributions.Normal(mu_q, logstd_q.clamp(-5, 2).exp())
    z = dist_q.rsample()
    feature = torch.cat([h, z], -1)
    # Illustrative losses; choose explicit observation / reward distributions.
    obs_loss = (decoder(feature) - obs[:, t]).square().flatten(1).mean(1)
    kl = torch.distributions.kl_divergence(dist_q, dist_p).sum(-1)
    loss = loss + (obs_loss + beta*kl).mean()
loss = loss/T
loss.backward()
torch.nn.utils.clip_grad_norm_(parameters, 100.0)
optimizer.step()
```

**This is not runnable as-is**: modules, targets, optimizer, normalization, reward loss, masks, sequence initialization, and actual dataset are intentionally left to implement. `obs_loss` uses unnormalized pixel MSE and may dominate KL unless scaled; choose proper likelihood/normalization. For initial frame the prior corresponds to a model-specific initial state, not learned from a prior action.

## Tests to write before training

1. Each prior and posterior returns `[B,32]` parameters and finite positive std.
2. When an observation is changed, posterior changes; with fixed h, prior does not.
3. Imagined rollout never reads future observations.
4. KL of identical distributions is approximately zero.
5. Step 0 / action indexing matches data (no shifted targets).
6. Episode termination resets recurrent state and is masked in loss.
7. One-step and open-loop metrics are computed with distinct procedures.

## Extensions

Replace Gaussian with straight-through categorical latents, add reward and continuation likelihood heads, KL balancing/free bits, overshooting, posterior seed sampling, λ-return critic and actor updates. Add one feature at a time and preserve a tested baseline.
