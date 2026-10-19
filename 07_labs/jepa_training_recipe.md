# Action-conditioned JEPA: minimal training protocol

Start with a toy image/video generator containing a ball, controllable acceleration and distractor texture. A good embedding should encode control-relevant velocity rather than texture. Create disjoint held-out videos and background styles.

## Student/context + teacher/target

```python
# Schematic PyTorch, not a complete runnable script.
z_t = online_encoder(obs_t)                 # [B,D]
with torch.no_grad():
    z_tp1_target = target_encoder(obs_tp1)   # [B,D]
a_emb = action_mlp(action_t)                 # [B,K]
z_pred = predictor(torch.cat([z_t, a_emb], -1))
loss_predict = (z_pred - z_tp1_target).square().mean()
loss = loss_predict + regularizer(z_t, z_tp1_target)
optimizer.zero_grad(); loss.backward(); optimizer.step()
with torch.no_grad():
    for target_p, online_p in zip(target_encoder.parameters(), online_encoder.parameters()):
        target_p.mul_(m).add_(online_p, alpha=1-m)
```

**Important:** This sketch does not instantiate `regularizer`; you must implement and test anti-collapse explicitly (e.g., VICReg-like variance/covariance) or validate a specific architecture's asymmetry/masking recipe. Apply teacher EMA **after** online optimization. Maintain identical architecture and matching parameter order for the EMA. A simple frame-level predictor is not the same as full I-JEPA's spatial masking design or V-JEPA's tubelet masking.

## Prevent accidental shortcuts

Do not feed future frames into context tokens. Keep temporal order. Test action shuffle: a world model should generally worsen when actions are mismatched with future transitions. Check *effective rank* of embeddings and downstream probes (position, velocity, collision) on held-out scenes; do not trust low latent training loss.

## Planning check

Encode a target goal observation, roll predicted latents for candidate action sequences, score terminal embedding distance, choose actions, and report **real environment target reaching**. If latent distance decreases but physical goal distance does not, the representation/loss is not sufficient for the controller.
