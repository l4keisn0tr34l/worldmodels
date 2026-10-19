# 18 — I-JEPA, V-JEPA, V-JEPA 2 and action conditioning

## I-JEPA: learn image representations without pixel reconstruction

[I-JEPA paper](https://arxiv.org/abs/2301.08243). Divide image into patches (as in a ViT), select a visible **context region** and one/more relatively large **target blocks**. The context encoder processes visible tokens. The target encoder, often an EMA of the context encoder, produces target representations (using full image input in common implementations, then gathering target positions). The predictor receives context features and mask/position tokens that say *where* each target is and attempts to predict the corresponding target embedding.

A conceptual example:

```text
image (book resting on table)
├─ context visible patches: table + half book
│  └─ context ViT --> context tokens
├─ target masked block: other half of book
│  └─ slow target ViT --> teacher embeddings
└─ predictor(context tokens, target position)
   └─ student-predicted target embeddings -> latent loss
```

Large target blocks discourage trivial interpolation from neighboring pixel values. Position tokens distinguish one hidden region from another. Target network is usually updated by exponential moving average:

\[\bar\theta\leftarrow m\bar\theta+(1-m)\theta.\]

Where `m≈1` changes the teacher slowly. That target receives no backprop through target outputs. The context/predictor are trained by latent regression. EMA target stability is useful but not magical; representation quality requires testing.

## V-JEPA: make masked prediction spatiotemporal

Replace image with a video `o_{1:T}`. Convert to **tubelets**, small patches extending across a few frames. Hide structured spatial/temporal blocks, encode context, predict target tubelet embeddings. Temporal position encodings tell the model when events occur; motion is necessary to predict many masked regions.

**Nuance:** masked video prediction can learn motion and temporal structure without being an explicitly causal **action-conditioned simulator**. Natural video includes moving actors but may not label the actions controlling them.

## V-JEPA 2: passive videos then embodied planning

[V-JEPA 2 paper](https://arxiv.org/abs/2506.09985). Broad high-level progression:

1. Self-supervised video representation learning on large video corpora using latent prediction.
2. Learn representations useful for understanding, temporal anticipation and other downstream tasks.
3. Post-train an **action-conditioned predictor** on robot interaction sequences so it estimates latent successors under candidate controls.
4. Given a goal observation/state embedding, score action-conditioned predicted trajectories and choose a sequence with latent MPC.

The important transition is not simply 'larger video Transformer': it is adding **action-grounding** that lets imagined futures depend on motor commands and match executable actions.

## A toy action-conditioned JEPA

Let `z_t=encoder(o_t)` and target `z_(t+1)^*=sg(target_encoder(o_(t+1)))`. Learn

\[\hat z_{t+1}=g_\phi(z_t,a_t),\quad L=\|\hat z_{t+1}-z_{t+1}^*\|^2+\lambda L_{anti-collapse}.\]

Planning to goal image `o_g`: encode `z_g=encoder(o_g)`, simulate each candidate action sequence through the learned `g`, select minimum `||ẑ_(t+H)−z_g||²` or shaped horizon cost. **Warning:** plain Euclidean latent distance is only valid as a control objective to the extent that the embedding geometry corresponds to task progress; validate it empirically.

## Passive video limits

Two actions can look like the same pixel transition; the same action can have varied physical consequences. Inferring an action model from action-free video needs extra assumptions or embodiment/action demonstrations. Passive learned correlations don't uniquely identify what happens under interventions. A video pretrained model can be a strong initialization while still requiring action data before dependable control.

## Proposed ablation sequence

A. No video pretraining, train action dynamics from scratch. B. Frozen pretrained image encoder + action predictor. C. Frozen pretrained video encoder + action predictor. D. Fine-tune encoder + action predictor with collapse monitoring. Keep robot interaction budget and data splits identical. This separates representation benefit from extra training data or compute.

**Check yourself:** Why might I-JEPA perform well on object recognition yet fail to predict how a robot's gripper moves? What does action-conditioned post-training add to V-JEPA?
