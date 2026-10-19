# 17 — Self-supervised learning and the problem JEPA addresses

## What does self-supervision mean?

A self-supervised task generates its target from raw unlabeled data, rather than requiring a human-annotated label. E.g., predict hidden patches, future frames, rotated views' embeddings, or the next token. It still has a **supervisory target**; it just doesn't need a manually labeled category.

The central choice is **what target** to predict:

- **Autoregression:** next data token given context; cross-entropy if tokens are categorical.
- **Masked autoencoder:** missing pixels/patches; typically reconstruction loss.
- **Contrastive:** pull compatible views together and push different examples apart; often negative pairs.
- **Invariance/redundancy methods:** align two transformed views with extra constraints to prevent degenerate embeddings (e.g., VICReg).
- **JEPA:** predict the *representation* of a missing/future target from a context representation.

## Why not predict pixels?

Imagine a camera view of an approaching car under flickering sunlight. Sunlight details may vary wildly but are not what matters to braking. Pixel prediction spends compute modeling texture and unpredictable lighting. An embedding predictor *could* learn speed, road geometry and other stable structure instead. This is a **design hope** requiring good masking, anti-collapse and evaluation—not a logical guarantee that latent prediction learns 'understanding'.

## JEPA's target

Let `x` = context (e.g., visible image patches) and `y` = target (e.g., masked region). A context encoder gives `s_x=f_θ(x)` and a target encoder gives `s_y=f_bar(y)` (with details of exactly which patches are passed depending on implementation). The predictor uses context plus target-location/auxiliary tokens:

\[\hat s_y=g_\phi(s_x,c),\qquad \mathcal L_{pred}=\|\hat s_y-\mathrm{sg}(s_y)\|^2.\]

Here `c` can carry spatial target positions, temporal offsets or actions. There is no decoder reconstructing raw y in the basic JEPA loss.

## Why the trivial solution is dangerous

If both encoders and predictor can learn to output constant vector `0`, the loss becomes 0 for every pair. The system has learned **nothing**, yet training loss is perfect. This is representation collapse. Preventing it is a central part of JEPA architecture design.

The exact mechanisms matter: target EMA/stop-gradient asymmetry, appropriate masking, predictor architecture, and/or explicit variance/distribution constraints. **EMA alone is not a mathematical proof that collapse is impossible.** See `19_collapse_vicreg_sigreg.md`.

## I-JEPA is not automatically a temporal world model

I-JEPA masks **spatial** image regions; it learns meaningful image features but does not necessarily learn controlled temporal transitions. V-JEPA uses videos and spatiotemporal masking to learn temporal features. To select actions, an agent needs action-conditioned dynamics or a policy/planner connecting representations with interventions.

## A surprising analogy to Transformers

In a language model, the predicted target is a word/token. In JEPA, the predicted target is an **embedding output by another network**. The training challenge isn't primarily attention mechanics; it's that your target representation moves during training and can collapse. A token ID cannot decide to become constant across every example; a learned target encoder can.

**Check yourself:** Show algebraically why `f(x)=c`, `f_bar(y)=c`, `g(c)=c` makes vanilla latent MSE zero. What extra information is missing when such a model is used for planning?
