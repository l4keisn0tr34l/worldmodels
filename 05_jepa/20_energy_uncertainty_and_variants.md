# 20 — JEPA as energy model, predictive information and uncertainty

## Energy-based interpretation

Define an energy that scores compatibility of context x and target y:

\[E_{\theta,\phi}(x,y)=\|g_\phi(f_\theta(x))-f_{\bar\theta}(y)\|^2.\]

A compatible x,y pair should have low energy. This resembles energy-based modeling, but a generic unnormalized energy `E(x,y)` is **not automatically a normalized probability distribution**. To obtain `p(y|x)∝exp(−E(x,y))` one needs a partition function `Z(x)=∫exp(−E(x,y))dy`, which may be computationally difficult and may not be trained in ordinary JEPA. 'JEPA is an EBM' is a useful interpretation, not an assertion that it computes calibrated likelihoods.

## Predictive information

A good representation of present observations should retain information useful for future observations. In information-theoretic language, we might aim to maximize `I(z_context;z_target)` while limiting irrelevant input detail. But the usual latent MSE JEPA objective does not directly calculate mutual information. The relationship is conceptual or depends on specific assumptions / objectives. A latent system can overfit shortcuts, e.g., camera timestamp, and score well without learning useful object dynamics.

## Deterministic point estimate versus multiple futures

If a hidden ball can exit occlusion left or right, a deterministic squared-error predictor often maps to a mean embedding. That 'in-between' vector may not correspond to either valid outcome. Solutions include a stochastic latent `u`, mixture output distribution, categorical future codes, diffusion in embedding space, ensembles, or conditional generative prediction:

\[\hat z_{t+1}=g_\phi(z_t,a_t,u_t),\quad u_t\sim p(u).\]

Now changing u may sample different plausible futures while holding observation/action fixed. For this to be useful in safety-critical planning, samples must represent *calibrated* uncertainty and maintain state/action consistency.

## Variational / probabilistic JEPA idea

Introduce a future-latent posterior `q_φ(u|context,future)` and an imagination prior `p_θ(u|context,action)`; predict target latent from sampled `u` and context. A generic loss is:

\[L=\mathbb E_{q}\big[\|\hat z_{future}(u)-\mathrm{sg}(z_{future})\|^2\big]+\beta D_{KL}(q(u\mid context,future)\|p(u\mid context,action))+L_{anti-collapse}.\]

This is **an illustrative design pattern**, not a universal formula for all probabilistic JEPA papers. Distinguish uncertainty in hidden outcomes from uncertainty in model parameters. Many possible latent distributions may fit the same observations; choosing Gaussian/categorical distributions is an inductive bias.

## Belief states connect classical filtering and JEPA

A probabilistic transition model predicts `b^-_(t+1)`. The next observation updates it via a likelihood or learned inference network. A deterministic JEPA embedding is closer to a predictive point representation than a full belief distribution. A model may keep a sufficiently informative recurrent memory to be useful under partial observation without providing a calibrated probabilistic posterior.

## Planning under energy/latent distances

A planner could minimize terminal energy between predicted future and a goal embedding:

\[A^*=\arg\min_A d(\hat z_{t+H}(A),z_g).\]

But shortcut representations are dangerous: a low embedding distance may not imply the goal is physically achieved. Evaluate success in real state terms, plus calibration, model bias, action feasibility, robustness under changed scene conditions.

**Check yourself:** If two futures are equally plausible, why does a single latent MSE prediction risk averaging? Is `exp(−E)` a probability without normalization?
