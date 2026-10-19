# Primary reading list and verified starting points

**Overview:** [Aman's World Models primer](https://aman.ai/primers/world-modeling/). The conceptual renderer/simulator/planner framing is prominent in the primer. For reinforcement-learning definitions, use Sutton & Barto's [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html).

## Learned latent dynamics and planning

- **Ha & Schmidhuber (2018), World Models** — https://arxiv.org/abs/1803.10122 — Vision/Memory/Controller, mixture-density RNN, dream controller.
- **Hafner et al., PlaNet** — https://arxiv.org/abs/1811.04551 — latent dynamics, RSSM, online planning, latent overshooting.
- **Hafner et al., Dreamer** — https://arxiv.org/abs/1912.01603 — behavior learning by imagination.
- **Schrittwieser et al., MuZero** — https://arxiv.org/abs/1911.08265 — learned latent reward/value/policy and tree search.
- **Chua et al., PETS** — https://arxiv.org/abs/1805.12114 — probabilistic dynamics ensembles and trajectory sampling.
- **Hansen et al., TD-MPC2** — https://arxiv.org/abs/2310.16828 — task-oriented continuous-control world models.

## Rendering and geometry

- **Ho et al., Video Diffusion Models** — https://arxiv.org/abs/2204.03458 — temporally structured generative video.
- **Peebles & Xie, Scalable Diffusion Models with Transformers** — https://arxiv.org/abs/2212.09748 — DiT architectural recipe.
- **Mildenhall et al., NeRF** — https://arxiv.org/abs/2003.08934 — continuous volumetric scene function.
- **Kerbl et al., 3D Gaussian Splatting** — https://arxiv.org/abs/2308.04079 — explicit neural scene primitives.
- **Sanchez-Gonzalez et al., Graph Network-based Simulators** — https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html — particle-based learned physics.
- **Pfaff et al., MeshGraphNets** — https://arxiv.org/abs/2010.03409 — mesh physical simulation.

## JEPA and self-supervised learning

- **Assran et al., I-JEPA** — https://arxiv.org/abs/2301.08243 — masked region representation prediction.
- **Assran et al., V-JEPA 2** — https://arxiv.org/abs/2506.09985 — large-scale video learning and action-conditioned planning.
- **Bardes et al., VICReg** — https://arxiv.org/abs/2105.04906 — explicit anti-collapse variance/covariance.
- **Balestriero & LeCun, LeJEPA** — https://arxiv.org/abs/2511.08544 — isotropic Gaussian distribution regularization.
- **LeCun, A Path Towards Autonomous Machine Intelligence** — https://openreview.net/forum?id=BZ5a1r-kVsf — programmatic motivation for predictive world models.

## New embodied paradigms

The [source primer's 'Vision-Language-Action, World Action Models, World Models' section](https://aman.ai/primers/world-modeling/) is the entry point for comparative coverage of FLUX 3 Action, Isaac 0.5, action carriers, training regimes and evaluation. Newer system-specific claims are most reliably assessed against those projects' **original primary papers, model cards and reported experimental setups**, rather than comparing cherry-picked headline success rates across different embodiments.

## Active skepticism

Do not treat paper abstract claims as established general law. Particularly scrutinize action-conditioned **causal** inference, passive-video-to-control transfer, physically accurate video worlds, collapse prevention guarantees, uncertainty calibration, and 'AGI/world understanding' rhetoric. These need specific validation, and results can vary by data, compute, task and evaluation protocol.
