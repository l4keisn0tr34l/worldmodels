# World Models — From First Principles to Research

**A self-contained, progressively rigorous course.** Built around [Aman's World Modeling primer](https://aman.ai/primers/world-modeling/), but rewritten as original teaching notes rather than a copy. The source's renderer / simulator / planner framing and its JEPA + embodied-model sections set the coverage.

**Your starting point:** you can follow basic RNNs, Transformers, attention, backpropagation, tensors, and embeddings. **Not assumed:** RL, POMDPs, state estimation, KL divergence, VAEs, diffusion, MPC, 3D representations, or JEPA.

## The through-line

An agent needs to answer four questions:

1. What is happening, even if I cannot see the full state? **Inference and belief.**
2. If I do *this*, what will happen? **Dynamics and world modeling.**
3. Which action will lead to a good outcome? **Planning or policy learning.**
4. Can I predict usefully without reconstructing every pixel? **Latent representations and JEPA.**

Start with the [linked table of contents](INDEX.md) and [full cart walkthrough](00_start_here/02_one_full_walkthrough.md). Read chapters in order through `02_latent_world_models`, then choose a deeper branch. Each chapter ends with a **check yourself** prompt, and the labs produce real numerical results. Equations define their symbols rather than relying on glossary lookup.

## Complete reading path

| Stage | Files | What you learn |
|---|---|---|
| Orient | `00_start_here/00_map.md`, `01_notation.md` | Fundamental questions, model taxonomy, notation |
| Prerequisites | `01_foundations/01`–`06` | RL, probability, POMDPs, information, autoencoders, variational inference |
| Dynamics | `02_latent_world_models/07`–`10` | State-space models, RSSMs, losses, Ha & Schmidhuber, PlaNet |
| Rendering / simulation | `03_renderer_and_simulator/11`–`13` | Diffusion/video; NeRF/3DGS; graph and physical simulators |
| Planning | `04_planning_and_control/14`–`16` | MPC/CEM/PETS, Dreamer, MuZero/TD-MPC2 |
| JEPA | `05_jepa/17`–`20` | Self-supervision, I-/V-JEPA, collapse, energy/probabilistic views |
| Research | `06_embodied_and_research/21`–`24` | VLAs/WAMs, evaluation, current directions, project design |
| Hands-on | `07_labs/` | Executable toy world model, CEM, tests, RSSM/JEPA training recipes |
| Practice | `08_exercises/` | Retrieval, derivations, debugging, solutions |
| Source map | `09_references/` | Papers, original primer crosswalk, glossary |

## Four exercises that show you've understood it

- Derive the Bayesian belief update from transition and observation likelihoods.
- Explain precisely why RSSM has *both* deterministic `h_t` and stochastic `z_t` and why training uses the posterior while imagination uses the prior.
- Train a dynamics predictor, show its one-step error vs free rollout error, then plan actions with it.
- Explain why **MSE between two freely trainable embeddings** can collapse and how JEPA variants avoid that.

## Study sequence (8 weeks, adjustable)

- Week 1: orientation + probability + MDP/POMDP + Bayesian filtering.
- Week 2: representations, KL/ELBO, deterministic vs probabilistic state-space models.
- Week 3: RSSM, model losses, Ha & Schmidhuber, PlaNet + toy lab.
- Week 4: planners, MPC/CEM/PETS, Dreamer, MuZero/TD-MPC2.
- Week 5: diffusion/renderers, NeRF/3DGS, object and graph simulators.
- Week 6: SSL, JEPA, I-JEPA, V-JEPA, collapse regularizers.
- Week 7: VLAs/WAMs, uncertainty, failure analysis, evaluation.
- Week 8: replicate an ablation, write research proposal, defend assumptions.

## How to use this repository

```bash
cd world_models_from_first_principles
python3 07_labs/toy_world_model.py
python3 07_labs/toy_cem_planner.py
python3 -m unittest discover -s 07_labs -p 'test_*.py'
```

Only the standard library is needed for the executable toy labs. The optional PyTorch recipes are instructional and require `torch` plus appropriate data.

**Caveat:** This is a broad, rigorous learning path, not a reproduction of the whole internet or every implementation detail of every cited model. The source's taxonomy is a useful *functional lens*, not the sole accepted way to classify models. A learned predictor is not automatically a causal or trustworthy simulator, and prediction accuracy by itself is not planning quality.
