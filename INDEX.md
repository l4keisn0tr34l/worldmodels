# Linked table of contents

Open any chapter directly from here. `README.md` explains the intended order; chapters can also be used as standalone reference notes.

## Start here

- [World Models — From First Principles to Research](README.md)

## 00_start_here

- [00 — What is a world model, really?](00_start_here/00_map.md)
- [01 — Notation and mental translations](00_start_here/01_notation.md)
- [02 — One complete worked example: from hidden world to planned action](00_start_here/02_one_full_walkthrough.md)

## 01_foundations

- [00 — Math refresher: exactly what the later equations need](01_foundations/00_math_refresher.md)
- [01 — Probability for world modeling: from zero to useful](01_foundations/01_probability.md)
- [02 — MDPs, rewards, policies, and model-based RL](01_foundations/02_mdp_rl.md)
- [03 — Partial observability, POMDPs, and Bayesian filtering](01_foundations/03_pomdp_beliefs.md)
- [04 — Latent state, embeddings, and the information you must keep](01_foundations/04_representations.md)
- [05 — VAEs, latent variables, the ELBO, and KL: fully unpacked](01_foundations/05_variational_inference.md)
- [06 — Dynamical systems, causality, and why 'predictive' is not enough](01_foundations/06_dynamics_and_identifiability.md)

## 02_latent_world_models

- [07 — RSSMs, dissected from the GRU you already know](02_latent_world_models/07_state_space_rssm.md)
- [08 — Training world models: objectives, implementation, and failures](02_latent_world_models/08_training_and_rollouts.md)
- [09 — 2018 World Models: V, M, C](02_latent_world_models/09_original_world_models.md)
- [10 — PlaNet: learning latent dynamics for planning from pixels](02_latent_world_models/10_planet.md)

## 03_renderer_and_simulator

- [11 — Renderer world models: image diffusion to interactive video](03_renderer_and_simulator/11_renderer_diffusion_video.md)
- [12 — Spatial state: NeRF, 3D Gaussians, geometry, and what they omit](03_renderer_and_simulator/12_3d_spatial_models.md)
- [13 — Object-centric models, graph networks, and learned physics](03_renderer_and_simulator/13_graph_physics.md)

## 04_planning_and_control

- [14 — Planning: MPC, CEM, PETS, risk and real decisions](04_planning_and_control/14_mpc_cem_pets.md)
- [15 — Dreamer: learning a policy from imagined trajectories](04_planning_and_control/15_dreamer.md)
- [16 — MuZero, TD-MPC, TD-MPC2, and task-oriented modeling](04_planning_and_control/16_muzero_tdmpc.md)

## 05_jepa

- [17 — Self-supervised learning and the problem JEPA addresses](05_jepa/17_ssl_and_predictive_learning.md)
- [18 — I-JEPA, V-JEPA, V-JEPA 2 and action conditioning](05_jepa/18_ijepa_vjepa.md)
- [19 — Representation collapse, VICReg, EMA, stop-gradient, SIGReg](05_jepa/19_collapse_vicreg_sigreg.md)
- [20 — JEPA as energy model, predictive information and uncertainty](05_jepa/20_energy_uncertainty_and_variants.md)

## 06_embodied_and_research

- [21 — Vision–Language–Action (VLA), World Models (WM), World Action Models (WAM)](06_embodied_and_research/21_vla_wam_world_action.md)
- [22 — Evaluation and failure modes: what actually counts as a better world model?](06_embodied_and_research/22_evaluation_failure_modes.md)
- [23 — Research landscape: how the pieces connect (not a ranking)](06_embodied_and_research/23_research_landscape.md)
- [24 — Build your own research project: three defensible blueprints](06_embodied_and_research/24_project_blueprints.md)
- [25 — Flow matching, diffusion actions and predictive control](06_embodied_and_research/25_flow_matching_and_actions.md)
- [26 — Semantic future percepts, Isaac 0.5 and scaling action-free video](06_embodied_and_research/26_semantic_predictive_models.md)

## 07_labs

- [Practical labs — small enough to understand every line](07_labs/README.md)
- [Action-conditioned JEPA: minimal training protocol](07_labs/jepa_training_recipe.md)
- [RSSM in PyTorch: a build recipe, not a fake turnkey implementation](07_labs/rssm_training_recipe.md)

## 08_exercises

- [Self-tests by stage — no peeking at answers first](08_exercises/00_checkpoints.md)
- [Detailed derivations and pencil-and-paper drills](08_exercises/01_derivations.md)
- [Solutions and reasoning — read after attempting the problems](08_exercises/02_solutions.md)
- [Paper reading and explanation template](08_exercises/03_paper_reading_template.md)

## 09_references

- [Glossary — quick retrieval (terms linked conceptually to chapters)](09_references/glossary.md)
- [Crosswalk: every major area of Aman's world-modeling primer](09_references/primer_crosswalk.md)
- [Primary reading list and verified starting points](09_references/reading_list.md)

