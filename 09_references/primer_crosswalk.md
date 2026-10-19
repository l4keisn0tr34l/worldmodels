# Crosswalk: every major area of Aman's world-modeling primer

**Source:** https://aman.ai/primers/world-modeling/ (consulted October 20, 2026).

This is a **topic coverage map**, not a line-by-line copy or reproduction. Sections of the original are extensive and sometimes contain many named systems. These notes aim to teach the reusable mechanisms underneath them. Follow the source for original figures and exhaustive system/paper-by-paper citations.

| Source primer topic | Explanatory chapters |
|---|---|
| Background, definitions, agent-world loop | `00_start_here/00_map.md`, `01_notation.md`, `01_foundations/02_mdp_rl.md` |
| Renderer vs simulator vs planner; unified models | `00_map`, `11_renderer_diffusion_video`, `12_3d_spatial_models`, `14_mpc_cem_pets` |
| Representation learning / temporal dynamics | `04_representations`, `06_dynamics_and_identifiability`, `07_state_space_rssm` |
| Partial observability and belief states | `03_pomdp_beliefs`, `07_state_space_rssm`, `20_energy_uncertainty_and_variants` |
| Object-centric and relational models | `13_graph_physics`, `18_ijepa_vjepa`, `23_research_landscape` |
| Renderer: diffusion conditioning / guidance / DiT / video | `11_renderer_diffusion_video`, `25_flow_matching_and_actions` |
| Interactive renderer models, controllability, evaluation | `11_renderer_diffusion_video`, `22_evaluation_failure_modes` |
| Simulator: NeRF, 3DGS, inverse graphics, spatial state | `12_3d_spatial_models` |
| Simulator: interaction networks, GNS, MeshGraphNets | `13_graph_physics` |
| World Models V-M-C, PlaNet, Dreamer | `09_original_world_models`, `10_planet`, `15_dreamer` |
| Planner: probabilistic dynamics / PETS / MPC / CEM | `14_mpc_cem_pets`, `06_dynamics_and_identifiability` |
| Planner: MuZero, TD-MPC, TD-MPC2 / task states | `16_muzero_tdmpc` |
| JEPA core vs generative and joint embedding | `17_ssl_and_predictive_learning` |
| I-JEPA masking, target encoder | `18_ijepa_vjepa` |
| V-JEPA, V-JEPA 2, action-conditioned adaptation | `18_ijepa_vjepa` |
| Advanced JEPA: object-centric, temporal, spatial | `13_graph_physics`, `18_ijepa_vjepa`, `20_energy_uncertainty_and_variants` |
| Collapse, stop-grad, EMA, VICReg, SIGReg/LeJEPA | `19_collapse_vicreg_sigreg` |
| Energy-based, predictive-information and variational JEPA | `20_energy_uncertainty_and_variants` |
| VLA, WM, WAM, future carriers, temporal horizons | `21_vla_wam_world_action`, `25_flow_matching_and_actions` |
| FLUX 3 Action (multimodal generation, action chunks, flow) | `21_vla_wam_world_action`, `25_flow_matching_and_actions` |
| Isaac 0.5 (semantic future percepts + action data) | `26_semantic_predictive_models` |
| Uncertainty, risk, evaluation contracts / deployment | `22_evaluation_failure_modes`, `14_mpc_cem_pets` |
| Open research questions and implementation patterns | `23_research_landscape`, `24_project_blueprints`, `07_labs` |

## What is deliberately **not** reproduced

- Original source figures, long copyrighted descriptions, or full system literature surveys.
- Full implementations of proprietary video/robot-action systems, or false claims that toy scripts reproduce Dreamer/V-JEPA.
- Exhaustive per-paper 2026 benchmarking numbers (which depend on different test sets and may change).

## Suggested way to read the original alongside this folder

Read a local foundational chapter first, then follow the matching source section for visual figures and paper-specific details. If the original introduces an unfamiliar equation, translate each symbol back to `00_start_here/01_notation.md` and track: what is known now, what is predicted, what distribution is being learned, and where gradients flow. For papers reporting robot capability, inspect closed-loop evaluation and action/physical time horizons before comparing performance.
