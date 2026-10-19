# 23 — Research landscape: how the pieces connect (not a ranking)

## A historical conceptual arc

**State estimation & control** (Kalman filtering, MPC) gave mathematical tools for latent state inference and action optimization. **Model-based RL** added learned transition/reward models. **World Models (2018)** popularized learned compressed environment dynamics and 'dreams'. **PlaNet** made online planning from pixels practical with RSSM. **Dreamer** learned actions in latent imagination. **MuZero** emphasized task-oriented latent search without observation reconstruction. **TD-MPC / TD-MPC2** optimized value-driven continuous control. **Diffusion/video models** scaled visual future generation. **NeRF/3DGS** offered richer spatial state. **JEPA/I-JEPA/V-JEPA** emphasized predicting representations rather than pixels. **WAM/VLA/semantic predictive approaches** explore coupling broad video or language representations with action.

This narrative is pedagogical; methods evolved in parallel and many ideas predate the papers that popularized them.

## Six active design questions

1. **What state should be predicted?** pixels, semantic objects, geometry, latent features, value-relevant abstractions or distribution of beliefs?
2. **What supervision?** action-annotated rollouts, passive video, self-supervision, preference/reward, physical simulator state?
3. **What uncertainty?** single predicted mean, stochastic dynamics, epistemic ensembles, multi-modal latent distributions?
4. **What does 'causal' mean?** action condition, intervention in a real environment, or identified mechanism? Keep separate.
5. **Where does planning compute live?** runtime search, value-guided sampling, learned actor, or jointly generated state-action trajectories?
6. **How do we falsify claims?** new embodiment, longer horizon, confounded actions, unseen surfaces/materials, changed camera, changed objectives?

## Avoid architecture-name cargo culting

If a dataset contains tabular sequential states and actions, a 2-layer GRU predicting next state can be a legitimate world model. A billion-parameter video model might be unnecessary. If images matter and actions are scarce, a video self-supervised encoder may help; test against strong frozen image/standard video representations. If object interactions determine safety, an explicit graph may improve inductive bias. 'Transformer' is a computational building block, not a full theory of environments.

## Suggested paper reading order

1. [World Models (2018)](https://arxiv.org/abs/1803.10122) — components and dream training.
2. [PlaNet (2018/2019)](https://arxiv.org/abs/1811.04551) — RSSM and latent MPC.
3. [Dreamer (2019)](https://arxiv.org/abs/1912.01603) — actor-critic from imagined trajectories.
4. [MuZero (2020)](https://arxiv.org/abs/1911.08265) — task-relevant planning latent.
5. [TD-MPC2 (2023/2024)](https://arxiv.org/abs/2310.16828) — scalable continuous control.
6. [I-JEPA (2023)](https://arxiv.org/abs/2301.08243) — representation-space targets.
7. [V-JEPA 2 (2025)](https://arxiv.org/abs/2506.09985) — video representation plus action grounding.
8. [LeJEPA (2025)](https://arxiv.org/abs/2511.08544) — distribution regularization for noncollapsed embeddings.

Use `../09_references/reading_list.md` for more thematic papers and primary sources, including diffusion, NeRF and graph simulation.

**Check yourself:** Pick a toy environment and justify choosing one state representation, one training objective, one controller and three falsification tests. If an architecture is selected only 'because it's SOTA', revise the design.
