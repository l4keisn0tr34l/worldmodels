# 22 — Evaluation and failure modes: what actually counts as a better world model?

## Evaluate against the intended contract

The primer's taxonomy is functional, so evaluate based on actual output:

| Contract | Strong metrics | Insufficient alone |
|---|---|---|
| Renderer | visual fidelity, temporal/identity coherence, viewpoint consistency, action adherence | single-frame FID / pretty videos |
| Simulator | state error, physical constraints, long rollout stability, counterfactual validity | next-frame pixel loss |
| Planner | real/simulator closed-loop return, safety, interactions to competence, latency | model loss without acting |
| JEPA encoder | probe accuracy, linear probe, temporal action transfer, noncollapse, downstream control | training latent prediction loss |
| WAM | task success, action feasibility, horizon/latency, physical future alignment | predicted video plausibility |

## Train/test split integrity

Always split by **episode/environment/scene/object combinations** appropriate for deployment, not random near-duplicate frames. For robotics, hold out object shapes, camera viewpoints, dynamics conditions and action sequences. For network/security telemetry, split by time, host/scenario/attack campaign as relevant to prevent cross-window leakage. Report exactly what was held out.

## The long-rollout problem

Plot error against horizon `k=1...H`. Expect errors to grow. Also report distribution of outcomes, not just mean: rare catastrophic failures can be hidden in average error. Measure physical constraints, state drift, object swaps, and calibration under actual closed-loop execution.

## Error attribution matrix

| Symptom | Likely issue | Test |
|---|---|---|
| Great one-step, poor 30-step | Exposure bias, unstable dynamics | Open-loop vs corrected rollouts |
| Same prediction under all actions | Action ignored / weak conditioning | Action intervention sensitivity test |
| Loss almost zero, embeddings constant | Representation collapse | Feature std / effective rank |
| Perfect images, impossible collisions | Appearance/physics mismatch | Explicit contact and constraint validation |
| Great model metrics, poor policy | Wrong decision-relevant state or model exploitation | Predicted vs actual chosen-action returns |
| High confidence but wrong on new task | Epistemic uncertainty underdetected | OOD calibration and ensembles |
| Excellent average success, unsafe tails | Risk hidden in aggregate | Constraint violation rate, CVaR/quantiles |

## Ablations that matter

Compare actions vs no action conditioning; correct shuffled actions vs real; image reconstruction vs future latent prediction; deterministic vs stochastic; short vs long imagination; KL/free-bits settings; policy with/without planner; pretrained vs scratch representation with matched data/compute. Report uncertainty bands across several seeds where feasible.

## Baselines before SOTA

Classical kinematics/constant-velocity; AR/GRU; persistence; random action; scripted planner; linear state-space / Kalman filter; supervised next-state model; matched-parameter MLP. A simple baseline can reveal flawed data alignment faster than complex architectures.

## Observational forecasting is not causal simulation

If an attacker often follows a port scan with credential attacks, a model can estimate next events conditioned on logs. It cannot conclude which countermeasure *would stop the attacker* unless defenses and counterfactual effects are learned/validated. Predicting attacker behavior ≠ proving a prevention action works.

## A model card for honest reporting

Document input modalities; decision-time available signals; output variables; training data; environment dynamics; action labels; randomness; evaluation split; baselines; calibration; rollout horizons; safety constraints; latency/hardware; known unsupported conditions; reported failures. Keep measured, predicted, proposed and simulated outputs clearly distinguished.

**Check yourself:** Design a falsification test for a model that claims 'understands physics' based on beautiful synthetic videos.
