# 21 — Vision–Language–Action (VLA), World Models (WM), World Action Models (WAM)

## Three different output contracts

A **VLA** is primarily an instruction-conditioned policy:

\[p_\theta(a_{t:t+K}\mid o_{\le t},\ell).\]

The input may include images, text instruction `ℓ`, robot state; output is executable action tokens or continuous action chunks. No explicit future-state prediction is mathematically required by that policy objective.

A **world model** predicts the conditional future under a hypothetical action:

\[p_\phi(z_{t+1:t+H}\mid z_t,a_{t:t+H-1}).\]

It may be useless at directly generating actions without a planner or policy. A **WAM** deliberately couples future-state prediction and action-generation under the same task context:

\[p_\psi(z_{t+1:t+H},a_{t:t+K}\mid z_{\le t},\ell).\]

This is an emerging research category and term, not a universally standardized benchmark-defined architecture. Some systems are joint (single network for futures/actions); others are cascaded (future video/semantic plan first, action decoding next). 'Predictive policy' does not by itself prove physical grounding.

## Action representations and embodiments

Different robots do not share a single action space: a gripper robot might output 7 joint velocities; an arm with hand articulation might output dozens of joint positions; digital actions are keyboard/mouse controls. Possible action carriers: discrete bins/tokens, end-effector poses, Cartesian deltas, joint commands, trajectories, diffusion/flow-based continuous actions.

An action can be **velocity**, **position target**, **torque**, or something else; model evaluation must state units and control frequency. Predicted actions need downstream tracking/controller integration. A pose prediction may not be dynamically reachable.

## Future carriers

- **Pixel/video carrier**: human-readable imagined sequence but expensive, sometimes physically wrong.
- **Geometric carrier**: object poses, point trajectories, depth, optical flow; more directly tied to motion but may miss contact/materials.
- **Semantic percept**: predicted object attributes and relations, e.g., 'cup held, lid open'. Needs grounding to measurable physical effects.
- **Latent embedding**: compact and scalable but difficult to inspect or evaluate.
- **Joint state-action tokens**: tightly coupled prediction but can be computationally expensive and difficult to calibrate.

## Three different horizons

1. **Prediction horizon H**: how far the predicted future goes.
2. **Action chunk K**: how many controls generated.
3. **Commit horizon C**: how many controls executed before next observation, generally `C≤K`.

An imagined 2-second future need not mean you drive blind for 2 seconds. Frequent replanning is how predictive systems reconcile foresight with closed-loop corrections.

## FLUX 3 Action vs Isaac 0.5 vs V-JEPA 2 (conceptual)

The source primer surveys several 2025–2026 embodied systems. **FLUX 3 Action** is a generative world-action formulation jointly predicting visual futures and control; its design balances rich video-action modeling with action latency and embodiment adaptation. **Isaac 0.5** uses semantic future percepts as predictive targets integrated into a policy-oriented backbone, illustrating a middle ground between explicit video and pure action-only control. **V-JEPA 2** learns video representations with latent prediction then uses robot action data to fit action-conditioned future predictions that feed a planner.

These are **different prediction carriers**, not a reliable rank-ordering of robot capability; reported tasks/data/latency vary and should not be compared without a common evaluation contract. See [source primer](https://aman.ai/primers/world-modeling/) and primary papers for study-specific measurements.

## Evaluating predictive control

Compare instruction completion, contact success, out-of-distribution robustness, action latency versus generated motion duration, replanning responsiveness, safety violations, physical feasibility and data efficiency. Check whether removing the future-prediction objective *actually worsens action outcomes* in controlled ablations.

**Check yourself:** A robot policy attends to a video encoder but only trains action imitation loss. Is it automatically a WAM? What experiment would support a claim that future prediction improves control?
