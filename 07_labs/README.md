# Practical labs — small enough to understand every line

## Lab A: train a real learned model

Run `python3 07_labs/toy_world_model.py` from the repository root. The environment is a 1D object with hidden-to-the-learner coefficients for damping and acceleration. You get **oracle state `(x,v)`**, action and subsequent state. The script collects random transitions, solves a linear least-squares problem (implemented directly via Gaussian elimination), and prints learned coefficients and **free-running** errors at steps 1, 5, 15 and 30. It is a learned **action-conditioned dynamics model** even though it uses no neural net. This deliberate simplicity separates dynamics learning from neural-architecture complexity.

**Experiments:** (1) reduce training transitions by 10x; (2) increase process noise; (3) train on actions in `[-0.2,0.2]` and test larger actions; (4) break action alignment by shuffling actions, then check action sensitivity; (5) increase horizon to 100. Explain the results. Caution: the environment's action is physically clamped at ±1, and extrapolating beyond this bound has no well-defined physical meaning in this code.

## Lab B: plan in the learned model

Run `python3 07_labs/toy_cem_planner.py`. It uses CEM to sample action sequences and scores the final predicted distance from `x=3`, velocity and action effort. It executes only the **first** action, reads true state, and replans. Compare to no action, random action, and sending all 16 planned actions without feedback. Make the learned acceleration artificially wrong to study model exploitation.

## Lab C: verify

Run `python3 -m unittest discover -s 07_labs -p 'test_*.py'` from root. This checks numerical regression, action effect, one-step prediction and MPC improvement. A passing test checks only this tiny simulator and **does not** establish robustness for real robotics.

## The gap to RSSM / JEPA

This toy environment exposes real velocity. A visual robot doesn't. To make it a POMDP, hide `v`, present noisy position readings, and implement Bayesian filtering, a recurrent state estimator, or RSSM. For an RSSM, a learned posterior would use the new observation while an imagination prior would not. For a JEPA, encode two consecutive observations using online and target encoders, predict the target embedding using action, prevent collapse, and plan in that feature space. See the two recipes below.

**Note:** All executable code is original and contains no external dependencies; you can inspect and modify every operation.
