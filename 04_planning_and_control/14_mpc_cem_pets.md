# 14 — Planning: MPC, CEM, PETS, risk and real decisions

## Counterfactual rollouts

At current inferred state `z_t`, a candidate action sequence `A=(a_t,...,a_(t+H−1))` produces an imagined trajectory under `f_θ`:

\[\hat z_{t+k+1}=f_\theta(\hat z_{t+k},a_{t+k}),\quad\hat z_t=z_t.\]

Score it by discounted predicted reward (or negative cost):

\[J(A)=\sum_{k=0}^{H-1}\gamma^k\hat r(\hat z_{t+k},a_{t+k})+\gamma^H\hat V(\hat z_{t+H}).\]

The terminal value `V` can cheaply approximate what happens *after* truncated planning horizon. If all you have is a short horizon, the agent can be greedy near the horizon end.

## MPC: repeated short commitments

1. Observe and estimate current state.
2. Generate candidate future action sequences.
3. Roll each through learned or analytic dynamics.
4. Score predictions; choose best candidate.
5. Execute **first action** (or safe prefix), not the whole sequence.
6. Observe again and replan.

This receding horizon corrects errors using real evidence. It cannot rescue a planner that chooses an unsafe first action due to model exploitation.

## CEM: optimization without action gradients

Cross-Entropy Method searches for high-return continuous sequences using a distribution:

\[A^{(i)}\sim\mathcal N(\mu,\mathrm{diag}(\sigma^2)).\]

At each iteration sample N candidates; evaluate J; sort; choose top K 'elite' trajectories; replace `μ` and `σ` with elite empirical mean/std; optionally smooth updates and clamp bounds. Repeat, then execute action from best or mean sequence. CEM is a stochastic black-box optimizer, not the classification cross-entropy loss even though its name also contains 'cross entropy'.

For discrete action spaces, optimize categorical probabilities instead of Gaussians. For very large action spaces, naive CEM may become intractable; use policy-guided proposals, hierarchical planning, gradient methods, or search.

## Uncertainty-aware PETS intuition

**PETS** combines probabilistic ensembles of dynamics models and trajectory sampling (including TS variants) with MPC. An ensemble has different parameter estimates; output distributions can express stochastic outcomes. During candidate rollout, sample plausible next states/models to estimate outcome distributions rather than treating one predicted trajectory as certain.

Crucially, **ensemble diversity** is only an approximation to epistemic uncertainty, and unreliable off-distribution predictions can still be overconfident. Evaluate calibration and actual control safety rather than asserting 'the ensemble knows what it doesn't know'.

## Expected reward is not enough for safety

Suppose A yields reward 10 surely; B yields 100 with probability .2 and −10 with probability .8. Expected B = 12, so a risk-neutral planner chooses B. A risk-sensitive decision-maker may prefer A. Options include constraints `P(collision)≤δ`, worst-case objectives, lower-confidence bounds, CVaR, and explicit hard safety constraints. Correct uncertainty estimates are essential; an uncalibrated learned risk score is not a probability.

## Model exploitation

Imagine a model that accidentally predicts reward rises when robot torque goes beyond observed ranges. Search explores many candidates and selects exactly that torque. Increasing candidate count can **worsen** performance by locating rarer model errors. Compare predicted vs actual returns for actions *chosen by the planner*, not just random held-out transitions. Limit action ranges and planning horizon; use epistemic uncertainty and real rollout validation where safe.

## Engineering trade-offs

| Approach | Compute spent | Typical strength | Weakness |
|---|---|---|---|
| Random shooting | At decision time | Simplest baseline | Sample inefficient |
| CEM/MPPI-style | At decision time | Continuous optimization | Expensive; model exploit |
| MCTS | At decision time | Discrete combinatorial search | Large branching / cost |
| Dreamer actor | Mostly training time | Fast deployment | Bias in imagination training |
| VLA/WAM direct actions | Mostly training time | Large-data and semantic control | May need feedback and action grounding |

**Check yourself:** Why might a 5-step MPC planner outperform a 30-step MPC planner despite optimizing a shorter horizon? How does a terminal value function change this comparison?
