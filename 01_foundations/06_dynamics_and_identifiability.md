# 06 — Dynamical systems, causality, and why 'predictive' is not enough

## State-space form

Classical dynamics use `s_{t+1}=F(s_t,a_t,w_t)` and observations `o_t=G(s_t,v_t)` with process noise `w_t` and measurement noise `v_t`. In linear-Gaussian models:

\[s_{t+1}=As_t+Ba_t+w_t,\quad o_t=Cs_t+v_t.\]

Here `A` evolves existing state, `B` describes action effects, `C` exposes state to sensor measurements. Learning the matrices is a basic form of learned world modeling. Kalman filtering computes the posterior distribution when linear-Gaussian assumptions hold; neural state-space models relax these assumptions.

### Hidden velocity example

If observation is only position `x_t`, one frame doesn't reveal velocity. Two or more frames let you estimate it. A Markov state might be `[x_t,v_t]`, with

\[v_{t+1}=v_t+\Delta t\,u_t,\qquad x_{t+1}=x_t+\Delta t\,v_{t+1}.\]

The ordering specifies semi-implicit Euler integration; another convention would use `v_t` for the position update. The distinction matters in code and rollout comparisons.

## Teacher forcing versus free rollout

One-step training may predict `z_{t+1}` given *actual* `z_t`. At deployment it predicts from *its own previous prediction*. Errors compound:

\[\hat z_{t+k+1}=f_\theta(\hat z_{t+k},a_{t+k}).\]

If f is Lipschitz with sensitivity L and one-step error ≤ε, rough error bound after H steps (under strong assumptions) is `ε Σ_{j=0}^{H−1}L^j`. When `L>1` even small errors may expand rapidly. Noise injection, multi-step objectives, recurrent memory, replanning, and shorter horizons are common mitigations. None guarantees physical consistency.

## Data confounding and causal actions

Suppose a driver only brakes on steep downhill roads. A dataset may show that braking correlates with increasing velocity. A passive predictor could learn the wrong relationship because road slope is hidden. True interventions mean comparing consequences while controlling the relevant state/confounders, roughly:

\[p(s'\mid s,\mathrm{do}(a))\quad\text{vs}\quad p(s'\mid s,a).\]

They match under suitable assumptions (e.g., the current state accounts for confounders and action assignment). They are not automatically equal from logged data. Thus 'action-conditioned' does **not** imply learned causal mechanisms.

## State aliasing and irreducible limits

If two physical states yield identical observation histories under available sensors, but react differently to an action, they are aliased. A deterministic encoder cannot distinguish them from that history. Add sensors, longer temporal context, active information gathering, or predictive uncertainty; don't blame only model capacity.

## What a scientifically useful model should pass

1. **Prediction:** forecasts held-out observed transitions better than baseline.
2. **Counterfactual validity:** changes when the action changes, in ways validated by interventions.
3. **Calibration:** forecast uncertainty matches empirical frequencies when possible.
4. **Rolling stability:** model remains useful when fed its own output.
5. **Decision utility:** planning or policy using it improves outcomes without safety regressions.
6. **OOD honesty:** errors or uncertainty meaningfully expose missing knowledge on novel states.

**Check yourself:** Why might one-step accuracy improve while MPC performance gets worse after another training epoch?
