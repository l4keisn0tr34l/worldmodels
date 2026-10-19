# 02 — One complete worked example: from hidden world to planned action

This connects the whole course using a one-dimensional robot cart. You can refer back to it while reading the architecture papers.

## Stage 1: the actual environment

The physical state is `s_t=(x_t,v_t)`, where `x` is location in meters and `v` is velocity in meters/second. The agent chooses acceleration-like input `a_t∈[-1,1]`. For a time interval Δt=1 and unitless damping coefficient 0.9, suppose:

\[v_{t+1}=0.9v_t+0.3a_t+w_t,\quad x_{t+1}=x_t+v_{t+1}.\]

Here `w_t` is process noise. Only position is observed: `o_t=x_t+n_t`, where `n_t` is measurement noise. The camera/sensor doesn't directly tell velocity. The control goal is `x_goal=3` with near-zero final velocity.

## Stage 2: observations are insufficient

One reading `o_t=1.0` is consistent with cart moving quickly toward the goal, standing still, or moving away. A snapshot is insufficient for reliable braking. History matters. A belief state `b_t=p(x_t,v_t|o_{0:t},a_{0:t−1})` resolves some uncertainty, using earlier positions/actions and a transition model.

In a classical system, you could use a Kalman filter if dynamics and noise are linear-Gaussian. In a learned world-model system, a GRU/Transformer approximates a useful summary of history; an RSSM makes stochastic uncertainty explicit through `z_t`.

## Stage 3: the learned representation

An encoder consumes sensor evidence (position history, maybe an image) and generates features. A deterministic recurrent memory `h_t` integrates past latent/action information. A prior `p(z_t|h_t)` predicts what the state might be before seeing new `o_t`. A posterior `q(z_t|h_t,o_t)` corrects with the current observation.

At a given time suppose `h_t` summarizes leftward motion and posterior is Gaussian for an unobserved velocity-like feature: `q(z_t|h_t,o_t)=N(0.4,0.1²)`. The prior predicted `N(0.5,0.2²)`. Their disagreement produces a KL training signal. The exact latent z may **not equal physical velocity**; this equality is just a didactic simplification.

## Stage 4: model learning

Real experience gives tuples `(o_t,a_t,o_(t+1),r_(t+1))`. Training has at least three jobs:

- Explain the current observation from latent state (`observation loss`).
- Align prior predictions to evidence-adjusted posterior state (`KL dynamics loss`).
- Predict reward/goal progress for action evaluation (`reward head`).

For this cart a reward might be `r_(t+1)=−(x_(t+1)−3)²−0.05a_t²`, with possible extra penalty for large velocity near the target. **Reward specification matters:** if you reward position only, a cart passing rapidly through the goal might appear successful despite failing to stop.

## Stage 5: imagination

We want to compare the following candidate 3-action sequences:

- `A=(+1,+1,+1)` aggressive acceleration.
- `B=(+1,0,−1)` accelerate then brake.
- `C=(0,0,0)` coast.

For each sequence start with the current posterior state estimate. Then run prior transitions under hypothetical actions; compute predicted rewards. The **future observation is not yet available**: the model cannot use future posterior states until that future becomes present.

As a numerical illustration with *fully observed* initial state `x=0,v=0`, **zero noise**, the known physical transition above gives:

| Sequence | Step 1 `(x,v)` | Step 2 `(x,v)` | Step 3 `(x,v)` |
|---|---|---|---|
| A (+1,+1,+1) | (.300,.300) | (.870,.570) | (1.683,.813) |
| B (+1,0,−1) | (.300,.300) | (.570,.270) | (.513,−.057) |
| C (0,0,0) | (0,0) | (0,0) | (0,0) |

These numbers are *analytic idealized* rollouts for illustration, not output from an actual learned RSSM. A might be better to reach x=3 within three steps, but its nonzero final speed matters. With a longer horizon and stopping constraint, B-like sequences might become preferable. This is why **objective and horizon** matter alongside model prediction quality.

## Stage 6: planning or dreaming

**PlaNet/MPC route:** sample many sequences, score via learned dynamics, execute first action of best sequence, receive new measurement and replan.

**Dreamer route:** use many imagined sequences to train actor/critic. At deployment the actor outputs an action without running a large candidate search each time.

**MuZero route:** learn state/reward/value predictions sufficient for search, even if predicted state cannot be decoded into a realistic image.

**JEPA route:** instead of reconstructing noisy camera frames, predict target encoder embeddings of future observations conditioned on action, then use goal-relevant distances or reward/value heads to guide control. This requires anti-collapse and action/goal validation.

## Stage 7: decide whether it really works

Evaluate on new trajectories with hidden initial velocity, sensor noise, novel damping, and changed goals. Plot one-step and open-loop errors; measure final goal distance, velocity, action cost and unsafe failures. Compare to a Kalman filter plus model predictive control and a GRU baseline. If latent MSE improves but the cart stops worse, claim only improved latent loss—not better control.

**Final challenge:** Which observation/action/likelihood assumptions break if the cart's tires start slipping unpredictably? How would you change inference, dynamics uncertainty and control?
