# 00 — What is a world model, really?

## Begin without neural networks

Imagine you're controlling a small robot in a room. You see a ball, a door, and part of a table. If you push the ball, you want to know where it will go. But you don't see the ball's mass, friction coefficient, or what is behind the table. You need **an internal hypothesis about hidden reality** and **a rule for how it changes when acted on**.

In a fully specified system, let `s_t` be the physical state, `a_t` the chosen action, and `o_t` the observation. The environment obeys a transition distribution `p(s_{t+1} | s_t, a_t)` and an observation distribution `p(o_t | s_t)`. You do not know either exactly. A learned model tries to approximate useful parts of these maps from experience.

The state is *not* necessarily the pixels. The world continues existing when the camera turns away. That is the distinction between **what exists** and **what is observed**.

## The agent loop

```text
                hidden environment state s_t
                   /                \
     observation p(o_t|s_t)      transition p(s_{t+1}|s_t,a_t)
               |                         ^
               v                         |
       observation o_t -> belief b_t -> choose action a_t
                                |          |
                         imagine futures -+
```

Most actual systems use a latent vector `z_t` learned from observations instead of known physical `s_t`. A minimal learned predictor is

\[\hat z_{t+1}=f_\theta(z_t,a_t).\]

This is useful but incomplete: is `z` informative, does it retain history, does it represent uncertainty, and can the planner exploit its errors? Those are the rest of the course.

## Predicting vs understanding vs acting

- **Recognition**: from an image, classify 'ball'. No temporal model required.
- **Forecasting**: from several images, predict next frame. No ability to intervene necessarily implied.
- **Action-conditioned prediction**: predict what changes under `push-left` vs `push-right`. Stronger for control.
- **Planning**: compare such counterfactual rollouts against a goal and choose an action.
- **A policy**: directly map belief/observation to action; it may be trained using a model, but need not call it at deployment.

A Transformer performing next-token prediction is a predictive sequence model. Is it a *world model*? That depends on whether its representation and transition semantics support the environment and use case; predicting text is not sufficient evidence of physically faithful counterfactual dynamics.

## Four kinds of systems, two taxonomies

Aman's primer uses the **functional taxonomy**:

| Output contract | Question | Example | What can go wrong? |
|---|---|---|---|
| Renderer | 'What might I see?' | Video diffusion | Beautiful but impossible physics |
| Simulator | 'How does the state change?' | Graph dynamics | Accurate visuals not guaranteed |
| Planner | 'What should I do?' | Latent MPC | Exploits incorrect transition model |
| JEPA-style predictive learner | 'What hidden features follow from context?' | I-JEPA / V-JEPA | Collapsed embeddings or poor control relevance |

The first three classify **system output**; the fourth describes a **representation-learning objective/architecture**, so it is intentionally not a mutually exclusive fourth category. A JEPA predictor may be used as a simulator; a planner may use a renderer; a renderer may contain a dynamics model.

A different, orthogonal classification: **deterministic vs probabilistic; image vs latent vs object vs geometry state; passive vs action-conditioned; online planning vs learned policy; model-based vs model-free**. Never infer one axis from another.

## One concrete comparison

A person throws a ball toward a wall.

- Renderer: generates a plausible-looking video of the throw and bounce.
- State simulator: predicts 3D position, velocity, rotation, collision, maybe uncertainty.
- Planner: selects a throwing angle to hit a target after the bounce.
- JEPA encoder/predictor: represents the video and predicts future embeddings, perhaps without generating the video.

**Check yourself:** A video model accurately predicts the next frame but responds identically whether the controller passes `steer-left` or `steer-right`. Is it ready for planning? Why not?
