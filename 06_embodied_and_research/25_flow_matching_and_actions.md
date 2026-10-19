# 25 — Flow matching, diffusion actions and predictive control

## Why use generative models for actions?

A robot can have several valid grasps or paths for the same instruction; an MSE action head averages these and may produce an awkward/invalid middle trajectory. Diffusion and flow-matching policies can model complex **conditional distributions over action chunks**, not only one deterministic action. Video-action systems sometimes generate visual and motor sequences together.

## Contrast two objectives

**Diffusion noise prediction:** sample a clean target `x_0`, noise it at index t and train a network to predict noise `ε` (see chapter 11).

**Flow matching:** define a probability path from simple base samples `x_0∼p_base` to data samples `x_1∼p_data`. For a simple conditional interpolation,

\[x_\tau=(1-\tau)x_0+\tau x_1,\qquad u_\tau=x_1-x_0,\quad \tau\sim U(0,1).\]

Learn a time-conditioned **vector field** `v_θ(x_τ,τ,c)` by squared error against desired velocity:

\[\mathcal L_{FM}=\mathbb E\|v_\theta(x_\tau,\tau,c)-u_\tau\|_2^2.\]

At inference, start from base sample and integrate ODE `dx/dτ=v_θ(x,τ,c)` toward a data-like sample. This linear pairing objective is a simplified example; practical flow matching uses different paths, couplings, solvers, and parameterizations. Distinguish **generation time τ** from **robot physical time t** and video frame index.

## Actions as generated trajectories

Let `A=(a_t,...,a_{t+K−1})` be an action chunk. An action-conditioned flow may model `p(A|o_{≤t},instruction)`. A world-action approach can couple it with predicted visual/semantic/latent future, e.g., model `p(A,Z_future|context)`. Training must define how the two modalities align in time, what is shared, and how predicted consequences constrain actions.

## What is action chunking for?

It amortizes expensive inference over K controls. But a robot usually commits only a prefix C before reobserving. Larger K can improve throughput and temporal consistency; smaller C improves responsiveness to unforeseen contact/failures. Report generation latency, physical control frequency, action chunk duration, commit duration, and effective throughput separately.

## Why this matters for FLUX 3 Action and modern VLAs

The primer highlights video-heavy pretraining, action midtraining across different embodiments, fine-tuning on a target robot, joint prediction of futures and actions, and reducing denoising/generation steps to meet robot latency constraints. The engineering problem is not merely to synthesize elegant futures: it is to obtain **feasible control commands quickly enough to act with feedback**. Flow-based action generation is an important alternative to autoregressive action-token prediction.

## Test the core hypothesis

Compare a deterministic action regressor vs a generative action-chunk model on truly multimodal demonstrations, holding data/encoder constant. Report mode coverage, collision/action feasibility, real task success, latency, and sensitivity to changed observations. A better action log-likelihood does not by itself imply better execution.

**Check yourself:** Why is generation-index `τ` in flow matching not interchangeable with physical time `t`? What practical trade-off appears when increasing denoising/integration steps for a robot control policy?
