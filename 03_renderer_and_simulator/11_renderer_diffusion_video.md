# 11 — Renderer world models: image diffusion to interactive video

## What is a renderer's contract?

Given state, observations, text, camera path, or hypothetical actions, synthesize what an observer would see. This is **observation generation**. An apparently real video is not proof of an accurate collision simulator, material model, or action-conditioned dynamics. Aman's [primer](https://aman.ai/primers/world-modeling/) uses 'renderer' as one of three functional roles.

## Diffusion from scratch: corrupt, then undo

A simple diffusion model defines a **forward noising process** on data `x_0`:

\[q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I).\]

Here `t` is a **diffusion noise index** from 1 to T, NOT the environment's physical time index. `α_t=1−β_t` and `\bar α_t=∏_{i=1}^t α_i` for noise schedule `β_t`.

One can sample directly:

\[x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad\epsilon\sim N(0,I).\]

A denoiser `ε_θ(x_t,t,c)` learns the added noise from the noisy input, noise step and conditioning `c` (e.g., text, prior frames, controls). Common loss:

\[\mathcal L_{noise}=\mathbb E_{x_0,\epsilon,t}\|\epsilon-\epsilon_\theta(x_t,t,c)\|_2^2.\]

Why does predicting noise help produce an image? Denoising gives a direction in data space toward cleaner, higher-likelihood samples under the learned reverse process. Generation starts at noise, iteratively applies a learned reverse update, and yields a plausible sample. The exact sampler and parameterization (epsilon, x0, velocity) vary.

## Latent diffusion

Raw megapixel denoising is costly. An image/video autoencoder maps `x_0` to a lower-dimensional latent `z_0`; diffusion works on `z`, and a decoder turns generated `z_0` back into pixels. Trade-off: compression saves compute but can discard precise details, including small features essential for control.

## How Transformers enter

A **Diffusion Transformer** (DiT) treats noisy latent patches as tokens, adds noise-step/conditioning information, and applies Transformer blocks to predict noise/velocity. You already know attention: the new idea is the **training target**, not a completely different attention operation.

For video, tokens include **space and time**. A full attention map over T×H×W patches is expensive: the cost of naive dense attention grows approximately quadratically in token count. Approaches include factorized spatial/temporal attention, causal frame chunks, local/window attention, temporal pooling and other compression.

## Video generation is harder

A video model must maintain: object identity; geometry/view consistency; occlusion persistence; plausible contact; accurate action response; long-range trajectory coherence. A frame that looks good in isolation can break any of these. A physically meaningful action-conditioned renderer should show different *valid* outcomes under different controls, not merely change style.

## Autoregressive versus parallel video generation

- **Autoregressive**: generate next tokens/frames/chunks given previous outputs; easy streaming but mistakes are fed back.
- **Diffusion over a clip**: denoise a block jointly, improving temporal consistency within the block; long episodes need stitching/conditioning.
- **Hybrid**: autoregressive latent dynamics for high-level change, diffusion decoder for detailed frames (a common design pattern).

### Conditioning and classifier-free guidance

A conditional denoiser can be evaluated with/without condition; combine:

\[\tilde\epsilon_\theta=\epsilon_\theta(x_t,t,\varnothing)+w[\epsilon_\theta(x_t,t,c)-\epsilon_\theta(x_t,t,\varnothing)].\]

Larger guidance scale `w` often increases condition adherence at the cost of diversity or sample quality. This is **generation guidance**, not MPC.

## Interactive world renderers

An interactive video generator receives actions repeatedly and predicts the next observation/chunk. Examples in the source primer include action-conditioned driving video systems, playable worlds from learned latent controls, and generative game environments. To qualify as useful **control simulation**, test whether the generator preserves the same hidden state through camera turns, keeps action effects persistent, handles multi-step feedback, and respects collisions. Rendering latency and controllability matter as much as aesthetics.

## Evaluate the right thing

Perceptual metrics (e.g., distributional realism) cannot alone test whether braking slows the vehicle. Evaluate rollout consistency, action adherence, geometric and physical plausibility, temporal identity, view consistency, and downstream agent performance. The **best-looking** generator need not be the best dynamics model.

**Check yourself:** Why can video generation be an excellent training-data generator yet a poor safety-critical digital twin? Why are diffusion noise steps and environment time steps different indices?
