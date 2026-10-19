# Glossary — quick retrieval (terms linked conceptually to chapters)

**Action:** control/intervention selected by an agent; can be discrete command, force, velocity, pose or action chunk.

**Action chunk:** multiple consecutive future motor commands produced in one inference.

**Action conditioning:** using a candidate action to predict resulting future; does not prove causal identifiability from confounded logs.

**Action horizon:** number of future actions generated. Not the number executed before feedback.

**Actor:** policy neural network producing actions.

**Aleatoric uncertainty:** uncertainty due to stochastic/ambiguous outcomes, given information and model assumptions.

**Anti-collapse mechanism:** architecture/regularizer preventing every sample embedding from becoming constant or too low-rank.

**Belief state:** distribution over hidden states conditioned on observation/action history.

**Bellman equation:** recursion connecting current value, immediate reward and future value.

**Bootstrap:** estimating a truncated future return with a value function instead of waiting for all rewards.

**Categorical latent:** stochastic latent represented by a discrete categorical distribution, sometimes as a one-hot vector.

**Causal intervention:** manipulation of action/variable, conceptually `do(a)`; different from observing naturally occurring action values.

**CEM:** Cross-Entropy Method, derivative-free trajectory optimizer by sampling and elite refitting; not classification cross entropy.

**Commit horizon:** how much of predicted action sequence is executed before sensing/replanning.

**Continuation head:** predicts whether an episode remains active, controlling value bootstrapping.

**Counterfactual:** alternative future under a hypothetical intervention; observational forecasting may not identify true counterfactual effects.

**Covariance:** statistical co-variation of random features, or spatial primitive shape in 3DGS depending on context.

**Critic:** network estimating value/expected return under policy.

**Decoder / renderer:** mapping latent/scene state to observations such as pixels.

**Diffusion model:** generative model often trained to reverse a progressive data corruption process.

**DiT:** Diffusion Transformer, attention-based denoising backbone over noisy latent patches.

**Dreamer:** learns an RSSM and trains an actor/critic on imagined latent trajectories.

**Dynamics:** rule/distribution describing how state changes under actions.

**ELBO:** evidence lower bound on log marginal likelihood, combining expected conditional likelihood and KL to prior.

**EMA teacher:** target encoder weights updated as slow moving average of online weights.

**Embedding:** learned vector/token representation; not necessarily interpretable physical state.

**Epistemic uncertainty:** uncertainty about model/parameters due to limited knowledge; difficult OOD.

**Equivariance:** changing input under transformation induces a predictable representation transformation, rather than no change.

**Flow matching:** train a time-dependent vector field transporting a simple distribution toward a data distribution.

**Free rollout/open-loop:** predict multiple steps using model-generated states rather than correcting with real future observations.

**Free bits:** KL regularization technique that avoids aggressively suppressing useful latent information below a threshold; exact variants differ.

**Graph network simulator:** predicts dynamics via node/edge representations and message passing.

**I-JEPA:** image masked-region representation prediction; not inherently an action-conditioned temporal model.

**Imagined trajectory:** future simulated inside learned dynamics given hypothetical actions.

**Inference posterior:** approximate distribution conditioned on new observation (RSSM `q`).

**JEPA:** Joint-Embedding Predictive Architecture; predict target representations rather than reconstruct raw signals.

**KL divergence:** asymmetric measure `E_q log(q/p)` comparing two distributions.

**Latent:** compressed/hidden learned representation; may not be directly observable or physically identifiable.

**Latent overshooting:** multi-step regularization encouraging long-range model prior predictions to agree with inferred state targets.

**Likelihood:** probability density/mass of observed signal under a specified model/hypothesis.

**Markov state:** representation where history offers no extra information for predicting next state beyond current state and action.

**MCTS:** Monte Carlo Tree Search, explores hypothetical action branches using estimates and visit statistics.

**MDP:** Markov Decision Process: states/actions/transitions/rewards/discount.

**Model bias:** decision-relevant error from an approximate dynamics/reward model.

**Model exploitation:** optimizer picks actions that appear good only due to model inaccuracies.

**MPC:** Model Predictive Control: plan, execute prefix, observe, replan.

**NeRF:** Neural Radiance Field, learned volumetric function from 3D point/view direction to density/color.

**Observation model:** probability of sensory observation given hidden state.

**PETS:** probabilistic-ensemble model-based RL with trajectory sampling.

**PlaNet:** learned latent dynamics from pixels plus online MPC/CEM.

**POMDP:** partially observable MDP with observation model; belief may summarize past.

**Policy:** distribution/function selecting actions from state, observation/history, or goal.

**Posterior collapse:** VAE-style failure where latent posterior moves toward prior and decoder ignores latent.

**Predictive sufficiency:** representation retains necessary information about selected future targets under future actions.

**Prior dynamics:** distribution over next latent before observing the next frame; used during imagination.

**Reparameterization:** differentiable sample construction (e.g. z=μ+σε) for gradient training.

**Renderer model:** primarily produces observations/images/video; fidelity may be appearance-focused.

**Representation collapse:** embedding learns constant or low-rank output across different data.

**Reward:** scalar outcome shaping objective, not the same as observation likelihood.

**RSSM:** Recurrent State-Space Model, typically combines deterministic recurrent memory and stochastic latent state with prior/posterior.

**Simulator:** transition/state model that represents how the environment changes; may be explicit physics or learned latent.

**SIGReg:** projected distribution-matching regularizer used in LeJEPA to encourage isotropic Gaussian feature distributions.

**Stop-gradient:** operation that passes values but blocks gradients along one computational path.

**Teacher forcing:** training multi-step model with true/interpreted previous states as input rather than its own predictions.

**TD learning:** temporal-difference learning uses observed reward plus bootstrapped next-state value.

**TD-MPC2:** continuous-control latent model predictive control with task-conditioned and stabilized training architecture.

**3D Gaussian Splatting:** explicit 3D Gaussian primitives optimized to synthesize views.

**V-JEPA:** video JEPA learning representation of masked spatiotemporal regions.

**VLA:** vision-language-action policy from observations and instruction to motor actions.

**WAM:** world action model combining prediction of future world state with action generation.

**World model:** predictive internal model of an environment, especially state transitions under actions; representations/output types vary.
