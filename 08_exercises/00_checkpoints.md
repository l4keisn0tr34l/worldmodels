# Self-tests by stage — no peeking at answers first

These questions are designed to expose weak understanding. Write your reasoning, not only a term definition. Suggested solutions are in `02_solutions.md`.

## Foundation: probability, MDP, partial observation

1. A hidden state is either A or B, prior `(0.2,0.8)`. A sensor reports '+'. Its likelihoods are `P(+|A)=0.8`, `P(+|B)=0.1`. Compute `P(A|+)`.
2. An agent sees a stationary image of a rolling ball. Why can't current pixels reliably determine the next position?
3. Explain the Markov property. Is it a property of *a chosen state representation*, *the laws of nature*, or both?
4. Suppose an action leads to reward 0 then reward 10 on the next step. Another leads to reward 8 and terminal. With `γ=0.6`, which is better, all else equal?
5. What does an observation model `p(o|s)` express? Distinguish it from `p(s'|s,a)`.
6. Why does `p(a|s)` not prove `p(s'|s,a)=p(s'|s,do(a))` in logged data?

## Latents, ELBO, RSSM

7. Why is a good reconstruction embedding not necessarily useful for control?
8. Write a VAE loss from a generative model `p(z)p(x|z)`, identify likelihood and KL.
9. Under a standard normal prior, compute KL of `q=N(1,1)` (one dimension).
10. Starting with `h_(t−1),z_(t−1),a_(t−1)`, write the RSSM update, prior, posterior and one reconstruction head.
11. In imagination at time t+3, can the model condition its latent sample on true `o_(t+3)`? Why?
12. Why can an RSSM use both recurrent and stochastic state rather than a GRU alone?
13. Explain posterior collapse vs representation collapse.
14. Why can 1-step MSE be excellent while a 50-step rollout fails?

## Rendering and simulators

15. Explain diffusion forward noising and why reverse denoising produces samples.
16. Distinguish diffusion noise time from video physical time.
17. Why can low video generation error still miss action-dependent physical dynamics?
18. What does NeRF's density mean? How does transmittance affect pixel color?
19. Distinguish a 3DGS primitive's covariance from a Bayesian covariance about a robot's state.
20. Write a graph message passing update and explain why summation is permutation invariant.

## Planning and learning in dreams

21. Describe one MPC cycle including which prefix is executed.
22. Why might longer horizon worsen action choice with a learned model?
23. Differentiate CEM and MCTS in action space and computation.
24. What does a critic value estimate help with when planning horizon is short?
25. Dreamer actor training begins from an RSSM posterior seed; how are the future states generated?
26. What does a λ-return balance? How does terminal probability affect bootstrapping?
27. How does MuZero differ from a pixel-level renderer during planning?
28. Explain model exploitation with a concrete invented error.

## JEPA and embodied systems

29. Why does MSE between trainable embeddings admit a constant solution?
30. In I-JEPA, why do target-location tokens and target-block size matter?
31. What changes between I-JEPA and V-JEPA? What is *still missing* for controllable simulation?
32. Explain the role of EMA target encoder without claiming it mathematically guarantees noncollapse.
33. Distinguish VICReg variance regularizer from covariance regularizer.
34. Does JEPA optimize mutual information explicitly? Under what caveats might that framing be useful?
35. State output contracts for VLA vs WM vs WAM.
36. Why must a model explicitly define H (predicted world), K (actions predicted), and C (actions committed)?
37. Flow-matching interpolation index vs physical time: why distinguish them?
38. What is missing when videos predict semantic 'grasp success' but real grasps slip?
39. Construct an experiment comparing action-free video pretraining vs no pretraining while avoiding dataset budget confounds.
40. Which measures would falsify a claim of causal counterfactual forecasting from network telemetry alone?

## Mastery task

Design an action-conditioned world model for a partially observed environment. Define hidden state, observations, actions, encoder, prior/posterior, loss, rollout, planner, uncertainty strategy, split and validation. Explain a failure mode *before* claiming results. If you cannot specify where future observation leakage would occur, revisit RSSM and filtering.
