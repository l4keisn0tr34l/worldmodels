# 01 — Notation and mental translations

These symbols are reused throughout. Different papers overload them, so **always read each chapter's definitions**.

| Symbol | Meaning | Shape / conceptual type |
|---|---|---|
| `t` | Real environment time step | Integer |
| `s_t` | True underlying state | Usually hidden; arbitrary structure |
| `o_t` | Observation (image, log, sensor reading) | Vector, image, multimodal |
| `a_t` | Agent action in interval `t→t+1` | Discrete or continuous |
| `r_t` | Reward for a transition, convention defined locally | Scalar |
| `z_t` | Learned stochastic / latent state | Vector or categorical variables |
| `h_t` | Deterministic recurrent memory state | RNN hidden vector |
| `b_t(s)` | Belief distribution `P(s_t=s | history)` | Distribution over possible states |
| `π(a|z)` | Policy selecting actions | Distribution |
| `V(z)` | Expected discounted future return | Scalar |
| `Q(z,a)` | Expected future return after first action `a` | Scalar |
| `γ` | Discount in [0,1] | Scalar |
| `H` | Planning/imagination horizon | Number of future transitions |
| `p_θ` | Learned generative / prior distribution | Normalized density or mass function |
| `q_φ` | Approximate posterior / encoder distribution | Normalized density or mass function |
| `f_θ` | Generic neural mapping | Depends on usage |
| `sg(x)` | Stop gradient | Forward value is x; derivative is 0 |
| `KL(q||p)` | Divergence, asymmetric | Nonnegative scalar |
| `E_q[f]` | Expected value under distribution q | Integral / sum |

**Timeline convention:** At `t`, the agent has received `o_t`. It uses current belief to choose `a_t`. The environment produces `s_{t+1}`, and then `o_{t+1}` arrives. Some RL papers write `r_t` for reward after `a_t`; others use `r_{t+1}`. Always check the indexing.

## Translate familiar RNN language

You know `h_t=GRU(h_{t-1},x_t)`. Here, `x_t` may concatenate the previous latent state and previous action. Thus `h_t` represents **accumulated dynamical context**, not 'the neural layers in between'. Its *value* changes over time; the GRU *parameters* are shared over time.

Likewise, a Transformer can consume tokens `(o_1,a_1,o_2,a_2,...)` or compressed patch tokens; masking determines which evidence can influence predictions. The same attention operation can serve prediction, encoding, or policy generation depending on inputs, targets, and loss.

## Three distinctions that save hours of confusion

1. **Latent is not necessarily hidden physical truth.** An embedding can be predictive without corresponding one-to-one to real-world variables.
2. **Sampling is not inference.** Generating from a prior is different from inferring a posterior conditioned on the current observation.
3. **Prediction horizon ≠ action chunk ≠ commit horizon.** A robot can predict 40 future frames, generate 16 actions, and execute only 4 before re-observing.

**Check yourself:** Explain the difference between changing `θ` during training and changing `h_t` during a rollout.
