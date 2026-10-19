# 16 — MuZero, TD-MPC, TD-MPC2, and task-oriented modeling

## Important reframe: simulate what is relevant to decisions

Must a planning model generate correct pixels? **No.** It may only need transitions that preserve reward, value, and action ranking. This is a different target from maximum-likelihood observation generation.

## MuZero: learned latent dynamics + tree search

MuZero uses three learned functions:

\[z_t^0=h_\theta(o_{\le t}),\quad(z_t^{k+1},\hat r_t^{k+1})=g_\theta(z_t^k,a^k),\quad(\hat p_t^k,\hat v_t^k)=f_\theta(z_t^k).\]

- `h`: representation of current observation history.
- `g`: hypothetical action → next hidden state + predicted immediate reward.
- `f`: hidden state → action prior logits + value estimate.

It searches hypothetical action branches with Monte Carlo Tree Search (MCTS). The network is trained to match reward, value and improved search-policy targets in latent rollouts, without needing pixel reconstruction. MCTS explores the *learned* model, not the real environment for every candidate. Its states need not decode into interpretable physical states.

**MCTS sketch:** from root, repeatedly select actions trading predicted quality and exploration (`Q+U`), expand a latent successor using `g`, get value/policy from `f`, backpropagate value up the visited path. Action is often chosen from root visit counts. Tree-search hyperparameters strongly affect behavior.

## TD-MPC: short-horizon latent control with value learning

Temporal Difference Model Predictive Control learns an encoder, latent dynamics, reward, Q/value and policy prior. It runs continuous-control trajectory optimization at decision time. A finite rollout score includes predicted rewards plus terminal `Q`/value, allowing shorter simulated horizons. It is deliberately **task-oriented**; pixel reconstruction isn't essential.

**Policy prior** seeds proposals in promising parts of continuous action space. **Temporal difference (TD)** learning teaches values from observed transitions using bootstrap targets, not only fully simulated rewards. Because TD value estimates can compensate for limited model horizon, the method spends less time modeling irrelevant detail.

## TD-MPC2

TD-MPC2 extends this into a scalable, more robust recipe for multitask continuous control; includes normalized latent representations (such as SimNorm), task conditioning, modernized reward/value objectives and shared architecture choices. Its central principle remains **learn task-relevant compact dynamics, plan short horizon, bootstrap with value**. [Paper](https://arxiv.org/abs/2310.16828).

## Compare what each architecture optimizes

| Agent | Model predicts | Action mechanism | Can omit images in future? |
|---|---|---|---|
| Ha–Schmidhuber | next compressed visual state | small controller | Yes |
| PlaNet | RSSM latent state, observation, reward | online CEM | Yes at plan time |
| Dreamer | RSSM latent dynamics, observation, reward | actor trained in imagination | Yes |
| MuZero | abstract latent successor, reward, policy/value | MCTS | Yes, intentionally |
| TD-MPC2 | task-relevant latent dynamics, reward and values | sampling-based MPC | Yes, intentionally |

## The trap of comparing only predictive accuracy

MuZero/TD-MPC's latent isn't guaranteed to correspond to a particular physical coordinate system. A latent may support high action value without explaining images or becoming transferrable to new tasks. Conversely, a beautiful renderer may be less useful for action choice than a small, ugly state predictor. **Report closed-loop performance** for planner families.

**Check yourself:** Why is a 'perfectly rendered' simulator not automatically more useful than a task-oriented MuZero-style dynamics model? Why can a short-horizon TD-MPC planner still consider distant consequences?
