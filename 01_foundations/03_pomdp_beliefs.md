# 03 — Partial observability, POMDPs, and Bayesian filtering

## Why your observation is not the state

A mobile robot sees an obstacle at time `t`. Next frame the obstacle is hidden behind a parked vehicle. If you condition only on current pixels, you may 'forget' it exists. The world remains Markov at the physical-state level, but the observed process is not necessarily Markov.

A **POMDP** adds an observation space `O` and observation model `Ω(o|s)` to the MDP. Instead of observing `s_t`, the agent obtains `o_t` drawn from `Ω`. Its sufficient statistic can be the belief:

\[b_t(s)=p(s_t=s\mid o_{0:t},a_{0:t-1}).\]

The complete interaction history becomes unwieldy. A belief compresses that history into the current distribution over possible states (under the assumed model). A recurrent hidden vector is *an approximation to a useful sufficient statistic*, not automatically a calibrated Bayesian belief.

## Derive the filter, step by step

Suppose we have `b_t(s)` and then choose action `a_t`. First predict next state before seeing the next observation:

\[b^-_{t+1}(s')=\sum_s P(s'\mid s,a_t)b_t(s).\]

That is the **prediction** or **time update**. Next observe `o_{t+1}` and apply Bayes:

\[b_{t+1}(s')=\frac{\Omega(o_{t+1}\mid s')b^-_{t+1}(s')}{\sum_u\Omega(o_{t+1}\mid u)b^-_{t+1}(u)}.\]

That is the **correction/measurement update**. For continuous state use an integral instead of a sum.

### Worked numeric filter

Let hidden location `L` or `R`, prior belief `(0.6,0.4)`. Action 'stay' has transition matrix (rows old state, columns new):

```text
        next L   next R
old L    0.8      0.2
old R    0.3      0.7
```

Prediction: `b^-(L)=0.6*0.8+0.4*0.3=0.60`; `b^-(R)=0.40`. A sensor then says 'left'. Likelihood `P(obs=left|L)=0.9`, `P(obs=left|R)=0.2`.

Unnormalized posterior `(0.54,0.08)`; normalize by `0.62`: posterior `(0.87097,0.12903)`. Notice how the new observation improves confidence without making it 100%.

## Filtering, smoothing, prediction

- **Filtering:** estimate state at `t` using observations through `t` only.
- **Smoothing:** estimate past state at `t` using future observations as well (offline training can do this; online action cannot).
- **Prediction:** infer states after `t` without seeing those future observations.

Using future observations in an online controller is **data leakage**. RSSM training can compute per-time posterior conditioned on current observations; future imagination must rely on prior transitions without future observations.

## Aleatoric vs epistemic uncertainty

**Aleatoric**: irreducible stochasticity or inherent ambiguity given available inputs (a person might turn either way). **Epistemic**: uncertainty in parameters/model because of insufficient knowledge or out-of-distribution inputs (never trained on wet roads). Both matter; a softmax score is not automatically a clean decomposition. Deep ensembles often help approximate epistemic variation, while probabilistic output models express conditional outcome spread.

**Check yourself:** Why does `GRU(o_t)` plus an MLP not necessarily produce a valid probabilistic belief, even though it encodes history?
