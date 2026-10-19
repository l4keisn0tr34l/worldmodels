# Detailed derivations and pencil-and-paper drills

## D1: Bayesian update with two states

Set `S={A,B}`, initial `b=(0.7,0.3)`. Transition matrix (old row → next column): `[[0.9,0.1],[0.4,0.6]]`. Sensor emits `y` with `P(y|A)=0.2`, `P(y|B)=0.8`. Compute prediction `b^-`, posterior `b^+`, and verify posterior sums to 1. Then repeat assuming no sensor arrives. State exactly where action would enter the transition matrix.

## D2: Bellman identity

Starting from `V^π(s)=E[r_(t+1)+γr_(t+2)+γ²r_(t+3)+...|s_t=s]`, isolate first reward and show the remaining sum equals `γ E[V^π(s_(t+1))]` when state is Markov. Write its sum-over-actions-and-successors form.

## D3: ELBO step by step

Start `p_θ(x)=∫p_θ(x,z)dz`. Multiply/divide by `q_φ(z|x)`. Take log, apply Jensen, factor `p(x,z)=p(x|z)p(z)`. Show `log p(x)=ELBO + KL(q(z|x)||p(z|x))` and explain why ELBO ≤ log evidence. What makes the posterior truly equal to prior?

## D4: two equally likely futures

`Y∈{−2,+2}` equally likely. Minimize `E[(Y−ŷ)²]` with respect to deterministic ŷ; compute the answer and its loss. Compare to a two-atom probability model that assigns 0.5 to each physically possible outcome. Why does low point-loss fail to preserve modes?

## D5: multi-step error

Assume `||f(s,a)−F(s,a)||≤ε` on a relevant region, and learned function F and true f are Lipschitz in state with factor L. Let `e_{k+1}≤L e_k+ε` starting `e_0=0`. Unroll for k=1...H and derive geometric series. Discuss L<1, L=1, L>1. What assumptions are unrealistic in a changing policy's OOD states?

## D6: CEM by hand

Four one-dimensional action candidates have predicted returns `[(−1,0),(−0.3,2),(0.2,4),(0.7,3)]` where pairs are `(action,return)`. Take top two elite returns. Calculate new mean and sample std of their actions; explain why CEM is derivative-free. How would clipping to action bounds affect elite statistics?

## D7: λ returns

At t=0: `r1=2,r2=4`, `γ=0.9`, continuation all 1, `V(s1)=3,V(s2)=5`. Compute 1-step and 2-step returns: `G^(1)=r1+γV(s1)`, `G^(2)=r1+γr2+γ²V(s2)`. Combine with finite `λ=0.5` via `(1−λ)G^(1)+λG^(2)`. What changes if trajectory terminates at t=1?

## D8: collapsed embeddings and VICReg

Given 3 samples with feature embeddings `[1,1]`, `[1,1]`, `[1,1]`, calculate variance and explain why invariance-only matching can be zero. Repeat with `[1,1]`, `[2,2]`, `[3,3]`: each dimension varies, but explain off-diagonal covariance. Which regularizer addresses each case?

## D9: action identifiability

Two hidden states look identical, but in one state braking reduces speed and in the other braking has no effect due to ice. If the observation history cannot distinguish the surfaces, can any deterministic point predictor produce correct outcomes for both? State a meaningful probabilistic prediction and an action/info-gathering strategy.

## D10: evaluation design

Propose train/val/test splits for a dataset of 10,000 correlated 1-second sensor windows drawn from 50 long real trajectories. Explain why shuffled window split is weak. Define one-step forecasting, 20-step rollout and action-conditioned intervention test separately. Which metric is not available if the dataset never varied action choices under comparable states?
