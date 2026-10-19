# 24 — Build your own research project: three defensible blueprints

Each proposal deliberately has a falsifiable hypothesis, baseline and measurable outcome. Do not call proposed components 'implemented' until they run and pass tests.

## Project A: learned dynamics + model predictive control

**Question:** Does modeling uncertainty improve closed-loop navigation under hidden motion? **Environment:** gridworld or 2D physics with moving obstacles, noisy partial observations, a few discrete/continuous actions. **Models:** constant-velocity, GRU point prediction, ensemble probabilistic dynamics. **Control:** CEM or discrete shooting MPC. **Primary outcome:** collision rate, success rate, calibration, out-of-distribution changes in obstacle speed; plot return as a function of horizon. **Key ablation:** identical planner without ensemble penalty vs with it. **Failure to look for:** ensemble overconfidence on wholly new physics.

## Project B: JEPA representation versus reconstruction

**Question:** Does latent predictive training preserve controlled motion more efficiently than reconstructing frames under distractor variation? **Environment:** moving shapes with changing textures; actions affect object velocity but not background texture. **Models:** matched-capacity next-frame AE/decoder, simple action-conditioned JEPA with anti-collapse, supervised latent state baseline. **Metrics:** linear probe velocity error, held-out action response, embedding variance/effective rank, OOD background changes, closed-loop target reaching. **Important fairness:** same observations/action budget, matched encoder capacity, explicit parameter counts, multiple seeds.

## Project C: predictive network security world model (careful framing)

**Question:** Can a dynamics model forecast future network state better than a recurrent forecaster, and can calibrated uncertainty improve defensive triage? **States:** aggregates of authentication events, host connection counts, alert classes, known topology; edges form a graph. **Actions:** include only *logged defender interventions* if they truly exist; absent interventions, this is forecasting rather than verified counterfactual defense. **Baselines:** last-value, linear dynamics, GRU/Transformer, graph sequence model. **Metrics:** time-split forecast accuracy by horizon, false positive rates, incident-class recall, reliability diagrams, decision time and constraint violations. **Research boundary:** you can forecast attacker progression from observational telemetry, but cannot claim an RL defender's interventions work unless they are tested causally in a controlled simulator or validated rollout setup. Explicitly distinguish offline evaluation from real blocking.

## Protocol for all three

1. Define question, desired failure, outcome and unit of analysis **before** training.
2. Implement minimal baseline, make sure data/action times are aligned.
3. Define disjoint train/validation/sealed-test splits (held-out trajectory/environment/time).
4. Fit dynamics and plot **open-loop** and **teacher-forced** losses separately.
5. Add planner/actor and report **closed-loop** outcome with controlled seeds.
6. Ablate one design choice at a time; report compute and action/data budget.
7. Keep a journal of mistaken assumptions and failure cases.

## How to defend a research choice aloud

> I chose an action-conditioned latent transition because my target is multi-step decision support, not photorealistic generation. My baseline is a GRU dynamics predictor; I measure error by horizon and the true closed-loop objective. I do not equate observational correlation with validated intervention effects. The main hypothesis is that [specific mechanism] improves [specific metric] under [specific held-out condition], and a negative result would be [specific falsifier].

**Check yourself:** Write the null hypothesis and most damaging dataset leakage risk for each proposed project.
