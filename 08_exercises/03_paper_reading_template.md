# Paper reading and explanation template

Copy this to a new Markdown file whenever reading a world-model paper. Try to answer in your own words before consulting secondary explanations.

## Paper

- Title, authors, date, link:
- One-sentence problem:
- What environment or data are modeled?
- What is observed, hidden, acted on and rewarded?
- What is the inference-time output contract: render, simulate, plan, policy, or combination?

## Model

- State representation and shape:
- Encoder / feature extractor:
- Transition function (conditioned on what?):
- Prior / posterior, if stochastic:
- Observation decoder? reward head? continuation head?
- Actor / critic / search algorithm?
- Which modules use RNNs, Transformers, attention or graphs? Why?

## Objective and experiment

- Write the loss precisely with defined random variables and gradient paths.
- Which targets are raw observations vs learned embeddings vs rewards/values?
- What would be a trivial/degenerate solution? How is it avoided?
- Data collection policy and source of action labels:
- Train/validation/test split and potential leakage:
- Baselines and matched compute/interaction budgets:
- Primary improvement and largest meaningful limitation:
- Does closed-loop evaluation support the headline claim?

## Research defense

- Strongest assumption:
- One plausible alternative explanation for results:
- Ablation that would distinguish explanations:
- Evidence vs hypothesis vs speculation:
- What would falsify this result in a simpler setup?
- A 60-second explanation to a peer:

## Glossary of uncertainties to flag

Use the phrases **verified by experiment**, **proposed**, **mathematically derived under assumptions**, **architectural intuition**, and **not established by this paper**. Do not slide between them.
