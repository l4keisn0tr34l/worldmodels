# 13 — Object-centric models, graph networks, and learned physics

## Why objects and edges?

A huge image embedding may predict motion but doesn't naturally expose *which* objects are interacting. A graph explicitly represents **entities** as nodes and **interactions** as edges. Examples: two colliding balls, fingers gripping a cup, mesh vertices connected by springs, particles in fluid.

Represent graph at time t as `G_t=(V_t,E_t)`. Node `v_i` may include position, velocity, mass, type. Edge `e_ij` may include relative position, distance, joint constraint, or contact features.

## Message passing from first principles

For an edge `j→i`, compute a message:

\[m_{ij}=\phi_e(v_i,v_j,e_{ij}).\]

Aggregate neighboring effects for destination node i:

\[\bar m_i=\sum_{j\in\mathcal N(i)}m_{ij}.\]

Update node state / predict acceleration:

\[\Delta v_i=\phi_v(v_i,\bar m_i).\]

Repeated message passing rounds propagate influence over graph neighborhoods. The sum (or mean) aggregation is permutation-invariant: changing node order doesn't change the aggregated multiset. A node's dynamics may use external fields (gravity), boundary conditions, and control inputs as additional features.

## Example: two balls and a spring

Node A `[x_A,v_A,m_A]`, B `[x_B,v_B,m_B]`. Edge includes separation `d=x_B−x_A` and spring rest length `L₀`. Classical force is `F_A=k(||d||−L₀) d/||d||` if attached spring obeys Hooke's law. A graph neural net can approximate this mapping from relative geometry and learned parameters without explicitly writing Hooke's law, but should be evaluated against invariant properties and unseen system configurations.

## From interaction networks to learned simulators

- **Interaction Networks:** explicitly compute pairwise relational effects, aggregate, then update objects.
- **Visual Interaction Networks:** add a perception front-end that infers object states from images before relational rollout.
- **Graph Network-based Simulators (GNS):** represent particles as graph nodes, connect local neighbors, encode-process-decode and predict future dynamics.
- **MeshGraphNets:** represent physical meshes and use mesh connectivity plus proximity-based edges; can approximate numerical solvers for cloth/fluid/other PDE-like systems.

When rolling out, learned acceleration/velocity predicts next positions (e.g., Euler integration). Each step depends on previous predicted states, so geometric drift and accumulating energy error are common.

## Physics-aware losses and constraints

Data fitting is not always enough. Depending on application, add constraints for conservation of mass/momentum, nonpenetration, contact complementarity, positive mass, boundary conditions, invariances/equivariance, or known differentiable integrators. Not all physical systems conserve mechanical energy (friction dissipates it), so indiscriminate 'energy conservation' penalties can make the model *less* correct.

## Object-centric JEPA link

Instead of pixel-patch embeddings, an encoder can produce object slots `z_t^(i)`. The predictor learns what happens to each object given neighboring slots, actions, and time. Masking an object and recovering its latent from neighbors encourages relational prediction. But object masking alone does not establish true causal identification; interventions, occlusion tests and held-out interaction configurations remain necessary.

## Evaluation

Measure multi-step position/velocity errors, constraint violations, collision validity, energy behavior appropriate to system, mesh distortion, generalization to numbers of particles, changing boundary conditions, and task utility. A low mean vertex error can hide catastrophic physically impossible penetrations.

**Check yourself:** Why is permutation-equivariance useful for particle simulations? Why might a learned graph model generalize poorly to a new material even if it generalizes to more particles?
