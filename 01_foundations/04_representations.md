# 04 — Latent state, embeddings, and the information you must keep

## A latent state isn't just compression

An autoencoder maps an observation `o_t` to a smaller `z_t=E(o_t)` and decodes `D(z_t)≈o_t`. That encourages preservation of visible features. But world modeling cares about information that explains **future consequences**, which may be invisible in the current image. A blurred patch containing object velocity is more important for future collision prediction than exact wall texture.

Two observations that look different may be equivalent for a task. Two nearly identical images may encode distinct futures if one includes a tiny traffic signal or a hidden velocity difference.

## Invariance vs equivariance

A representation `f` is invariant to transformation `T` when `f(Tx)=f(x)`. It is equivariant when `f(Tx)=T' f(x)` for a corresponding feature transformation `T'`. For object recognition, illumination invariance may be good. For robot control, invariance to object position is disastrous: actions depend on where objects are. 'Invariant to nuisance, sensitive to dynamics' is a **task-specific assumption**, not universally given.

## Predictive sufficiency

A state representation `z_t` is predictively sufficient for a target if, for relevant future action sequences,

\[p(o_{t+1:t+H},r_{t+1:t+H}\mid o_{0:t},a_{0:t+H-1}) \approx p(o_{t+1:t+H},r_{t+1:t+H}\mid z_t,a_{t:t+H-1}).\]

Interpretation: once the latent state is known, looking again at the entire past should not materially change future predictions. Exact sufficiency may be impossible in partial observation; the target itself depends on use case. Task-dependent predictive states can be much smaller than perfect reconstructions.

## Bottlenecks and information

A bottleneck discourages passing all raw detail forward. A naive bottleneck `min ||z||²` can destroy information. An information bottleneck conceptually balances predictive usefulness against compressing input information:

\[\max_{f}\ I(Z;Y)-\beta I(Z;X).\]

Here `X` is past/context, `Y` a future target, and `I` is mutual information. This is a conceptual principle; a standard JEPA MSE loss **is not mathematically identical** to directly maximizing that information objective without additional assumptions.

## Pixel, object, geometry, belief or task-state?

| Latent form | Great for | Major limitation |
|---|---|---|
| Image patches | Flexible visual learning | Too many tokens; implicit physics |
| Whole-image vector | Compact classification | Can discard spatial relationships |
| Objects / slots | Contacts, interactions | Hard to detect, track, bind identities |
| Graph nodes/edges | Relational dynamics | Requires meaningful graph construction |
| 3D points/meshes/fields | View consistency, geometry | Often not complete physics |
| Probabilistic belief | Occlusion and ambiguity | Inference may be approximate/expensive |
| Task-oriented latent | Efficient planning | May ignore transferable world properties |

## Practical test for learned states

Don't assess embedding quality only by reconstruction. Hold out trajectories and probe: predict velocity, collision, reward, hidden-state variables; measure long-horizon errors; check whether planned actions transfer to the real/simulated environment; intervene on an action and ask if the latent future responds correctly.

**Check yourself:** Could a state have perfect image reconstruction but be a poor state for planning? Construct a two-frame example.
