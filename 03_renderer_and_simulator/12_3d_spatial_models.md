# 12 — Spatial state: NeRF, 3D Gaussians, geometry, and what they omit

## Why model 3D at all?

In 2D video, the same object can look very different from different camera views. A geometrically consistent representation allows rendering new viewpoints, estimating occupied space, locating objects, and perhaps editing structure. But a static scene representation does **not** automatically know contact, dynamics, mass, friction, or how actions change the world.

## The geometric basics

A 3D point in world coordinates `X_w=(x,y,z)` maps to camera coordinates `X_c=RX_w+t`, and to image pixel homogeneous coordinates `u~K X_c`, where `R` is rotation, `t` translation and `K` camera intrinsics. Divide by the third coordinate to get pixel coordinates. Multi-view consistency requires calibrated cameras and correct poses (or jointly learned estimates). Depth ambiguity is a fundamental reason a single image is not a full 3D state.

## NeRF: learn a continuous scene function

A **Neural Radiance Field** learns a mapping of position and view direction to **density** and **view-dependent color**:

\[F_\theta(\mathbf x,\mathbf d)=(\sigma,\mathbf c).\]

To render a pixel, cast a ray `r(t)=o+td` from camera through that pixel. Sample positions along it. At each sample, density `σ` controls how likely light is blocked and color `c` contributes to the pixel. Conceptual volume-rendering equation:

\[C(r)=\int_{t_n}^{t_f} T(t)\sigma(r(t))c(r(t),d)dt,\quad T(t)=\exp\left(-\int_{t_n}^t\sigma(r(s))ds\right).\]

`T(t)` is light transmittance: the fraction surviving attenuation from the camera to t. Because rendering is differentiable, compare generated pixel colors against posed reference images and update network weights. Positional encoding helps represent fine detail (similar high-frequency motivation to Fourier features).

**Capability:** novel-view synthesis of a learned scene. **Not guaranteed:** correct geometry everywhere, editability, collision behavior, temporal dynamics, physically plausible material properties.

## 3D Gaussian Splatting: explicit spatial primitives

Represent a scene as many anisotropic (directionally stretched) 3D Gaussians. Each primitive contains center `μ_i`, covariance `Σ_i`, opacity `α_i`, and appearance/color parameters. Project Gaussians into camera space and alpha-composite their contributions in depth order:

\[C=\sum_i T_i\alpha_i c_i.\]

This is not the same as a normal-distribution belief about uncertain world state. Here Gaussians are **geometric rendering primitives**; covariance describes each splat's spatial footprint. 3DGS can be fast to render and comparatively explicit, but a rendered collection of splats is not automatically a watertight rigid-body mesh suitable for precise collision detection.

## Other state representations

| Representation | Contents | Strength | Typical weakness |
|---|---|---|---|
| Point cloud | 3D samples | Sensor-friendly | No direct surface topology |
| Voxel grid | Occupancy at cells | Simple collision/space | Memory-heavy at high resolution |
| Mesh | Vertices/faces | Surface topology | Hard remeshing / topology changes |
| Signed distance field | Distance to surface | Collision/query geometry | Difficult with dynamics/materials |
| NeRF | Continuous radiance/density | Novel views | Geometry not always physically valid |
| 3DGS | Spatial Gaussian primitives | Fast visual rendering | Physical constraints not native |
| Object/scene graph | Entities + relations | Editable semantics | Requires correct detections and binding |

## Bridging rendering, state and physical simulation

A system may reconstruct a scene with NeRF/3DGS, infer objects/materials/collision meshes, then run an external dynamics engine for actions. This is an engineering pipeline; the physical consistency does not automatically emerge from radiance reconstruction. 'Text-to-3D' similarly generates plausible assets, not validated physics.

**Check yourself:** A NeRF gives perfect images from every camera but objects pass through walls when pushed. Did it fail as a renderer, spatial representation, or dynamics simulator? Explain which contract was tested.
