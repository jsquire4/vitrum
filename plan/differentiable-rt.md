# Differentiable Ray Tracing — Inverse Rendering ("reference image → matching 3D scene")

> **Status: implemented with an explicitly narrow adjoint scope.** Finite-difference
> sessions ship on pt-webgl2 and pt-webgpu. pt-webgpu path replay is certified only
> for camera-visible material `emissive` under the session's one-bounce compatibility
> checks. Every other parameter, including `baseColor` and `roughness`, resolves to
> finite differences with a structured diagnostic.

## What this actually is

**Input:** a reference image (photo, render, stained-glass swatch, product shot).
**Output:** a 3D scene that, when rendered, looks like that image.

Differentiable RT is **not** the whole "image → 3D" product. It is the *differentiable
simulator in the loop* that turns "create this scene" from guess-and-check into gradient
descent — once you already have something to optimize. vitrum has the forward path
(scene → BVH → path trace → image); diff-RT adds the backward path (which pixels are
sensitive to which material texels / lights, and by how much).

## The three tiers (easy-ish → genuinely hard)

| Tier | Given upfront | Optimize | Example |
|------|---------------|----------|---------|
| **A — Material / lighting fit** | Geometry + camera fixed | Textures, BSDF params, light pos/intensity | "This Cornell mesh should look like this photo" |
| **B — Parametric scene fit** | A template scene (panel shape, frame, sun angle as knobs) | Continuous params + textures | "This stained-glass layout should match this reference" |
| **C — Full image → 3D** | Just the image | Geometry, materials, lighting, maybe camera | "Here's a photo of a room — build the scene" |

Diff-RT is the engine for **A and B**. **C** needs A/B *plus* a geometry-proposal stage
(ML reconstruction, multi-view, user scaffolding, or a generative 3D model) — that is an
ecosystem problem, not a path-tracer problem. vitrum is one layer in a Tier-C system, not
the whole thing.

**The de-niche-ifying insight:** the first shippable wow is **Tier B on a stained-glass
template**, not arbitrary photo → full scene. Tier B gives diff-RT a concrete first
customer (the existing stainedGlass workflow: designer has a reference swatch/photo, wants
the 3D panel to match) instead of competing as a general moonshot.

## The loop

```
reference image
       ↓
   loss(image_rendered, image_target)      ← L2 or L1
       ↓
   ∂loss/∂θ  for each optimizable θ          ← THIS is what diff-RT provides
       ↓
   update θ (Adam, etc.)
       ↓
   re-render → repeat until match
```

The shipping contract optimizes finite scalar, vec2, and RGB material/emitter fields.
Texture variables and perceptual losses are not public modes.

Current API shape — the host owns the optimization cadence and calls `step()`:

```ts
const session = engine.createInverseSession({
  target: { data: referenceRgb, width, height, channels: 3 },
  loss: 'l2',
  method: 'finite-difference',
  parameters: [
    { path: 'materials.panel-1.baseColor', kind: 'rgb' },
    { path: 'emitters.sun.intensity', kind: 'scalar' },
  ],
});
for (let i = 0; i < 500; i += 1) {
  const step = await session.step();
  onProgress(step.loss, step.preview);
}
session.dispose();
```

Under the hood: forward render → loss → finite-difference probes, or the certified
pt-webgpu emissive adjoint → Adam update. The host owns cadence; one non-reentrant
`step()` performs one transactional iteration.

## Why it's hard in a path tracer (and in the browser)

Path tracing is full of **discontinuities**: ray hits triangle A vs B (small move → totally
different result), specular-vs-diffuse sample choice, Russian-roulette survival. Naive
autograd through the whole kernel breaks at those jumps. Production approaches:

- **Path replay** — freeze the random choices from the forward pass; differentiate only the
  continuous shading. (Common in research PT differentiators; Mitsuba 3 / dr.jit is the open
  reference template.)
- **Reparameterization** — smooth the boundaries (math-heavy).
- **Hybrid** — differentiable rasterizer / NeRF for coarse fit, PT for refinement (how many
  real image-to-3D pipelines work).

## Two engine-specific notes (vitrum-grounded, 2026-05-29)

These change the cost estimate and pin the main risk:

1. **The optimizer and a scoped Phase-1 adjoint ship.**
   The NRC fused MLP kernel is a WGSL training kernel with a *full backward pass*:
   i32 fixed-point atomic gradient accumulation, mixed-precision Adam, workgroup-resident
   tiles, validated GPU==CPU-analytic to ~9.5e-7. `pt-webgpu` also has a
   path-replay inverse pass. The native full-render gate
   (`tools/gpu-env/inverse-fit-deno.ts`) certifies material `emissive`: all three
   channels match finite-difference signs with 1.9–2.7% relative error on lavapipe.
   `baseColor` and `roughness` differed by 13–22%, so they are deliberately routed
   to finite differences. What remains unbuilt is broad adjoint parity:
   indirect/path-space derivatives, stochastic path
   reuse, spectral/volume/layered material gradients, transmission/thickness/displacement,
   and the visibility-discontinuity cases that need finite-difference, reparameterization,
   or explicit unsupported diagnostics beyond Phase 1. The old "8-12 weeks" estimate
   should be read as the broader parity project, not the current scoped adjoint baseline
   (`packages/walkaround-hybrid/src/neural/nrc/`, `packages/pt-webgpu/src/inverse/`).

2. **The memory wall forces path replay in-browser.** A backward/adjoint pass needs per-pixel
   path state to replay against. The BDPT eye-stack (also this session) is direct evidence of
   the ceiling: per-pixel path state hit 506 MiB at 1080p×depth-8 and had to be capped at
   384 MiB with a loud fallback (`GpuResources.BDPT_EYE_STACK_MAX_BYTES`). So **path replay**
   (re-trace forward with frozen RNG inside the backward — trade compute for memory) is the
   *only* tractable in-browser option. Conveniently, it is also what sidesteps the
   discontinuity problem in Phase 1. Phase-1 scoping (fixed paths, replay, continuous-shading
   gradients only) isn't merely clean — it's mandatory.

## The architectural fork that decides the effort's size

**Backward-in-WGSL next to pt-webgpu (all-in-browser, host-dumb)** vs **export to
dr.jit / PyTorch**. In-browser aligns with the lifecycle contract *and* is now viable
because of the NRC infra, but it is the decision that gates whether this is an ~8-week or a
much-longer effort. Browser constraint: 10k optimization steps with a side PyTorch stack is
a non-starter unless forward *and* backward both live in WebGPU compute — which favors the
in-browser fork. **This fork should be settled before Phase 1 starts.**

## Phased roadmap

| Phase | Scope | Native to | Notes |
|-------|-------|-----------|-------|
| **0 — Prove the loop** | Finite differences on a bounded material/emitter vector. | vitrum | **Shipped** on pt-webgl2 and pt-webgpu with transactional lifecycle and strict validation. |
| **1 — Differentiable shading on fixed paths** | One bounce, fixed triangle hit; camera-visible emissive path replay. | vitrum | **Shipped for `emissive` only.** Base-color/roughness candidates remain FD until the native gradient gate agrees. |
| **2 — Texture optimization** | Gradients w.r.t. texture maps (spatial params). | vitrum | Where "paint the target look" starts to feel magic. |
| **3 — Lighting + environment** | Sun direction, HDRI scaling. | vitrum | Huge for "match this photo's mood." |
| **4 — Template scenes (stained glass)** | Parametric panel + lead came + glass layers; reference drives full layout fit. | product + vitrum | **The Tier-B headline / first shippable wow.** |
| **5 — External geometry proposal** | Plug in image-to-mesh / depth / Gaussian splats; vitrum refines. | ecosystem + vitrum | **The Tier-C headline.** vitrum is the refinement layer, not the proposer. |

Phases 0–3 are vitrum-native; 4 is product + vitrum; 5 is ecosystem + vitrum.

## Where vitrum already helps (before diff-RT exists)

- **Scene contract** — materials, emitters, env are structured, addressable params.
- **pt-webgpu** — compute-friendly, progressive; good for repeated render-and-compare.
- **Hero materials** — spectral, thin-film, transmission (exactly what you'd match in glass).
- **NRC / neural path** — adjacent online-training muscle; *and* the built optimizer (above).
- **Incremental `updatePrimitive`** — push optimized textures back each iteration.
- **Two-backend split** — walkaround for a fast proxy loss during convergence, pt-webgpu for
  the final fit.

The optimization API, finite-difference engines, Adam state, and pt-webgpu emissive
adjoint are present. Missing work is promotion of additional adjoint fields through the
same native full-render gradient gate, plus any future texture/geometry parameter model.

## Feasibility filter (the project's own rule, applied)

| Requirement | Status |
|-------------|--------|
| Public algorithm | ✅ Path-replay diff-PT — papers + dr.jit/Mitsuba 3 reference |
| Portable to WebGPU/WGSL | ✅ Forward already there; backward is engineering, optimizer half built |
| Not RTX-locked | ✅ |
| Browser precedent | ⚠️ Few full diff-PT in browser — the browser-novelty *is* the contribution; NRC online-training is the adjacent proof it's tractable |

Not cargo-cult: ambitious engineering with a clear literature trail, not "wait for vendor."

## Bottom line

"Reference image → matching 3D render" is the right north star for diff-RT in vitrum, with
honest framing: **diff-RT = the physics-accurate optimizer that makes matching believable
once you have geometry**; full one-photo magic = diff-RT *plus* a geometry-proposal stage.
First win is **Tier B on a stained-glass template**, then **Tier A everywhere** — not
arbitrary photo → full scene. Capture now (done); execute only after the locked fidelity +
validation priorities clear and a host asks for Tier B.
