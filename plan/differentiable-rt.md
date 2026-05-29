# Differentiable Ray Tracing — Inverse Rendering ("reference image → matching 3D scene")

> **Status: FRONTIER, gated & de-prioritized** per `roadmap.md` §0.5. This is a captured
> phased plan, **not** scheduled work. The near-term priority order is unchanged
> (fidelity grind + WSL-GPU validation + finishing the in-flight waves). Nothing here
> should preempt that. Phase 0 (finite-difference) is the one piece cheap enough to
> prototype anytime; everything past it waits on a host signalling demand for the Tier-B
> workflow. See `plan/tier4-vision-not-yet.md` for the gating rationale.

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
   loss(image_rendered, image_target)      ← L2, SSIM, perceptual (LPIPS)
       ↓
   ∂loss/∂θ  for each optimizable θ          ← THIS is what diff-RT provides
       ↓
   update θ (Adam, etc.)
       ↓
   re-render → repeat until match
```

Optimizable `θ`: albedo / roughness / transmission textures (or low-res param grids),
light direction / intensity / environment map; later vertex offsets, camera extrinsics,
thin-film thickness.

Illustrative future API (not in repo — shape only):

```ts
const session = engine.createInverseSession({
  targetImage: refBitmap,
  loss: 'lpips',                 // or 'l2', 'ssim'
  parameters: [
    { path: 'materials.panel-1.albedo', kind: 'texture' },
    { path: 'emitters.sun.intensity',  kind: 'scalar'  },
  ],
});
for (const step of session.optimize({ maxSteps: 500 })) {
  onProgress(step.loss, step.preview);
}
```

Under the hood: forward render → loss pass → adjoint/backward through pt-webgpu → param
update. Host stays dumb; vitrum owns the loop (consistent with the lifecycle contract).

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

1. **The optimizer half is already built.** The NRC fused MLP kernel (landed this session)
   is a WGSL training kernel with a *full backward pass*: i32 fixed-point atomic gradient
   accumulation, mixed-precision Adam, workgroup-resident tiles, validated GPU==CPU-analytic
   to ~9.5e-7. That is exactly the gradient-accumulation + optimizer + WGSL-autodiff
   discipline the inverse loop needs. **What remains unbuilt is the adjoint *integrator*** —
   differentiating the path-trace kernel itself — which is the hard half. The roadmap's
   "8–12 weeks" estimate should be revised *down* now that the optimizer/grad plumbing
   exists and is proven (`packages/walkaround-hybrid/src/neural/nrc/`).

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
| **0 — Prove the loop** | Finite-difference / coordinate descent on a tiny param vector (one material RGB + one light). No autograd. | vitrum (pt-webgpu only) | Ugly but validates the inverse-session UX/API. Cheap; available anytime. |
| **1 — Differentiable shading on fixed paths** | One bounce, fixed triangle hit; optimize albedo/roughness. Path replay + backward through the BSDF. | vitrum | The core tier-4 item, scoped tight. Needs the architectural fork decided first. |
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

What's missing is the **optimization API** (`createInverseSession`) and the **adjoint
integrator** (gradient machinery for the forward kernel). The optimizer/grad-accumulation
substrate now exists (NRC); the adjoint render pass does not.

## Feasibility filter (the project's own rule, applied)

| Requirement | Status |
|-------------|--------|
| Public algorithm | ✅ Path-replay diff-PT, LPIPS loss — papers + dr.jit/Mitsuba 3 reference |
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
