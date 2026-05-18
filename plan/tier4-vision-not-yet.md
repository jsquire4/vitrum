# Tier 4 — Vision (NOT EXECUTABLE)

> ⚠️ **DO NOT DISPATCH AS A PLAN.** ⚠️
>
> This document captures moonshot ideas for the renderer's eventual SOTA
> direction. It is **gated** behind:
>
> 1. Tiers 1, 2, and 3 fully landed.
> 2. Validated in the `~/projects/stainedGlass` consumer app — i.e.,
>    the drop-in API actually works for a real host, hero examples
>    actually render, and we've used the library ourselves long enough
>    to know what _we_ would change.
> 3. An explicit "go" from the user.
>
> Until those three conditions hold, this doc is reference only. No
> agent dispatch, no per-item planning, no commitment to scope or order.

---

## Why these ideas are captured here

The post-Tier-3 conversation surfaced three legitimate "phenomenal
renderer" directions and ~6 supporting moonshots. They're worth writing
down so the thinking isn't lost, but they're emphatically NOT next steps.

The discipline:

1. Prove vitrum works as a drop-in library (Tier 3) by integrating it
   into stainedGlass.
2. Live with the integration. Find the friction points the spec doesn't
   predict.
3. Iterate on the API based on real consumer pain, not theoretical
   ergonomic concerns.
4. Only then turn to Tier 4 — and re-evaluate which ideas still matter
   given what stainedGlass actually needs.

Most of these ideas may turn out to be the wrong bets after stainedGlass
exposes the actual gaps. Capture is for later-evaluation, not for
later-execution.

---

## The headline composition (Path 1 + Path 3)

The most defensible "set vitrum apart" combination is **unified
progressive convergence with neural radiance caching**.

### Path 1 — Unified progressive convergence

**Concept:** When the user stops interacting (camera idle > 250ms), the
walkaround engine's denoised output becomes the t=0 prior for a path-
traced accumulator. PT samples blend in via temporal MIS until fully
converged. When the user moves again, vitrum smoothly hands back to
walkaround. The two backends share scene state, BVH, materials — so the
handoff is invisible.

**Why it matters:** No browser engine ships this. Three-gpu-pathtracer
gives you "wait 30 seconds for a clean image"; Three.js gives you
"interactive but no GI". Vitrum could give you "interactive when moving,
photoreal when paused, no visible transition." That changes the workflow
for designers (the stainedGlass user) — they stop waiting, start
iterating.

**What's required:**

- Shared scene state across backends. Already mostly there via
  `@vitrum/core/Scene` post-extraction; the M6 cross-package BVH
  consistency makes shared BVH possible.
- Temporal MIS at the integrator-output level. Walkaround's output
  treated as a "cheap importance distribution" for the first sample
  of the PT integrator. Standard one-sample MIS, but applied at the
  integrator level, which is novel.
- A "frame-blend layer" above both engines. New
  `@vitrum/progressive-engine` package that owns the transition state
  machine. Subscribes to camera-motion events; toggles which backend
  is producing; cross-fades the temporal accumulator.
- Per-backend "freeze and resume" support. Walkaround's SVGF history
  has to pause cleanly when PT takes over, then resume from the same
  state when PT yields back.

**Estimated work:** 4–6 weeks. Single contiguous sprint.

### Path 3 — Neural radiance caching (Müller 2021)

**Concept:** A tiny MLP (4-layer, 64-neuron) queries radiance at any
point + outgoing direction. Trained _online during rendering_ on PT
samples the user has already paid for. Radiance-cache hits skip remaining
bounces — at bounce N, query the cache; if confidence > threshold,
terminate and use the cached value; else continue tracing.

Müller 2021 reports 5–10× effective bounce reduction in NVIDIA RTXGI.
**No browser implementation exists.**

**Why it matters:** This is what makes the Path 1 PT phase actually
fast. Without it, "converge to PT" might take 5–10 seconds for a hero
scene. With it, 1–2 seconds. The visible behavior to the user is
"sharpens immediately" instead of "sharpens slowly." That's the
difference between cool and shippable.

**What's required:**

- Tiny MLP forward pass in WGSL. ~200 lines (much simpler than the
  U-Net scaffold M2.3 deleted).
- Hash-grid frequency encoding for the input position (Müller's "Instant
  Neural Graphics Primitives" feature encoding).
- **The novel piece:** online training in WGSL. Adam optimizer, ~1k
  weights. Each completed PT path contributes a training sample
  (radiance at sample point + observed total irradiance). Few hundred
  steps per frame. Backprop in compute is achievable but not standard;
  needs careful WGSL shader authoring.
- Cache-hit-vs-trace branch in the PT integrator.

**Estimated work:** 8–12 weeks. Higher risk than Path 1 — online training
in WGSL is novel territory for the browser.

### Path 1 + Path 3 combined

Total ~3–4 months of focused work. Result: a renderer that does not
exist anywhere on the web. The Tier 1 + Tier 2 + Tier 3 foundations make
this credible — the math discipline is in place; the API contract
supports backend swapping; the test infrastructure catches regressions
the moonshots will inevitably introduce.

---

## Path 2 — ReSTIR PT (the SOTA-leap)

**Concept:** Generalize ReSTIR from direct/indirect GI sampling to
_full path-space integration_. Bitterli + Wyman 2022 "ReSTIR PT" + Wyman
SIGGRAPH 2023 GRIS framework. Maintain reservoirs over _paths_ (not just
lights or visible-point hits), reuse them temporally and spatially.

**Why it matters:** 5–20× faster convergence on hard light transport
(caustics, deep indirect, complex glass). Researchers cite this. **Zero
browser implementations.**

**Why it's not in the headline:** Path 1+3 gives the user a more visible
behavioral change (interactive → photoreal smooth handoff) than ReSTIR
PT (faster convergence on already-convergent paths). For an external
audience, Path 1+3 is the demo; ReSTIR PT is the technical talk at a
conference.

**What's required:**

- Path reservoir storage — beyond the current per-pixel ReSTIR-DI/GI
  reservoirs (M9), store full path samples per pixel.
- Generalized RIS with Talbot-style pairwise MIS over paths.
- Reconnection vertex Jacobian for the path-space domain. The M9
  ReSTIR-GI implementation already has this for the visible-point ↔
  sampled-point reconnection; generalizes to arbitrary path vertices.
- Visibility re-validation for chosen path samples on reuse.

**Estimated work:** 6–10 weeks. Major shader: `restirPt.wgsl.ts`.
Significant extension of M9 ReSTIR-GI infrastructure.

---

## The workflow-changing bet (cached light field)

Standalone idea worth considering even without Path 1+3:

**Persistent cross-session light field.** PT samples accumulated for a
static scene write to IndexedDB (or a server). User reopens the app at
session N+1; scene loads with the converged light field from session N
as a baseline. Frame 1 looks like frame 10000 from yesterday. User makes
an edit; only the affected region locally invalidates and re-converges.

**Why it matters:** This is the change that turns rendering from a
per-session start-from-zero task into a continuous artifact. Designers
stop waiting at the start of every session. The render _evolves_ across
their workflow.

**What's required:**

- Serialization for the walkaround DDGI atlas + RC cascades + ReSTIR
  history (or, if we have Path 1, the PT accumulator state).
- Versioning so a scene-change correctly invalidates the cache.
- Locality-aware invalidation: changing one mesh shouldn't blow away
  the entire light field; only the radiance contributions that
  reference that mesh need re-converging.
- IndexedDB storage adapter (browsers have ~50–100 MB limits; need
  compression / pruning).

**Estimated work:** 3–4 weeks for naive whole-scene cache; 8–10 weeks
for locality-aware invalidation.

**Why this is in Tier 4:** the value is workflow-shaped, and we won't
know if it matters until we actually use vitrum from stainedGlass for a
while. Maybe stainedGlass already iterates fast enough that cross-session
caching is gilding. Maybe it's the single feature that flips the
designer's experience. We won't know until we live the integration.

---

## Compounding moonshots (smaller, fit alongside the headline)

### MNEE caustics (RFE-05 deferred)

**Concept:** Manifold Next Event Estimation per Hanika et al. 2015.
Path-space mutation that tunnels through specular vertices to find a
manifold-correct path connecting a non-specular point to a light. Makes
swimming-pool caustics, glass refraction patterns, and water rim-light
work in real-time.

**Why it matters:** Caustics are the killer demo for a path tracer.
Without them, glass scenes look stippled. With them, photoreal. The
deferred RFE-05 spec is in `external_requests/05-manifold-nee.md`.

**Estimated work:** 2–3 weeks. Fits in the existing dielectric branch
of `pathTraceBruteforce.wgsl.ts`.

**Risk:** MNEE is fragile — converges only when a manifold path exists
and the seeding is good. May need fallback to brute-force NEE for
unconvergent cases.

### Wavefront path tracing (Aila + Laine 2010)

**Concept:** Split the megakernel `pathTraceBruteforce.wgsl.ts` (1789
lines) into per-stage compute kernels operating on path queues. Each
stage (intersection, BSDF sample, NEE, miss handling) is a focused
kernel. SIMD lanes stay coherent within a kernel; better hardware
utilization.

**Why it matters:** Typical 30–50% speedup over megakernel on real
scenes. Also makes the kernel maintainable — the 1789-line file is hard
to extend.

**Estimated work:** 4–6 weeks. Architectural change; touches the
integrator deeply.

**Risk:** WebGPU's compute-queue dispatch overhead may eat the savings
for small-batch dispatches. Needs careful batching.

### Sky-occluded importance sampling (HDR sun + clouds)

**Concept:** Outdoor scenes today use uniform-importance sky sampling.
Real sky has a sun (delta-ish bright disc) + clouds (HDR continuous).
Importance-sample the HDR (already built in the existing environment
PDF) + split-importance the sun. Drastically reduces variance for
outdoor scenes.

**Estimated work:** 1–2 weeks. Drop-in upgrade to the existing
environment importance sampler.

### Live differentiable rendering

**Concept:** Flip the integrator to support gradients with respect to
material parameters (or geometry, or light positions). User paints onto
a surface; renderer learns the texture / material parameters that
produce that result.

**Why it matters:** Niche but turns vitrum from a viewer into a creative
authoring tool. Inverse-rendering workflows.

**Estimated work:** 8–12 weeks. The integrator needs autograd
infrastructure; not trivial in WGSL.

**Why deferred:** stainedGlass might want this, might not. Depends on
the workflow.

### WebXR + foveated rendering

**Concept:** With WebXR + eye tracking, render high quality only at the
gaze point. PT in the foveal region; walkaround periphery. Path
reservoirs concentrated where the eye looks.

**Estimated work:** 4–6 weeks (mostly WebXR plumbing).

**Why deferred:** stainedGlass is unlikely to be a VR app. Bet only if
a VR-shaped consumer surfaces.

### Deferred shading on top of GI

**Concept:** Separate "what light arrives here" (GI) from "how the
surface responds" (BSDF). Cache the irradiance / radiance field; let
the host swap material BSDFs without re-running GI. Standard in offline
renderers; rare in browser real-time.

**Why it matters:** Material editing in real-time without recomputing
GI. Designers can iterate on material look while keeping lighting fixed.

**Estimated work:** 3–4 weeks. The DDGI atlas already approximates a
position-conditional irradiance cache; needs extending to a true
deferred path.

---

## What stainedGlass integration will tell us

These are the questions only real usage answers:

1. **Does the unified progressive Path 1 actually feel seamless,** or
   does the handoff produce a visible blink? Won't know until we try.
2. **Does the cached light field workflow change matter,** or do
   designers iterate fast enough that cold-start is fine?
3. **Are caustics a "must-have" or "nice-to-have" for stained glass?**
   The name suggests yes, but the actual workflow may treat them as a
   final-render-only concern.
4. **What's the right convergence target for "paused" mode?** 1 second?
   5 seconds? 30 seconds? Different answers shape Path 1 + 3 priorities
   completely.
5. **How much GPU memory budget does the consumer have?** Path 1
   doubles persistent texture footprint (PT accumulator + walkaround
   atlases simultaneously). Path 3 adds the MLP weights. Cached light
   field adds IndexedDB storage. If budget is tight, some of these
   become non-starters.

The discipline: **answer these questions by integrating + using vitrum
from stainedGlass first**, then re-evaluate which Tier 4 ideas matter.

---

## Suggested re-entry process when this doc becomes executable

If/when the user lifts the gate on Tier 4:

1. Re-read this doc.
2. List which ideas still feel right given the stainedGlass integration
   experience.
3. For each surviving idea, write a per-item plan with the same
   discipline as `sweep-2026-05-11-decisions.md` (decisions locked,
   per-file scope, behavior-preserving tests, dependencies).
4. Pick a single headline (recommend: Path 1 if convergence speed
   matters more; Path 3 if perf matters more; cached light field if
   workflow matters most).
5. Schedule as Tier 4.A; defer the rest to Tier 4.B+ subdivisions.
6. Same execution model as Tier 1+2+3 — orchestrator + parallel sonnet
   agents + audit loops + retrospective.

The pattern is proven now. The only thing missing is the data from
stainedGlass to know what to actually pick.

---

## What "phenomenal" actually means, captured for the future

If, two years from now, someone asks "what was vitrum trying to be?",
the answer captured here is:

**The browser GI library that disappears.** No visible "PT mode" /
"realtime mode" toggle. No 30-second wait for a clean image. No
designer staring at a progress bar wondering if it's frozen. Just:
move the camera, iteration is responsive; let go, the image sharpens to
photoreal in seconds; come back tomorrow, your scene picks up where it
left off.

The math is just the table stakes for that. The Tier 4 ideas are the
tactics. The goal is the workflow change for the designer at the other
end.

That's what we're trying to build. We're not there yet. Tier 1+2+3 +
stainedGlass integration tells us how close we got with the foundation.
Tier 4 is what we'd do if the foundation proved it could carry the
weight.
