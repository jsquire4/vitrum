# vitrum — Proposed roadmap (unified)

**Status:** planning document (not a commitment schedule).  
**Date:** 2026-05-28  
**Audience:** maintainers deciding what to build next for **interactive real-time GI** and **hero path tracing**, including **graceful degradation** across GPU classes.

This document unifies prior discussions about maturity labels, remaining deep-pipeline work, performance tiering, frontier algorithms, and hardware validation. It deliberately **does not** cover npm publish, release governance, or cross-host marketing evidence—those are separate programs.

---

## 0.5 Locked decisions — 2026-05-28 maintainer session

These supersede conflicting guidance elsewhere in this doc. Recorded so nothing is forgotten (the priority is *tracking*, not schedule).

**North star: fidelity.** Maximum hero-stack fidelity is the paramount goal. **No deferring or removing fidelity features** — if the contract promises it, implement it.

1. **BMFR denoiser → IMPLEMENT (do NOT remove).** The `denoiser` union advertises `'bmfr'`; the contract must be honored. Keep the union entry as the standing reminder until a real BMFR module ships in `shared-denoisers`. (Overrides §6.5 / §8.5 "implement *or* remove".)
2. **pt-webgl spectral → IMPLEMENT real Jakob-Hanika** (replace the `jakob-hanika-placeholder`). No deferring. Coefficient math is unit-testable now; render-validate on the GPU env.
3. **Phase 0 prioritized, ahead of the fidelity grind:** finalize the **adapter profile to 100% as a library export** (`probeAdapterProfile` from `@vitrum/engine`/`@vitrum/core`) + **quality presets** (ultra/high/medium/low, §4.3) + **hybrid lite tier** (§5.2). Enables graceful degradation (Class A–E).
4. **WSL-GPU validation environment = a passed-off side-project, sequenced FIRST** (before the fidelity grind). Goal: a headless WebGPU adapter in WSL clearing hybrid limits (≥16 storage buffers / ≥8 storage textures) so fidelity captures + V1–V8 run locally without manual Windows-Chrome. Lives in `tools/gpu-env/`. **Sequencing rule: implement fidelity features → validate on this env (never ship fidelity blind).**
5. **Fidelity grind = the major focus** once the GPU env exists. Promote `plan/renderer-fidelity-matrix.md` rows to `supported` with GPU captures.
6. **Frontier stays in the plan (so it's not lost), but de-prioritized.** The walkaround↔PT handoff (§8.1 Path 1) is a "fun trick," **not** rated above NRC / ReSTIR-PT; do it now only if the freeze/resume temporal-MIS state machine proves trivial, else defer in favor of fidelity. NRC, ReSTIR-PT/GRIS, cached light field all remain tracked (§8).
7. **Work on `main` directly** for remediation (no feature branches); no remote push without explicit instruction.

**SHIPPED — 2026-05-28 fidelity-combination wave (merged + pushed to `origin/main`):** Locked items 1 (BMFR) + 2 (pt-webgl spectral) are DONE — real Koskela-2019 BMFR in `shared-denoisers`; Jakob-Hanika coeffs now CONSUMED in pt-webgl shading (were uploaded-but-dead). Item 3 (Phase 0) shipped — adapter-profile export + quality presets + hybrid lite tier + resolutionFactor — with two post-audit fixes (resolutionFactor composite upscale; `ddgiUpdateDivisor` made load-bearing → 2→32 preset spread, **default cadence now stride 2**). Plus, from an algorithm-combination fitness review (find techniques combined where they shouldn't be): **SVGF-real dropped from pt-webgpu** (converged tracer → `oidn-final`; SVGF is real-time-only — `unsupported` on both converged backends), **RC⊕ReSTIR-GI fixed-scalar blend → per-pixel confidence balance heuristic**, **PPG finished** (gi-ris now guides via defensive MIS — was train-only), GPU normal-skinning confirmed + bindMatrix CPU-fallback gate, duplicate-DDGI-sun dedup. **Radiometric changes are unit-pinned; many are now GPU-VALIDATED on dzn/lavapipe, the rest pending real-GPU A/B (`HARDWARE-VALIDATION-NEEDS.md` V1–V26).** Item 4 (WSL-GPU env) remains the validation blocker — the lavapipe PNG render-capture adapter is the `wsl-gpu` sibling project's Task 1.

**Stale-claim corrections (verified against current code 2026-05-28):**
- **pt-webgpu GPU BDPT light-subpath is DONE** (`bdptExtendLightSubpath` @compute pass shipped + dispatched) — remove from §6.2 / §8.3 "remaining."
- **GPU skinning: DONE** — positions AND normals (inverse-transpose via `GPU_SKIN_BVH_WITH_NORMALS_WGSL` / `mat3InverseTranspose`) shipped; non-identity-bind meshes fall back to CPU `solveSkin`. (§7 "polish" closed; GPU A/B = V11.)
- **W4-c / W4-e are done;** `HybridEngine.ts` is large-but-cohesive (LOC-target residue only).
- Sweep follow-ups in `HARDWARE-VALIDATION-NEEDS.md` V1–V8 (Möller-unify, BDPT tangent, DDGI radiometry/border, stained-glass, directional→DDGI, instancing, T9-stepC) are code-complete + unit-pinned; only GPU render/compile validation is outstanding (the V8 §10.2 typo "Molller" → "Möller").

---

## Table of contents

1. [Terminology: two rendering stacks](#1-terminology-two-rendering-stacks)
2. [Where the repo is today](#2-where-the-repo-is-today)
3. [Where it needs to go (north star)](#3-where-it-needs-to-go-north-star)
4. [Hardware classes and graceful degradation](#4-hardware-classes-and-graceful-degradation)
5. [Remaining work — real-time stack](#5-remaining-work--real-time-stack)
6. [Remaining work — hero stack](#6-remaining-work--hero-stack)
7. [Shared platform work (both stacks)](#7-shared-platform-work-both-stacks)
8. [Frontier techniques (documented in-repo, not shipped)](#8-frontier-techniques-documented-in-repo-not-shipped)
9. [Broader research landscape (feasibility-filtered)](#9-broader-research-landscape-feasibility-filtered)
10. [Hardware validation program](#10-hardware-validation-program)
11. [Suggested phases and dependencies](#11-suggested-phases-and-dependencies)
12. [Related documents](#12-related-documents)

---

## 1. Terminology: two rendering stacks

vitrum is **not** one renderer. It is one **contract** (`@vitrum/core` `Engine`) with **two product-grade rendering stacks** plus a facade:

| Name (this doc) | Package(s) | What it does | Frame semantics |
|-----------------|--------------|--------------|-----------------|
| **Real-time stack** | `@vitrum/walkaround-hybrid` (+ `@vitrum/walkaround-rc` opt-in) | WebGPU **real-time GI**: DDGI probes + ReSTIR DI/GI + GTAO + temporal denoise (à-trous-variance default; SVGF-real / neural / OIDN opt-in). Optional RC, PPG. | **One fresh stochastic frame per tick** — not multi-SPP convergence. Targets ~60 fps on capable GPUs for moderate scenes. |
| **Hero stack** | `@vitrum/pt-webgl` (WebGL2 + forked `three-gpu-pathtracer`) and `@vitrum/pt-webgpu` (native WebGPU PT peer) | **Converged path tracing** — multi-bounce, hero materials, spectral/thin-film/BDPT/caustics (tier-dependent). | **Accumulates samples** until `samplesTarget` or pause; quality improves over seconds. |
| **Facade** | `@vitrum/engine` | `createEngine()` / `attachVitrum()` / `VitrumCanvas` — picks backend from `prefer`, WebGPU probe, triangle budget, TLAS need. | Host-owned loop; engine does not own device lifetime. |

**Critical distinction for “in-browser 3D that feels alive”:**

- The real-time stack is **not** path tracing at 60 fps. It composes **game-industry-style** techniques (DDGI, ReSTIR, GTAO, SVGF-family denoising).
- The hero stack is **path tracing**. It can look photographic but is the wrong tool for smooth walk-around on a phone unless heavily denoised and sample-budgeted.

**Maturity labels (accurate as of 2026-05-26 signoffs):**

- **Release-candidate track:** `@vitrum/engine`, `@vitrum/walkaround-hybrid`, `@vitrum/pt-webgl` — pipeline shipped, mechanical tests green, backend maturity matrix “strong” on deep integration.
- **Peer hero backend, fidelity-experimental:** `@vitrum/pt-webgpu` — contract-strong, deep audit closed; many renderer rows still `experimental` until GPU gap-closure promotes them.
- **Not “alpha” / “pre-alpha prototype”** for the primary surface — those terms are stale in scattered docs; prefer **RC track**, **experimental feature tier**, or **pre-1.0 API**.

---

## 2. Where the repo is today

### 2.1 Real-time stack — shipped core

- Full pass chain: RIS → temporal/spatial ReSTIR-DI → ReSTIR-GI (half-res) → shade → denoise → temporal accumulation → composite (~18 registered passes; W1 registry refactor).
- DDGI coherent physical model (M7): receiver applies `albedo·π⁻¹` once; probe Halton rotation; Lambertian cosine blend.
- GTAO half-res + bilateral upsample; per-channel SVGF on direct + indirect (`svgf-real` opt-in).
- W8: RC opt-in via `HybridEngineOptions.rcEnabled` + MIS in shade.
- PPG: CPU sTree/dTree + GPU guide/update kernels (opt-in); kd-tree sTree descent (W9); dTree compaction fix (2026-05-28 sweep).
- Incremental scene: `updatePrimitive` (transform / positions / material / topology paths), `updateEmitter`, `updateLighting`, `setSize`.
- Scale-aware defaults from scene AABB via `createEngine()`.
- Adapter contract: `HYBRID_WEBGPU_REQUIRED_LIMITS` — notably `maxStorageTexturesPerShaderStage: 8`, `maxStorageBuffersPerShaderStage: 16`.

### 2.2 Real-time stack — gaps (product + perf, not missing GI math)

| Gap | Impact |
|-----|--------|
| ~~**No hybrid lite tier**~~ **SHIPPED (2026-05-28, see §0.5)** | `HYBRID_LITE_LIMITS` + `tier:'lite'` gating + `hybridLiteCapable` adapter probe. |
| ~~**`resolutionFactor` not wired in hybrid**~~ **SHIPPED** | `FrameInput.quality.resolutionFactor` wired per-frame in `HybridEngine`. |
| ~~**No shipped quality presets**~~ **SHIPPED** | `QualityTier` `ultra/high/medium/low` presets; `createEngine` applies `recommendedRealtimeTier`. |
| **Mobile / iGPU validation deferred** | Dev WSL2 SwiftShader: `hybridCanRun: false`; radiometric evidence blocked without hardware WebGPU. |
| **WebGPU required for realtime** | `prefer: 'realtime'` with no WebGPU → **`pt-webgl` fallback** (progressive PT, not realtime GI). |
| **THREE coupling** | BVH/DDGI still rooted in THREE scene graph; `hostScene/types.ts` is a seam, not a second binding. |

### 2.3 Hero stack — shipped core

- **pt-webgl:** Production path via fork; BDPT GPU light-subpath; incremental transform/positions/material/emitter; caustic strategies; OIDN-final; MRT G-buffer (fork); material LOD / lobeMask (fork).
- **pt-webgpu:** Progressive compute PT; **full vs lite** trace tier from adapter limits; TLAS; spectral hero-λ, layered MIS, bounded emitters; BDPT v1 (GPU `bdptExtendLightSubpath` compute pass + kernel evaluate; CPU fill is now test-oracle only); OIDN-final; aux G-buffers; 120+ package tests; deep audit closed.

### 2.4 Hero stack — gaps

| Gap | Impact |
|-----|--------|
| **Fidelity matrix mostly `experimental`** | Hero spectral, thin-film, SSS, caustics, multi-emitter, BDPT — implemented mechanically; GPU acceptance captures pending for `supported` promotion. |
| ~~**pt-webgpu GPU BDPT light-subpath**~~ **SHIPPED (see §0.5)** | `bdptExtendLightSubpath` @compute pass dispatched; CPU fill is test-oracle only. |
| **SVGF-real on pt-webgl** | Row: `unsupported` on WebGL2 path. |
| **Topology-changing animation** | Incremental patches strong; vertex-count / index changes still force full `setScene()` on PT backends. |
| **Not in `auto` for small scenes** | `auto` picks walkaround &lt;500k tris; large scenes → pt-webgpu. Hero PT is explicit `quality` / `quality-webgpu`. |

### 2.5 What is *not* a gap (clarification)

- **Existence** of real-time GI or hero PT — both exist.
- **Baseline correctness** of pt-webgpu deep audit — closed; remaining work is promotion, perf, and product tiering.
- **Contract surface** — `engine.capabilities`, `incrementalPatchSupport`, branded textures (re-landed 2026-05-24) are in place.

---

## 3. Where it needs to go (north star)

### 3.1 Real-time stack north star

**A host can run a WebGPU 3D experience (walk, orbit camera, animate transforms) where lighting feels alive at interactive rates**, with automatic degradation on weaker GPUs—without requiring a 4090.

Success looks like:

- **Tier A (discrete GPU / strong WebGPU):** 60 fps target at 1080p for sub-500k triangles (README claim); stable pass chain under camera motion.
- **Tier B (integrated GPU / mobile WebGPU):** 30–60 fps at reduced internal resolution, fewer GI samples, slower DDGI updates—**still realtime GI**, not PT slideshow.
- **Tier C (no WebGPU or below hybrid limits):** Explicit, honest fallback—not silent black canvas or pretending PT accumulation is “realtime.”

### 3.2 Hero stack north star

**A host can converge to photographic stills** (material fidelity, glass/caustics/spectral where enabled) with truthful capability reporting and tier-appropriate features.

Success looks like:

- Rows in `plan/renderer-fidelity-matrix.md` promoted to `supported` with GPU captures.
- pt-webgpu at parity with pt-webgl for hero workflows where WebGPU is available.
- Clear **quality** vs **quality-webgpu** semantics in the facade.

### 3.3 Cross-stack north star (optional, Tier 4)

**Interactive when moving, photographic when still** — walkaround ↔ PT handoff (see §8). Not required for “professional library,” but is the differentiated moonshot.

---

## 4. Hardware classes and graceful degradation

### 4.1 Detection pipeline (proposed product behavior)

Today:

- `detectGpu()` — WebGPU vs WebGL2 availability.
- `pickBackend(prefer, hasWebGPU, triangleCount, needsTlas)` — facade routing.
- `tools/benchmark-runner` adapter probe — `ptWebgpuFullTier`, `ptWebgpuLiteTier`, `hybridCanRun`, buffer/texture limits.
- pt-webgpu: `selectPtWebgpuTraceTier(device)` → `full` | `lite` | throw.

**Proposed unified probe** (library export, not only benchmark-runner):

```
AdapterProfile {
  hasWebGPU: boolean
  hybridCapable: boolean      // meets HYBRID_WEBGPU_REQUIRED_LIMITS
  ptWebgpuTier: 'full' | 'lite' | 'none'
  maxStorageBuffersPerStage: number
  maxStorageTexturesPerStage: number
  isSoftwareAdapter: boolean  // SwiftShader-class heuristic
  recommendedRealtimeTier: 'ultra' | 'high' | 'medium' | 'low' | 'unavailable'
  recommendedHeroBackend: 'pt-webgpu-full' | 'pt-webgpu-lite' | 'pt-webgl' | 'none'
}
```

Host flow:

1. Probe once at engine creation (or canvas mount).
2. Select **stack** (realtime vs hero) from user `prefer` + profile.
3. Select **internal quality tier** from profile + optional user override.
4. Re-probe on `adapter` loss/restored (rare; document host behavior).

### 4.2 Hardware class matrix

| Class | Typical devices | WebGPU | Real-time stack | Hero stack |
|-------|-----------------|--------|-----------------|------------|
| **A — Strong discrete GPU** | Desktop dGPU, Apple M-series Pro/Max, recent mobile flagships | Yes, hybrid capable | **Primary target:** full pass graph, 1080p+, 60 fps goal for &lt;500k tris | pt-webgpu **full** or pt-webgl; all experimental features testable |
| **B — Integrated / low-power dGPU** | Intel iGPU, Ryzen APU, older mobile | Often yes; may be borderline on storage textures | **Degrade:** internal res 0.5–0.75, single spatial pass, lower DDGI update rate, disable RC/PPG/neural by default | pt-webgpu **lite** or pt-webgl; fewer bounces live; lower `samplesTarget` |
| **C — Mobile WebGPU (capable)** | iOS 18+ Safari, Android Chrome WebGPU | Yes if limits pass probe | **Mobile preset:** 720p internal, half spatial neighbors, DDGI stride ↑, `targetFrameIntervalMs` cap | Usually **pt-webgl** or pt-webgpu lite; hero convergence “on tap” not walk-around |
| **D — WebGPU present, below hybrid limits** | SwiftShader, some software stacks, future tight adapters | Yes but `hybridCanRun: false` | **Unavailable** — do not init HybridEngine; show UX + offer hero or raster fallback | pt-webgpu **lite** if limits allow; else pt-webgl |
| **E — No WebGPU** | Older browsers, locked-down environments | No | **Unavailable** — `prefer: 'realtime'` → pt-webgl (accumulating PT), label clearly in UI | **pt-webgl only** (quality path) |

### 4.3 Degradation dimensions (orthogonal knobs)

Apply per **real-time tier** (hybrid) or **hero tier** (PT):

| Dimension | Ultra | High | Medium | Low |
|-----------|-------|------|--------|-----|
| Internal resolution scale | 1.0 | 0.85 | 0.67 | 0.5 |
| ReSTIR spatial passes | 2 | 2 | 1 | 1 |
| Spatial neighbors | 5 | 5 | 3 | 3 |
| ReSTIR-GI half-res M_GI scale | full | full | reduced | minimal |
| DDGI probe update stride | default | default | 2× | 4× |
| GTAO | on | on | on | off or quarter-res |
| Denoiser | atrous-variance | atrous-variance | atrous-variance | atrous only |
| RC / PPG / neural | opt-in | off default | off | off |
| `targetFrameIntervalMs` | null (uncapped) | null | 20 | 33 |

Hero PT equivalents:

| Dimension | Ultra | High | Medium | Low |
|-----------|-------|------|--------|-----|
| `resolutionFactor` | 1.0 | 0.75 | 0.5 | 0.5 |
| `samplesTarget` (interactive) | user | reduced | low | very low |
| `maxBounces` live preview | max | max | 3–5 | 2–3 |
| Denoiser | OIDN / svgf-real | atrous / accumulate | accumulate | accumulate |
| pt-webgpu tier | full | full | lite | lite / webgl |
| BDPT / caustics / spectral | on | on | selective | off |

### 4.4 What “graceful” means (UX contract)

- **Never** initialize hybrid on `hybridCanRun: false` — fail at construction with actionable error + suggested `prefer: 'quality'`.
- **Never** imply 60 fps on Class C/D without measurement — document tier targets as goals validated per device class.
- **Degrade quality before dropping frames silently** — lower res / fewer passes before stutter; optional `onFrame` stats for host HUD.
- **Preserve camera motion responsiveness** — temporal reset on move (already); tier may increase `temporalAccumAlpha` on low tier for stability.

---

## 5. Remaining work — real-time stack

Organized as **implementation themes**. Each theme should land with mechanical tests + hardware capture on at least one Class A and one Class B/C device.

### 5.1 P0 — Adapter profile + quality presets (library completeness)

**Problem:** Hosts cannot ask “what will run here?” or select a preset.

**Deliverables:**

- Export `probeAdapterProfile(device | adapter)` from `@vitrum/engine` or `@vitrum/core`.
- `HybridEngineOptions.qualityTier?: 'ultra' | 'high' | 'medium' | 'low'` mapping to §4.3 tables.
- Wire `FrameInput.quality.resolutionFactor` → internal render size in hybrid (or document-only rejection with clear error).
- `createEngine()` applies recommended tier when `advanced` omits overrides.
- Unit tests: preset → expected UBO/pass flags (characterization).

**Evidence:** Adapter probe JSON in CI artifact; manual matrix on 2–3 real GPUs.

### 5.2 P0 — Hybrid lite tier (weak WebGPU)

**Problem:** pt-webgpu has full/lite; hybrid does not—weak adapters hard-fail or run full cost.

**Deliverables:**

- Define `HYBRID_LITE_LIMITS` + reduced bind layout (fewer simultaneous storage textures, optional RC/PPG stripped).
- Second shade variant or `hybridParams.enabledFeatures` bitfield in UBO.
- `hybridCapable` vs `hybridLiteCapable` in adapter profile.

**Risk:** Shader permutation explosion — prefer runtime UBO gating over N× pipeline objects where possible.

### 5.3 P1 — Performance optimization (same algorithms)

**Problem:** Pass graph is correct but not fully cost-optimized.

**Deliverables (prioritize by profiling on Class A hardware):**

- Profile pass timestamps (`timestamp-query` opt-in already sketched).
- Reduce redundant primary casts in spatial ReSTIR (neighbor validation strategy).
- Optional fuse / skip passes per tier (§4.3).
- DDGI update budgeting tied to tier.
- PPG: run guide pass every N frames on medium/low tiers.

**Evidence:** `tools/benchmark-runner` frame-time scenarios (`PR-hybrid-200k-static`, material/emitter churn) with budget thresholds per tier.

### 5.4 P1 — Mobile + iGPU validation harness

**Problem:** Claims untested on phones and Intel iGPU.

**Deliverables:**

- Document minimum WebGPU limits for hybrid lite vs full.
- Capture set: 720p walkthrough, 30 fps goal Class C, 60 fps goal Class A.
- Heat / thermal note in README (honest “sustained vs burst” framing).

### 5.5 P2 — Host ergonomics for “game-like” hosts

**Deliverables:**

- Example: first-person / orbit walk loop with `updatePrimitive({ transform })` each frame + `FrameStats` HUD.
- Document animation matrix (refresh `plan/archive/animation-support-status.md` claims against current `HybridEngine`).

### 5.6 P2 — THREE decoupling (M4, not blocking tiering)

**Deliverables:** Continue `hostScene/types.ts` + non-THREE BVH root — enables Babylon/custom hosts. Does not change GI math.

### 5.7 Explicit non-goals (real-time stack)

- Full path tracing at 60 fps on mobile (see §8 ReSTIR PT).
- WebGL2 realtime GI fallback (enormous scope; Class E uses hero PT or host raster).
- Physics / networking / scene graph (host responsibility).

---

## 6. Remaining work — hero stack

### 6.1 P0 — Fidelity promotion program

**Problem:** `plan/renderer-fidelity-matrix.md` rows are `experimental`; promotion requires GPU evidence.

**Process per row:**

1. Mechanical tests (existing).
2. `tools/benchmark-runner` scenario + `VITRUM_GPU_CAPTURE=1`.
3. Reference PNG + hash + perf ms/sample in `tools/reference-renders/`.
4. Flip row to `supported` with signoff note.

**Rows (initial matrix):** hero-λ, spectral Beer–Lambert, thin-film TMM, Cauchy dispersion, layered transmission MIS, SSS/translucent, multi-emitter, caustic strategies, BDPT, material-fields parity (pt-webgpu), SVGF-real (pt-webgpu).

**Owner docs:** `plan/archive/gap-closure-acceptance-matrix.md`, `HARDWARE-VALIDATION-NEEDS.md`.

### 6.2 P1 — pt-webgpu throughput

**Deliverables:**

- ~~GPU BDPT light-subpath pass (match pt-webgl fork path).~~ **DONE** — `bdptExtendLightSubpath` @compute pass shipped + dispatched (see §0.5).
- Wavefront / split-kernel integrator evaluation (§8) — only if profiling shows megakernel bound.
- Cheaper `traceAny` after D2 unification (shadow rays).

### 6.3 P1 — Hero interactive degradation

**Deliverables:**

- Document `prefer: 'quality'` live preview defaults (low SPP, `resolutionFactor`, denoiser).
- Optional `EngineOptions` hero preview profile mirroring §4.3 hero table.
- pt-webgpu lite: document disabled features (caustics, BDPT, etc.).

### 6.4 P2 — Animation / topology

**Deliverables:**

- BVH leaf rebuild without full pipeline teardown (walkaround partial; PT follow-up).
- Broader topology incremental on pt-webgl/pt-webgpu where safe.

### 6.5 P2 — Contract denoisers

**Deliverables:** ~~Implement or remove **BMFR** from public union (today type-only).~~ **DONE** — real Koskela-2019 BMFR shipped in `shared-denoisers` + `BmfrDenoiser` in walkaround.

### 6.6 Explicit non-goals (hero stack)

- Replacing fork entirely before pt-webgpu parity is proven in production hosts.
- Hardware RT dependency (WebGPU ray tracing extension) as baseline.

---

## 7. Shared platform work (both stacks)

| Item | Real-time | Hero | Notes |
|------|-----------|------|-------|
| Adapter profile API | ✓ | ✓ | Single probe |
| `engine.capabilities` truth | ✓ | ✓ | Already improved; keep in sync with tiers |
| TLAS / BLAS packer | ✓ | ✓ | Shared `packSceneFromCore`; RC moving-instance refit (PR-5.3) |
| GPU skinning compute | polish | polish | CPU `solveSkin` shipped (C1) |
| OIDN / neural / svgf-real | opt-in | opt-in | shared-denoisers |
| Examples + hero viewer | ✓ | ✓ | Game-like example benefits both |
| Complexity / bind-group dedup | ✓ | ✓ | Ongoing hygiene (2026-05-28 sweep) |

---

## 8. Frontier techniques (documented in-repo, not shipped)

These are **algorithm evolution** stages—not graceful degradation. Sourced primarily from `plan/tier4-vision-not-yet.md` (gated: integrate vitrum in a real host before committing). Listed with **intent**, **deps**, and **browser feasibility**.

### 8.1 Tier 4 headline — interactive ↔ hero composition

#### Path 1 — Unified progressive convergence (walkaround → PT handoff)

- **Idea:** Camera idle &gt; ~250 ms → denoised walkaround output seeds PT accumulator; temporal MIS blends PT samples; motion → hand back to walkaround without visible pop.
- **Why:** Unique web UX—“alive while moving, photographic when still.”
- **Requires:** Shared scene/BVH; frame-blend state machine (`@vitrum/progressive-engine` proposed); per-backend freeze/resume of SVGF history + PT accum.
- **Estimate (in-repo):** 4–6 weeks.
- **Feasibility:** High — no new GI math; integration heavy.
- **Stacks:** Both.

#### Path 3 — Neural radiance caching (Müller 2021)

- **Idea:** Small MLP + hash-grid encoding; online training from PT paths; terminate paths on cache hit.
- **Why:** Makes Path 1 PT phase fast (1–2 s vs 5–10 s).
- **Requires:** WGSL forward + online Adam-style train; integrator branch.
- **Estimate:** 8–12 weeks; higher risk.
- **Feasibility:** Medium-hard — online train in WGSL is rare.
- **Stacks:** Hero primary; benefits Path 1.

#### Path 2 — ReSTIR PT / GRIS (path-space resampling)

- **Idea:** Reservoirs over full paths, not just lights/visible points; Bitterli & Wyman 2022, Wyman SIGGRAPH 2023 GRIS.
- **Why:** 5–20× on caustics, deep indirect, complex glass.
- **Requires:** Path reservoir buffers; Talbot pairwise MIS; reconnection Jacobians (extend ReSTIR-GI machinery).
- **Estimate:** 6–10 weeks; major `restirPt.wgsl.ts`.
- **Feasibility:** Hard — **zero browser implementations** per tier-4 doc.
- **Stacks:** Could target hero (pt-webgpu) or eventually real-time if denoised hard enough.

### 8.2 Workflow / persistence

#### Cached light field (cross-session)

- **Idea:** Serialize DDGI atlas + RC + ReSTIR history (or PT accum) to IndexedDB; invalidate locally on edits.
- **Estimate:** 3–4 weeks naive; 8–10 weeks locality-aware.
- **Feasibility:** Medium — storage limits, versioning.
- **Stacks:** Real-time primary; hero optional.

### 8.3 Hero-focused techniques (tier 4 + RFEs)

| Technique | Source | Status today | Frontier work |
|-----------|--------|--------------|---------------|
| **MNEE caustics** | RFE-05, tier 4 | `experimental`; API plumbed | Runtime verification + fallback to brute NEE |
| **Wavefront PT** | tier 4 | Megakernel pt-webgpu | Split kernels; 30–50% potential; WebGPU dispatch overhead risk |
| **Sky / sun split sampling** | tier 4 | HDRI IS exists | Split-importance sun + clouds; 1–2 weeks |
| **BDPT GPU light subpath** | WG signoff | **GPU `bdptExtendLightSubpath` compute pass shipped** (CPU fill now test-oracle only) | done — see §0.5 |
| **Deferred GI + material swap** | tier 4 | DDGI ≈ irradiance cache | True deferred: re-shade without recomputing GI |
| **Differentiable rendering** (inverse rendering: reference image → matching scene) | tier 4 | — | Adjoint integrator in WGSL; optimizer/grad half already built (NRC); first win = Tier-B stained-glass template fit, not arbitrary photo→scene. Phased plan: [`plan/differentiable-rt.md`](./differentiable-rt.md) |
| **WebXR + foveated** | tier 4 | — | VR-only bet |

### 8.4 Real-time-focused frontier

| Technique | Notes |
|-----------|-------|
| **ReSTIR PT on walkaround** | Unlikely as full 60fps solution; conceivable as low-SPP + heavy denoise research branch |
| **Better many-light** | ReGIR-style structures; current emitter caps are bounded arrays |
| **SER / advanced resampling** | Research-active; evaluate after ReSTIR-GI hardening |

### 8.5 Explicitly deferred / remove from union

- ~~**BMFR denoiser** — in type union, not in `shared-denoisers`; implement or delete.~~ **DONE** — real Koskela-2019 BMFR shipped in `shared-denoisers` (2026-05-28).
- **Hardware RT (WebGPU ray tracing)** — track Chromium experiments; do not baseline vitrum on it.

### 8.6 Suggested evolution order (algorithm stage gate)

**Stage 1 (product):** §5 P0–P1 + §6 P0 — tiers, lite hybrid, fidelity promotion.  
**Stage 2 (UX differentiation):** Path 1 handoff.  
**Stage 3 (hero speed):** Path 3 NRC OR wavefront PT (pick based on profiling).  
**Stage 4 (hard transport):** ReSTIR PT / MNEE production hardening.  
**Stage 5 (workflow):** Cached light field.

Re-evaluate after stainedGlass (or another host) answers tier-4 “usage questions” in `plan/tier4-vision-not-yet.md` §“What stainedGlass integration will tell us”.

---

## 9. Broader research landscape (feasibility-filtered)

vitrum’s rule: **public algorithm, portable to WGSL/WebGPU compute, not RTX-locked.** Below: directions **not** fully captured in §8 but relevant for roadmap planning.

| Direction | Relevance to vitrum | Feasibility note |
|-----------|---------------------|------------------|
| **ReSTIR PT / GRIS / SER** | Hero caustics/glass; maybe future realtime | High shader/memory cost; no browser precedent |
| **ReGIR / many lights** | Large emissive worlds | Needs new data structures beyond bounded emitter buffers |
| **Neural denoisers / upscalers** | Partially shipped (U-Net, OIDN) | Upscaler ≠ denoiser; could help Class B/C |
| **Radiance caches / irradiance volumes** | DDGI + RC are cousins | Engineering to unify and invalidate correctly |
| **Path guiding (PPG, neural guiding)** | PPG opt-in shipped | Neural guiding needs training pipeline |
| **Blue-noise / Owen scrambling** | QMC already (Sobol, etc.) | Incremental quality win |
| **Ray cones / footprint filtering** | Aliasing in PT | Medium; hero stack |
| **Mesh shaders, SER, SER** | D3D12/ Vulkan | **Not** WebGPU baseline |
| **DLSS / FSR** | Native APIs | Not in browser; neural upscale possible but separate product |
| **WebGPU `rayTracing` extension** | Future optional fast path | Chromium experimental; not portable today |

**Optimization without new papers (high ROI):** hybrid tiering, pass scheduling, traversal specialization, BSDF lobe LOD (fork already has `liteMode`), bind-group dedup, DDGI update budgeting — see §5.3.

---

## 10. Hardware validation program

Code can land without hardware; **claims** cannot.promote without it.

### 10.1 Environments

| Environment | Role |
|-------------|------|
| **CI (SwiftShader / Vitest browser)** | Mechanical tests, WGSL compile, bind layout parity — **not** radiometric truth for hybrid |
| **Class A desktop (dGPU)** | Gold captures; 60 fps hybrid budgets; pt-webgpu full tier |
| **Class B iGPU** | Lite tier + degradation tables |
| **Class C mobile WebGPU** | Mobile preset validation |
| **Class E WebGL2-only** | pt-webgl hero paths |

### 10.2 Commands (existing)

- `npm run verify:mechanical` — typecheck + tests + shader smoke.
- `npm run baseline:wave0` / `hardening:wave4` — gate bundles.
- `VITRUM_PROBE_START_SERVER=1 npm run benchmark:pt-webgpu-adapter-probe` — adapter caps.
- `tools/benchmark-runner` + `VITRUM_GPU_CAPTURE=1` — scenario captures.
- See `HARDWARE-VALIDATION-NEEDS.md` for PR-specific A/B list (Möller-Trumbore, BDPT tangent, hybrid radiometry).

### 10.3 Acceptance artifacts per milestone

For each tier or feature promotion:

1. Scenario ID + seed + resolution.
2. Before/after PNG hashes (where applicable).
3. Frame time ms (hybrid) or ms/sample (hero).
4. Adapter profile JSON attached.
5. Pass/fail recorded in gap-closure or PR signoff doc.

### 10.4 Blocking truths (documented 2026-05-28)

On SwiftShader-class hosts:

- `hybridCanRun: false` → **do not** validate walkaround radiometry there.
- pt-webgpu often **lite tier only** → BDPT GPU subpath / full caustics blocked.

**First step on a real GPU machine:** probe must show `hybridCanRun: true` and `ptWebgpuFullTier: true` before trusting hybrid or full-tier captures.

---

## 11. Suggested phases and dependencies

```
Phase 0 — Foundation (parallel)
├── Adapter profile API + documentation
├── Hybrid quality presets (ultra/high/medium/low)
└── Fidelity promotion playbook (hero rows)

Phase 1 — Real-time productization (depends Phase 0 probe)
├── Wire resolutionFactor in hybrid ✓ DONE (2026-05-28)
├── Hybrid lite tier + hybridLiteCapable probe ✓ DONE (2026-05-28)
├── Benchmark budgets per tier (Class A + one Class C device)
└── Game-like example + animation doc refresh

Phase 2 — Hero hardening (parallel with Phase 1)
├── GPU gap-closure captures (matrix rows)
├── pt-webgpu BDPT GPU subpath
└── BMFR implement-or-remove ✓ DONE (implemented, 2026-05-28)

Phase 3 — Optimization pass (profiling-driven)
├── Hybrid pass cost reduction
├── pt-webgpu traceAny / BSDF LOD alignment
└── PPG scheduling per tier

Phase 4 — Frontier (gated on host integration feedback)
├── Path 1 walkaround ↔ PT handoff
├── Path 3 NRC OR wavefront PT (choose one)
└── ReSTIR PT / cached light field (optional)
```

**Dependency rules:**

- Do not market mobile 60 fps until Phase 1 Class C captures exist.
- Do not promote fidelity rows without Phase 2 captures.
- Do not start Phase 4 until a real host validates Phase 1 UX.

---

## 12. Related documents

| Document | Role |
|----------|------|
| `README.md` | Public maturity: release-candidate track |
| `CLAUDE.md` / `AGENTS.md` | Agent brief; maturity labels |
| `plan/archive/backend-maturity-matrix-2026-05-26-archived-2026-05-30.md` | Technical maturity by package (archived) |
| `plan/renderer-fidelity-matrix.md` | Hero feature truth table |
| `plan/tier4-vision-not-yet.md` | Moonshots (gated) |
| `plan/differentiable-rt.md` | Inverse rendering (reference image → matching scene); phased, gated frontier |
| `HARDWARE-VALIDATION-NEEDS.md` | GPU-only validation backlog |
| `plan/gpu-validation-followup-20260528.md` | Post-sweep GPU deferrals |
| `packages/walkaround-hybrid/README.md` | Real-time stack features |
| `packages/pt-webgpu/README.md` | Experimental boundary + tiers |
| `items_to_fix.md` | Historical; Sections A–E closed |
| `CREDITS.md` | Algorithm provenance |

---

## Revision history

| Date | Note |
|------|------|
| 2026-05-28 | Initial unified roadmap from maintainer discussion (maturity, two stacks, degradation, frontier, validation). |
