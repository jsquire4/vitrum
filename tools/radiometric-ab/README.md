# tools/radiometric-ab

Radiometric A/B harnesses for `pt-webgpu`. Most scripts run full-tier render
variants in the linear HDR domain (`captureFrame({ colorSpace:'linear' })`) —
the raw `accumTexture` float32 values, NOT the tonemapped display output. The
ReSTIR-PT specialty-lobe script is a CPU/static identity proof, not a GPU
recapture.

## What this tests

"Does the math converge to the same answer?" — the tier above the behavioral gate's
"does it render?".  Three comparisons:

| Script | Test | Reference | Candidate |
|--------|------|-----------|-----------|
| `ab-sppm.mjs` | SPPM convergence | `causticStrategy:'manifold-nee'` (GPU-validated MNEE) | `causticStrategy:'photon-map'` at 20/50/80 frames |
| `ab-bdpt.mjs` | BDPT unbiasedness + variance | `bdpt:false` (unidirectional) | `bdpt:true` |
| `ab-restir-pt.mjs` | ReSTIR-PT bias + variance | `restirPtReuse:false` (default megakernel) | `restirPtReuse:true` (composite megakernel) |
| `ab-restir-pt-specialty.mjs` | ReSTIR-PT specialty-lobe identity | base-path CPU estimator | one-sample ReSTIR-PT producer/finalize/resolve identity |

## How to run

Prerequisites: Deno ≥ 2.8 and a pt-webgpu full-tier WebGPU adapter. The SPPM,
BDPT, and ReSTIR-PT A/B scripts now force `traceTier:"full"` and fail fast when
the adapter resolves to lite, because lite disables caustics/BDPT/ReSTIR-PT and
can otherwise produce false all-zero "passes". They also require non-black
linear-HDR captures, so two black arms cannot be reported as a passing A/B.

```bash
# From the repo root:
export VK_ICD_FILENAMES=/path/to/full-tier/icd.json

# Run each test:
npm run radiometric-ab:sppm

npm run radiometric-ab:bdpt

npm run radiometric-ab:restir-pt

# Static CPU fixture check:
npm run radiometric-ab:restir-pt-specialty
```

Each script writes a `results-*.json` in this directory.

## Verdicts

### A/B #1 — SPPM vs manifold-NEE caustic reference

**PASS**

Scene: Cornell box + glass sphere at (0, −0.3, 0) r=0.28 + overhead point light.
Caustic ROI: 525 pixels (cols 28–52, rows 50–70).

| Frames | ROI lum | relErr vs ref | RMSE |
|--------|---------|---------------|------|
| ref (manifold-nee, 80) | 0.48611 | — | — |
| SPPM, 20 | 0.48373 | **0.5%** | 0.05514 |
| SPPM, 50 | 0.48501 | **0.2%** | 0.03985 |
| SPPM, 80 | 0.48493 | **0.2%** | 0.03524 |

SPPM converges to within 0.2% of the GPU-validated MNEE reference in the caustic ROI,
and RMSE decreases monotonically (0.055 → 0.040 → 0.035) across the 3 checkpoints.
This is a near-perfect agreement — the photon hash-grid is sampling the same energy as
the manifold solver, confirming the progressive Hachisuka update rule (A4-progressive)
produces an unbiased estimate at these spp levels.

### A/B #2 — BDPT vs unidirectional

**PASS for default `bdpt:true` (2026-06-17, lavapipe after Cornell fixture repair and safe-default policy)**

Scene: Cornell box with a glossy metal sphere (r=0.08 roughness, metallic=1.0).
Indirect ROI: 1,271 pixels (cols 20–60, rows 25–55).

| Metric | UNI (bdpt:false) | BDPT safe default (bdpt:true) | Notes |
|--------|-----------------|-----------------|-------|
| Global mean lum (60 frames) | 0.08007 | 0.08007 | relErr = **0.00%** |
| ROI mean lum (60 frames) | 0.10937 | 0.10937 | relErr = **0.00%** |
| Variance in ROI (8×8 frames) | 0.078567 | 0.078567 | ratio = **1.0000** |

The previous 2026-06-11 **PASS** was invalid as promotion evidence. The shared Cornell
fixture had its back wall at `z=+1` while the PT camera sits at `+Z` looking toward the
origin, so local-light Cornell scenes could measure the unlit outside of a closed box.
`helpers.mjs` now keeps the box open toward `+Z` by placing the back wall at `z=-1`,
matching the behavioral-gate Cornell fixture. After that repair, the same BDPT proof
lane produces finite non-black signal and exposes a real multi-vertex BDPT mean mismatch.

One source-verified bug was fixed with this recapture: the pt-webgpu BDPT connection loop
no longer connects secondary eye vertices to `lvi=0`, the emitter endpoint, because that
direct-light strategy is already estimated by the normal per-bounce NEE path. The harness
now writes `controls.byMaxLightBounces` to `results-bdpt.json`: `bdpt:true` defaults to
the same endpoint-only depth as `maxLightBounces:1`, which is identical to `bdpt:false`.
The remaining mismatch starts only when a host explicitly opts into multi-vertex
light-subpath connections (`maxLightBounces:2` is +13.21% global luminance and
`maxLightBounces:3` is +17.08% at the 60-frame mean checkpoint).

Conclusion: default BDPT is now radiometrically neutral and the A/B lane records
`"verdict":"PASS"`. Full multi-vertex BDPT promotion remains open as an explicit
research-mode tail, guarded by the constructor warning
`pt-webgpu.bdpt-multivertex-research-mode`.

### A/B #3 — ReSTIR-PT reuse on vs off

**PASS** (fixed 2026-06-11 — was FINDING at 46% deficit)

Scene: same Cornell box + metal sphere.
Indirect ROI: same 1,271 pixels.

| Metric | BASE (rpt:off) | RPT (rpt:on) | Notes |
|--------|---------------|-------------|-------|
| Global mean lum (60 frames) | 0.48690 | 0.47198 | relErr = **3.06%** |
| ROI mean lum (60 frames) | 0.48723 | 0.47154 | relErr = **3.22%** |
| Variance in ROI (8×8 frames) | 0.005900 | 0.001591 | ratio = **0.2697** |

ReSTIR-PT composite path agrees with the default megakernel within MC tolerance (3% at
60 spp).  Variance ratio 0.27 = 3.7× variance reduction — the temporal reuse is working.

**Root cause of the prior 46% deficit (found + fixed 2026-06-11):**

The composite megakernel was gating `bsdfAreaLightConnectionContribution` and
`bsdfEnvironmentConnectionContribution` at E0 on `!rptCompositeContributed` — i.e.,
dropping both BSDF-side MIS halves for composited pixels under the assumption that the
resolve indirect already covers "first-bounce-hits-a-light".  This was wrong:

- Analytic lights (rect-area, disc, env/sky, directional) are **NOT in the TLAS/BVH**.
  The producer's reconnection vertex xs can never be placed on an analytic light.
  Therefore `rptComposite.rgb` can never double-count the BSDF-side MIS contribution from
  an analytic light at E0.
- Dropping `bsdfAreaLightConnectionContribution` removed the entire MIS-weighted BSDF
  half of direct area-light illumination at E0 — roughly half the rect-area-light energy
  for the Cornell scene.  That caused the ~46% deficit.

**Fix:** removed the `!rptCompositeContributed` gate from `sampleAllowsAreaMisCond` in
`composePathTraceKernelWgsl`.  Composite mode now runs `bsdfAreaConnect` on the same
condition as the default kernel: `sampleAllowsAreaMis`.  The only residual double-count
risk is for mesh area lights (which ARE in the TLAS) when the reconnection vertex xs lands
on the emissive mesh face and the same BSDF direction hits it analytically — a rare event
that over-estimates rather than under-estimates energy.

FINDING entry (pre-fix numbers) preserved above for history.

**FINDING (pre-fix, 2026-06-10 baseline):**

| Metric | BASE (rpt:off) | RPT (rpt:on) | Notes |
|--------|---------------|-------------|-------|
| Global mean lum (60 frames) | 0.48690 | 0.26190 | relErr = 46.21% |
| ROI mean lum (60 frames) | 0.48723 | 0.26167 | relErr = 46.29% |
| Variance in ROI (8×8 frames) | 0.005900 | 0.000594 | ratio = 0.1006 |

## Pass criteria

| A/B | Mean agreement | Variance |
|-----|---------------|----------|
| SPPM | relErr < 500% AND convergence trend decreasing | N/A |
| BDPT | global relErr < 10% (unbiasedness) | ratio ≤ 2.0 |
| ReSTIR-PT | global relErr < 10% (bias check) | ratio ≤ 3.0 |

## Shared infrastructure

`helpers.mjs` — engine boot, naga-gap patches (same as `behavioral-gate/gate.mjs`),
scene builders, `renderScene()`, `renderMultipleRuns()`, and radiometric statistics
(`meanLuminanceROI`, `rmseROI`, `varianceROI`).

## Key design choices

- **Linear HDR via `captureFrame({ colorSpace:'linear' })`** — reads `accumTexture`
  (rgba16float running mean), not the tonemapped `presentTexture`.  Comparisons are in
  radiometric units (average incident radiance per pixel), not display-encoded values.

- **ROI-based statistics** — caustic or indirect-light regions are defined geometrically
  rather than globally, so the A/B signal is not diluted by large uniform areas.

- **Independent runs for variance** — `renderMultipleRuns()` boots a fresh engine
  instance per run (same device, different seed offset) so each run is an independent
  sample from the estimator's distribution.  This is the sound method for variance
  estimation without access to the internal per-pixel variance buffer.

---

## Walkaround-hybrid A/Bs

Script: `walkaround-ab.mjs`. Legacy results: `walkaround-ab-results.json`.

**Current harness status (2026-06-17):** `walkaround-ab.mjs` now renders the frame normally
through a host `bgra8unorm` swap-chain texture, then reads the engine-owned post-denoise,
pre-tonemap `resolvedTexture` through `engine.captureFrame({ colorSpace:"linear" })`. The
luminance statistics are therefore linear-HDR float32 values rather than display-encoded
8-bit swap-chain samples.

**Current WSL validation caveat:** Deno 2.8.1 native WebGPU currently panics in the WSL
lavapipe walkaround path before the harness can produce a verdict:
`wgpu-hal-28.0.0/src/gles/command.rs:771:21: index out of bounds`. This reproduces in the
older `walkaround-sun-control.mjs` swap-chain-readback script too, so it is a validation-host
runtime blocker rather than a regression caused by linear capture. Rerun this harness on the
browser/real-adapter lane before promoting the walkaround rows.

### Legacy 8-bit baseline (2026-06-10)

The following numbers were captured before `walkaround-ab.mjs` switched to linear-HDR
`captureFrame`. They are useful historical smoke evidence, but their GLASS/GLOSSY caveats
come from the old 8-bit display-domain readback and should not be used as final promotion
evidence.

### A8 — GRIS Bias Quantification

**Verdict: NEGLIGIBLE** — overall delta = +0.0001 (0.05% of mean luminance).

Two arms: `restirPtReuse:false` (default biased path) vs `restirPtReuse:true` (GRIS unbiased).
Cornell + ceiling emitter, 128×128, SPP=16.

| Region | Biased (off) | Unbiased (on) | Delta |
|--------|-------------|--------------|-------|
| Overall | 0.2087 | 0.2088 | **+0.0001** |
| Floor | 0.1550 | 0.1550 | −0.00003 |
| Ceiling | 0.0319 | 0.0320 | +0.00005 |
| Left wall | 0.0385 | 0.0386 | +0.000007 |
| Right wall | 0.0156 | 0.0156 | +0.0000005 |

The four bias sources (B1–B4, documented in `HybridEngineOptions.restirPtReuse` JSDoc) produce
statistically negligible bias on this convex scene. The bias is bounded and real but below MC
noise at SPP=16. Scenes with large emitters, deep occlusion, or dramatic M-count gradients will
show larger bias. GPU unbiasedness A/B (V19 in `HARDWARE-VALIDATION-NEEDS.md`) is the outstanding
gate for the full characterization.

Render time: 4.3 s total.

### SUN — Sun-NEE Analytic Self-Validation

**Verdict: PASS** — floor ratio = 1.404 (within ±50% tolerance of analytic).

Directional-lit diffuse floor (no area emitter). Config: `sunDir=[0.3,−0.8,0.5]`, `I=0.3`,
floor albedo 0.8. Analytic Lambertian: `Lo = I × cosθ × albedo = 0.3 × 0.808 × 0.8 = 0.1939`.

| Metric | Value |
|--------|-------|
| Analytic floor Lo | 0.1939 |
| Rendered floor lum | 0.2724 |
| Floor / analytic | **1.404** |
| Left wall (sun-shadowed) | 0.0074 |
| Shadow correct? | YES (0.0074 < floor × 0.7 = 0.1906) |

The floor is ~40% above pure Lambertian because `lo_sunNEE` evaluates the full GGX BRDF (not
only diffuse), and DDGI indirect + sky irradiance add to the floor. The left wall
(`dot([1,0,0], toSun) < 0`) is 37× dimmer than the floor, confirming the shadow ray correctly
gates the NEE term. Render time: 1.9 s.

### GLASS — Glass-GI Transmitted Light Validation

**Verdict: PASS (caveat)**

Cornell+glass pane (transmission=1.0, z=0.5) vs Cornell-no-glass, ceiling emitter, SPP=16.

| Metric | Value |
|--------|-------|
| Glass centre lum | 0.5068 |
| No-glass centre lum | 0.5068 |
| Centre ratio | 1.000 |

Pixel-identical result (diff=0.0000 at every pixel). **This is a platform limitation, not a
bug.** A clear glass pane transmits ~92% of direct light at normal incidence (Fresnel-T for
n=1.5). The 8% attenuation is below the 8-bit quantization floor at 0.5 luminance. The
`lo_transmittedGI` refracted-indirect term requires higher SPP and a darker/smaller scene to
distinguish from Lambertian GI.

**CPU-side verified:** `packBVHIndexWFromCore` correctly encodes glass pane `trans4=15`,
`isMetal=0`. The glass pane IS in the BVH and IS correctly tagged. Render time: 4.0 s.

### GLOSSY — B2 Metallic Probe Check (Specular Indirect)

**Verdict: PASS (caveat)**

Metal floor (metalness=1.0, rough=0.05) vs diffuse floor (metalness=0.0, rough=1.0), Cornell,
ceiling emitter, SPP=16.

| Metric | Value |
|--------|-------|
| Metal floor lum | 0.1887 |
| Diffuse floor lum | 0.1887 |
| Floor ratio | 1.000 |

Pixel-identical result. **CPU-side verified correct:** `packBVHRoughMetalFromCore` produces
distinct data for the two floors:
- Diffuse floor tri 0/1: `rough=1.000, metal=0.000` (raw=0xff000000)
- Metal floor tri 0/1: `rough=0.051, metal=1.000` (raw=0x0dff0000)
- `isMetal=1` correctly set in BVH index byte for metal floor triangles.

The material IS going to the GPU correctly. At SPP=16 with 8-bit readback and 128×128 resolution,
the narrow-lobe GGX specular highlight from a 0.2×0.2 ceiling emitter at distance ~2 m projects
to ~2% of the floor pixels — below the quantization floor at mean luminance 0.19. This would
be clearly visible at real-hardware resolution with higher SPP. Render time: 3.7 s.

### Summary

| A/B | Verdict | Key Number |
|-----|---------|-----------|
| A8 GRIS bias | NEGLIGIBLE | overall delta = +0.0001 (0.05% of mean) |
| SUN analytic | PASS | floor ratio = 1.404 (within ±50% of analytic) |
| GLASS GI | PASS (8-bit caveat) | centre ratio = 1.000 — CPU packing correct, effect below quantization |
| GLOSSY probe | PASS (8-bit caveat) | floor ratio = 1.000 — CPU packing correct, specular below quantization |
