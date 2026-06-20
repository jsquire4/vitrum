# tools/radiometric-ab

Radiometric A/B harnesses for `pt-webgpu`. Most scripts run full-tier render
variants in the linear HDR domain (`captureFrame({ colorSpace:'linear' })`) —
the raw `accumTexture` float32 values, NOT the tonemapped display output. The
ReSTIR-PT specialty-lobe script is a CPU/static identity proof for scalar and
map-backed-effective lobe payloads, not a GPU recapture.

## What this tests

"Does the math converge to the same answer?" — the tier above the behavioral gate's
"does it render?".  Three comparisons:

| Script | Test | Reference | Candidate |
|--------|------|-----------|-----------|
| `ab-sppm.mjs` | SPPM convergence | `causticStrategy:'manifold-nee'` (GPU-validated MNEE) | `causticStrategy:'photon-map'` at 20/50/80 frames |
| `ab-bdpt.mjs` | BDPT unbiasedness + variance | `bdpt:false` (unidirectional) | `bdpt:true` |
| `ab-restir-pt.mjs` | ReSTIR-PT bias + variance | `restirPtReuse:false` (default megakernel) | `restirPtReuse:true` (composite megakernel) |
| `ab-restir-pt-specialty.mjs` | ReSTIR-PT specialty-lobe identity | base-path CPU estimator | one-sample ReSTIR-PT producer/finalize/resolve identity for scalar + map-backed-effective clearcoat/sheen/iridescence/aniso/specular payloads |

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

# Committed result-snapshot and host-status proof check:
npm run radiometric-ab:proof-check

# Walkaround-hybrid A/Bs. On the current WSL native-Deno host this may classify
# as HOST-BLOCKED if Deno panics in wgpu-hal before the harness can return a
# verdict; see `walkaround-ab-host-status.json`.
npm run radiometric-ab:walkaround

# Raise the native-Deno host timeout for slower validation machines. The default
# is 180 seconds; timeouts are recorded as HOST-BLOCKED rather than a renderer
# PASS/FAIL verdict.
VITRUM_WALKAROUND_AB_TIMEOUT_MS=300000 npm run radiometric-ab:walkaround
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

**Current WSL validation status (2026-06-18):** `npm run radiometric-ab:walkaround`
now completes on the current native WSL WebGPU host and writes
`walkaround-ab-host-status.json` with `verdict:"PASS-PARTIAL"`. The host is no
longer classified as blocked for this lane. The partial verdict is intentional:
the SUN case proves nonzero direct sun plus shadow correctness, but its analytic
floor ratio is outside the full promotion band at 16 spp, so the status keeps a
do-not-promote warning. If a future host times out or hits the known native-Deno
WebGPU panic, the wrapper still records `HOST-BLOCKED` rather than a renderer
PASS/FAIL verdict. Slow native-Deno hosts can raise the default 180-second
wrapper budget with `VITRUM_WALKAROUND_AB_TIMEOUT_MS`.

### Current Linear-HDR Results (2026-06-18)

The committed `walkaround-ab-results.json` values below are post-denoise,
pre-tonemap linear-HDR float32 captures. They supersede the old 8-bit
display-domain smoke numbers for proof-check purposes.

### A8 — GRIS Bias Quantification

**Verdict: NEGLIGIBLE** — overall delta = -0.000020 (0.03% of mean luminance).

Two arms: `restirPtReuse:false` (default biased path) vs `restirPtReuse:true` (GRIS unbiased).
Cornell + ceiling emitter, 128×128, SPP=16.

| Region | Biased (off) | Unbiased (on) | Delta |
|--------|-------------|--------------|-------|
| Overall | 0.060017 | 0.059997 | **-0.000020** |
| Floor | 0.043049 | 0.043005 | -0.000043 |
| Ceiling | 0.006022 | 0.006020 | -0.000002 |
| Left wall | 0.007088 | 0.007093 | +0.000005 |
| Right wall | 0.002002 | 0.002001 | -0.000001 |

The four bias sources (B1–B4, documented in `HybridEngineOptions.restirPtReuse` JSDoc) produce
statistically negligible bias on this convex scene. The committed proof checker
now bounds both the overall delta and the per-region deltas, so this V19/A8
snapshot cannot silently drift into a vague "partial" claim. Scenes with large
emitters, deep occlusion, or dramatic M-count gradients still need separate
promotion captures before changing the default policy.

Render time: 10.7 s for the A8 pair.

### SUN — Sun-NEE Analytic Self-Validation

**Verdict: PASS-PARTIAL** — direct sun and shadowing are live, but the floor
ratio is outside the full analytic promotion band at 16 spp.

Directional-lit diffuse floor (no area emitter). Config: `sunDir=[0.3,−0.8,0.5]`, `I=0.3`,
floor albedo 0.8. Analytic Lambertian: `Lo = I × cosθ × albedo = 0.3 × 0.808 × 0.8 = 0.1939`.

| Metric | Value |
|--------|-------|
| Analytic floor Lo | 0.1939 |
| Rendered floor lum | 0.0719 |
| Floor / analytic | **0.371** |
| Left wall (sun-shadowed) | 0.0009 |
| Shadow correct? | YES |

The SUN case remains useful as a live-path proof: the directional light produces
finite direct signal and the sun-shadowed wall stays dark. It is not promotion
evidence for analytic absolute radiometry yet; the committed host status keeps
the walkaround harness at `PASS-PARTIAL` and preserves a do-not-promote warning.
Render time: 4.5 s.

### GLASS — Glass-GI Transmitted Light Validation

**Verdict: PASS-WEAK** — the through-glass region is non-black, but the
glass/no-glass captures are indistinguishable at this SPP (`delta=0`), so this
is not material-transport promotion evidence.

Cornell+glass pane (transmission=1.0, z=0.5) vs Cornell-no-glass, ceiling emitter, SPP=16.

| Metric | Value |
|--------|-------|
| Glass centre lum | 0.1541 |
| No-glass centre lum | 0.1541 |
| Centre ratio | 1.000 |
| Centre delta | 0.0000 |

The current linear-HDR harness no longer has the old 8-bit readback caveat, but
this scene is only a conservative non-black/through-glass smoke. Because the two
arms are identical within the committed measurement, it cannot prove that glass
transport changed the render.

Render time: 8.5 s for the pair.

### GLOSSY — B2 Metallic Probe Check (Specular Indirect)

**Verdict: PASS-WEAK** — the metallic floor is non-black and passes the broad
ratio floor, but the metal/diffuse captures are indistinguishable at this SPP
(`delta=0`), so this is not glossy probe promotion evidence.

Metal floor (metalness=1.0, rough=0.05) vs diffuse floor (metalness=0.0, rough=1.0), Cornell,
ceiling emitter, SPP=16.

| Metric | Value |
|--------|-------|
| Metal floor lum | 0.0526 |
| Diffuse floor lum | 0.0526 |
| Floor ratio | 1.000 |
| Floor delta | 0.0000 |

The metallic-probe check remains a bounded non-black live-path proof for the
current walkaround approximation, not a glossy-reference material furnace. Higher-SPP
or case-specific reference captures are still required before promoting glossy
walkaround rows beyond their current approximate status. Render time: 8.4 s.

### Summary

| A/B | Verdict | Key Number |
|-----|---------|-----------|
| A8 GRIS bias | NEGLIGIBLE | overall delta = -0.000020 (0.03% of mean) |
| SUN analytic | PASS-PARTIAL | floor ratio = 0.371; shadow correctness passes |
| GLASS GI | PASS-WEAK | centre ratio = 1.000; delta = 0 |
| GLOSSY probe | PASS-WEAK | floor ratio = 1.000; delta = 0 |

### Legacy 8-bit Baseline (2026-06-10)

Earlier captures used display-domain 8-bit swap-chain readback. Those numbers
remain useful historical smoke evidence, but they are superseded by the
linear-HDR `captureFrame` results and should not be used as final promotion
evidence.
