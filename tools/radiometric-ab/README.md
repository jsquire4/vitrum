# tools/radiometric-ab

Radiometric A/B harnesses for `pt-webgpu`. Most scripts run full-tier render
variants in the linear HDR domain (`captureFrame({ colorSpace:'linear' })`) —
the raw `accumTexture` float32 values, NOT the tonemapped display output. The
Sobol lane intentionally runs on the WSL-available lite tier as a bounded
correctness/convergence proxy; it is not default-promotion evidence. The
ReSTIR-PT specialty-lobe script is a CPU/static identity proof for scalar and
map-backed-effective lobe payloads, not a GPU recapture.

## What this tests

"Does the math converge to the same answer?" — the tier above the behavioral gate's
"does it render?".  Comparisons:

| Script | Test | Reference | Candidate |
|--------|------|-----------|-----------|
| `ab-sppm.mjs` | SPPM convergence | `causticStrategy:'manifold-nee'` (GPU-validated MNEE) | `causticStrategy:'photon-map'` at 20/50/80 frames |
| `ab-bdpt.mjs` | BDPT unbiasedness + variance | `bdpt:false` (unidirectional) | `bdpt:true` |
| `ab-restir-pt.mjs` | ReSTIR-PT bias + variance | `restirPtReuse:false` (default megakernel) | `restirPtReuse:true` (composite megakernel) |
| `ab-restir-pt-glossy-research.mjs` | ReSTIR-PT glossy research finding | `restirPtReuse:false` | `restirPtReuse:true` with `experimentalGlossyReuse:true` |
| `ab-sobol.mjs` | Sobol equal-frame RMSE proxy | higher-frame PCG | `sampling:'sobol'` at the same frame count as low-frame PCG |
| `ab-restir-pt-specialty.mjs` | ReSTIR-PT specialty-lobe identity | base-path CPU estimator | one-sample ReSTIR-PT producer/finalize/resolve identity for scalar + map-backed-effective clearcoat/sheen/iridescence/aniso/specular payloads |

## How to run

Prerequisites: Deno ≥ 2.8 and a pt-webgpu full-tier WebGPU adapter. The SPPM,
BDPT, and ReSTIR-PT A/B scripts now force `traceTier:"full"` and fail fast when
the adapter resolves to lite, because lite disables caustics/BDPT/ReSTIR-PT and
can otherwise produce false all-zero "passes". The Sobol script forces
`traceTier:"lite"` because its current committed role is WSL-recappable proof
that the opt-in sampler is bounded and non-regressing, not evidence for changing
the default. All GPU A/Bs require non-black linear-HDR captures, so two black
arms cannot be reported as a passing A/B.

```bash
# From the repo root:
export VK_ICD_FILENAMES=/path/to/full-tier/icd.json

# Run each test:
npm run radiometric-ab:sppm

npm run radiometric-ab:bdpt

npm run radiometric-ab:restir-pt

npm run radiometric-ab:restir-pt-glossy-research

npm run radiometric-ab:sobol

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
| ref (manifold-nee, 80) | 0.88287 | — | — |
| SPPM, 20 | 0.66015 | **25.2%** | 0.26320 |
| SPPM, 50 | 0.67388 | **23.7%** | 0.26914 |
| SPPM, 80 | 0.67626 | **23.4%** | 0.23915 |

SPPM trends toward the GPU-validated MNEE reference in the caustic ROI and
finishes within the deliberately loose proof threshold (23.4% final ROI error,
threshold <500%). RMSE is noisy at these low checkpoint counts but improves by
the 80-frame checkpoint. This is convergence evidence for the progressive
Hachisuka update rule, not a claim of final equal-energy promotion.

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
`"verdict":"PASS"`. The opt-in multi-vertex branch is a structured
non-promotion finding: `createPTEngine_WebGPU()` requires
`bdptOptions.experimentalMultiVertex:true`, emits
`pt-webgpu.bdpt-multivertex-research-mode`, and includes
`promotionReady:false`, blocker
`not-weighted-against-regular-eye-path-strategy`,
`multi-vertex-light-subpath-strategies-weighted-against-regular-eye-path-strategy`,
and evidence path `tools/radiometric-ab/results-bdpt.json` in the structured
warning details and committed result snapshot.
Full multi-vertex BDPT promotion requires a redesigned estimator that weights
those strategies against the regular eye-path strategy instead of adding them
as an unweighted research sidecar.

### A/B #3 — ReSTIR-PT reuse on vs off

**PASS** (recaptured 2026-06-21 on the repaired Cornell fixture)

Scene: same Cornell box + metal sphere.
Indirect ROI: same 1,271 pixels.

| Metric | BASE (rpt:off) | RPT (rpt:on) | Notes |
|--------|---------------|-------------|-------|
| Global mean lum (60 frames) | 0.08007 | 0.08640 | relErr = **7.91%** |
| ROI mean lum (60 frames) | 0.10937 | 0.12535 | relErr = **14.61%** |
| Variance in ROI (8×8 frames) | 0.078567 | 0.073875 | ratio = **0.9403** |

ReSTIR-PT composite path agrees with the default megakernel within the global
10% tolerance at 60 spp, and the independent-run ROI variance is not worse
than the base estimator. This capture supersedes the old 2026-06-11 PASS
numbers, which were taken before the Cornell fixture repair moved the back wall
to the visible `z=-1` side.

**2026-06-21 repaired-scene finding + fix:** the first recapture on the repaired
fixture exposed an over-bright RPT arm (`globalRelErr=108.40%`,
`varRatio=7.7143`). Source read showed the producer admitted the scene's rough
metal sphere (`metallic=1`, `roughness=0.08`) into temporal/spatial reuse. The
default producer now admits only diffuse-safe visible vertices
(`metallic <= 0.05 && roughness >= 0.35`); glossy/metallic visible-vertex reuse
is still available behind `restirPtReuseOptions.experimentalGlossyReuse:true`
for research captures, but is not part of the default radiometric proof.

**2026-06-22 glossy research artifact:** `ab-restir-pt-glossy-research.mjs`
now captures the opt-in branch as a committed non-promotion finding:
`results-restir-pt-glossy-research.json` records `verdict:"FINDING"`,
`globalRelErr=108.42%`, `roiRelErr=297.66%`, and `varRatio=7.7140`.
This proves the branch is measured and intentionally kept out of the default
path; it does not promote glossy/metallic visible-vertex reuse.

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
| Sobol | global/ROI RMSE ratio ≤ 1.5 vs equal-frame PCG proxy | elapsed ratio ≤ 20.0 on WSL lite |

### A/B #4 — Sobol equal-frame RMSE proxy

**PASS as bounded opt-in evidence, not a default-promotion claim** (captured
2026-06-21 on WSL lavapipe lite tier).

The harness compares `sampling:'sobol'` against low-frame PCG at the same
12-frame budget, using a 40-frame PCG image as the reference. It covers the
Cornell indirect scene and the caustic-floor stress scene. The Sobol frame key
now preserves a monotonic low 16-bit sample index while using high bits as the
scramble seed, so the source-level sampler is no longer just a hashed random
index stream.

Current result: Sobol stays within the committed 1.5x global/ROI RMSE envelope
on both scenes, but it does not beat PCG enough to justify default promotion.
Keep Sobol opt-in until a full-tier real-adapter equal-time capture shows a
clear convergence win.

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

**Current WSL validation status (2026-06-22):** the latest committed native-Deno
status is `PASS-PARTIAL` from a completed full-suite run. It records SUN as `PASS`
with receiver ratio = 0.99948; the wrapper keeps the aggregate status
partial because GLOSSY remains `FINDING`. The wrapper can still record
`HOST-BLOCKED` if Deno/wgpu-hal panics or times out before a verdict; slow
native-Deno hosts can raise the default 180-second wrapper budget with
`VITRUM_WALKAROUND_AB_TIMEOUT_MS`.

**High-SPP native recaptures (2026-06-22):** `npm run radiometric-ab:walkaround-all-spp64`
preserves a separate `walkaround-ab-all-spp64.json` / `walkaround-ab-all-spp64-status.json`
artifact at 128×128×64 SPP for A8, SUN, GLASS, and GLOSSY together. It keeps
A8 `NEGLIGIBLE`, SUN `PASS`, GLASS `PASS`, and GLOSSY `FINDING`; this strengthens
the native proof while keeping the row `PASS-PARTIAL`. The older
`npm run radiometric-ab:walkaround-glossy-spp64` single-case lane remains useful
for quick focused recapture of the rich-material GI blocker.

The harness now accepts `VITRUM_WALKAROUND_AB_CASES=a8,sun,glass,glossy` to
rerun a subset of cases while preserving the other committed results. The SUN
fixture compares a diffuse-only visible directional receiver against
`Lo = I * cos(theta) * albedo / pi` and disables sky, GTAO, denoising, and the
sun shadow ray for that case. Shadow-visibility promotion remains a separate
transport proof.

### Preserved Linear-HDR Results (2026-06-22)

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

Render time: 16.1 s for the A8 pair.

### SUN — Sun-NEE Analytic Self-Validation

**Verdict: PASS** — direct sun on the visible diffuse receiver matches the
analytic Lambertian value within the committed proof band.

Directional-lit diffuse visible receiver (no area emitter). Config:
`primaryLightDir=[0,0,1]`, sun travel direction `[0,0,-1]`, `I=0.3`,
receiver albedo 0.8. The shader face-forwards the visible back-wall receiver
normal to `[0,0,1]`, so `cosθ=1`. Analytic Lambertian:
`Lo = I × cosθ × albedo / π = 0.3 × 1.0 × 0.8 / π = 0.076394`.

| Metric | Value |
|--------|-------|
| Analytic receiver Lo | 0.076394 |
| Rendered receiver lum | 0.076355 |
| Receiver / analytic | **0.99948** |
| Side window diagnostic | 0.076355 |
| Shadow assertion authored? | NO |

The prior SUN fixture claimed to sample a floor and left-wall shadow region, but
CPU picking showed those screen windows hit the visible back-wall receiver with
this harness camera. The fixture now records that truth explicitly and validates
the direct-sun BRDF path only. Shadow correctness is not inferred from this row.
Render time: 7.0 s.

### GLASS — Glass-GI Transmitted Light Validation

**Verdict: PASS** — the through-glass region is non-black, above the conservative
ratio threshold, and materially different from the no-glass control in the
committed linear-HDR capture.

Cornell+glass pane (transmission=1.0, z=1.5, Beer-tinted) vs Cornell-no-glass,
ceiling emitter, SPP=16.

| Metric | Value |
|--------|-------|
| Glass centre lum | 2.8516 |
| No-glass centre lum | 0.1539 |
| Centre ratio | 18.523 |
| Centre delta | 2.6976 |

The current linear-HDR harness no longer has the old 8-bit readback caveat, and
the visible through-glass crop now proves a material transport delta for this
bounded scene. It is still one low-SPP validation scene, not a global promotion
for every transparent transport case.

Render time: 15.4 s for the pair.

### GLOSSY — B2 Metallic Probe Check (Specular Indirect)

**Verdict: FINDING** — the material path is live and visibly changes the render,
but this low-SPP capture is a do-not-promote rich-material GI result.

Metal visible back-wall center crop (metalness=1.0, rough=0.05) vs diffuse
visible back-wall center crop (metalness=0.0, rough=1.0), Cornell, ceiling
emitter, SPP=16. Older result snapshots expose this metric as `floorLum` /
`floorRatio`; the sampled region is not the geometric floor.

| Metric | Value |
|--------|-------|
| Metal sample-region lum | 0.0009 |
| Diffuse sample-region lum | 0.1715 |
| Sample ratio | 0.005 |
| Sample delta | -0.1706 |

The metallic-probe check proves the authored roughness/metalness path is not a
dead branch: the diffuse control keeps the same base color and the measured
delta isolates material behavior. It is still not a glossy-reference material
furnace. The walkaround DDGI atlas stores cosine-weighted irradiance, not
GGX-filtered radiance, and the low-SPP mirror arm can legitimately reflect a
darker scene direction than the diffuse control. The committed result carries
`promotion.defaultReady:false`, blocker
`ddgi-irradiance-cache-not-ggx-filtered-radiance`, and required evidence
`material-furnace-reference-ab-and-browser-real-adapter-recapture`.
Render time: 13.9 s.

### Summary

| A/B | Verdict | Key Number |
|-----|---------|-----------|
| A8 GRIS bias | NEGLIGIBLE | overall delta = -0.000020 (0.03% of mean) |
| SUN analytic | PASS | receiver ratio = 0.99948 |
| GLASS GI | PASS | centre ratio = 18.523; delta = 2.6976 |
| GLOSSY probe | FINDING | sample ratio = 0.005; material-effect observed |

### Legacy 8-bit Baseline (2026-06-10)

Earlier captures used display-domain 8-bit swap-chain readback. Those numbers
remain useful historical smoke evidence, but they are superseded by the
linear-HDR `captureFrame` results and should not be used as final promotion
evidence.
