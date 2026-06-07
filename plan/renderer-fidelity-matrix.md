# Renderer fidelity matrix (living document)

Date: 2026-05-26 (WG program signoff — see `plan/archive/WG-signoff-2026-05-26-archived-2026-05-28.md`)

This matrix tracks **truthful** renderer capability claims for `@vitrum/pt-webgl`
(WebGL2 / fork-backed) vs `@vitrum/pt-webgpu` (WGSL prototype advancing toward parity).

## Legend

| Tag | Meaning |
|-----|--------|
| supported | Implemented + unit tests + captured runtime evidence |
| approximate | Known simplification; documented |
| experimental | May change; incomplete MIS / sampling |
| unsupported | Not implemented |

## Feature rows (initial)

| Feature | pt-webgl | pt-webgpu | Mechanical evidence | Runtime evidence | Notes |
|---------|----------|-----------|---------------------|------------------|-------|
| Hero-wavelength + CMF accumulation | experimental | supported | `heroWavelengthPlumbing.test.ts`; `spectral.test.ts` (66); `shared-samplers/wgsl/heroWavelength*.ts` | dzn (RTX 4090) spectral ON/OFF A/B (re-run-confirmed by me): hero-λ+CMF engaged, glass chroma red-leaning [0.456,0.266,0.278] vs neutral OFF [0.333,0.307,0.360], distinct+finite, **0% out-of-gamut negatives** @256spp; baseline `tools/reference-renders/baseline/ptwgpu-spectral-hero.png` (sha `e656e280…`, 512²/256spp/seed 4242, 177 ms/sample) | WebGPU hero-λ MIS + `heroWavelengthToRgb`; pt-webgl fork stays `experimental` (V9 Jakob-Hanika has 59–72% negatives at software-GL spp) |
| Spectral Beer–Lambert (packed μ) | experimental | supported | `scenePack.materials.test.ts`; `spectral.test.ts` | dzn (RTX 4090) μ-curve present-vs-absent A/B with the hero pipeline held FIXED (re-run-confirmed): green-peaked packed μ shifts transmitted light magenta — glass greenDeficit Δ(present−absent)=+0.0254; baseline shares `ptwgpu-spectral-hero.png`. (V23 also GPU-validated chromatic Beer-Lambert via the `attenuationColor` medium walk.) | WebGPU 32-bin grid + hero-λ `sampleMaterialSpectralMu`; pt-webgl `experimental` |
| Multi-layer thin film TMM | experimental | supported | `wgslContract.test.ts`; per-λ TMM in kernel when spectral on | dzn (RTX 4090) hue-vs-angle A/B (re-run-confirmed): thin-film ON shows angle-dependent chroma drift Δ(R/G)=0.118 Δ(B/G)=0.225 = **18820× OFF's flat 0.000** (iridescence present); `thinFilmTmmRt` consumed at `shadePrologue.wgsl.ts:93,105-107` (verified by code-read). Baseline `tools/reference-renders/baseline/ptwgpu-thinfilm-angle.png` (sha `5a33d83…`, 640²/256spp/seed 9001, 217 ms/sample) | 8-layer stack + incident IOR; RGB-3λ (630/540/460) fallback when spectral off; pt-webgl `experimental` |
| Cauchy dispersion | experimental | supported | `materialPacking` Abbe + `cauchyIorAtLambda` in WGSL | dzn (RTX 4090) Abbe-set-vs-absent A/B on a DEDICATED smooth-glass-slab rig (re-run-confirmed): wavelength-dependent IOR alters the refracted hue, meanAbsChromaΔ(on−off) 0.127 over 96.2% of through-glass px; baseline `tools/reference-renders/baseline/ptwgpu-cauchy-dispersion.png` (sha `6b009505…`, 512²/512spp/seed 9001). NOTE: the in-`rfe08` A/B is confounded (roughness 0.75 blurs dispersion + μ masks it) — the SCENARIO is the smooth slab (`scripts/cauchy-dispersion-ab.ts`), not rfe08 | Requires spectral extension + `dispersionAbbeNumber` (gate ≥1.0); pt-webgl `experimental` |
| Layered front/back + transmission MIS | experimental | supported | `wgslContract.test.ts` (`activeLayerWeightRgb`, η² PDF) | dzn (RTX 4090) front/back + transmission-MIS A/B (re-run-confirmed): layered ON makes front-view & back-view chroma diverge \|Δ\|=0.0593 (**×1016 OFF**) through the η²-PDF refraction path, and each face's layer alters its view (ON−OFF front 0.052 / back 0.007); `activeLayerWeightRgb` consumed at `shadePrologue.wgsl.ts:76` (verified by code-read). Baseline `tools/reference-renders/baseline/ptwgpu-layered-front.png` (sha `e14e99e…`, 512²/320spp/seed 1337, 139 ms/sample) | WG-4; per-face surface-absorption layer; pt-webgl `experimental` |
| SSS / translucent panels | experimental | supported | `wgslContract.test.ts` (`isTranslucent` gate) | dzn (RTX 4090) mixed-panel toggle A/B (re-run-confirmed): SSS LOCALIZED to the flagged panel — LEFT/RIGHT Δ-luma ratio **1190:1** (translucent-flag flips LEFT; opaque-control RIGHT unregressed at Δ 0.0001); baseline `tools/reference-renders/baseline/ptwgpu-sss-mixed-panels.png` (sha `8bebc9a1…`, 512²/256spp/seed 2027). V23 chromatic Beer-Lambert slab A/B (amber R>G>B / cyan B>G>R) GPU-validated | Derived translucent flag (transmission + scatteringCoeff/σ_a); pt-webgl `experimental` |
| Multi emitter direct lighting | experimental | supported | `scenePack.test.ts`; `wgslContract.test.ts`; `lightTreeImportance.test.ts`; `lightTreeConsumption.test.ts` | dzn (RTX 4090) baseline `tools/reference-renders/baseline/cornell-manylights.png` (sha256 `e14d0ae…`, 512×512/256spp/seed 6121, 202.9 ms/sample); unbiased + 60% variance-reduction GPU-validated (HARDWARE-VALIDATION-NEEDS V22, re-confirmed: variance test exists + runs) | Full tier: power-weighted light-tree NEE (Conty Estévez & Kulla 2018); lite keeps uniform pick |
| Material fields parity (cornell) | supported | supported | `scenePack.test.ts`; `capturePtWebgpu.mjs` | dzn (RTX 4090) strict-hash re-capture == committed `tools/reference-renders/baseline/ptwgpu-parity-material-fields.png` BYTE-FOR-BYTE (PSNR 999 dB; 1280×720/512spp/seed 777, 231 ms/sample); dzn↔lavapipe 68.4 dB | WG-0 baseline; pt-webgpu strict-hash confirmed on full-tier dzn (2026-06-04) |
| Caustic strategies | approximate | supported | `factoryCapabilities.test.ts` | MNEE caustic GPU-validated vs DETERMINISTIC references (not the cone-search): reflection (analytic mirror-image, ratio 0.881), refraction + 2-vertex glass-slab chain (forward-traced Snell, 0.987 / 0.996); dzn (RTX 4090) full-tier baseline `tools/reference-renders/baseline/mnee-glass-slab.png` (sha256 `60117577…`, 1280×720/1024spp/seed 27182, 116.4 ms/sample; caustic ON−OFF +14.6/255, ~24 ms/sample). Commits 8cd52cf/893ef00/f38e881 | pt-webgl fork caustic modes are phenomenological and report `pt-webgl-caustic-approximate`; pt-webgpu `manifold-nee` is the validated reference caustic path. pt-webgpu `photon-map` is FORMALLY APPROXIMATE/stylized (reports `pt-webgpu-photon-map-approximate`): GPU A/B vs the same forward-traced oracle (dzn RTX-4090, 2026-06-07) recovers only ~21% of true caustic energy / fires on ~1% of caustic pixels (vs MNEE 98.7% / 99.4%), with a hardcoded world-unit `gatherRadius=0.35` (~6× firing-rate swing under a radiometrically-null scene rescale) + a flat `1+0.25·transmission` brightness fudge (~20% of its reported energy) — see `wsl-gpu/captures/queue-2026-06-07/photon-map/RESULTS.md`. Full tier only; lite disables. SCENARIO: point-light `mnee-glass-slab` — the old `rfe05` is area-lit so MNEE correctly no-ops there |
| SVGF-real denoiser | unsupported | unsupported | `unsupportedDenoiserDegrade.test.ts` (warns + degrades to no-denoise) | n/a | Converged tracer → `oidn-final`; SVGF is real-time-only. Both converged backends warn on `'svgf-real'`; the real SVGF impl lives in `shared-denoisers` for the realtime walkaround stack. |
| BDPT (eye↔light connections) | experimental | supported | `forkUniformBridge.test.ts`, `bdptPlumbing.test.ts` | pt-webgpu GPU `bdptExtendLightSubpath` @compute (V18/V25 GPU-validated: compiles+dispatches+renders, 0 device errors); dzn (RTX 4090) BDPT-ON baseline `tools/reference-renders/baseline/cornell-bdpt-on.png` (512²/256spp/seed 1337, 204 ms/sample), non-black+finite, fireflies ≤ unidirectional. The bidirectional VARIANCE-WIN demo (vs unidir) needs a small/hidden-emitter caustic scene — a follow-up, NOT a promotion gate (playbook #11 correctness-gates BDPT). | pt-webgpu promoted correctness-gated (playbook #11); pt-webgl fork stays `experimental` pending a GL capture path; variance-win A/B is the finer follow-up |

## Evidence gates

- Mechanical: `npm run typecheck`, `npm test`, `npm run fork-shader-smoke` (sibling fork).
- GPU: `tools/benchmark-runner` with `VITRUM_GPU_CAPTURE=1` and a capture adapter.
  Rows must not be promoted to `supported` until this produces non-null hashes,
  perf fields, and PASS status for the matching acceptance scenario.
- **Correctness vs perf (2026-05/06 GPU waves):** several rows have their *correctness*
  GPU-validated on dzn/lavapipe per `HARDWARE-VALIDATION-NEEDS.md` (Möller V7, BDPT
  V18/V25, many-light V22, SSS V23, camera-emitter V26) but stay `experimental` here
  because promotion additionally requires a perf field + a strict committed-baseline hash
  on a real GPU. "Hardware capture pending" in a row means that step, not "nothing ran."

See also `plan/gap-closure-acceptance-matrix.md`.
