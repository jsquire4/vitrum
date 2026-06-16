# Renderer fidelity matrix (living document)

Last graded: 2026-06-15 (post-Road reconciliation pass — rows below re-verified against source).

This matrix tracks **truthful** renderer capability claims for `@vitrum/pt-webgl2`
(native WebGL2) and `@vitrum/pt-webgpu` (WebGPU-native).

> **Note:** The former `@vitrum/pt-webgl` (fork-backed) column was removed with
> commit `e14000c` (THREE removal). `@vitrum/pt-webgl2` is the replacement native
> backend. Its fidelity rows are graded from current source truth, not aspirational
> state.

## Legend

| Tag | Meaning |
|-----|---------|
| supported | Implemented + unit tests + captured runtime evidence |
| approximate | Known simplification; documented |
| experimental | May change; incomplete MIS / sampling |
| unsupported | Not implemented |

## Feature rows

| Feature | pt-webgl2 (WebGL2) | pt-webgpu (WebGPU) | Mechanical evidence | Runtime evidence | Notes |
|---------|--------------------|--------------------|---------------------|------------------|-------|
| Hero-wavelength + CMF accumulation | experimental | supported | pt-webgl2: `uploadGapGuard.test.ts` pins `uSpectralRendering`, `uCmfX/Y/Z`, CDFs, and integrals upload when `spectral:true`; pt-webgpu: `heroWavelengthPlumbing.test.ts`; `spectral.test.ts` (66); `shared-samplers/wgsl/heroWavelength*.ts` | pt-webgl2: runtime A/B capture pending. pt-webgpu dzn (RTX 4090) spectral ON/OFF A/B (re-run-confirmed): hero-λ+CMF engaged, glass chroma red-leaning [0.456,0.266,0.278] vs neutral OFF [0.333,0.307,0.360]; baseline `tools/reference-renders/baseline/ptwgpu-spectral-hero.png` (sha `e656e280…`, 512²/256spp/seed 4242) | pt-webgl2 H2 dead-uniform blocker is closed in source/tests; row stays `experimental` until pt-webgl2 spectral runtime A/B has a committed reference. |
| Spectral Beer–Lambert (packed μ) | experimental | supported | pt-webgl2: `materialsTexture.test.ts` pins 32-sample μ grid in `s20..`; GLSL consumes `spectralAttenuationMuHero`; pt-webgpu: `scenePack.materials.test.ts`; `spectral.test.ts` | pt-webgl2: runtime A/B capture pending. pt-webgpu dzn (RTX 4090) μ-curve present-vs-absent A/B (re-run-confirmed): green-peaked packed μ shifts transmitted light magenta — Δ(present−absent)=+0.0254 | pt-webgl2 no longer has the old CMF/μ dead-path blocker, but remains unpromoted without a visual Beer-Lambert reference. |
| Multi-layer thin film TMM | experimental | supported | pt-webgl2: `materialsTexture.test.ts` pins thin-film payload; `thin_film_tmm.glsl.js` is consumed from `bsdf_functions.glsl.js`; pt-webgpu: `wgslContract.test.ts`; per-λ TMM in kernel | pt-webgl2: runtime A/B capture pending. pt-webgpu dzn (RTX 4090) hue-vs-angle A/B (re-run-confirmed): thin-film ON shows angle-dependent chroma drift Δ(R/G)=0.118; baseline `tools/reference-renders/baseline/ptwgpu-thinfilm-angle.png` | pt-webgl2 layered/thin-film uniforms are now packed and consumed; row remains `experimental` pending angle/hue A/B. |
| Cauchy dispersion | experimental | supported | pt-webgl2: `uploadGapGuard.test.ts` pins `iorCauchyA/B/C` upload when `spectral:true`; GLSL applies Cauchy delta with per-material `dispersionStrength`; pt-webgpu: `materialPacking` Abbe + `cauchyIorAtLambda` in WGSL | pt-webgl2: runtime A/B capture pending. pt-webgpu dzn (RTX 4090) Abbe-set-vs-absent A/B (re-run-confirmed): wavelength-dependent IOR alters refracted hue, meanAbsChromaΔ 0.127; baseline `tools/reference-renders/baseline/ptwgpu-cauchy-dispersion.png` | pt-webgl2 requires spectral mode, but no longer has an unsupported hero-path blocker; needs dispersion visual promotion. |
| Layered front/back + transmission MIS | approximate | supported | pt-webgl2: material packer/GLSL consume front/back layer transmission and roughness; `engineContract.test.ts` grades front/back layers approximate; pt-webgpu: `wgslContract.test.ts` (`activeLayerWeightRgb`, η² PDF) | pt-webgl2: runtime A/B capture pending. pt-webgpu dzn (RTX 4090) front/back A/B (re-run-confirmed): layered ON diverges chroma \|Δ\|=0.0593 (×1016 OFF); baseline `tools/reference-renders/baseline/ptwgpu-layered-front.png` | pt-webgl2 consumes layer tint/roughness but not per-layer normal-map subfields, so `approximate` is the truthful grade. |
| SSS / translucent panels | approximate | supported | pt-webgl2: `materialsTexture.test.ts` pins `sssSigmaT`/`sssSigmaS`/aniso lanes; `composeTraceGlsl.test.ts` pins the SSS branch and σ_s/σ_t albedo derivation; pt-webgpu: `wgslContract.test.ts` (`isTranslucent` gate) | pt-webgl2: runtime A/B capture pending. pt-webgpu dzn (RTX 4090) mixed-panel toggle A/B (re-run-confirmed): SSS LOCALIZED to flagged panel — LEFT/RIGHT Δ-luma ratio 1190:1; baseline `tools/reference-renders/baseline/ptwgpu-sss-mixed-panels.png` | pt-webgl2 uses per-material SSS fields and now consumes `scatteringCoefficientRGB` as authored per-channel σ_s; still `approximate` until the scalar-majorant WebGL single-scatter model has visual promotion. |
| Multi emitter direct lighting | approximate | supported | `scenePack.test.ts`; `wgslContract.test.ts`; `lightTreeImportance.test.ts`; `lightsTexture.test.ts`; `composeTraceGlsl.test.ts` | dzn (RTX 4090) baseline `tools/reference-renders/baseline/cornell-manylights.png` (sha256 `e14d0ae…`, 512×512/256spp/seed 6121) | pt-webgl2: the old `1/N` selector-bias note is stale; `randomLightSample()` is power-weighted and `directLightContribution()` uses the shared selector variate. Grade remains `approximate` until unequal-power/mixed-emitter visual A/B promotion proves the full direct-light estimator, not because the old selector bug is still present. |
| Material fields parity (cornell) | supported | supported | `scenePack.test.ts`; `capturePtWebgpu.mjs` | dzn (RTX 4090) strict-hash re-capture == committed `tools/reference-renders/baseline/ptwgpu-parity-material-fields.png` BYTE-FOR-BYTE (PSNR 999 dB; 1280×720/512spp/seed 777) | WG-0 baseline; material packing structural parity confirmed. |
| Caustic strategies | approximate | supported | `factoryCapabilities.test.ts` | MNEE GPU-validated vs DETERMINISTIC references: reflection ratio 0.881, refraction+2-vertex glass 0.987/0.996; baseline `tools/reference-renders/baseline/mnee-glass-slab.png` | pt-webgl2: heuristic photon-map caustic (`causticStrategy:'manifold-nee'` option exists but routes to a phenomenological GLSL path, NOT Newton-solve MNEE — see H7-e / D2; grade 'approximate'). pt-webgpu `manifold-nee` is the validated reference. |
| SVGF-real denoiser | unsupported | unsupported | `unsupportedDenoiserDegrade.test.ts` (warns + degrades to no-denoise) | n/a | Converged tracer → `oidn-final`; SVGF is real-time-only. Both converged backends warn on `'svgf-real'`. |
| BDPT (eye↔light connections) | approximate | supported | `bdptDriver.test.ts`; `composeTraceGlsl.test.ts` | pt-webgpu GPU-validated (V18/V25); baseline `tools/reference-renders/baseline/cornell-bdpt-on.png` | pt-webgl2: the old inert-driver note is stale. `bdpt:true` now issues ordered light-subpath build passes before the eye pass, and `bdpt:false`/no-light cases are structurally pinned. Grade remains `approximate` until pt-webgl2 BDPT has visual A/B promotion against simple emitter scenes. |

## Evidence gates

- Mechanical: `npm run typecheck`, `npm test`.
- GPU: `tools/benchmark-runner` with `VITRUM_GPU_CAPTURE=1` and a capture adapter.
  Rows must not be promoted to `supported` until this produces non-null hashes,
  perf fields, and PASS status for the matching acceptance scenario.
- **Note:** `npm run fork-shader-smoke` was removed with the `@vitrum/pt-webgl` fork
  deletion (commit `e14000c`). The pre-push T1 GPU smoke (`wsl-gpu` lavapipe + dzn)
  remains the primary compile-time gate for the runtime pass graph.
- **Correctness vs perf:** several rows have correctness GPU-validated on dzn/lavapipe
  per `HARDWARE-VALIDATION-NEEDS.md` but stay `experimental` because promotion additionally
  requires a perf field + a strict committed-baseline hash on a real GPU.

See also `plan/gap-closure-acceptance-matrix.md` and `items_to_fix.md` §H for
pending fixes that will upgrade these rows.
