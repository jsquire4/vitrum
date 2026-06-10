# Renderer fidelity matrix (living document)

Last graded: 2026-06-09 (post-H remediation pass — all rows re-verified against source).

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
| Hero-wavelength + CMF accumulation | unsupported | supported | `heroWavelengthPlumbing.test.ts`; `spectral.test.ts` (66); `shared-samplers/wgsl/heroWavelength*.ts` | dzn (RTX 4090) spectral ON/OFF A/B (re-run-confirmed): hero-λ+CMF engaged, glass chroma red-leaning [0.456,0.266,0.278] vs neutral OFF [0.333,0.307,0.360]; baseline `tools/reference-renders/baseline/ptwgpu-spectral-hero.png` (sha `e656e280…`, 512²/256spp/seed 4242) | pt-webgl2: spectral GLSL present (fork heritage) but CMF uniforms not uploaded (`glResources.ts` — H2 pending W3); hero-λ path dead. Grade 'unsupported' until H2 lands. |
| Spectral Beer–Lambert (packed μ) | unsupported | supported | `scenePack.materials.test.ts`; `spectral.test.ts` | dzn (RTX 4090) μ-curve present-vs-absent A/B (re-run-confirmed): green-peaked packed μ shifts transmitted light magenta — Δ(present−absent)=+0.0254 | Same gate as hero-λ: μ uniforms not uploaded in pt-webgl2. |
| Multi-layer thin film TMM | unsupported | supported | `wgslContract.test.ts`; per-λ TMM in kernel | dzn (RTX 4090) hue-vs-angle A/B (re-run-confirmed): thin-film ON shows angle-dependent chroma drift Δ(R/G)=0.118; baseline `tools/reference-renders/baseline/ptwgpu-thinfilm-angle.png` | pt-webgl2: fork GLSL has thin-film code but layered uniforms not uploaded. |
| Cauchy dispersion | unsupported | supported | `materialPacking` Abbe + `cauchyIorAtLambda` in WGSL | dzn (RTX 4090) Abbe-set-vs-absent A/B (re-run-confirmed): wavelength-dependent IOR alters refracted hue, meanAbsChromaΔ 0.127; baseline `tools/reference-renders/baseline/ptwgpu-cauchy-dispersion.png` | pt-webgl2: requires spectral hero-λ path (unsupported). |
| Layered front/back + transmission MIS | unsupported | supported | `wgslContract.test.ts` (`activeLayerWeightRgb`, η² PDF) | dzn (RTX 4090) front/back A/B (re-run-confirmed): layered ON diverges chroma \|Δ\|=0.0593 (×1016 OFF); baseline `tools/reference-renders/baseline/ptwgpu-layered-front.png` | pt-webgl2: fork layered GLSL present but uniforms not uploaded. |
| SSS / translucent panels | unsupported | supported | `wgslContract.test.ts` (`isTranslucent` gate) | dzn (RTX 4090) mixed-panel toggle A/B (re-run-confirmed): SSS LOCALIZED to flagged panel — LEFT/RIGHT Δ-luma ratio 1190:1; baseline `tools/reference-renders/baseline/ptwgpu-sss-mixed-panels.png` | pt-webgl2: fork SSS GLSL present (`composeTraceGlsl.ts:470`) but `u_volumeDensity` not uploaded — dead code path. |
| Multi emitter direct lighting | approximate | supported | `scenePack.test.ts`; `wgslContract.test.ts`; `lightTreeImportance.test.ts` | dzn (RTX 4090) baseline `tools/reference-renders/baseline/cornell-manylights.png` (sha256 `e14d0ae…`, 512×512/256spp/seed 6121) | pt-webgl2: lights.count upload fixed (H1, 2026-06-09). Residual MIS bias: selection pdf uses `1/N` (not stored power-weighted prob) — single-light scenes unbiased; multi-light with unequal powers has approximate MIS (H4 pending W3 for full fix). Grade 'approximate' until H4 resolves. |
| Material fields parity (cornell) | supported | supported | `scenePack.test.ts`; `capturePtWebgpu.mjs` | dzn (RTX 4090) strict-hash re-capture == committed `tools/reference-renders/baseline/ptwgpu-parity-material-fields.png` BYTE-FOR-BYTE (PSNR 999 dB; 1280×720/512spp/seed 777) | WG-0 baseline; material packing structural parity confirmed. |
| Caustic strategies | approximate | supported | `factoryCapabilities.test.ts` | MNEE GPU-validated vs DETERMINISTIC references: reflection ratio 0.881, refraction+2-vertex glass 0.987/0.996; baseline `tools/reference-renders/baseline/mnee-glass-slab.png` | pt-webgl2: heuristic photon-map caustic (`causticStrategy:'manifold-nee'` option exists but routes to a phenomenological GLSL path, NOT Newton-solve MNEE — see H7-e / D2; grade 'approximate'). pt-webgpu `manifold-nee` is the validated reference. |
| SVGF-real denoiser | unsupported | unsupported | `unsupportedDenoiserDegrade.test.ts` (warns + degrades to no-denoise) | n/a | Converged tracer → `oidn-final`; SVGF is real-time-only. Both converged backends warn on `'svgf-real'`. |
| BDPT (eye↔light connections) | unsupported (inert) | supported | `bdptPlumbing.test.ts` | pt-webgpu GPU-validated (V18/V25); baseline `tools/reference-renders/baseline/cornell-bdpt-on.png` | pt-webgl2: `bdpt:true` option exists, shader compiles, but the host driver (`bdptAdvanceFrame`) is never called — BDPT is crash-fixed but inert (H5 / D1; grade 'unsupported'). Decision D1: option gates off honestly; driver is road-to-100. |

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
