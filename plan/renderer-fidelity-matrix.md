# Renderer fidelity matrix (living document)

Date: 2026-05-26 (WG program signoff — see `plan/WG-signoff-2026-05-26.md`)

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
| Hero-wavelength + CMF accumulation | experimental | experimental | `heroWavelengthPlumbing.test.ts`; `shared-samplers/wgsl/heroWavelength*.ts`; opt-in extension | `rfe08-13-spectral-payload` preset → hardware capture | WebGPU: hero-λ MIS + `heroWavelengthToRgb`; opt-in `vitrum.ptWebgpu.spectralHeroWavelength` |
| Spectral Beer–Lambert (packed μ) | experimental | experimental | `scenePack.test.ts`; `scenePack.materials.test.ts` | `rfe08` hardware capture pending | WebGPU: 32-bin grid + hero-λ `sampleMaterialSpectralMu` |
| Multi-layer thin film TMM | experimental | experimental | `wgslContract.test.ts`; per-λ TMM in kernel when spectral on | `rfe14-thinfilm-angle-shift` hardware capture pending | 8-layer stack + incident IOR; single-λ evaluation at hero |
| Cauchy dispersion | experimental | experimental | `materialPacking` Abbe + `cauchyIorAtLambda` in WGSL | hardware capture pending | Requires spectral extension + `dispersionAbbeNumber` |
| Layered front/back + transmission MIS | experimental | experimental | `wgslContract.test.ts` (`activeLayerWeightRgb`, η² PDF) | `rfe03-layered-front-back` preset (`backend: pt-webgpu`) | WG-4 landed 2026-05-26 |
| SSS / translucent panels | experimental | experimental | `wgslContract.test.ts` (`isTranslucent` gate) | `rfe07-11-sss-mixed-panels` preset | Derived translucent flag (transmission + scatteringCoeff) |
| Multi emitter direct lighting | experimental | experimental | `scenePack.test.ts`; `wgslContract.test.ts` | hardware capture pending | Full tier: bounded emitter arrays |
| Material fields parity (cornell) | supported | experimental | `scenePack.test.ts`; `capturePtWebgpu.mjs` | `tools/reference-renders/baseline/ptwgpu-parity-material-fields.png` | WG-0 baseline committed; strict hash on GPU host |
| Caustic strategies | experimental | experimental | `factoryCapabilities.test.ts` | hardware capture pending | Full tier only; lite tier disables |
| SVGF-real denoiser | unsupported | unsupported | `shared-denoisers` + hybrid `svgfReal.ts` | `denoiser: 'svgf-real'` on walkaround-hybrid | pt-webgpu explicitly rejects; pt-webgl uses atrous/OIDN |
| BDPT (eye↔light connections) | experimental | unsupported | `forkUniformBridge.test.ts` (pt-webgl) | fork `FEATURE_BDPT` + host light-path texture | WG-7 deferred on WebGPU |

## Evidence gates

- Mechanical: `npm run typecheck`, `npm test`, `npm run fork-shader-smoke` (sibling fork).
- GPU: `tools/benchmark-runner` with `VITRUM_GPU_CAPTURE=1` and a capture adapter.
  Rows must not be promoted to `supported` until this produces non-null hashes,
  perf fields, and PASS status for the matching acceptance scenario.

See also `plan/gap-closure-acceptance-matrix.md`.
