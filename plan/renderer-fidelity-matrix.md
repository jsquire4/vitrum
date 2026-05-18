# Renderer fidelity matrix (living document)

Date: 2026-05-10

This matrix tracks **truthful** renderer capability claims for `@vitrum/pt-webgl`
(WebGL2 / fork-backed) vs `@vitrum/pt-webgpu` (WGSL prototype advancing toward parity).

## Legend

| Tag          | Meaning                                              |
| ------------ | ---------------------------------------------------- |
| supported    | Implemented + unit tests + captured runtime evidence |
| approximate  | Known simplification; documented                     |
| experimental | May change; incomplete MIS / sampling                |
| unsupported  | Not implemented                                      |

## Feature rows (initial)

| Feature                            | pt-webgl     | pt-webgpu    | Mechanical evidence                                                                                                    | Runtime evidence                                                 | Notes                                                                                |
| ---------------------------------- | ------------ | ------------ | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Hero-wavelength + CMF accumulation | experimental | approximate  | `npm run fork-shader-smoke`; `packages/pt-webgl/src/__tests__/forkUniformBridge.test.ts`                               | `plan/gap-closure-artifacts-2026-05-10.json` currently `BLOCKED` | WebGPU uses RGB probes for thin-film / spectral visualization                        |
| Spectral Beer–Lambert (packed μ)   | experimental | experimental | `npm run fork-shader-smoke`; `packages/pt-webgpu/src/__tests__/scenePack.test.ts`                                      | `plan/gap-closure-artifacts-2026-05-10.json` currently `BLOCKED` | WebGL: fork `transmissionAttenuationHero`; WebGPU: grid + `sampleMaterialSpectralMu` |
| Multi-layer thin film TMM          | experimental | approximate  | `npm run fork-shader-smoke`; `packages/pt-webgpu/src/__tests__/wgslContract.test.ts`                                   | `plan/gap-closure-artifacts-2026-05-10.json` currently `BLOCKED` | WebGPU: incident IOR, extinction k absorption factor, angle-dependent phase scale    |
| Cauchy dispersion                  | experimental | unsupported  | `npm run fork-shader-smoke`                                                                                            | `plan/gap-closure-artifacts-2026-05-10.json` currently `BLOCKED` | WebGPU lacks full Cauchy bridge in WGSL                                              |
| Multi emitter direct lighting      | experimental | experimental | `packages/pt-webgpu/src/__tests__/scenePack.test.ts`; `packages/pt-webgpu/src/__tests__/wgslContract.test.ts`          | `plan/gap-closure-artifacts-2026-05-10.json` currently `BLOCKED` | WebGPU: bounded arrays + uniform counts                                              |
| Caustic strategies                 | experimental | experimental | `packages/pt-webgl/src/__tests__/capabilities.test.ts`; `packages/pt-webgpu/src/__tests__/factoryCapabilities.test.ts` | `plan/gap-closure-artifacts-2026-05-10.json` currently `BLOCKED` | API + uniforms; quality varies by scene                                              |

## Evidence gates

- Mechanical: `npm run typecheck`, `npm test`, `npm run fork-shader-smoke` (sibling fork).
- GPU: `tools/benchmark-runner` with `VITRUM_GPU_CAPTURE=1` and a capture adapter.
  Rows must not be promoted to `supported` until this produces non-null hashes,
  perf fields, and PASS status for the matching acceptance scenario.

See also `plan/gap-closure-acceptance-matrix.md`.
