# Remediation checklist (complexity / verification sweep)

**Started:** 2026-05-10. **Completed:** 2026-05-10. Verify with `npm run typecheck && npm test`.

## Phase A — `@vitrum/core`

- [x] **A1** Document or align `Material` mutability vs readonly scene types (normative JSDoc on `Material`).
- [x] **A2** `detectGpu()`: optional `publishToWindow` / callback so `window.__WG__` is not mandatory.
- [x] **A3** `probeWebGPU` / `WgpuProbeResult`: add `adapterKind: 'hardware' | 'swiftshader' | 'unknown'`; keep `isHardwareGpu` as `@deprecated` compat.
- [x] **A4** Reduce `any` in `wgpuSupport` adapter probing (minimal structural types).

## Phase B — `@vitrum/shared-bvh`

- [x] **B1** Runtime guard / clear error when `MeshBVH` `_roots` buffer shape is wrong (version pin note).
- [x] **B2** `SceneBvh.update`: use `traverseVisible` to match `buildSceneBVH`; fix misleading JSDoc.
- [x] **B3** Optional `onSlowRebuild` callback on `SceneBvh` instead of only `console.warn`.
- [x] **B4** Document multi-material `material[0]` limitation in `snapshotPreBuildMaterials` (JSDoc).

## Phase C — `@vitrum/shared-samplers`

- [x] **C1** `buildBDPTStrategyPDFs`: JSDoc matches actual strategy support (simplified connection PDFs; boundary cases `s=0` / `t=0` / `k=s`).
- [x] **C2** `bdpt.test.ts`: assert documented PDF pattern; add comment on MIS partial-sum test.
- [x] **C3** `rgbToApproxSpectralCoefficients` + `@deprecated` alias `rgbToSpectralCoefficients`.
- [x] **C4** `sampleEquiAngular`: optional `sceneTMax` / `degenerateFallbackLength`.
- [x] **C5** Canonical `OCTAHEDRAL_CORE_WGSL` + extend `HAMMERSLEY_WGSL` with shared `rotateAngleAxis` (DDGI uses it).

## Phase D — `@vitrum/pt-webgpu`

- [x] **D1** Depend on `@vitrum/shared-samplers`; drop duplicate `hammersley.wgsl.ts` / `octahedral.wgsl.ts`; re-export compat names from `index.ts`.
- [x] **D2** (Deferred / larger) Dedup `traceClosest` vs `traceAny` analytic+BVH bodies via composable WGSL fragments.

## Phase E — `@vitrum/walkaround-hybrid`

- [x] **E1** Gate `WalkaroundGPUPipeline` init `console.log` behind `debug` / `verbose` from engine.
- [x] **E2** Gate `pipelineCompiler` success `console.log` behind verbose.
- [x] **E3** `cascadeDispatch`: typed WebGPU backend guard instead of raw `as any`.
- [x] **E4** `nodeMaterialUpgrade`: `Record<string, unknown>` copy loop (remove `as any`).
- [x] **E5** `rc/bvhCompute` `packCascadeMaterials`: `instanceof` branches instead of unsafe cast.
- [x] **E6** (Deferred) `ppgSample.wgsl` O(N) → spatial acceleration follow-up.

## Phase F — `@vitrum/babylon-bindings`

- [x] **F1** `@deprecated` on `sceneFromBabylonScene` with migration note; keep throw until implementation exists.

## Hygiene

- [x] **H1** `CHANGELOG.md`: note API additions/changes (deprecated spectral alias, `detectGpu` options, `probeWebGPU` field, WGSL import path, pipeline logging).
