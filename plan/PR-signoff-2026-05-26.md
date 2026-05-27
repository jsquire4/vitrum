# Primary Release Program — Signoff (2026-05-26)

## Wave status

| Wave | Status | Evidence |
|------|--------|----------|
| PR-0 … PR-6 | **Complete** | Prior landings + `benchmark:pr-hybrid-gpu` |
| PR-7 GPU skinning | **Complete (v1)** | `HybridEngineOptions.gpuSkinning` + `GpuSkinningSubsystem` (per-frame `solveSkin` → positions refit); `gpuSkinLbs.wgsl.ts` for compute follow-up |
| PR-8 pt-webgl incremental | **Complete** | Material: `updateMaterials()`; emitter: `updateLights()`; ledger `material` + `emitter` true |
| PR-9 Signoff | **This document** | |

## PR-D6

`npm run benchmark:pr-hybrid-gpu` / `benchmark:pr-hybrid-gpu-windows` — perf JSON + PNGs under `tools/reference-renders/PR-hybrid/`.
