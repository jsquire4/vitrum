# `_staging/` — legacy host-app reference

Files here are **not** shipped in `@vitrum/*` packages. They remain as **read-only reference** from an older host that embedded the renderer next to Redux/React UI.

## Canonical implementations (use these)

| Concern | Package / location |
|--------|---------------------|
| WebGL2 path tracing `Engine` | `@vitrum/pt-webgl` (wraps sibling repo `three-gpu-pathtracer` via `file:` dependency) |
| WebGPU hybrid GI `Engine` | `@vitrum/walkaround-hybrid` (`HybridEngine`) |
| THREE → core `Scene` | `@vitrum/three-bindings` (`sceneFromThreeJS`) |
| Minimal PT demo | `examples/cornell-box` |

## Remaining files (snapshot)

| Path | Disposition |
|------|-------------|
| `PathTracingLayer.tsx`, `PathtracerSceneSync.tsx`, `PTStage.tsx`, … | Host React/Redux — **do not** copy into packages; use `createPTEngine_WebGL2` + `setScene` instead. |
| `walkaround/*Stage.tsx`, `RestirStage.tsx`, `RcStage.tsx`, `WalkaroundDebugBridge.tsx` | Host shells — engine logic lives in `packages/walkaround-hybrid`. |
| `lightingIntensityTable.ts` | **Removed** from staging — use `packages/pt-webgl/src/lightingIntensityTable.ts`. |
| `PTPostProcessing.tsx`, `PTDeviceLostBoundary.tsx`, presets, hooks | Host-only or future `examples/*` helpers. |

When migrating further: **delete** each file here after its behavior is replaced in `packages/` or `examples/`, and update this table.
