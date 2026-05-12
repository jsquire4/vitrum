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

> **Note:** All React/Redux host-app files in `legacy-source/` share the
> disposition "Host-only — do not extract"; only files with non-trivial
> migration notes are individually listed below. Everything else follows the
> blanket rule.

| Path | Disposition |
|------|-------------|
| `PathTracingLayer.tsx`, `PathtracerSceneSync.tsx`, `PathtracerDebugBridge.tsx`, `PTStage.tsx` | Host React/Redux — **do not** copy into packages; use `createPTEngine_WebGL2` + `setScene` instead. |
| `PTPostProcessing.tsx`, `PTDeviceLostBoundary.tsx` | Host-only; may serve as `examples/*` reference. |
| `cameraLookPresets.ts`, `outdoorHdri.ts`, `outdoorScenePresets.ts`, `ptEnvironment.ts` | Host-only config/presets — not library concerns. |
| `lighting/usePTPipelineConfig.ts`, `lighting/usePTSampleTarget.ts`, `lighting/renderers/sunPathTraced.tsx` | Host React hooks — engine config belongs in `EngineOptions`, not in hook form. |
| `walkaround/WalkaroundStage.tsx`, `walkaround/HybridLayeredStage.tsx`, `walkaround/engineRegistry.ts` | Host shells — engine logic lives in `packages/walkaround-hybrid`. |
| `walkaround/engines/rc/RcStage.tsx`, `walkaround/engines/restir/RestirStage.tsx`, `walkaround/engines/restir/WalkaroundDebugBridge.tsx`, `walkaround/engines/restir/walkaroundBridgeTypes.ts` | Host shells/types — no extraction needed. |
| `walkaround/useSceneBVH.ts` | Duplicate of `walkaround/lib/useSceneBVH.ts` — **delete the top-level copy** when host app is updated. |
| `walkaround/lib/useSceneBVH.ts` | Canonical host hook; use `@vitrum/shared-bvh` for any library BVH work. |
| `lightingIntensityTable.ts` | **Removed** from staging — use `packages/pt-webgl/src/lightingIntensityTable.ts`. |

When migrating further: **delete** each file here after its behavior is replaced in `packages/` or `examples/`, and update this table.
