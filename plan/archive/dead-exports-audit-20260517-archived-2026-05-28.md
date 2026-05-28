# Dead-export audit — 2026-05-17

> **ARCHIVED 2026-05-28.** The 2026-05-18 status block already notes these numbers are stale: post-audit cleanup dropped ~100 dead exports; as of 2026-05-18 knip reported only ~2 intentional design surfaces. The `walkaround-rc` + `stained-glass-extensions` package splits further redistributed counts. Use `npx knip --include exports --reporter compact` from the repo root for a live picture. Current source of truth: the live codebase.

> **Read-only audit.** No exports were removed or renamed. Findings only.
>
> **Status (2026-05-18):** the numbers below are stale. The post-audit
> cleanup sweep (c76b476, a8b312c, 5f72a0f, a786a27, 2e0b9cc, 1cdd534,
> plus follow-ups landed across 2026-05-18) dropped or demoted ~100 dead
> exports across `walkaround-hybrid`, `shared-denoisers`, `pt-webgpu`,
> `pt-webgl`, `three-bindings`, and `core`. The post-W8-extraction
> walkaround-rc + walkaround-hybrid split also redistributes counts. Run
> `npx knip --include exports --reporter compact` from the workspace
> root for a live picture — as of 2026-05-18 it reports only ~2 unused
> exports, both intentional design surfaces (`TUNABLE_DEFINITIONS`
> diagnostic table, `wgslModules.ts` central WGSL barrel).

This audit enumerates every named export from each workspace package's
`src/index.ts`, then searches the workspace for consumers. The intent is to
identify dead weight in the public surface before vitrum reaches 1.0 and the
contract is locked in.

## Methodology

- **Scope:** Every named export from `packages/<pkg>/src/index.ts`, including
  wildcard (`export *`) re-exports resolved to their source files.
- **Search corpus:** All `.ts` and `.tsx` files (including `*.wgsl.ts`) under
  `packages/`, `examples/`, `tools/`, and `_staging/`. `node_modules/`, `dist/`,
  and `.next/` are excluded.
- **Pattern:** Word-boundary `grep -E '\b<name>\b'` against the full corpus.
  Each match counts the *file* once (we report files-with-matches, not raw
  line counts).
- **Classification rules** (per export, by file-hit bucket):
  - `LIVE` — at least one hit in `other_pkg`, `examples/`, or `tools/`
    (i.e., a real cross-package or host consumer).
  - `LIVE-TEST-ONLY` — only test files import it (no other-package /
    example / tool consumer).
  - `LIVE-STAGING-ONLY` — only `_staging/legacy-source/` consumers; the export
    dies with legacy-source.
  - `LIVE-TEST+STAGING` — both test and staging consumers; production code does
    not use it.
  - `DEAD` — no consumer outside the home package's own source files.

`self` = file-hits inside the home package's `packages/<pkg>/` tree
(excluding tests, which are bucketed under `tests`).

## Top-line totals

| Classification | Count |
|---|---|
| **Total exports** | 410 |
| LIVE | 119 |
| LIVE-TEST-ONLY | 123 |
| LIVE-STAGING-ONLY | 18 |
| LIVE-TEST+STAGING | 5 |
| **DEAD** | 145 |

## Per-package summary

| Package | Total | LIVE | LIVE-TEST-ONLY | LIVE-STAGING-ONLY | LIVE-TEST+STAGING | DEAD |
|---|---|---|---|---|---|---|
| core | 50 | 37 | 0 | 1 | 0 | 12 |
| dev | 20 | 5 | 7 | 0 | 0 | 8 |
| engine | 21 | 14 | 3 | 0 | 0 | 4 |
| pt-webgl | 59 | 8 | 13 | 16 | 0 | 22 |
| pt-webgpu | 7 | 3 | 0 | 0 | 0 | 4 |
| shared-bvh | 9 | 5 | 1 | 1 | 1 | 1 |
| shared-denoisers | 59 | 23 | 16 | 0 | 0 | 20 |
| shared-samplers | 58 | 8 | 44 | 0 | 0 | 6 |
| three-bindings | 18 | 7 | 2 | 0 | 0 | 9 |
| walkaround-hybrid | 109 | 9 | 37 | 0 | 4 | 59 |

## Per-package detail

### @vitrum/core

| Export | Classification | Consumers (file count) | Self | Other-pkg | Tests | Staging | Examples | Tools |
|---|---|---|---|---|---|---|---|---|
| `Vec2` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `Vec3` | LIVE | 18 | 2 | 8 | 5 | 0 | 3 | 0 |
| `Vec4` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `Mat4` | LIVE | 9 | 2 | 4 | 1 | 0 | 2 | 0 |
| `SceneNodeId` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `SpectralCurve` | LIVE | 4 | 1 | 1 | 2 | 0 | 0 | 0 |
| `SurfaceAbsorptionLayer` | LIVE | 3 | 1 | 1 | 1 | 0 | 0 | 0 |
| `ThinFilmLayer` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `ThinFilmStack` | LIVE | 3 | 1 | 1 | 1 | 0 | 0 | 0 |
| `Material` | LIVE | 27 | 1 | 17 | 5 | 2 | 2 | 0 |
| `TextureRef` | LIVE | 2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `MeshPrimitive` | LIVE | 4 | 1 | 2 | 1 | 0 | 0 | 0 |
| `InstancedMeshPrimitive` | LIVE | 3 | 1 | 1 | 1 | 0 | 0 | 0 |
| `AnalyticPrimitive` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `AnalyticShape` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `ScenePrimitive` | LIVE | 9 | 2 | 7 | 0 | 0 | 0 | 0 |
| `SceneEmitter` | LIVE | 12 | 2 | 10 | 0 | 0 | 0 | 0 |
| `EmitterBase` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `DirectionalEmitter` | LIVE | 2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `DiscAreaEmitter` | LIVE | 3 | 1 | 2 | 0 | 0 | 0 | 0 |
| `RectAreaEmitter` | LIVE | 3 | 1 | 2 | 0 | 0 | 0 | 0 |
| `PointEmitter` | LIVE | 2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `SpotEmitter` | LIVE | 2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `MeshAreaEmitter` | LIVE | 2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `SceneEnvironment` | LIVE | 6 | 2 | 3 | 1 | 0 | 0 | 0 |
| `HdriEnvironment` | LIVE | 3 | 1 | 2 | 0 | 0 | 0 | 0 |
| `ProceduralSkyEnvironment` | LIVE | 4 | 1 | 3 | 0 | 0 | 0 | 0 |
| `NoneEnvironment` | LIVE | 2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `Scene` | LIVE | 69 | 3 | 34 | 21 | 3 | 8 | 0 |
| `FrameQualitySettings` | LIVE | 2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `FrameInput` | LIVE | 16 | 3 | 9 | 2 | 0 | 2 | 0 |
| `Viewport` | LIVE | 2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `FrameOutput` | LIVE | 9 | 2 | 5 | 2 | 0 | 0 | 0 |
| `BackendTexture` | LIVE | 2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `BackendTextureFormat` | LIVE | 2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `EngineState` | LIVE | 9 | 1 | 4 | 4 | 0 | 0 | 0 |
| `EngineCapabilities` | LIVE | 10 | 2 | 4 | 4 | 0 | 0 | 0 |
| `Engine` | LIVE | 21 | 2 | 11 | 4 | 0 | 4 | 0 |
| `EngineDebugSurface` | LIVE | 10 | 1 | 8 | 1 | 0 | 0 | 0 |
| `FrameStats` | LIVE | 12 | 1 | 8 | 2 | 0 | 1 | 0 |
| `ProgressStats` | LIVE | 9 | 1 | 5 | 2 | 0 | 1 | 0 |
| `EngineFactory` | LIVE | 4 | 1 | 3 | 0 | 0 | 0 | 0 |
| `EngineOptions` | LIVE | 5 | 1 | 4 | 0 | 0 | 0 | 0 |
| `GpuDetection` | LIVE-STAGING-ONLY | 5 | 2 | 0 | 0 | 3 | 0 | 0 |
| `DetectGpuOptions` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `WgpuAdapterKind` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `WgpuProbeResult` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `detectGpu` | LIVE | 6 | 2 | 2 | 0 | 2 | 0 | 0 |
| `probeWebGPU` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `isSwiftShaderAdapter` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |

### @vitrum/dev

| Export | Classification | Consumers (file count) | Self | Other-pkg | Tests | Staging | Examples | Tools |
|---|---|---|---|---|---|---|---|---|
| `FrameStats` | LIVE | 12 | 4 | 5 | 2 | 0 | 1 | 0 |
| `ProgressStats` | LIVE | 9 | 2 | 4 | 2 | 0 | 1 | 0 |
| `EngineDebugSurface` | LIVE | 10 | 7 | 2 | 1 | 0 | 0 | 0 |
| `DebuggableEngine` | LIVE-TEST-ONLY | 11 | 9 | 0 | 2 | 0 | 0 | 0 |
| `FrameTimeHUD` | LIVE-TEST-ONLY | 5 | 3 | 0 | 2 | 0 | 0 | 0 |
| `RingBuffer` | LIVE-TEST-ONLY | 5 | 3 | 0 | 2 | 0 | 0 | 0 |
| `FrameTimeHUDProps` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `DDGIAtlasViewer` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `DDGIAtlasViewerProps` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `BVHVisualizer` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `BVHVisualizerProps` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `GISignalSplit` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `GISignalSplitProps` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `DenoiserABToggle` | LIVE | 5 | 3 | 1 | 1 | 0 | 0 | 0 |
| `DenoiserABToggleProps` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `MaterialInspector` | LIVE | 6 | 3 | 2 | 1 | 0 | 0 | 0 |
| `MaterialInspectorProps` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `attachDebugOverlays` | LIVE-TEST-ONLY | 5 | 3 | 0 | 2 | 0 | 0 | 0 |
| `AttachDebugOverlaysOptions` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `DebugOverlaysHandle` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |

### @vitrum/engine

| Export | Classification | Consumers (file count) | Self | Other-pkg | Tests | Staging | Examples | Tools |
|---|---|---|---|---|---|---|---|---|
| `createEngine` | LIVE | 9 | 5 | 1 | 2 | 0 | 1 | 0 |
| `pickBackend` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `deriveScaleDefaults` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `CreateEngineOptions` | DEAD | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| `EnginePreference` | LIVE | 4 | 3 | 0 | 0 | 0 | 1 | 0 |
| `ScaleDefaults` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `computeSceneAABB` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `SceneAABB` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `attachVitrum` | LIVE | 8 | 4 | 0 | 1 | 0 | 3 | 0 |
| `AttachVitrumOptions` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `AttachVitrumHandle` | LIVE | 7 | 4 | 0 | 0 | 0 | 3 | 0 |
| `Engine` | LIVE | 21 | 4 | 9 | 4 | 0 | 4 | 0 |
| `EngineState` | LIVE | 9 | 1 | 4 | 4 | 0 | 0 | 0 |
| `EngineCapabilities` | LIVE | 10 | 1 | 5 | 4 | 0 | 0 | 0 |
| `Scene` | LIVE | 69 | 5 | 32 | 21 | 3 | 8 | 0 |
| `ScenePrimitive` | LIVE | 9 | 2 | 7 | 0 | 0 | 0 | 0 |
| `SceneEmitter` | LIVE | 12 | 1 | 11 | 0 | 0 | 0 | 0 |
| `Material` | LIVE | 27 | 1 | 17 | 5 | 2 | 2 | 0 |
| `FrameInput` | LIVE | 16 | 4 | 8 | 2 | 0 | 2 | 0 |
| `FrameOutput` | LIVE | 9 | 1 | 6 | 2 | 0 | 0 | 0 |
| `Viewport` | LIVE | 2 | 1 | 1 | 0 | 0 | 0 | 0 |

### @vitrum/pt-webgl

| Export | Classification | Consumers (file count) | Self | Other-pkg | Tests | Staging | Examples | Tools |
|---|---|---|---|---|---|---|---|---|
| `MAX_TILE_GRID` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `TileVariancePass` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `computeAdaptiveTileRepeatFactors` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `linearTileIndexFromVarianceReadPixelsPy` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `ZERO_SAMPLE_COUNT_EPSILON` | LIVE-TEST-ONLY | 5 | 4 | 0 | 1 | 0 | 0 | 0 |
| `readAccumulationRgbFloat` | LIVE | 4 | 2 | 0 | 1 | 0 | 1 | 0 |
| `accumulationFloatRgbaToRgb` | LIVE-TEST-ONLY | 5 | 3 | 0 | 2 | 0 | 0 | 0 |
| `HDR_ACCUM_GOLDEN_BASE64` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `HDR_ACCUM_GOLDEN_EXPECTED_RGB_DIVIDE` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `HDR_ACCUM_GOLDEN_PIXEL_COUNT` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `HDR_ACCUM_GOLDEN_BYTE_LENGTH` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `decodeHdrAccumGoldenBin` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `hdrAccumGoldenBinFromBase64` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `applyFrameToPerspectiveCamera` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `packCameUBO` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `CameSegment` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `CameNode` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `CameUploadOptions` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `CamePackedUBO` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `PTEngineWebGL2` | LIVE | 6 | 2 | 1 | 2 | 0 | 1 | 0 |
| `createPTEngine_WebGL2` | LIVE | 8 | 2 | 1 | 3 | 0 | 2 | 0 |
| `PTEngineWebGL2Options` | LIVE | 3 | 2 | 1 | 0 | 0 | 0 | 0 |
| `PTEngineWebGL2QualityMode` | LIVE | 3 | 2 | 0 | 0 | 0 | 1 | 0 |
| `PTEngineWebGL2Telemetry` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `PTEngineWebGL2FrameOutput` | LIVE | 3 | 2 | 0 | 0 | 0 | 1 | 0 |
| `PT_TARGET_SAMPLES` | LIVE-STAGING-ONLY | 2 | 1 | 0 | 0 | 1 | 0 | 0 |
| `PT_TARGET_SAMPLES_BASE` | LIVE-STAGING-ONLY | 2 | 1 | 0 | 0 | 1 | 0 | 0 |
| `PT_TARGET_SAMPLES_FIXTURES` | LIVE-STAGING-ONLY | 2 | 1 | 0 | 0 | 1 | 0 | 0 |
| `PT_PREVIEW_BOUNCES` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `PT_FINAL_BOUNCES` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `PT_FILTERED_GLOSSY_FACTOR` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `PT_RESOLUTION_FACTOR` | LIVE-STAGING-ONLY | 2 | 1 | 0 | 0 | 1 | 0 | 0 |
| `PT_LOW_RES_SCALE` | LIVE-STAGING-ONLY | 2 | 1 | 0 | 0 | 1 | 0 | 0 |
| `PT_PREVIEW_OPTIONS` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `PT_FINAL_OPTIONS` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `PT_SUN_DISTANCE` | LIVE-STAGING-ONLY | 2 | 1 | 0 | 0 | 1 | 0 | 0 |
| `SUN_ANGULAR_RADIUS` | LIVE | 3 | 1 | 1 | 0 | 1 | 0 | 0 |
| `PT_SUN_DISC_DIAMETER` | LIVE-STAGING-ONLY | 2 | 1 | 0 | 0 | 1 | 0 | 0 |
| `PT_SUN_AREA_INTENSITY` | LIVE-STAGING-ONLY | 2 | 1 | 0 | 0 | 1 | 0 | 0 |
| `bakeSkyEquirect` | LIVE-STAGING-ONLY | 4 | 2 | 0 | 0 | 2 | 0 | 0 |
| `clearSkyEquirectCache` | LIVE-STAGING-ONLY | 3 | 2 | 0 | 0 | 1 | 0 | 0 |
| `debounceMsForEditRate` | LIVE-STAGING-ONLY | 3 | 2 | 0 | 0 | 1 | 0 | 0 |
| `PT_DEBOUNCE_MS_NORMAL` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `PT_DEBOUNCE_MS_BURST` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `computeLightingState` | LIVE | 6 | 2 | 2 | 0 | 2 | 0 | 0 |
| `LightingState` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `LightingStateInputs` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `skyParamsFor` | LIVE-STAGING-ONLY | 4 | 2 | 0 | 0 | 2 | 0 | 0 |
| `worldSunPosition` | LIVE-STAGING-ONLY | 3 | 2 | 0 | 0 | 1 | 0 | 0 |
| `SUN_LIGHT_DISTANCE` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `SkyParams` | LIVE-STAGING-ONLY | 10 | 4 | 0 | 0 | 6 | 0 | 0 |
| `COLOR_TEMP_HEX` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `SUN_INTENSITY` | LIVE-STAGING-ONLY | 4 | 3 | 0 | 0 | 1 | 0 | 0 |
| `getSunIntensity` | LIVE-STAGING-ONLY | 5 | 3 | 0 | 0 | 2 | 0 | 0 |
| `pointIntensityFromLumens` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `rectAreaIntensityFromLumens` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `ForkBridgeBdptOptions` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `ForkBridgeCausticOptions` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `driveForkMaterialUniforms` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |

### @vitrum/pt-webgpu

| Export | Classification | Consumers (file count) | Self | Other-pkg | Tests | Staging | Examples | Tools |
|---|---|---|---|---|---|---|---|---|
| `PT_WEBGPU_COMMON_WGSL` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `HAMMERSLEY_WGSL` | LIVE | 4 | 2 | 2 | 0 | 0 | 0 | 0 |
| `OCTAHEDRAL_CORE_WGSL` | LIVE | 5 | 2 | 3 | 0 | 0 | 0 | 0 |
| `summarizeScene` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `SceneSummary` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `PTEngineWebGPUOptions` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `createPTEngine_WebGPU` | LIVE | 3 | 1 | 0 | 1 | 0 | 1 | 0 |

### @vitrum/shared-bvh

| Export | Classification | Consumers (file count) | Self | Other-pkg | Tests | Staging | Examples | Tools |
|---|---|---|---|---|---|---|---|---|
| `SceneBVHCommonResult` | LIVE-TEST+STAGING | 4 | 1 | 0 | 1 | 2 | 0 | 0 |
| `SceneBVHCommonOpts` | LIVE-STAGING-ONLY | 3 | 1 | 0 | 0 | 2 | 0 | 0 |
| `buildSceneBVH` | LIVE | 15 | 3 | 7 | 2 | 3 | 0 | 0 |
| `validateBvhEncoding` | LIVE-TEST-ONLY | 2 | 1 | 0 | 1 | 0 | 0 | 0 |
| `SceneBvhBuffers` | LIVE | 2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `SceneBvhOptions` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `SceneBvh` | LIVE | 5 | 2 | 3 | 0 | 0 | 0 | 0 |
| `OCTAHEDRAL_WGSL` | LIVE | 3 | 1 | 2 | 0 | 0 | 0 | 0 |
| `BvhIndexStride` | LIVE | 3 | 2 | 1 | 0 | 0 | 0 | 0 |

### @vitrum/shared-denoisers

| Export | Classification | Consumers (file count) | Self | Other-pkg | Tests | Staging | Examples | Tools |
|---|---|---|---|---|---|---|---|---|
| `ATROUS_WGSL` | LIVE | 3 | 1 | 2 | 0 | 0 | 0 | 0 |
| `TEMPORAL_ACCUM_WGSL` | LIVE | 2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `WELFORD_VARIANCE_WGSL` | LIVE | 7 | 3 | 3 | 1 | 0 | 0 | 0 |
| `WELFORD_VARIANCE_VERSION` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `ATROUS_VARIANCE_WGSL` | LIVE | 7 | 4 | 1 | 2 | 0 | 0 | 0 |
| `ATROUS_VARIANCE_COMPUTE_WORKGROUP_SIZE` | DEAD | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| `ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT` | LIVE-TEST-ONLY | 6 | 5 | 0 | 1 | 0 | 0 | 0 |
| `ATROUS_VARIANCE_DEFAULT_ATROUS_ITERATIONS` | LIVE | 6 | 3 | 2 | 0 | 0 | 1 | 0 |
| `ATROUS_VARIANCE_MAX_ATROUS_ITERATIONS` | LIVE | 4 | 3 | 0 | 0 | 0 | 1 | 0 |
| `ATROUS_VARIANCE_FRAME_COUNT_INPUT_GUARD_MAX` | LIVE | 5 | 3 | 0 | 1 | 0 | 1 | 0 |
| `ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES` | LIVE | 6 | 4 | 1 | 1 | 0 | 0 | 0 |
| `ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES` | LIVE | 6 | 4 | 1 | 1 | 0 | 0 | 0 |
| `ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS` | LIVE | 6 | 4 | 1 | 1 | 0 | 0 | 0 |
| `packAtrousVarianceAtrousUniforms` | LIVE | 6 | 4 | 1 | 1 | 0 | 0 | 0 |
| `packAtrousVarianceVarianceUniforms` | LIVE | 6 | 4 | 1 | 1 | 0 | 0 | 0 |
| `AtrousVarianceAtrousUniforms` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `AtrousVarianceVarianceUniforms` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `AtrousVarianceVarianceBindGroupLayout` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `AtrousVarianceAtrousBindGroupLayout` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `denoiseFinal` | LIVE | 4 | 2 | 0 | 1 | 0 | 1 | 0 |
| `preloadOIDNModel` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `clearOIDNCache` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `OIDNDenoiseInputs` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `OIDNDenoiseOptions` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `HDR_LUMINANCE_BILATERAL_WGSL` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `HDR_LUMINANCE_BILATERAL_ENTRY` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `HDR_LUMINANCE_BILATERAL_WORKGROUP_SIZE` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `runHdrLuminanceBilateralWebGPU` | LIVE | 3 | 2 | 0 | 0 | 0 | 1 | 0 |
| `HDR_LUMINANCE_BILATERAL_DEFAULT_SIGMA_LUMINANCE` | LIVE | 3 | 2 | 0 | 0 | 0 | 1 | 0 |
| `HdrLuminanceBilateralWebGPUOptions` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `runAtrousVarianceWebGPU` | LIVE | 8 | 4 | 1 | 2 | 0 | 1 | 0 |
| `assertAtrousVarianceWebGPUBufferShapes` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `ATROUS_VARIANCE_SYNTHETIC_GBUFFER_DEFAULTS` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `AtrousVarianceWebGPUOptions` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `AtrousVarianceSyntheticGbufferFallback` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `SVGF_REPROJECTION_WGSL` | LIVE | 5 | 3 | 1 | 1 | 0 | 0 | 0 |
| `SVGF_REAL_REPROJECTION_WORKGROUP_SIZE` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `SVGF_VARIANCE_FROM_MOMENTS_WGSL` | LIVE | 5 | 3 | 1 | 1 | 0 | 0 | 0 |
| `SVGF_HISTORY_MIN_FOR_MOMENTS` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `SVGF_VARIANCE_FROM_MOMENTS_WORKGROUP_SIZE` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `SVGF_7X7_SPATIAL_FALLBACK_WGSL` | LIVE | 5 | 3 | 1 | 1 | 0 | 0 | 0 |
| `SVGF_SPATIAL_FALLBACK_HISTORY_THRESHOLD` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `SVGF_7X7_SPATIAL_FALLBACK_WORKGROUP_SIZE` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `SVGF_REAL_DEFAULT_ALPHA_MIN` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `SVGF_REAL_DEFAULT_SIGMA_DEPTH` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `SVGF_REAL_DEFAULT_SIGMA_NORMAL` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `SVGF_REAL_DEFAULT_ATROUS_ITERATIONS` | LIVE | 5 | 3 | 2 | 0 | 0 | 0 | 0 |
| `SVGF_REAL_MAX_ATROUS_ITERATIONS` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `SVGF_REPROJ_UNIFORMS_SIZE_BYTES` | LIVE | 5 | 3 | 1 | 1 | 0 | 0 | 0 |
| `SVGF_REPROJ_DEFAULT_UNIFORMS` | LIVE | 5 | 3 | 1 | 1 | 0 | 0 | 0 |
| `packSVGFReprojUniforms` | LIVE | 5 | 3 | 1 | 1 | 0 | 0 | 0 |
| `SVGFReprojUniforms` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `runSVGFRealWebGPU` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `svgfReprojCPU` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `svgfVarianceFromMomentsCPU` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `svgf7x7FallbackCPU` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `SVGFRealWebGPUOptions` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `SVGFReprojCPUInput` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `SVGFReprojCPUOutput` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |

### @vitrum/shared-samplers

| Export | Classification | Consumers (file count) | Self | Other-pkg | Tests | Staging | Examples | Tools |
|---|---|---|---|---|---|---|---|---|
| `HAMMERSLEY_WGSL` | LIVE | 4 | 1 | 3 | 0 | 0 | 0 | 0 |
| `OCTAHEDRAL_CORE_WGSL` | LIVE | 5 | 2 | 3 | 0 | 0 | 0 | 0 |
| `buildLightTree` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `packLightTreeForGPU` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `LightTreeNode` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `LightTreeBuildInput` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `balanceHeuristic` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `powerHeuristic` | LIVE | 7 | 2 | 1 | 4 | 0 | 0 | 0 |
| `mixturePdf` | LIVE-TEST-ONLY | 4 | 2 | 0 | 2 | 0 | 0 | 0 |
| `evaluateHG` | LIVE-TEST-ONLY | 4 | 2 | 0 | 2 | 0 | 0 | 0 |
| `sampleHG` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `pdfHG` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `sampleEquiAngular` | LIVE-TEST-ONLY | 4 | 2 | 0 | 2 | 0 | 0 | 0 |
| `EquiAngularSample` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `EquiAngularOptions` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `rgbToApproxSpectralCoefficients` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `rgbToSpectralCoefficients` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `evaluateSpectrum` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `BDPT_KIND_LIGHT` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `BDPT_KIND_EYE` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `BDPT_KIND_CONNECTION` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `BDPT_KIND_INVALID` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `BDPT_VERTEX_FLOATS` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `BDPT_VERTEX_BYTES` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `BDPT_MAX_LIGHT_BOUNCES` | LIVE | 4 | 2 | 1 | 1 | 0 | 0 | 0 |
| `BDPT_MAX_EYE_BOUNCES` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `packBDPTVertex` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `unpackBDPTVertex` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `BDPTVertex` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `bdptConnectionMIS_partial` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `buildBDPTStrategyPDFs_partial` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `geometricTermG` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `buildBDPTStrategyPDFs_full` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `bdptConnectionMIS_full` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `BDPTFullVertex` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `CIE_X_TABLE` | LIVE | 5 | 3 | 1 | 1 | 0 | 0 | 0 |
| `CIE_Y_TABLE` | LIVE | 5 | 3 | 1 | 1 | 0 | 0 | 0 |
| `CIE_Z_TABLE` | LIVE | 5 | 3 | 1 | 1 | 0 | 0 | 0 |
| `CIE_D65_TABLE` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `CIE_LAMBDA_MIN` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `CIE_LAMBDA_MAX` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `CIE_LAMBDA_STEP` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `CIE_TABLE_LENGTH` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `sampleCMF` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `xyzToLinearSRGB` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `sampleHeroWavelength` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `wavelengthToRGB` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `Y_CMF_INTEGRAL` | LIVE | 4 | 2 | 1 | 1 | 0 | 0 | 0 |
| `HERO_LAMBDA_MIN` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `HERO_LAMBDA_MAX` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `cauchyIOR` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `abbeNumber` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `CAUCHY_CROWN_GLASS` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `CAUCHY_FLINT_GLASS` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `CAUCHY_LEAD_CRYSTAL` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `FRAUNHOFER_D_NM` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `FRAUNHOFER_F_NM` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `FRAUNHOFER_C_NM` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |

### @vitrum/three-bindings

| Export | Classification | Consumers (file count) | Self | Other-pkg | Tests | Staging | Examples | Tools |
|---|---|---|---|---|---|---|---|---|
| `sceneFromThreeJS` | LIVE | 12 | 5 | 2 | 1 | 0 | 4 | 0 |
| `vitrumSceneToThree` | LIVE | 14 | 3 | 4 | 7 | 0 | 0 | 0 |
| `disposeVitrumThreeSceneRoot` | LIVE | 6 | 2 | 1 | 3 | 0 | 0 | 0 |
| `applyEnvironment` | LIVE | 4 | 2 | 1 | 1 | 0 | 0 | 0 |
| `loadGltfScene` | LIVE | 4 | 2 | 0 | 1 | 0 | 1 | 0 |
| `LoadedGltf` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `GltfCamera` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `LoadGltfSceneOptions` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `VITRUM_USER_DATA_KEYS` | LIVE | 5 | 4 | 0 | 0 | 0 | 1 | 0 |
| `extractThreePbrScalars` | LIVE | 5 | 2 | 1 | 2 | 0 | 0 | 0 |
| `PBR_DEFAULTS_DEFAULT` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `colorToVec3` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `convertMaterial` | LIVE-TEST-ONLY | 5 | 4 | 0 | 1 | 0 | 0 | 0 |
| `convertBasicMaterial` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `PbrScalars` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `PbrDefaults` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `ThreeStdMat` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `ThreePhysMat` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |

### @vitrum/walkaround-hybrid

| Export | Classification | Consumers (file count) | Self | Other-pkg | Tests | Staging | Examples | Tools |
|---|---|---|---|---|---|---|---|---|
| `WalkaroundBVHSceneRoot` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `WalkaroundDDGIScene` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `WalkaroundThreeHostScene` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `HybridEngine` | LIVE | 26 | 11 | 9 | 6 | 0 | 0 | 0 |
| `createWalkaroundEngine_Hybrid` | LIVE | 4 | 2 | 1 | 0 | 0 | 1 | 0 |
| `HybridEngineOptions` | LIVE | 10 | 8 | 2 | 0 | 0 | 0 | 0 |
| `DDGI` | LIVE | 56 | 26 | 13 | 12 | 5 | 0 | 0 |
| `DDGIOptions` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `DDGIFrameInputs` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `ProbeUpdatePass` | DEAD | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| `ProbeUpdatePassOptions` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `ProbeGrid` | DEAD | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| `ProbeGridDims` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `ProbeGridParams` | DEAD | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| `IRR_CELL` | LIVE-TEST-ONLY | 7 | 6 | 0 | 1 | 0 | 0 | 0 |
| `VIS_CELL` | LIVE-TEST-ONLY | 6 | 5 | 0 | 1 | 0 | 0 | 0 |
| `BORDER` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `IRR_STRIDE` | DEAD | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| `VIS_STRIDE` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `ATLAS_LAYOUT_WGSL_LOCALS` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `DDGI_SAMPLE_WGSL` | DEAD | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| `DDGILight` | DEAD | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| `WalkaroundGPUPipeline` | LIVE | 20 | 11 | 2 | 5 | 2 | 0 | 0 |
| `HYBRID_WEBGPU_REQUIRED_LIMITS` | LIVE | 6 | 2 | 0 | 3 | 0 | 1 | 0 |
| `HYBRID_WEBGPU_REQUIRED_FEATURES` | LIVE | 6 | 2 | 0 | 3 | 0 | 1 | 0 |
| `PipelineFrameInputs` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `buildReSTIRSceneBVH` | LIVE-TEST-ONLY | 7 | 3 | 0 | 4 | 0 | 0 | 0 |
| `disposeSceneBVH` | LIVE-TEST+STAGING | 8 | 3 | 0 | 4 | 1 | 0 | 0 |
| `SceneBVHBuffers` | LIVE-TEST+STAGING | 7 | 4 | 0 | 2 | 1 | 0 | 0 |
| `COMMON_WGSL` | LIVE | 14 | 10 | 1 | 3 | 0 | 0 | 0 |
| `RIS_WGSL` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `TEMPORAL_WGSL` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `SPATIAL_WGSL` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `SHADE_WGSL` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `COMPOSITE_VERT_WGSL` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `COMPOSITE_FRAG_WGSL` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `upgradeToNodeMaterial` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `SURFACE_TEXTURE_ID` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `SurfaceTextureName` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `SurfaceTextureId` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `CASCADE_DIMS` | LIVE-TEST-ONLY | 6 | 5 | 0 | 1 | 0 | 0 | 0 |
| `CASCADE_COUNT` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `allocateCascades` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `disposeCascades` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `fillCascadeDebug` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `CascadeDim` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `CascadeBuffers` | DEAD | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| `buildRCSceneBVH` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `RCSceneBVH` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `RCBvhBuildOpts` | DEAD | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `RCDispatcher` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `dispatchCascadePasses` | LIVE-TEST+STAGING | 4 | 2 | 0 | 1 | 1 | 0 | 0 |
| `disposeSharedDispatcher` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `RCDispatchOpts` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `CascadeBufferManager` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `GIReceiver` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `GIReceiverExclusionPredicate` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `GIReceiverOptions` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `buildWalkaroundLightingNode` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `WalkaroundLightingNodes` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `applyDDGIShading` | LIVE-TEST+STAGING | 10 | 8 | 0 | 1 | 1 | 0 | 0 |
| `PROBE_RAY_CAST_WGSL` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `CASCADE_MERGE_WGSL` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `FrameResourceOptions` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `buildSTree` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `findSTreeLeaf` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `sTreeAccumulate` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `splitOverflowLeaves` | LIVE-TEST-ONLY | 5 | 4 | 0 | 1 | 0 | 0 | 0 |
| `resetAccumulators` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `aabbContains` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `buildEmptyDTree` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `findDTreeLeaf` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `dTreeSample` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `dTreePdf` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `refineDTree` | LIVE-TEST-ONLY | 5 | 4 | 0 | 1 | 0 | 0 | 0 |
| `sumLeafSolidAngles` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `sumLeafPdfIntegrals` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `computeMISWeights` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `PPG_CELL_SPLIT_THRESHOLD` | LIVE-TEST-ONLY | 5 | 4 | 0 | 1 | 0 | 0 | 0 |
| `PPG_DTREE_FLUX_FRACTION` | LIVE-TEST-ONLY | 5 | 4 | 0 | 1 | 0 | 0 | 0 |
| `PPG_DTREE_MERGE_FRACTION` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `PPG_DTREE_MAX_DEPTH` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `PPG_DTREE_INITIAL_DEPTH` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `PPG_MIS_ALPHA` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `PPG_MIS_ALPHA_MIN` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `PPG_MIS_ALPHA_MAX` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `PPG_MAX_SPATIAL_CELLS` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `AABB` | LIVE | 26 | 11 | 9 | 5 | 0 | 1 | 0 |
| `STreeNode` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `DTreeNode` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `DTree` | DEAD | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| `STree` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `PPGModelHandle` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `PPG_UPDATE_WGSL` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `PPG_GUIDE_WGSL` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `InferenceGraph` | LIVE-TEST-ONLY | 6 | 5 | 0 | 1 | 0 | 0 | 0 |
| `buildUNetSpec` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `WALKAROUND_DENOISER_UNET_SPEC` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `UNetSpec` | DEAD | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| `LayerSpec` | LIVE-TEST-ONLY | 5 | 4 | 0 | 1 | 0 | 0 | 0 |
| `LayerKind` | DEAD | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `LayerWeightLayout` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `LayerParams` | DEAD | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `loadWeightsFromArrayBuffer` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |
| `serializeWeightsToArrayBuffer` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `VITRUM_MODEL_MAGIC` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `VITRUM_MODEL_VERSION` | LIVE-TEST-ONLY | 3 | 2 | 0 | 1 | 0 | 0 | 0 |
| `ModelWeights` | LIVE-TEST-ONLY | 5 | 4 | 0 | 1 | 0 | 0 | 0 |
| `LayerWeights` | LIVE-TEST-ONLY | 4 | 3 | 0 | 1 | 0 | 0 | 0 |

## Safe-to-delete shortlist (high-confidence DEAD)

These exports have zero consumers anywhere in the workspace (other-pkg, tests, staging, examples, tools all zero). They are safe candidates for deletion in the next refactor pass.

**Caveat:** none of these are reachable via dynamic `import(varName)` patterns — see Limitations below. The only non-string dynamic import in the workspace is the ONNX runtime loader in `shared-denoisers/oidnBridge.ts`, which imports an external package, not a workspace export.

**@vitrum/core** (12):

- `Vec2`
- `Vec4`
- `SceneNodeId`
- `ThinFilmLayer`
- `AnalyticPrimitive`
- `AnalyticShape`
- `EmitterBase`
- `DetectGpuOptions`
- `WgpuAdapterKind`
- `WgpuProbeResult`
- `probeWebGPU`
- `isSwiftShaderAdapter`

**@vitrum/dev** (8):

- `FrameTimeHUDProps`
- `DDGIAtlasViewerProps`
- `BVHVisualizerProps`
- `GISignalSplitProps`
- `DenoiserABToggleProps`
- `MaterialInspectorProps`
- `AttachDebugOverlaysOptions`
- `DebugOverlaysHandle`

**@vitrum/engine** (4):

- `CreateEngineOptions`
- `ScaleDefaults`
- `SceneAABB`
- `AttachVitrumOptions`

**@vitrum/pt-webgl** (22):

- `MAX_TILE_GRID`
- `TileVariancePass`
- `computeAdaptiveTileRepeatFactors`
- `applyFrameToPerspectiveCamera`
- `CameUploadOptions`
- `CamePackedUBO`
- `PTEngineWebGL2Telemetry`
- `PT_PREVIEW_BOUNCES`
- `PT_FINAL_BOUNCES`
- `PT_FILTERED_GLOSSY_FACTOR`
- `PT_PREVIEW_OPTIONS`
- `PT_FINAL_OPTIONS`
- `PT_DEBOUNCE_MS_NORMAL`
- `PT_DEBOUNCE_MS_BURST`
- `LightingState`
- `LightingStateInputs`
- `SUN_LIGHT_DISTANCE`
- `COLOR_TEMP_HEX`
- `pointIntensityFromLumens`
- `rectAreaIntensityFromLumens`
- `ForkBridgeBdptOptions`
- `ForkBridgeCausticOptions`

**@vitrum/pt-webgpu** (4):

- `PT_WEBGPU_COMMON_WGSL`
- `summarizeScene`
- `SceneSummary`
- `PTEngineWebGPUOptions`

**@vitrum/shared-bvh** (1):

- `SceneBvhOptions`

**@vitrum/shared-denoisers** (20):

- `ATROUS_VARIANCE_COMPUTE_WORKGROUP_SIZE`
- `AtrousVarianceVarianceBindGroupLayout`
- `AtrousVarianceAtrousBindGroupLayout`
- `HDR_LUMINANCE_BILATERAL_WORKGROUP_SIZE`
- `HdrLuminanceBilateralWebGPUOptions`
- `ATROUS_VARIANCE_SYNTHETIC_GBUFFER_DEFAULTS`
- `AtrousVarianceWebGPUOptions`
- `AtrousVarianceSyntheticGbufferFallback`
- `SVGF_REAL_REPROJECTION_WORKGROUP_SIZE`
- `SVGF_VARIANCE_FROM_MOMENTS_WORKGROUP_SIZE`
- `SVGF_SPATIAL_FALLBACK_HISTORY_THRESHOLD`
- `SVGF_7X7_SPATIAL_FALLBACK_WORKGROUP_SIZE`
- `SVGF_REAL_DEFAULT_ALPHA_MIN`
- `SVGF_REAL_DEFAULT_SIGMA_DEPTH`
- `SVGF_REAL_DEFAULT_SIGMA_NORMAL`
- `SVGF_REAL_MAX_ATROUS_ITERATIONS`
- `runSVGFRealWebGPU`
- `SVGFRealWebGPUOptions`
- `SVGFReprojCPUInput`
- `SVGFReprojCPUOutput`

**@vitrum/shared-samplers** (6):

- `LightTreeNode`
- `EquiAngularSample`
- `EquiAngularOptions`
- `rgbToApproxSpectralCoefficients`
- `FRAUNHOFER_F_NM`
- `FRAUNHOFER_C_NM`

**@vitrum/three-bindings** (9):

- `LoadedGltf`
- `GltfCamera`
- `LoadGltfSceneOptions`
- `colorToVec3`
- `convertBasicMaterial`
- `PbrScalars`
- `PbrDefaults`
- `ThreeStdMat`
- `ThreePhysMat`

**@vitrum/walkaround-hybrid** (59):

- `WalkaroundBVHSceneRoot`
- `WalkaroundDDGIScene`
- `WalkaroundThreeHostScene`
- `DDGIOptions`
- `DDGIFrameInputs`
- `ProbeUpdatePass`
- `ProbeUpdatePassOptions`
- `ProbeGrid`
- `ProbeGridDims`
- `ProbeGridParams`
- `BORDER`
- `IRR_STRIDE`
- `VIS_STRIDE`
- `ATLAS_LAYOUT_WGSL_LOCALS`
- `DDGI_SAMPLE_WGSL`
- `DDGILight`
- `PipelineFrameInputs`
- `RIS_WGSL`
- `TEMPORAL_WGSL`
- `SPATIAL_WGSL`
- `COMPOSITE_VERT_WGSL`
- `COMPOSITE_FRAG_WGSL`
- `upgradeToNodeMaterial`
- `SurfaceTextureName`
- `SurfaceTextureId`
- `allocateCascades`
- `disposeCascades`
- `fillCascadeDebug`
- `CascadeDim`
- `CascadeBuffers`
- `buildRCSceneBVH`
- `RCSceneBVH`
- `RCBvhBuildOpts`
- `disposeSharedDispatcher`
- `GIReceiverExclusionPredicate`
- `GIReceiverOptions`
- `WalkaroundLightingNodes`
- `FrameResourceOptions`
- `resetAccumulators`
- `aabbContains`
- `findDTreeLeaf`
- `dTreeSample`
- `computeMISWeights`
- `PPG_DTREE_MERGE_FRACTION`
- `PPG_DTREE_MAX_DEPTH`
- `PPG_MIS_ALPHA`
- `PPG_MIS_ALPHA_MIN`
- `PPG_MIS_ALPHA_MAX`
- `PPG_MAX_SPATIAL_CELLS`
- `STreeNode`
- `DTreeNode`
- `DTree`
- `STree`
- `PPGModelHandle`
- `PPG_GUIDE_WGSL`
- `UNetSpec`
- `LayerKind`
- `LayerWeightLayout`
- `LayerParams`

## LIVE-STAGING-ONLY — dies when `_staging/legacy-source/` is removed

These exports are kept alive *only* by `_staging/legacy-source/` references. Per `_staging/README.md`, that staging tree contains host-app files intentionally not extracted. Once it is finally deleted, these exports become DEAD.

- `@vitrum/core` → `GpuDetection`
- `@vitrum/pt-webgl` → `PT_TARGET_SAMPLES`
- `@vitrum/pt-webgl` → `PT_TARGET_SAMPLES_BASE`
- `@vitrum/pt-webgl` → `PT_TARGET_SAMPLES_FIXTURES`
- `@vitrum/pt-webgl` → `PT_RESOLUTION_FACTOR`
- `@vitrum/pt-webgl` → `PT_LOW_RES_SCALE`
- `@vitrum/pt-webgl` → `PT_SUN_DISTANCE`
- `@vitrum/pt-webgl` → `PT_SUN_DISC_DIAMETER`
- `@vitrum/pt-webgl` → `PT_SUN_AREA_INTENSITY`
- `@vitrum/pt-webgl` → `bakeSkyEquirect`
- `@vitrum/pt-webgl` → `clearSkyEquirectCache`
- `@vitrum/pt-webgl` → `debounceMsForEditRate`
- `@vitrum/pt-webgl` → `skyParamsFor`
- `@vitrum/pt-webgl` → `worldSunPosition`
- `@vitrum/pt-webgl` → `SkyParams`
- `@vitrum/pt-webgl` → `SUN_INTENSITY`
- `@vitrum/pt-webgl` → `getSunIntensity`
- `@vitrum/shared-bvh` → `SceneBVHCommonOpts`

## LIVE-TEST+STAGING — production code does not use these

Tests and `_staging` consumers only; no production package or example imports these.

- `@vitrum/shared-bvh` → `SceneBVHCommonResult`
- `@vitrum/walkaround-hybrid` → `disposeSceneBVH`
- `@vitrum/walkaround-hybrid` → `SceneBVHBuffers`
- `@vitrum/walkaround-hybrid` → `dispatchCascadePasses`
- `@vitrum/walkaround-hybrid` → `applyDDGIShading`

## Limitations

This audit's classification is best-effort and conservative. The following
cases would not be caught by a word-boundary symbol search across `.ts` and
`.tsx` files:

1. **Truly-dynamic dynamic imports.** A single non-string dynamic import exists
   in the workspace: `shared-denoisers/oidnBridge.ts:296` uses
   `new Function('id', 'return import(id)')` to lazy-load `onnxruntime-web`.
   That target is an external package, not a workspace export, so it does
   *not* affect any DEAD classification here. All other dynamic imports in
   the workspace use string literals and are visible to grep.
2. **String-templated module specifiers.** The audit can't see
   `import(\`@vitrum/${pkg}\`)` patterns. None observed, but worth disclosing.
3. **Cross-file member-access matches.** Word-boundary search will match
   `obj.SomeName` even though that's a property access, not an import.
   Inspection of the DEAD set found no false-positive matches of this form.
4. **Comment-only / JSDoc-only references.** A symbol mentioned only in a
   doc-comment is counted as a hit. This *inflates* `total` (favours LIVE
   classification), so it produces false negatives on the DEAD list — not
   false positives.
5. **Type-only re-exports from another package.** If a host app re-exports a
   vitrum type and then uses the re-exported alias, the original name may
   appear unused. Not observed in the audit corpus, but possible in
   downstream apps.
6. **WGSL string usage.** A symbol whose string value is embedded into a WGSL
   shader template (e.g., constant numeric values used by `${IRR_CELL}`)
   would still match by name because the constant is referenced by the JS
   identifier. Symbols that flow only as numeric literals into WGSL without
   being referenced by JS identifier would be miscounted — but no such case
   exists in the audited surface (every WGSL-baked constant is also accessed
   by name elsewhere in JS).
7. **Tooling / build scripts outside the audited corpus.** This audit covers
   `packages/`, `examples/`, `tools/`, and `_staging/`. Top-level build
   scripts (`scripts/`, vitest configs, vite configs) are not searched — if
   any reference a workspace export by name, they would be missed. A spot
   check of `vite.config.ts` and `vitest.config.ts` files found no
   workspace-export imports outside the audited corpus.
8. **Same-name exports in multiple packages.** When two packages export the
   same name (e.g., `FrameStats` is exported from both `@vitrum/core` and
   `@vitrum/dev`, `HAMMERSLEY_WGSL` from both `@vitrum/shared-samplers` and
   `@vitrum/pt-webgpu`), each package is classified independently using
   its own `self` boundary. A consumer that imports the symbol from
   either source counts as an `other_pkg` hit for *both* home packages.
   This is the conservative choice — it favours LIVE classifications.

## Reproduction

The audit script and intermediate artifacts live at
`/tmp/dead-exports-audit/` (not committed):

- `exports.txt` — enumerated 410 named exports (one per line, `pkg|name`).
- `scan.sh` — bash + grep word-boundary scanner; produces `results.csv`.
- `results.csv` — pipe-separated per-export breakdown
  (`pkg|name|total|self|other_pkg|tests|staging|examples|tools|classification`).
- `build_report.sh` — generates this report from `results.csv`.
