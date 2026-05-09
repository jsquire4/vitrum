# `_staging/` — legacy source for decomposition

This directory holds renderer source files imported from a host application that previously embedded the engine code directly. **Nothing here is the canonical implementation yet.** Files are read-only reference; the agent's job is to extract them into `packages/<name>/src/` with whatever restructuring + renaming + interface-fitting is appropriate.

## What's in `legacy-source/`

```
legacy-source/src/rendering/scene/
├── walkaround/                          # ← @vitrum/walkaround-hybrid (the crown jewel)
│   ├── HybridLayeredStage.tsx           # React stage wrapper — extract host-agnostic engine
│   ├── WalkaroundStage.tsx              # Older walkaround stage; may be deprecated by HybridLayered
│   ├── applyDDGIShading.ts
│   ├── bvhCompute.ts                    # ← @vitrum/shared-bvh
│   ├── cascadeDispatch.ts               # Radiance Cascades dispatch
│   ├── cascadePyramid.ts                # Radiance Cascades data structure
│   ├── ddgiAtlasLayout.ts               # DDGI probe atlas
│   ├── ddgiSampleWgsl.ts
│   ├── engineRegistry.ts                # Walkaround engine selector
│   ├── engines/
│   │   ├── rc/                          # Radiance Cascades standalone engine
│   │   └── restir/                      # ReSTIR DI engine — main implementation
│   │       ├── RestirStage.tsx
│   │       ├── WalkaroundDebugBridge.tsx
│   │       ├── WalkaroundGPUPipeline.ts # The 1287-LOC god file (a known refactoring target)
│   │       ├── bvhCompute.ts            # Different from outer bvhCompute — engine-specific
│   │       └── shaders/                 # WGSL shader code
│   ├── giReceiver.ts
│   ├── gpuDetection.ts                  # WebGPU support detection — could move to @vitrum/core
│   ├── lib/
│   │   ├── bvhCommon.ts                 # ← @vitrum/shared-bvh
│   │   ├── bvhCommon.test.ts            # tests come along too
│   │   ├── nodeMaterialUpgrade.ts
│   │   ├── useSceneBVH.ts               # Note: also in outer dir — pick one
│   │   └── wgpuSupport.ts
│   ├── probeGrid.ts                     # DDGI probe grid math
│   ├── probeUpdatePass.ts
│   ├── sceneBvh.ts
│   ├── useCascadeBuffers.ts
│   ├── useDDGI.ts
│   ├── useHybridLayeredGI.ts            # The hook that orchestrates DDGI+RC+ReSTIR
│   ├── useSceneBVH.ts                   # Also in lib/ — pick one
│   ├── walkaroundDiffuseLighting.ts
│   └── wgsl/                            # Shared WGSL fragments
│       ├── hammersley.wgsl.ts           # ← @vitrum/shared-samplers
│       ├── octahedral.wgsl.ts           # ← @vitrum/shared-bvh (used by DDGI atlas)
│       ├── probeUpdateBlend.wgsl.ts
│       └── probeUpdateRays.wgsl.ts
│
├── PathTracingLayer.tsx                 # ← @vitrum/pt-webgl host wrapper.
├── PathtracerSceneSync.tsx              # Scene → setScene plumbing
├── PathtracerDebugBridge.tsx            # window.__PT__ debug bridge — keep dev-only
├── PTStage.tsx                          # React stage wrapper
├── PTPostProcessing.tsx                 # Bloom + DoF + CA + Vignette + Grain
├── PTDeviceLostBoundary.tsx             # React error boundary — host concern
├── cameraLookPresets.ts                 # Three documentary/cinematic/architectural presets
├── pathtracerConstants.ts               # PT_PREVIEW + PT_FINAL configs
├── ptDebounce.ts                        # Camera-change debounce
├── ptEnvironment.ts                     # IBL setup (HDRI loader + procedural sky bake)
├── ptIblBaker.ts                        # Procedural sky → equirect texture bake
├── lightingState.ts                     # computeLightingState — single source of truth for sun/sky params
├── skyParams.ts                         # Preetham sky parameters
├── outdoorHdri.ts                       # HDRI URL bucket dispatch
├── outdoorScenePresets.ts               # 4 user-selectable HDRI presets
├── lightingIntensityTable.ts            # SUN_INTENSITY buckets + PT_IBL_INTENSITY
├── lighting/
│   ├── usePTPipelineConfig.ts           # Picks PT_PREVIEW vs PT_FINAL config
│   ├── usePTSampleTarget.ts             # Per-scene sample target tier
│   └── renderers/
│       └── sunPathTraced.tsx            # ShapedAreaLight sun (Phase 1 deliverable)
```

## What was deliberately NOT included

- **The vendor fork** — lives at `~/projects/three-gpu-pathtracer/` as a sibling repo (remote: `github.com/jsquire4/three-gpu-pathtracer`). When `@vitrum/pt-webgl` first wraps it, decide whether to git-submodule it, leave it as a sibling repo and reference via `file:` npm path, or move its source under vitrum's tree (and lose the independent git history). Surface the choice to the user.
- **Domain-specific code** — anything specific to a particular content domain (panel/cell/came/glass-material concepts). That belongs in a future domain-specific package, not vitrum.
- **App lifecycle / React glue** — `StudioScene.tsx`, `StudioCameraRig.tsx`, `StageOrbitControls.tsx`, etc. are host concerns. The host calls `engine.renderFrame()`; the host is not the engine.
- **Tests** — `*.test.ts` files mostly travelled with their source (e.g., `bvhCommon.test.ts`). When you extract a file to `packages/X/src/`, move its test to `packages/X/src/__tests__/` (or `packages/X/test/`).

## What to do with this directory

The agent should:

1. **Read each file's responsibilities** (`grep -l <thing>` against `_staging/`).
2. **Decide the target package** per file (most files have hints in the tree above).
3. **Copy + adapt** into `packages/<name>/src/`. Adaptations include:
   - Replace host-app-specific imports (e.g., `@/store/...`, `@/types/properties`) with engine-agnostic equivalents from `@vitrum/core`
   - Replace any `window.__WGPU__` global bridge with a library-internal device handle held by the engine instance
   - Replace React lifecycle assumptions (mount/unmount, useEffect) with explicit lifecycle methods on the `Engine` interface (`init`, `setScene`, `renderFrame`, `pause`, `resume`, `dispose`)
   - Preserve tests, citations, and historical comments
4. **Verify**: run `tsc --noEmit` after each package's extraction; package compiles in isolation against `@vitrum/core`.
5. **Delete from `_staging/`** as files migrate, so `_staging/` shrinks toward empty. Empty `_staging/` = extraction phase complete.

## Order of extraction (recommended)

Per `plan/sprint-0-api-contract.md` Sprint 0 step 2 onwards, in dependency order:

1. **`@vitrum/shared-bvh`** — `walkaround/lib/bvhCommon.ts`, `walkaround/lib/wgpuSupport.ts`, `walkaround/wgsl/octahedral.wgsl.ts`, `walkaround/sceneBvh.ts`. Foundation.
2. **`@vitrum/shared-samplers`** — `walkaround/wgsl/hammersley.wgsl.ts`, anything sampling-related lifted from RestirStage's WGSL. Foundation.
3. **`@vitrum/three-bindings`** — needs to be sketched first to unblock everything else. Walks a `THREE.Scene` and emits a `Scene` per `@vitrum/core/scene.ts`.
4. **`@vitrum/pt-webgl`** — wraps the existing PathTracingLayer + the fork. Engine-stub already exists; flesh out.
5. **`@vitrum/walkaround-hybrid`** — the big one. Lifts most of `walkaround/` into the package. The `WalkaroundGPUPipeline` god file should be split per the locked refactoring plan in `plan/glorious-hybrid.md`.
6. **`@vitrum/shared-denoisers`** — extract à-trous + variance accumulator into a denoiser interface. Future SVGF/BMFR/OIDN bridge plug into the same interface.

## Tracking

When a `_staging/` file's content has been adapted into a `packages/<name>/src/` file, delete it from `_staging/` and note the migration in the package's README. Don't leave duplicates around.
