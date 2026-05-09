# Extraction Plan — `_staging/legacy-source/` → `@vitrum/*`

**Status**: planning  
**Date**: 2026-05-09  
**Depends on**: Sprint 0 complete (checked: `packages/core/src/*.ts` locked, `packages/pt-webgl/src/index.ts` stub committed, `tsc --noEmit` clean)  
**Author**: planning agent, spot-checked against actual files per CLAUDE.md verification protocol

---

## 1. Status and Framing

### What's already done

Sprint 0 is complete. Verified state:

- `@vitrum/core/src/{scene,frame,engine}.ts` — locked contract types.
- `@vitrum/pt-webgl/src/index.ts` — stub implementing `Engine`, every method throws `Not implemented`. Factory accepts `WebGL2RenderingContext`.
- `@vitrum/three-bindings/src/index.ts` — stub (`sceneFromThreeJS` throws).
- All 8 package `package.json` files wired in npm workspace.
- `tsc --noEmit` clean across workspace.

The 2 remaining Sprint 0 checklist items (host-app bridge hooks, Sprint 0 committed with doc update) are explicitly deferred to the first host-app integration (no host app currently imports `@vitrum/*`).

### What this plan covers

The extraction phase moves renderer source from `_staging/legacy-source/src/rendering/scene/` into `packages/<name>/src/`. This is not a feature sprint — no new algorithms, no visual changes. It is the prerequisite that gives Phase 6 Sprints 1–N a clean package home to land their implementations.

**Extraction is a prerequisite to Phase 6 Sprint 1.** Sprint 1's DoD (bounces 3, resolutionFactor 0.5, HDRI 404 fix) must land in `@vitrum/pt-webgl/src/`, not `_staging/`. Until pt-webgl is fleshed out with actual three-gpu-pathtracer wiring, Sprint 1 cannot deliver a rendering change.

### Scope of `_staging/legacy-source/`

Verified by filesystem read: 47 TypeScript/TSX files + 12 WGSL-as-TypeScript files. The TSX React wrappers (`PathTracingLayer.tsx`, `HybridLayeredStage.tsx`, `RestirStage.tsx`, `PTStage.tsx`, etc.) are NOT in staging — they are host-app files and stay in the host app. They are referenced in some analyst findings but are not extraction targets.

---

## 2. Goals and Non-Goals

### Goals

- Move every library-grade file from `_staging/` into its target `packages/<name>/src/`.
- Strip host-app coupling from each file: replace `@/store/...` Redux imports, React lifecycle hooks, host-specific type imports.
- Each extracted package compiles with `tsc --noEmit` in isolation against `@vitrum/core`.
- No `@/...` import paths in any extracted file.
- Delete each file from `_staging/` as it migrates (no duplicates).
- Move test files (e.g., `lib/bvhCommon.test.ts`) to `packages/<name>/src/__tests__/`.

### Non-Goals

- **No refactoring for its own sake.** Known tech debt items (`WalkaroundGPUPipeline` god file, `lightingIntensityTable.ts` split, `useSceneBVH` consolidation, D1–D6 from `path-tracer-library-readiness.md`) are noted as riders within this plan's effort estimates but are NOT required for extraction to be declared done. They must not block extraction.
- **No new features or algorithm changes.** Extraction is a structural move, not a capability sprint.
- **No host-app responsibilities assumed.** The host remains the owner of: React lifecycle, Redux state, domain composition (panel/cell/came concepts), scene assembly, UI controls.
- **No npm publish.** Local-only via workspace `file:` links until prime-time.
- **No upstream PRs.** The `three-gpu-pathtracer` fork stays local.

---

## 3. Extraction Strategy

### Recommended order (with rationale for one change)

`_staging/README.md` recommends: shared-bvh → shared-samplers → three-bindings → pt-webgl → walkaround-hybrid → shared-denoisers.

This plan **confirms that order with one addition**: extract `gpuDetection.ts` to `@vitrum/core` as the first sub-step of `shared-bvh`, because `shared-bvh` depends on GPU-detection for the WebGPU path and `@vitrum/core` is the natural home for a device-capability utility with no GPU-backend coupling.

Rationale for the existing order:

1. **shared-bvh first** — `walkaround-hybrid` depends on it heavily (`bvhCommon.ts` is the foundation of every BVH builder). Three-bindings doesn't need it yet (stub walks the scene graph).
2. **shared-samplers second** — depends on `@vitrum/core` only. Needed by `walkaround-hybrid` for Hammersley.
3. **three-bindings third** — needed by the host before Sprint 1 can wire up. `sceneFromThreeJS` must produce `@vitrum/core`'s `Scene` type that pt-webgl consumes.
4. **pt-webgl fourth** — Sprint 1 lands here. Must be ready before Sprint 1 kickoff.
5. **walkaround-hybrid fifth** — the largest extraction. Depends on shared-bvh + shared-samplers. Sprint 9 (adaptive sampling, walkaround-only) lands here.
6. **shared-denoisers last** — its sources (atrous, temporalAccum WGSL) only make sense after walkaround-hybrid is extracted; they are consumed by WalkaroundGPUPipeline.

---

## 4. Per-Package Extraction Sub-Plans

### 4.1 `@vitrum/core` — `gpuDetection.ts` rider

**Files moving in:**

| Source | Target |
|---|---|
| `walkaround/gpuDetection.ts` | `packages/core/src/gpuDetection.ts` |

**Transformations needed:**

- `gpuDetection.ts` is self-contained (verified: imports only from `detectGpu`, no host-app deps). Remove the `__resetGpuDetectionForTests` export from the production barrel; move it to a `.test-utils.ts` adjacent file per `path-tracer-library-readiness.md` C3 finding.
- Export `HardwareVerdict` and `getCachedGpuDetection` from `packages/core/src/index.ts`.

**Definition of done:**

- `tsc --noEmit` clean.
- No `@/...` imports.
- `gpuDetection.ts` deleted from `_staging/`.

**Dependencies:** None — first step, zero dependencies.

**Effort:** 0.5 days.

**Risks:** Low. The file is clean per the path-tracer-library-readiness audit ("confirmed clean").

---

### 4.2 `@vitrum/shared-bvh`

**Files moving in:**

| Source | Target |
|---|---|
| `walkaround/lib/bvhCommon.ts` | `packages/shared-bvh/src/bvhCommon.ts` |
| `walkaround/lib/bvhCommon.test.ts` | `packages/shared-bvh/src/__tests__/bvhCommon.test.ts` |
| `walkaround/lib/wgpuSupport.ts` | `packages/shared-bvh/src/wgpuSupport.ts` |
| `walkaround/wgsl/octahedral.wgsl.ts` | `packages/shared-bvh/src/wgsl/octahedral.wgsl.ts` |
| `walkaround/sceneBvh.ts` | `packages/shared-bvh/src/sceneBvh.ts` |

**Files NOT extracted (stay host-only):**

- `walkaround/bvhCompute.ts` — the outer RC-specific BVH packer; this wraps `bvhCommon.ts` but uses TSL `StorageBufferAttribute`. It belongs in `@vitrum/walkaround-hybrid` as part of the RC path.
- `walkaround/engines/restir/bvhCompute.ts` — ReSTIR-specific post-packing (UV-pack-into-position-w, RGBA8 emitter list). Belongs in `@vitrum/walkaround-hybrid`.

**Transformations needed:**

- `bvhCommon.ts`: strip plan-section references (`§3.1`, `M0–M5`, worktree paths, old commit hashes) per E1 finding in `path-tracer-library-readiness.md`. Replace "Per-branch consumers" with "Per-engine consumers". No import changes needed — the file has no host-app imports.
- `wgpuSupport.ts`: self-contained (verified: no host-app imports). Export from `packages/shared-bvh/src/index.ts`.
- `sceneBvh.ts`: check for host-app imports; replace any `@/...` paths with `@vitrum/core` equivalents or internal imports.
- `bvhCommon.test.ts`: move to `__tests__/`; update relative import paths.

**Note on `lib/useSceneBVH.ts`:** This file is a React hook (`useState`, `useEffect`). It belongs in the host app's bridge layer, NOT in `@vitrum/shared-bvh`. It is the canonical version (79 LOC, generalized opts) — during walkaround-hybrid extraction, it becomes the blueprint for a non-React equivalent. Do not extract it here.

**Definition of done:**

- `tsc --noEmit` clean for `packages/shared-bvh/`.
- No React imports, no `@/...` imports, no Redux imports.
- `bvhCommon.test.ts` passes (`npm test --workspace=@vitrum/shared-bvh`).
- Source files deleted from `_staging/walkaround/lib/` and `_staging/walkaround/wgsl/` (where moved).

**Dependencies:** `@vitrum/core` (4.1 must land first for `gpuDetection` import).

**Effort:** 1.5–2 days (including stale-comment cleanup).

**Risks:** Medium. `bvhCommon.ts` is described as "great test coverage; E1+C3 nits only" — nit cleanup is low-risk. The `DEFAULT_FILTER` accepts `MeshStandardMaterial` (hardcoded sentinel per B6 finding) — document this as a rider fix, not a blocking issue.

---

### 4.3 `@vitrum/shared-samplers`

**Files moving in:**

| Source | Target |
|---|---|
| `walkaround/wgsl/hammersley.wgsl.ts` | `packages/shared-samplers/src/wgsl/hammersley.wgsl.ts` |

**Files NOT here yet (Phase 6 sprints land them later):**

- Sobol QMC sequence generation — Sprint 3.
- Light tree CDF construction — Sprint 3.
- HG phase function — Sprint 7.
- Equi-angular PDF — Sprint 7.
- Jakob+Hanika spectral upsampling — Sprint 8b.
- Welford variance struct — Sprint 9 rider.

**Transformations needed:**

- `hammersley.wgsl.ts`: verified "confirmed clean" in the path-tracer-library-readiness audit. No host-app imports. Export the WGSL string constant from `packages/shared-samplers/src/index.ts`.
- Create minimal `packages/shared-samplers/src/index.ts` barrel.

**Definition of done:**

- `tsc --noEmit` clean for `packages/shared-samplers/`.
- File deleted from `_staging/walkaround/wgsl/`.

**Dependencies:** `@vitrum/core` only.

**Effort:** 0.5 days.

**Risks:** Low.

---

### 4.4 `@vitrum/three-bindings`

**Files moving in:** None from `_staging/` — this package has no legacy source to extract. The stub (`sceneFromThreeJS` throws) already exists at `packages/three-bindings/src/index.ts`.

**What needs to happen:** Flesh out `sceneFromThreeJS` to walk a `THREE.Scene` and convert meshes/lights to `@vitrum/core`'s `Scene`, `ScenePrimitive`, `SceneEmitter` types. This is NOT a mechanical file move — it is new implementation work that Sprint 1 requires.

**Transformations needed:**

- Implement `sceneFromThreeJS(threeScene: THREE.Scene): Scene`.
- Handle `THREE.Mesh` + `THREE.MeshPhysicalMaterial` → `Material` + `TriangleMeshPrimitive`.
- Handle `THREE.DirectionalLight` → `SceneEmitter` with `kind: 'directional'`.
- Handle `THREE.PointLight` → `SceneEmitter` with `kind: 'point'`.
- Throw on unsupported types (instanced meshes, custom shaders) — Sprint 1 adds support incrementally.

**Definition of done:**

- `tsc --noEmit` clean.
- `sceneFromThreeJS` converts a minimal scene (1 mesh, 1 directional light) without throwing.
- Unit test exists for the minimal case.

**Dependencies:** `@vitrum/core`.

**Effort:** 1–2 days (implementation, not extraction).

**Risks:** Low for the stub; grows as scene type coverage grows. Sprint 1 only needs directional light + MeshPhysicalMaterial.

---

### 4.5 `@vitrum/pt-webgl`

This is the highest-priority extraction for Phase 6 because Sprint 1 lands here.

**Files moving in:**

| Source | Target | Notes |
|---|---|---|
| `pathtracerConstants.ts` | `packages/pt-webgl/src/constants.ts` | Split: library-grade parts only (see below) |
| `ptIblBaker.ts` | `packages/pt-webgl/src/iblBaker.ts` | Needs renderer coupling decision (Q-PT-2) |
| `ptDebounce.ts` | `packages/pt-webgl/src/debounce.ts` | Pure function, no coupling |
| `lightingState.ts` | `packages/pt-webgl/src/lightingState.ts` | Replace `SkyParams` import |
| `skyParams.ts` | `packages/pt-webgl/src/skyParams.ts` | Partial — see below |
| `lightingIntensityTable.ts` | `packages/pt-webgl/src/lightingIntensityTable.ts` | PARTIAL — split required |

**Files staying host-only (not extracted):**

- `PathTracingLayer.tsx`, `PathtracerSceneSync.tsx`, `PathtracerDebugBridge.tsx`, `PTStage.tsx`, `PTPostProcessing.tsx`, `PTDeviceLostBoundary.tsx` — React/R3F lifecycle, host concern.
- `outdoorHdri.ts`, `outdoorScenePresets.ts` — host-asset URL tables.
- `ptEnvironment.ts` — R3F lifecycle wrapper around the IBL baker.
- `lighting/usePTPipelineConfig.ts`, `lighting/usePTSampleTarget.ts` — Redux selectors.
- `lighting/renderers/sunPathTraced.tsx` — R3F + host `SunLight` type coupling.
- `cameraLookPresets.ts` — keyed by host `CameraLook` enum. (See Q-PT-5.)

**Transformations per file:**

**`pathtracerConstants.ts` → `packages/pt-webgl/src/constants.ts`:**
- `PT_TARGET_SAMPLES`, `PT_TARGET_SAMPLES_BASE`, `PT_TARGET_SAMPLES_FIXTURES`, `PT_BOUNCES`, `PT_FILTERED_GLOSSY_FACTOR`, `PT_RESOLUTION_FACTOR`, `PT_LOW_RES_SCALE` → library-grade exports. These become defaults wired into `createPTEngine_WebGL2` options defaults.
- `PTPipelineConfig`, `PT_PREVIEW`, `PT_FINAL` → export as `PT_PREVIEW_OPTIONS` and `PT_FINAL_OPTIONS` of type `Partial<EngineOptions>`.
- Timing budgets (`PT_HONEYCOMB_TIMING_BUDGET_MS`, etc.) → host-side concern (these are e2e test parameters, not engine config). Do NOT extract; note in a comment in the source file.

**`lightingIntensityTable.ts` — SPLIT REQUIRED:**
- Lines 1–73 (pure physics functions: `COLOR_TEMP_HEX`, `SUN_INTENSITY`, `getSunIntensity`, `pointIntensityFromLumens`, `rectAreaIntensityFromLumens`) → `packages/pt-webgl/src/lightingIntensityTable.ts`.
- Lines 75–111 (`PT_IBL_INTENSITY`, `PT_BACKGROUND_INTENSITY`) — depend on host `BackdropMode` type via `import type { BackdropMode } from '@/store/uiSlice'`. These STAY host-only. Verified by reading the file (line 75 confirmed: `import type { BackdropMode } from '@/store/uiSlice'`; line 94 uses `satisfies Record<BackdropMode, number>`).

**`skyParams.ts`:**
- `SkyParams` interface → collapse into `@vitrum/core`'s `ProceduralSkyEnvironment` (or keep as a pt-webgl–internal type if `ProceduralSkyEnvironment` isn't yet in core). `skyParamsFor()` and `worldSunPosition()` → `packages/pt-webgl/src/skyParams.ts`.

**`lightingState.ts`:**
- `computeLightingState` is a pure function. Replace `SkyParams` import with `ProceduralSkyEnvironment` from core (or the pt-webgl–internal `SkyParams` type from above).

**`ptDebounce.ts`:**
- Pure function; no changes needed.

**`ptIblBaker.ts`:**
- Depends on `SkyParams` → replace as above.
- Accepts `THREE.WebGLRenderer` (verified: line 108 `renderer: THREE.WebGLRenderer`). This is a DECISION-PENDING-USER (Q-PT-2 below).

**Sun geometry constants from `lighting/renderers/sunPathTraced.tsx`:**
- `PT_SUN_DISTANCE`, `SUN_ANGULAR_RADIUS`, `PT_SUN_DISC_DIAMETER`, `PT_SUN_AREA_INTENSITY` → `packages/pt-webgl/src/sunGeometry.ts`. These are pure constants with no React coupling.

**Definition of done:**

- `tsc --noEmit` clean for `packages/pt-webgl/`.
- No `@/...` imports, no React imports, no Redux imports.
- `PT_PREVIEW_OPTIONS` and `PT_FINAL_OPTIONS` are exported and consumable by Sprint 1.
- Source files for extracted portions deleted from `_staging/`.

**Dependencies:** `@vitrum/core`, `@vitrum/shared-samplers` (if any sampler utilities are needed).

**Effort:** 2–4 days. The split of `lightingIntensityTable.ts` and the `SkyParams` collapse are the riskiest steps.

**Risks:** Medium.
- The `ptIblBaker.ts` / `THREE.WebGLRenderer` coupling is the hardest coupling to break (DECISION-PENDING-USER Q-PT-2).
- Q-PT-4: `PT_PREVIEW` ↔ `PT_FINAL` mode switching requires `Engine.updateOptions()` or `reset(newOptions)`. The current contract has neither. This needs resolution before Sprint 1 — see Decision-Pending-User section.

---

### 4.6 `@vitrum/walkaround-hybrid`

The largest, highest-complexity extraction. Covers DDGI + RC + ReSTIR DI pipeline.

**Files moving in:**

| Source | Target |
|---|---|
| `walkaround/engines/restir/WalkaroundGPUPipeline.ts` | `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts` |
| `walkaround/engines/restir/bvhCompute.ts` | `packages/walkaround-hybrid/src/restir/bvhCompute.ts` |
| `walkaround/engines/restir/shaders/common.wgsl.ts` | `packages/walkaround-hybrid/src/shaders/common.wgsl.ts` |
| `walkaround/engines/restir/shaders/ris.wgsl.ts` | `packages/walkaround-hybrid/src/shaders/ris.wgsl.ts` |
| `walkaround/engines/restir/shaders/temporal.wgsl.ts` | `packages/walkaround-hybrid/src/shaders/temporal.wgsl.ts` |
| `walkaround/engines/restir/shaders/spatial.wgsl.ts` | `packages/walkaround-hybrid/src/shaders/spatial.wgsl.ts` |
| `walkaround/engines/restir/shaders/shade.wgsl.ts` | `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts` |
| `walkaround/engines/restir/shaders/composite.wgsl.ts` | `packages/walkaround-hybrid/src/shaders/composite.wgsl.ts` |
| `walkaround/useHybridLayeredGI.ts` | `packages/walkaround-hybrid/src/HybridEngine.ts` (de-React-ified; becomes the `Engine` implementation body) |
| `walkaround/useDDGI.ts` | `packages/walkaround-hybrid/src/ddgi/DDGI.ts` (de-React-ified) |
| `walkaround/probeUpdatePass.ts` | `packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts` |
| `walkaround/probeGrid.ts` | `packages/walkaround-hybrid/src/ddgi/probeGrid.ts` |
| `walkaround/ddgiAtlasLayout.ts` | `packages/walkaround-hybrid/src/ddgi/ddgiAtlasLayout.ts` |
| `walkaround/ddgiSampleWgsl.ts` | `packages/walkaround-hybrid/src/ddgi/ddgiSampleWgsl.ts` |
| `walkaround/sceneBvh.ts` | `packages/walkaround-hybrid/src/bvh/sceneBvh.ts` |
| `walkaround/wgsl/probeUpdateRays.wgsl.ts` | `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts` |
| `walkaround/wgsl/probeUpdateBlend.wgsl.ts` | `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateBlend.wgsl.ts` |
| `walkaround/lib/nodeMaterialUpgrade.ts` | `packages/walkaround-hybrid/src/lib/nodeMaterialUpgrade.ts` |

**RC subsystem (see DECISION-PENDING-USER Q-WA-3):**
- `walkaround/cascadePyramid.ts`, `walkaround/cascadeDispatch.ts`, `walkaround/useCascadeBuffers.ts`, `walkaround/bvhCompute.ts`, `walkaround/applyDDGIShading.ts`, `walkaround/giReceiver.ts`, `walkaround/walkaroundDiffuseLighting.ts`
- These belong in `@vitrum/walkaround-hybrid` IF the RC engine is extracted. Deferred pending Q-WA-3 decision.

**Files NOT extracted (stay host-only):**

- `walkaround/HybridLayeredStage.tsx` — React stage wrapper (469 LOC). Host concern; the host keeps this.
- `walkaround/WalkaroundStage.tsx` — older React stage.
- `walkaround/engines/rc/RcStage.tsx` — host React stage.
- `walkaround/engines/restir/RestirStage.tsx`, `WalkaroundDebugBridge.tsx`, `walkaroundBridgeTypes.ts` — host debug bridge + window globals augmentation.
- `walkaround/engineRegistry.ts` — Redux-coupled engine selector.
- `walkaround/lib/useSceneBVH.ts` — React hook; see 4.2 note above.
- `walkaround/useSceneBVH.ts` — RC's older hook; becomes a thin host-side wrapper.

**Transformations needed:**

**`WalkaroundGPUPipeline.ts`:**
- Already exports `HYBRID_WEBGPU_REQUIRED_LIMITS` as a library-grade constant (verified: line 53). This becomes a top-level export of `@vitrum/walkaround-hybrid`.
- `PipelineFrameInputs.sunDirection` / `.sunIntensity` rename to `primaryLightDir` / `primaryLightIntensity` per C1 in path-tracer-library-readiness.md.
- Export `HYBRID_WEBGPU_REQUIRED_LIMITS` from the package's `index.ts`.
- The god-file split is DECISION-PENDING-USER (Q-WA-6). See below.

**`probeUpdatePass.ts`:**
- Imports `LightSource from '../lighting/lightSourceTypes'` (verified: line 21). Replace with a package-internal `DDGILight` interface defining only the fields actually used. Verified fields used: `l.kind` (string: `'sun' | 'fixture' | 'teaLight'`), `l.intensity`, and via unsafe cast `l.position`. This interface should be declared in `packages/walkaround-hybrid/src/ddgi/types.ts`. LOC count verified: 615 lines (not the "400–600" estimate — exact count is 615).
- The `StorageTexture` import from `three/webgpu` and `backend.device` access couple this to Three.js WebGPU renderer internals. This is known per Q-WA-9 and is accepted — `@vitrum/walkaround-hybrid` will carry `three/webgpu` as a peer dep on the DDGI path.

**`useHybridLayeredGI.ts` → `HybridEngine.ts`:**
- Strip React hooks (`useEffect`, `useState`, `useRef`, `useCallback`). Replace with explicit `initialize()`, `renderFrame()`, `dispose()` lifecycle methods implementing the `Engine` interface.
- Remove Redux imports (`useAppSelector`, `selectMount`, etc.). Callers pass these values as constructor or method arguments.
- Verify: the hook currently accepts `scene`, `bvh`, `sunDirection`, `ddgiEnabled`, `rcEnabled`, `device` — these become constructor opts or factory parameters.

**`useDDGI.ts` → `DDGI.ts`:**
- Same de-React treatment. Strip `useRef`/`useEffect`/`useCallback`.
- Becomes a class with `initialize()`, `update(frameInputs)`, `dispose()` methods.

**Definition of done:**

- `tsc --noEmit` clean for `packages/walkaround-hybrid/`.
- No `@/...` imports (only `three`, `three/webgpu` peer deps, and `@vitrum/*` internal deps).
- No React imports (no `useState`, `useEffect`, `useRef`, `useCallback`, `useMemo`).
- `HYBRID_WEBGPU_REQUIRED_LIMITS` exported from package index.
- `probeUpdatePass.ts` uses only library-internal `DDGILight` type (no host `LightSource`).
- Source files deleted from `_staging/walkaround/`.

**Dependencies:** `@vitrum/core`, `@vitrum/shared-bvh`, `@vitrum/shared-samplers`.

**Effort:** 5–10 days. The de-React-ification of `useHybridLayeredGI.ts` and `useDDGI.ts` is the most difficult transformation.

**Risks:** High.
- Q-WA-2 (missing RC shaders — confirmed blocking; see below).
- Q-WA-7 (TSL vs raw WebGPU in `cascadeDispatch.ts` — confirmed).
- Q-WA-9 (`three/webgpu` peer dep on DDGI path — confirmed, accepted).

---

### 4.7 `@vitrum/shared-denoisers`

**Files moving in:**

| Source | Target |
|---|---|
| `walkaround/engines/restir/shaders/atrous.wgsl.ts` | `packages/shared-denoisers/src/wgsl/atrous.wgsl.ts` |
| `walkaround/engines/restir/shaders/temporalAccum.wgsl.ts` | `packages/shared-denoisers/src/wgsl/temporalAccum.wgsl.ts` |

**Note:** These WGSL modules are imported by `WalkaroundGPUPipeline.ts` (verified: line 32, `import { ATROUS_WGSL } from './shaders/atrous.wgsl'`; line 33 `import { TEMPORAL_ACCUM_WGSL }`). The extraction order requires `shared-denoisers` to be stubbed (barrel + empty WGSL exports) before `walkaround-hybrid` compiles. Alternatively, extract walkaround-hybrid first and leave these two files in place as internal modules, then migrate them to `shared-denoisers` as a follow-up. See Section 6 (sequencing) for the recommended approach.

**Transformations needed:** None — these are pure WGSL string constants, no host-app coupling.

**Definition of done:**

- `tsc --noEmit` clean for `packages/shared-denoisers/`.
- `WalkaroundGPUPipeline.ts` imports from `@vitrum/shared-denoisers`, not from `./shaders/`.

**Dependencies:** Must be stubbed before walkaround-hybrid extraction references it; full extraction happens after walkaround-hybrid.

**Effort:** 0.5 days (the WGSL migration is trivial; the cross-package import update is the only work).

**Risks:** Low once walkaround-hybrid is stable.

---

## 5. Resolved Decisions vs. Deferred Questions

### Resolved decisions (locked by evidence)

**RD-1: Outer `useSceneBVH.ts` is not the canonical version.**  
Confirmed by reading both files. `walkaround/lib/useSceneBVH.ts` (79 LOC, generalized `SceneBVHCommonOpts` + `optsRef` pattern) is canonical. `walkaround/useSceneBVH.ts` (58 LOC, RC-specific, typed against `SceneBVH` from the outer `bvhCompute.ts`) is the older RC-specific variant. The outer version has one caller — RC's BVH consumers. During extraction, the outer version becomes a thin host-side wrapper (calls the lib version, adapts the type). The lib version moves to `@vitrum/shared-bvh` as a blueprint (the de-React-ified equivalent) rather than the hook itself.

**RD-2: RC shader files `probeRayCast.wgsl` and `cascadeMerge.wgsl` are NOT in `_staging/`.**  
Verified by filesystem search: zero results for these files anywhere under `_staging/`. `cascadeDispatch.ts` imports them from `../../shaders/walkaround/probeRayCast.wgsl` (line 24, read directly). These files must be staged from the host app before RC extraction can proceed. **RC extraction is hard-blocked until these are staged.**

**RD-3: `lightingIntensityTable.ts` must be split.**  
Verified by reading the file. Lines 1–73: pure physics (no host imports). Lines 75–111: `import type { BackdropMode } from '@/store/uiSlice'` at line 75; `satisfies Record<BackdropMode, number>` at line 94. The `BackdropMode`-typed maps (`PT_IBL_INTENSITY`, `PT_BACKGROUND_INTENSITY`) stay host-only; the physics functions extract to pt-webgl.

**RD-4: `probeUpdatePass.ts` LOC is 615 (not the analyst's "likely 400–600").**  
Verified by `wc -l`. Actual count: 615. Scoping adjusted accordingly.

**RD-5: `WalkaroundGPUPipeline.ts` LOC is 1287 (confirms analyst).**  
Verified by `wc -l`. Count confirmed.

**RD-6: `LightSource` coupling in `probeUpdatePass.ts` uses only `kind`, `intensity`, and `position` (via unsafe cast).**  
Verified by reading lines 370–410. The `_uploadLights` method switches on `l.kind` (string union `'sun' | 'fixture' | 'teaLight'`) and uses `l.intensity` and `(l as unknown as { position? }).position`. A package-internal `DDGILight` interface with `kind: string`, `intensity: number`, `position?: { x: number; y: number; z: number }` is sufficient.

**RD-7: Plan docs do NOT prescribe a concrete split for `WalkaroundGPUPipeline.ts`.**  
Confirmed by reading `glorious-hybrid.md` and `layered-hybrid-v2.md`. Neither document prescribes splitting the god file into sub-modules. The split proposal is analyst-inferred only. This is DECISION-PENDING-USER (Q-WA-6).

**RD-8: `cascadeDispatch.ts` uses TSL (confirmed).**  
Verified by reading the file: imports `StorageBufferAttribute, WebGPURenderer` from `three/webgpu` and `storage, sampler, texture, instanceIndex` from `three/tsl` (lines 18–20). This is a different paradigm from `WalkaroundGPUPipeline.ts`'s raw WebGPU. Confirms Q-WA-7 is real and requires a decision.

---

### DECISION-PENDING-USER items

These require user sign-off before the relevant extraction step proceeds.

---

**DECISION-PENDING-USER: Q-PT-2 — IBL baker and the `THREE.WebGLRenderer` requirement**

**Problem**: `ptIblBaker.ts` (line 108) accepts `renderer: THREE.WebGLRenderer` — not a raw `WebGLRenderingContext`. The current `createPTEngine_WebGL2` factory accepts `WebGL2RenderingContext`. The baker needs the full renderer object because it uses `CubeCamera`, `WebGLRenderTarget`, and `renderer.render()`.

**Option A**: Engine factory accepts both — add an optional `renderer?: THREE.WebGLRenderer` to `PTEngineWebGL2Options`. The engine holds the renderer reference and passes it to the baker internally.  
_Pros_: Minimal change to the existing stub; backwards-compatible.  
_Cons_: `@vitrum/pt-webgl` now carries `three` as a required peer dep (it already would for the PT pipeline anyway, but the factory contract exposes it).

**Option B**: Baker is a standalone exported function; host passes the renderer separately.  
_Pros_: Factory contract stays clean (just `WebGL2RenderingContext`); baker is composable.  
_Cons_: Host must thread the renderer through two separate API calls.

**Option C**: Engine creates its own `THREE.WebGLRenderer` from the raw `WebGLRenderingContext`.  
_Pros_: Clean factory contract.  
_Cons_: Engine would own the renderer lifecycle, violating the "host owns lifecycle" design principle.

**Recommendation**: Option A. `@vitrum/pt-webgl` already depends on `three` (the fork wraps Three.js internals); adding `renderer?: THREE.WebGLRenderer` to options is the least-disruptive path.

---

**DECISION-PENDING-USER: Q-PT-4 — PT_PREVIEW ↔ PT_FINAL switching and the Engine contract**

**Problem**: The host currently switches between `PT_PREVIEW` and `PT_FINAL` config by re-mounting the `<Pathtracer>` component with different props. After extraction, the `Engine` contract has `reset()` but no `updateOptions()` method. Without one, the host must `dispose()` + recreate the engine on mode switch — which reloads BVH/shaders and causes a visible stall.

**Option A**: Add `updateOptions(opts: Partial<EngineOptions>): void` to the `Engine` interface in `@vitrum/core`.  
_Pros_: Clean contract; backends implement for the fields they support.  
_Cons_: Any addition to `@vitrum/core`'s `Engine` interface is a breaking change for all future backends.

**Option B**: Add `reset(opts?: Partial<EngineOptions>): void` overload semantics — existing `reset()` clears accumulator; `reset(newOpts)` also patches config.  
_Pros_: Builds on existing method; less surface area.  
_Cons_: Overloads on presence/absence of argument are ambiguous TypeScript.

**Option C**: Keep `Engine` contract unchanged; expose a `PTEngineWebGL2` subclass method `applyConfig(cfg: PTPipelineConfig)` not on the `Engine` interface.  
_Pros_: No core contract change; implementation detail of the backend.  
_Cons_: Host must hold a `PTEngineWebGL2` reference, not the `Engine` interface — leaks the backend type.

**Recommendation**: Option A. The Sprint 0 doc explicitly lists `updateOptions()` (via `updatePrimitive`/`updateEmitter` analogues) as a known TBD. A general `updateOptions()` on `Engine` is the honest contract extension. Sprint 1 kickoff is the right moment to lock this.

---

**DECISION-PENDING-USER: Q-WA-3 — RC subsystem extraction scope**

**Problem**: The Radiance Cascade engine (`cascadeDispatch.ts`, `cascadePyramid.ts`, `useCascadeBuffers.ts`, outer `bvhCompute.ts`, `applyDDGIShading.ts`, `giReceiver.ts`, `walkaroundDiffuseLighting.ts`) is used by the standalone `'rc'` walkaround engine (`RcStage.tsx`) but is classified as "Path A orphan" in `layered-hybrid-v2.md`. The active hybrid engine (`HybridLayeredStage`) runs RC as a compute prerequisite — but is blocked on Q-WA-2 (missing shaders).

**Additional blocker**: `cascadeDispatch.ts` uses TSL (`storage()`, `wgslFn`, `compute()`) — different paradigm from `WalkaroundGPUPipeline.ts`'s raw WebGPU. Verified by reading the file. This means `@vitrum/walkaround-hybrid` would carry both a raw-WebGPU path (ReSTIR) and a TSL path (RC), or the TSL path must be rewritten to raw WebGPU during extraction.

**Option A**: Extract RC subsystem as-is into `@vitrum/walkaround-hybrid` (accept TSL peer dep).  
_Pros_: Preserves the standalone `'rc'` engine; RC composition in hybrid works when missing shaders are staged.  
_Cons_: TSL + raw WebGPU coexistence is messy; blocked on Q-WA-2.

**Option B**: Defer RC extraction to a separate sprint after hybrid-hybrid core is extracted. Stage it under `packages/walkaround-hybrid/src/rc/` but don't wire it up until Q-WA-2 is resolved.  
_Pros_: Unblocks DDGI + ReSTIR extraction immediately; RC staging happens in parallel.  
_Cons_: The standalone `'rc'` engine remains host-side longer.

**Option C**: Do not extract RC. Mark the standalone `'rc'` walkaround engine as host-only; document that Phase 6 Sprint 9 walkaround work targets only the hybrid path.  
_Pros_: Simplest; Q-WA-2 and Q-WA-7 become non-issues.  
_Cons_: Loses library-grade RC; future walkaround improvements must stay in the host app for the RC path.

**Recommendation**: Option B. Extract DDGI + ReSTIR first (they're independent of the missing shaders). Stage RC files in `packages/walkaround-hybrid/src/rc/` but mark them `// NOT YET WIRED` until Q-WA-2 shaders are staged and Q-WA-7 is resolved.

---

**DECISION-PENDING-USER: Q-WA-6 — WalkaroundGPUPipeline god-file split**

**Problem**: `WalkaroundGPUPipeline.ts` is 1287 LOC verified. The analyst proposed splitting it into 7 sub-modules. This split is not prescribed by any plan document (confirmed by reading `glorious-hybrid.md` and `layered-hybrid-v2.md`).

**Option A**: Extract as-is (one file). Split can be done as a follow-on refactor sprint.  
_Pros_: Fastest path to extraction; no risk of semantic changes during split.  
_Cons_: The god file grows as Phase 6 sprints add features; refactoring later under active development is harder.

**Option B**: Split during extraction per the analyst's proposed module boundaries (7 sub-modules: stripped public API, resourceManager, bindGroupLayouts, bindGroupBuilders, pipelineCompiler, timestampQueries, uboUpdater).  
_Pros_: Clean architecture for future Phase 6 work; each Sprint lands in a bounded file.  
_Cons_: Adds 3–5 days to extraction; introduces risk of semantic changes during a structural move.

**Option C**: Partial split — extract only `timestampQueries.ts` and `uboUpdater.ts` (the cleanest, most self-contained pieces, ~160 LOC total) and leave the rest as-is.  
_Pros_: Some benefit, bounded risk.  
_Cons_: Doesn't address the core complexity issue.

**Recommendation**: Option A for now. The god file is well-tested; the extraction itself is the priority. Schedule the split as a dedicated Phase 6 Sprint after extraction completes and before Sprint 9 (walkaround adaptive sampling) lands new code in this file.

---

**DECISION-PENDING-USER: Q-PT-1 — IBL baker sun position: unit vs. non-unit Preetham**

**Problem**: `ptIblBaker.ts` takes `SkyParams` which includes a raw (non-unit) `sunPosition` from the Preetham sky model. Core's `ProceduralSkyEnvironment` (when it exists) would likely carry a unit `sunDirection`. The baker's shader may require the non-unit position for Preetham's atmospheric scattering math.

**Option A**: Core's `ProceduralSkyEnvironment` exposes a raw Preetham sun position (non-unit); baker uses it directly.  
_Pros_: Correct for Preetham math.  
_Cons_: Pollutes the core type with a Preetham-specific convention.

**Option B**: Core carries a unit `sunDirection`; baker converts internally (`sunDirection.multiplyScalar(someDistance)`).  
_Pros_: Clean core contract.  
_Cons_: Requires verifying the baker shader doesn't depend on the non-unit distance.

**Recommendation**: Option B pending verification of the baker's GLSL. Read `ptIblBaker.ts`'s shader code (particularly `CUBE_TO_EQUIRECT_FRAG`) before locking. The equirect bake shader (verified lines 74–101) uses cube-map texture lookup — it doesn't use `sunPosition` directly in the fragment shader. The Preetham parameters feed the `Sky` object's uniforms; Three.js `Sky` accepts its own positional parameters. This suggests Option B is viable, but the Sky uniform setup code in `ptIblBaker.ts` (not yet read) must be checked.

---

## 6. Sequence and Parallelization

### Phase 1 — Foundation (serial, days 1–4)

```
Step 1:  @vitrum/core + gpuDetection rider (0.5 days)
Step 2:  @vitrum/shared-bvh (1.5–2 days)
Step 3:  @vitrum/shared-samplers (0.5 days)
```

Steps 1–3 are serial (each depends on the previous). No parallelization opportunity — the foundation must stabilize before consumers build on it.

### Phase 2 — PT track + three-bindings (parallel, days 4–8)

```
Track A:  @vitrum/three-bindings implementation (1–2 days)
Track B:  @vitrum/pt-webgl extraction (2–4 days)
```

These are independent (pt-webgl doesn't need three-bindings; three-bindings doesn't need pt-webgl). Run in parallel. Both depend on Phase 1.

**Synchronization point**: Both must be complete before Sprint 1 can be executed. Sprint 1 requires a working `sceneFromThreeJS` (from three-bindings) and a fleshed-out `createPTEngine_WebGL2` (from pt-webgl).

### Phase 3 — shared-denoisers stub (0.5 days, serial after Phase 1)

Stub `@vitrum/shared-denoisers` with barrel exports for atrous and temporalAccum WGSL (empty string constants initially). This unblocks walkaround-hybrid's `tsc --noEmit` clean.

### Phase 4 — walkaround-hybrid (serial, 5–10 days, after Phase 1 + Phase 3)

```
Step 1:  Extract DDGI subsystem (probeUpdatePass, probeGrid, ddgiAtlasLayout, ddgiSampleWgsl, wgsl/) (2–3 days)
Step 2:  Extract ReSTIR pipeline (WalkaroundGPUPipeline + its shaders) (2–4 days)
Step 3:  De-React useHybridLayeredGI → HybridEngine class (2–3 days)
Step 4:  RC subsystem staging (per Q-WA-3 decision) (1–2 days if Option B)
```

Steps 1–3 can overlap partially (DDGI and ReSTIR shaders are independent), but `HybridEngine.ts` depends on both being stable.

### Phase 5 — shared-denoisers full migration (0.5 days, after Phase 4)

Move atrous + temporalAccum WGSL from `walkaround-hybrid/src/shaders/` to `shared-denoisers`; update imports.

### Gantt summary

```
Days 1–2:   shared-bvh + shared-samplers + core rider
Days 2–6:   [Track A] three-bindings  |  [Track B] pt-webgl
Day 3:      shared-denoisers stub (parallel with Track A/B)
Days 6–16:  walkaround-hybrid (serial, 4 steps)
Day 16:     shared-denoisers full migration
```

**Estimated total elapsed time: 14–18 days** with the parallel Phase 2 tracks.

---

## 7. Verification Protocol

### Per-package verification

After each package's extraction is complete:

1. **`tsc --noEmit`** must pass for the package in isolation: `npx tsc --noEmit -p packages/<name>/tsconfig.json`.
2. **No `@/...` imports** in any extracted file: `grep -r "@/" packages/<name>/src/` must return zero results.
3. **No React imports**: `grep -r "from 'react'" packages/<name>/src/` must return zero results (except if explicitly host-bridge code, which should not exist in library packages).
4. **No Redux imports**: `grep -r "useAppSelector\|useDispatch\|createSlice" packages/<name>/src/` must return zero results.
5. **Tests pass** (where applicable): `npm test --workspace=@vitrum/<name>`.
6. **File deleted from `_staging/`**: the original source file is absent from `_staging/` after migration.

### Workspace-level verification

After all extractions complete:

1. `tsc --noEmit` across the entire workspace (all 8 packages).
2. `_staging/legacy-source/` contains only host-app-only files (TSX React wrappers, Redux selectors, host-asset URLs). Ideally empty except for those.
3. A minimal integration test: create an in-memory `THREE.Scene`, call `sceneFromThreeJS`, call `createPTEngine_WebGL2`, call `engine.setScene(scene)` — this should throw `Not implemented: setScene` (confirming the wire-up compiles; the throw is expected until Sprint 1).

---

## 8. Interaction with Phase 6 Sprints

| Sprint | Extraction prerequisite | What lands in `@vitrum/*` |
|---|---|---|
| **Sprint 1** (PT preview perf, 3.5 hrs) | pt-webgl + three-bindings extractions complete | `PT_PREVIEW_OPTIONS` defaults wired in `createPTEngine_WebGL2`; `THREE.WebGLRenderer` integration; `resolutionFactor: 0.5` option honored; HDRI 404 fix lands in `ptIblBaker.ts` |
| **Sprint 2** (per-cell luminance, 1 day) | walkaround-hybrid extracted (at least DDGI + ReSTIR BVH packer) | `cellPower` added to `engines/restir/bvhCompute.ts` in walkaround-hybrid |
| **Sprint 3** (sampling theory, 5.5 days) | pt-webgl stable | New `@vitrum/shared-samplers` exports: light tree CDF, mixture PDF. `pt-webgl` fork patches for mixture PDF in GLSL |
| **Sprint 4** (BSDF cost, 3.5 days) | pt-webgl stable | Fork-internal patches; no new `@vitrum/*` files but pt-webgl builds against updated fork |
| **Sprint 5** (analytic came, 5 days) | pt-webgl stable; `@vitrum/core`'s `AnalyticShape` extended | `AnalyticShape` discriminated union gets `'came-segment'` kind in core; pt-webgl's `supportedAnalyticShapes` set updated; `sunGeometry.ts` constants consumed |
| **Sprint 6** (spatial filter, 5 days) | shared-denoisers fully extracted | New denoiser class in `@vitrum/shared-denoisers` implementing a hexagonal edge-stopping filter |
| **Sprint 7** (volume + SSS, 7 days) | pt-webgl + shared-samplers stable | HG phase function in shared-samplers; fork patches for volume in pt-webgl |
| **Sprint 8** (RGB-3λ, 5 days) | pt-webgl stable | Fork patches; Jakob+Hanika rider in shared-samplers |
| **Sprint 9** (walkaround convergence, 5 days) | walkaround-hybrid fully extracted | `WelfordVariance` struct in `common.wgsl`; adaptive sampling dispatch in `HybridEngine.ts`; checkerboard resolve in walkaround-hybrid |

---

## 9. Open Risks

### RISK-1 (HIGH): Missing RC shaders block RC extraction — confirmed

`cascadeDispatch.ts` imports `probeRayCast.wgsl` and `cascadeMerge.wgsl` from `../../shaders/walkaround/` (verified lines 24–25). These paths are NOT relative to `_staging/` — they reference the host app's shader directory. Neither file exists anywhere under `_staging/` (verified by filesystem search). The RC extraction (and RC integration into the hybrid engine) is hard-blocked until these files are staged from the host app. This does not block DDGI + ReSTIR extraction.

**Mitigation**: Stage these two WGSL files from the host app before beginning Step 4 of walkaround-hybrid extraction.

### RISK-2 (HIGH): TSL vs. raw WebGPU paradigm in RC subsystem — confirmed

`cascadeDispatch.ts` uses Three.js TSL (`storage()`, `compute()`, `wgslFn`) while `WalkaroundGPUPipeline.ts` uses raw WebGPU (`device.createShaderModule()`, `GPUComputePipeline`). Both paradigms cannot coexist cleanly in a library without accepting Three.js WebGPU renderer as a required peer dep for ALL walkaround-hybrid consumers (including those that only want ReSTIR). Decision pending Q-WA-3.

**Mitigation**: Q-WA-3 Option B (stage RC separately, wire up after decision) keeps this from blocking DDGI + ReSTIR extraction.

### RISK-3 (MEDIUM): `probeUpdatePass.ts` Three.js WebGPU renderer internals coupling

Verified: `probeUpdatePass.ts` imports `StorageTexture from 'three/webgpu'` (line 17) and accesses `renderer.backend.device` (lines 119–121). `@vitrum/walkaround-hybrid` will carry `three/webgpu` as a peer dep on the DDGI path. This is an accepted known cost (Q-WA-9) but means the package cannot be used without Three.js even for pure WebGPU consumers.

**Mitigation**: Document clearly in the package README that `three/webgpu` is required for DDGI. ReSTIR-only usage path (if the user creates the `GPUDevice` themselves) should not require importing Three.js — keep the DDGI initialization optional via a gate in `HybridEngine.ts`.

### RISK-4 (MEDIUM): Engine contract missing `updateOptions()` for PT_PREVIEW ↔ PT_FINAL switching

Sprint 1 requires mode-switching. Current `Engine` interface has only `reset()`. If Q-PT-4 Option A is selected, a contract change must land before Sprint 1 implementation. A contract change in `@vitrum/core` breaks all downstream stubs.

**Mitigation**: Resolve Q-PT-4 first; update `@vitrum/core`, then update all stubs (they all throw anyway). This is a 1-day coordination task, not a risk to the overall timeline.

### RISK-5 (LOW-MEDIUM): Stale plan-section references in extracted files

`path-tracer-library-readiness.md` E1 identifies ~15 stale comments across `lib/bvhCommon.ts`, `lib/useSceneBVH.ts`, `lib/nodeMaterialUpgrade.ts`, `lib/wgpuSupport.ts`, `walkaround/useSceneBVH.ts`, `walkaround/bvhCompute.ts`, `engines/restir/bvhCompute.ts`. These reference plan sections (`§3.1`, `M0–M5`), worktree paths, and old commit hashes that no longer exist. If left in place, they mislead future implementers.

**Mitigation**: Clean these as part of extraction (not as a separate wave). Total estimated scope: ~1 hour per file.

### RISK-6 (LOW): `_staging/` contains no RC `.tsx` React stages

The RC stages (`RcStage.tsx`, `HybridLayeredStage.tsx`, etc.) are not in `_staging/`. The analyst and this plan treat them as host-only. Confirmed: the `_staging/` file listing shows no `.tsx` files at all. The staging directory correctly contains only library-candidate files.

---

## Appendix A: Complete Staging File Disposition Table

| File | Target | Status |
|---|---|---|
| `pathtracerConstants.ts` | `@vitrum/pt-webgl/src/constants.ts` (partial) | Extract |
| `ptIblBaker.ts` | `@vitrum/pt-webgl/src/iblBaker.ts` | Extract (pending Q-PT-2) |
| `ptDebounce.ts` | `@vitrum/pt-webgl/src/debounce.ts` | Extract |
| `lightingState.ts` | `@vitrum/pt-webgl/src/lightingState.ts` | Extract |
| `skyParams.ts` | `@vitrum/pt-webgl/src/skyParams.ts` (partial) | Extract |
| `lightingIntensityTable.ts` | `@vitrum/pt-webgl/src/lightingIntensityTable.ts` (lines 1–73 only) | Partial extract + split |
| `cameraLookPresets.ts` | Host-only | Do not extract |
| `outdoorHdri.ts` | Host-only | Do not extract |
| `outdoorScenePresets.ts` | Host-only | Do not extract |
| `ptEnvironment.ts` | Host-only | Do not extract |
| `lighting/usePTPipelineConfig.ts` | Host-only (Redux) | Do not extract |
| `lighting/usePTSampleTarget.ts` | Host-only (Redux) | Do not extract |
| `walkaround/gpuDetection.ts` | `@vitrum/core/src/gpuDetection.ts` | Extract |
| `walkaround/lib/bvhCommon.ts` | `@vitrum/shared-bvh/src/bvhCommon.ts` | Extract |
| `walkaround/lib/bvhCommon.test.ts` | `@vitrum/shared-bvh/src/__tests__/bvhCommon.test.ts` | Extract |
| `walkaround/lib/wgpuSupport.ts` | `@vitrum/shared-bvh/src/wgpuSupport.ts` | Extract |
| `walkaround/lib/nodeMaterialUpgrade.ts` | `@vitrum/walkaround-hybrid/src/lib/nodeMaterialUpgrade.ts` | Extract |
| `walkaround/lib/useSceneBVH.ts` | Host-only (React hook; blueprint for library non-React equivalent) | Do not extract as hook |
| `walkaround/wgsl/hammersley.wgsl.ts` | `@vitrum/shared-samplers/src/wgsl/hammersley.wgsl.ts` | Extract |
| `walkaround/wgsl/octahedral.wgsl.ts` | `@vitrum/shared-bvh/src/wgsl/octahedral.wgsl.ts` | Extract |
| `walkaround/wgsl/probeUpdateRays.wgsl.ts` | `@vitrum/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts` | Extract |
| `walkaround/wgsl/probeUpdateBlend.wgsl.ts` | `@vitrum/walkaround-hybrid/src/ddgi/wgsl/probeUpdateBlend.wgsl.ts` | Extract |
| `walkaround/sceneBvh.ts` | `@vitrum/shared-bvh/src/sceneBvh.ts` | Extract |
| `walkaround/probeGrid.ts` | `@vitrum/walkaround-hybrid/src/ddgi/probeGrid.ts` | Extract |
| `walkaround/probeUpdatePass.ts` | `@vitrum/walkaround-hybrid/src/ddgi/probeUpdatePass.ts` | Extract (replace `LightSource` import) |
| `walkaround/ddgiAtlasLayout.ts` | `@vitrum/walkaround-hybrid/src/ddgi/ddgiAtlasLayout.ts` | Extract |
| `walkaround/ddgiSampleWgsl.ts` | `@vitrum/walkaround-hybrid/src/ddgi/ddgiSampleWgsl.ts` | Extract |
| `walkaround/useDDGI.ts` | `@vitrum/walkaround-hybrid/src/ddgi/DDGI.ts` | Extract + de-React |
| `walkaround/useHybridLayeredGI.ts` | `@vitrum/walkaround-hybrid/src/HybridEngine.ts` | Extract + de-React |
| `walkaround/useSceneBVH.ts` | Host-only (RC's older hook) | Do not extract |
| `walkaround/cascadePyramid.ts` | `@vitrum/walkaround-hybrid/src/rc/cascadePyramid.ts` (staged, not wired) | Pending Q-WA-3 |
| `walkaround/cascadeDispatch.ts` | `@vitrum/walkaround-hybrid/src/rc/cascadeDispatch.ts` (staged, not wired) | Pending Q-WA-2 + Q-WA-3 |
| `walkaround/useCascadeBuffers.ts` | `@vitrum/walkaround-hybrid/src/rc/useCascadeBuffers.ts` | Pending Q-WA-3 |
| `walkaround/bvhCompute.ts` | `@vitrum/walkaround-hybrid/src/rc/bvhCompute.ts` | Pending Q-WA-3 |
| `walkaround/applyDDGIShading.ts` | `@vitrum/walkaround-hybrid/src/rc/applyDDGIShading.ts` | Pending Q-WA-3 |
| `walkaround/giReceiver.ts` | `@vitrum/walkaround-hybrid/src/rc/giReceiver.ts` | Pending Q-WA-3 |
| `walkaround/walkaroundDiffuseLighting.ts` | `@vitrum/walkaround-hybrid/src/rc/walkaroundDiffuseLighting.ts` | Pending Q-WA-3 |
| `walkaround/engineRegistry.ts` | Host-only (Redux) | Do not extract |
| `walkaround/engines/restir/WalkaroundGPUPipeline.ts` | `@vitrum/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts` | Extract (god-file split pending Q-WA-6) |
| `walkaround/engines/restir/bvhCompute.ts` | `@vitrum/walkaround-hybrid/src/restir/bvhCompute.ts` | Extract |
| `walkaround/engines/restir/walkaroundBridgeTypes.ts` | Host-only (window globals) | Do not extract |
| `walkaround/engines/restir/shaders/common.wgsl.ts` | `@vitrum/walkaround-hybrid/src/shaders/common.wgsl.ts` | Extract |
| `walkaround/engines/restir/shaders/ris.wgsl.ts` | `@vitrum/walkaround-hybrid/src/shaders/ris.wgsl.ts` | Extract |
| `walkaround/engines/restir/shaders/temporal.wgsl.ts` | `@vitrum/walkaround-hybrid/src/shaders/temporal.wgsl.ts` | Extract |
| `walkaround/engines/restir/shaders/spatial.wgsl.ts` | `@vitrum/walkaround-hybrid/src/shaders/spatial.wgsl.ts` | Extract |
| `walkaround/engines/restir/shaders/shade.wgsl.ts` | `@vitrum/walkaround-hybrid/src/shaders/shade.wgsl.ts` | Extract |
| `walkaround/engines/restir/shaders/composite.wgsl.ts` | `@vitrum/walkaround-hybrid/src/shaders/composite.wgsl.ts` | Extract |
| `walkaround/engines/restir/shaders/atrous.wgsl.ts` | `@vitrum/shared-denoisers/src/wgsl/atrous.wgsl.ts` | Extract (after walkaround-hybrid stable) |
| `walkaround/engines/restir/shaders/temporalAccum.wgsl.ts` | `@vitrum/shared-denoisers/src/wgsl/temporalAccum.wgsl.ts` | Extract (after walkaround-hybrid stable) |
