# Vitrum Complexity Sweep — 2026-05-09

## Summary
- Files scanned: 62 TypeScript source files (excludes `_staging/`, `node_modules`, `dist`)
- Total LOC: 12,694 (raw); estimated ~8,000 code-only lines after blanks + comments stripped
- HOT hotspots: 2
- WARM hotspots: 4
- COOL hotspots: 4

---

## Per-package size summary

| Package | File count | Total LOC | Largest file (LOC) |
|---|---|---|---|
| `walkaround-hybrid` | 34 | 8,908 | `shaders/common.wgsl.ts` (832) |
| `shared-bvh` | 5 | 1,233 | `bvhCommon.ts` (569) |
| `core` | 7 | 909 | `engine.ts` (201) |
| `pt-webgl` | 8 | 732 | `iblBaker.ts` (224) |
| `three-bindings` | 1 | 354 | `index.ts` (354) |
| `shared-denoisers` | 3 | 200 | `wgsl/atrous.wgsl.ts` (114) |
| `shared-samplers` | 2 | 57 | `wgsl/hammersley.wgsl.ts` (~50) |
| `pt-webgpu` | 1 | 8 | `index.ts` (8) |

---

## HOT hotspots (immediate refactor recommended)

### HOT-1 DDGI UBO placeholder layout duplicated across two files

**Files**:
- `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts` lines 503–511
- `packages/walkaround-hybrid/src/pipeline/resourceManager.ts` lines 173–181

**Metrics**: Two verbatim identical 12-float DDGI UBO packs (origin + spacing + dims + atlas dims), written inline as `placeholder[3] = 24`, `placeholder[8] = 1`, etc., with no shared constant or helper function.

**Issue**: The WGSL `DDGIGridUniform` struct layout (origin vec3f + f32 spacing + vec3u dims + padding + 4 atlas size floats = 64 bytes) is encoded independently in both files. If the WGSL struct gains a field, both sites must change in lockstep — and they are 300+ lines apart in different files with no cross-reference comment.

`WalkaroundGPUPipeline.setDDGIInputs(null)` (line 502) packs the placeholder fresh every time it is called (i.e., every frame when DDGI is off), allocating a new `Float32Array(16)` per call in the hot path.

**Suggested decomposition**:
1. Move the zero-DDGI UBO pack into `resourceManager.ts` as an exported `buildDDGIPlaceholderUBO(): Float32Array` helper (4 lines).
2. Replace both duplicated blocks with a single call to that helper.
3. Cache the result in `WalkaroundGPUPipeline` so `setDDGIInputs(null)` reuses a pre-built `ArrayBuffer` instead of allocating per call.

**Estimated effort**: 1–2 hours.

---

### HOT-2 `three-bindings/src/index.ts` — single-file package with no decomposition room

**File**: `packages/three-bindings/src/index.ts` (354 LOC, only file in the package's `src/`)

**Metrics**: 354 LOC, 1 file, 12 imports from `@vitrum/core`, 2 private type aliases (`ThreeStdMat`, `ThreePhysMat`), one exported function `sceneFromThreeJS` that is 145 LOC long and contains 6 `if` branches + 4 nested guard checks + 1 `traverse` call.

**Issue**: This is not a complexity crisis, but the single-file constraint means there is no test surface for individual sub-converters. The `convertMesh`, `convertMaterial`, `resolveEnvironment`, and each emitter-type conversion are private module-scope functions — none are exported or independently testable. When Sprint 2 adds more light types or material kinds (clearcoat sheen, subsurface), this file will grow without a natural split point.

The `sceneFromThreeJS` traverse callback (lines 204–337) throws on unsupported types rather than skipping with a warning, which is the right strictness for now but will need per-type opt-in control as the type list grows.

**Suggested decomposition**: Extract `materialConverters.ts`, `meshConverter.ts`, and `lightConverters.ts` as separate modules before the Sprint 2 light-tree work begins. `index.ts` re-exports `sceneFromThreeJS` only. Each converter becomes independently testable. No API surface changes.

**Estimated effort**: 2–3 hours.

---

## WARM hotspots (refactor at next change)

### WARM-1 `probeUpdatePass.ts` — 468 code-lines, 3-pipeline god method

**File**: `packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts` (640 LOC, ~468 code)

**Metrics**: `ProbeUpdatePass` class has 12 private methods + 3 public methods. `init()` is ~120 code lines. `runFrame()` is ~90 code lines and orchestrates buffer reallocations, BVH version-check, per-frame uniform uploads (5 separate `_upload*` calls), and 3 compute dispatches. Every `_upload*` method allocates a fresh `Float32Array` on each call.

**Issue**: The `_uploadMaterials` method (lines 336–378) is a standalone material-struct packer using the `DDGIMaterial` WGSL layout (64 bytes = 16 floats per entry). `rc/bvhCompute.ts` has a separate `packCascadeMaterials` function with a different WGSL layout (also 16 floats, but different field order). Neither references the other. As long as the layouts stay different, this is justified. Watch this if DDGI and RC ever share a material format.

The pre-allocated `GPUResources` placeholder buffers at construction use magic sizes (`makeBuffer(12, RO)` for position data, `makeBuffer(64 * 64, UB)` for materials) that have no named constants. If DDGI's material count exceeds 64 or the stride changes, these are the wrong sizes with no compile-time protection.

**Suggested action**: Name the magic buffer sizes (`DDGI_MAX_MATERIALS = 64`, `DDGI_MATERIAL_STRIDE_BYTES = 64`). Extract a `DDGIBufferSizes` constants object. Allocate placeholder buffers from those constants. The method structure is otherwise proportional to the algorithmic complexity — no splitting recommended yet.

**Estimated effort**: 1 hour.

---

### WARM-2 `restir/bvhCompute.ts` — 377 code-lines, `buildEmitterList` is 130-line loop

**File**: `packages/walkaround-hybrid/src/restir/bvhCompute.ts` (704 LOC, ~377 code)

**Metrics**: Top-level `buildSceneBVH` is well-decomposed into 5 named steps. The private `buildEmitterList` function (lines 529–703) is 130 code-lines: a single `for (let t = 0; t < triCount; t++)` loop with two `if/else` branches (emissive path → transmissive path), inline `THREE.Vector3` temp objects declared in the enclosing scope (`_va`, `_vb`, `_vc`, `_ab`, `_ac`, `_cross`), and an inner transmissive-panel sub-block with 5 multiplications and a `skipEmitter` userData check.

The `packBVHIndexW` function (lines 387–452) contains a magic inline default: `let r = 153, g = 148, b = 140` (lines 405, 477) — a hardcoded warm-gray fallback color in two separate functions, with no shared constant.

**Issue**: The hardcoded warm-gray `(153, 148, 140)` appears verbatim in both `packBVHIndexW` (line 405) and `packBVHBeerColors` (line 477). These must match; if the default color changes, both must update. Extract as `DEFAULT_OPAQUE_COLOR_RGBA8 = { r: 153, g: 148, b: 140 }`.

`buildEmitterList` is at the edge of readable for a single function but not yet over it. The loop logic is sequential and well-commented. Flag for extraction (into `classifyEmitter(tri, mat, options)` + `packEmitterStruct(emitterData)`) if it grows further.

**Estimated effort**: 30 minutes for the constant extraction; 2–3 hours if full emitter-list decomposition is desired.

---

### WARM-3 `rc/cascadeDispatch.ts` — `_buildHandles` is a 175-line setup method

**File**: `packages/walkaround-hybrid/src/rc/cascadeDispatch.ts` (508 LOC, ~329 code)

**Metrics**: `RCDispatcher._buildHandles` (lines 320–493) is 173 code-lines. It creates cast pipelines (5), merge pipelines (4), uniform buffers, bind groups, and the env-texture fallback in one sequential block. All of this is one-time setup, so the length is linear (no deep nesting), but it is difficult to read as a unit: the env-texture fallback (lines 351–390) is embedded mid-function between BVH buffer extraction and the cast-pipeline loop.

`buildCascadeUniformDataInto` (lines 91–126) directly packs a `Float32Array` using bare numeric indices `d[0]`, `ui[8]`, etc. The comment provides the layout, but the actual field names exist only in the comment, not as named offsets.

**Issue**: No structural splitting needed. Two targeted improvements: (1) extract the env-texture fallback into a private `_buildEnvBinding(device, opts)` method — 40 lines to a named helper; (2) add named offset constants for `buildCascadeUniformDataInto` analogous to how `uboUpdater.ts` documents the UBO layout in its header comment.

**Estimated effort**: 1–2 hours.

---

### WARM-4 `WalkaroundGPUPipeline.renderFrame` — inline ping-pong alpha logic

**File**: `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts` (520 LOC, method at lines 242–464)

**Metrics**: `renderFrame` is 222 LOC. The temporal accumulator logic (lines 386–418) computes camera delta, applies the `camMoveSq > 1.0` threshold, and picks `alpha` — all inline in the frame method.

**Issue**: The magic threshold `1.0` (line 396) is the only unnamed constant in the frame path. It is documented in a comment but has no named constant. When the threshold needs tuning (it has been tuned before), a reader must search for `1.0` among many numeric literals.

The `setDDGIInputs(null)` path (lines 497–511) packs a fresh `Float32Array(16)` on every call, which is in the warm path when DDGI is disabled. This is the same HOT-1 issue from a different call site.

**Suggested action**: `const CAM_MOVE_RESET_THRESHOLD_SQ = 1.0;` at file top. Fix HOT-1 to resolve the hot-path allocation.

**Estimated effort**: 30 minutes.

---

## COOL hotspots (FYI only)

### COOL-1 `shared-bvh/bvhCommon.ts` — well-structured at 569 LOC

**File**: `packages/shared-bvh/src/bvhCommon.ts` (569 LOC, ~261 code)

`buildSceneBVH` is 240 code-lines but organized as 8 numbered sequential steps with clear phase comments. No function exceeds 60 code-lines. The depth nesting stays at ≤2 levels (one `if`, one loop). Comment density is high (~45% of total lines are comments/docs) — appropriate for a shared algorithmic module. No action needed.

---

### COOL-2 `ddgi/probeGrid.ts` — magic spacing constant `worldSpacing = 24`

**File**: `packages/walkaround-hybrid/src/ddgi/probeGrid.ts` (203 LOC)

Line 37: `worldSpacing: number = 24;`. The value 24 is the default probe spacing in world inches. The same value `24` is used as the placeholder UBO spacing in `resourceManager.ts` (line 173) and `WalkaroundGPUPipeline.setDDGIInputs(null)` (line 503). These three must match. Currently they are all hardcoded separately.

**Note**: They will stay in sync as long as the default spacing is never changed. If the default ever changes, all three sites require simultaneous update. A shared `DDGI_DEFAULT_PROBE_SPACING = 24` constant in `ddgi/types.ts` would eliminate the risk.

**Estimated effort**: 20 minutes.

---

### COOL-3 `HybridEngine.ts` — 658 LOC, justified size

**File**: `packages/walkaround-hybrid/src/HybridEngine.ts` (658 LOC, ~380 code)

The class has 15 private fields, 10 methods, and a module-scope factory function. The largest method, `renderFrame` (lines 273–445), is 172 LOC with one DDGI branch, one debug branch, one matrix-copy block, and one `pipeline.renderFrame()` delegation. The DDGI atlas wire block (lines 360–387) packs the grid-params `ArrayBuffer` inline using named offsets (`f32[0]`, `u32[4]`), with a comment matching the layout. This is the correct level of documentation.

The `_initPipeline` method contains a polling loop with `setTimeout` — the only async retry pattern in the codebase. The loop is bounded (5s timeout, 50ms intervals) and well-commented. Not a complexity concern.

No action needed.

---

### COOL-4 `common.wgsl.ts` — 832-line WGSL string, length is intrinsic

**File**: `packages/walkaround-hybrid/src/shaders/common.wgsl.ts` (832 LOC)

The TypeScript wrapper is 3 lines (import comment + `export const COMMON_WGSL = ...` + closing backtick). All remaining lines are WGSL shader code. The WGSL contains the BVH traversal kernel, full GGX BRDF, PCG RNG, reservoir pack/unpack, and emitter sampling — all algorithmically necessary for a path tracer. Splitting the WGSL string across files would require a string-concatenation step in `pipelineCompiler.ts`, which already does this for per-pass shaders. This is algorithmic length, not accidental complexity. No action needed.

---

## Did the 7-way split work?

**Yes — the split achieved genuine separation of concerns.**

`WalkaroundGPUPipeline.ts` (520 LOC) imports from exactly 6 of its 7 sibling files (all except `uboUpdater.ts` which is imported transitively). Read the actual import block at lines 26–55: each import is narrow — `updateUBO` from `uboUpdater`, `compilePipelines` from `pipelineCompiler`, 4 builder functions + `UboRef` from `bindGroupBuilders`, `BGLCache` from `bindGroupLayouts`, 6 timestamp functions from `timestampQueries`, and resource management from `resourceManager`.

The class itself is a thin coordinator: `initialize()` calls `uploadBuffer`, `createFrameResources`, `compilePipelines`, `initTimestampQueries`. `renderFrame()` calls `updateUBO`, 7 `build*BindGroup` calls, then sequences GPU encoder calls. `dispose()` calls `destroyFrameResources`, `disposeTimestampState`. There is no complex logic in the class itself — all algorithmic work lives in the modules.

The split is **not cosmetic**. Each module is independently readable:
- `resourceManager.ts` (222 LOC): knows nothing about shaders or bind groups.
- `bindGroupLayouts.ts` (154 LOC): knows nothing about frame resources.
- `bindGroupBuilders.ts` (244 LOC): knows about layouts and resources, knows nothing about pipelines.
- `pipelineCompiler.ts` (148 LOC): knows about shaders and layouts, knows nothing about resources.
- `timestampQueries.ts` (188 LOC): self-contained, no pipeline dependencies.
- `uboUpdater.ts` (53 LOC): trivially small, pure function.

One structural note: `WalkaroundGPUPipeline.ts` still holds all BVH buffer fields (8 `GPUBuffer` private fields, lines 139–146) and orchestrates `uploadBuffer` for each at initialization. Those fields could move into `resourceManager.ts` as part of a `SceneResources` bundle, but this is a polish-level refactor, not a complexity problem.

---

## Magic numbers / unnamed constants

Notable instances, in priority order:

| File | Line(s) | Value | Issue |
|---|---|---|---|
| `restir/bvhCompute.ts` | 405, 477 | `153, 148, 140` | Default warm-gray RGBA duplicated in two functions |
| `pipeline/resourceManager.ts` | 173–181 | `24, 1, 1, 1, 1` | DDGI placeholder UBO spacing + dims (duplicated in WalkaroundGPUPipeline.ts:503–511) |
| `pipeline/WalkaroundGPUPipeline.ts` | 503–511 | `24, 1, 1, 1, 1` | Same DDGI placeholder UBO (see HOT-1) |
| `ddgi/probeGrid.ts` | 37 | `24` | Default probe spacing, appears in 3 files |
| `pipeline/WalkaroundGPUPipeline.ts` | 396 | `1.0` | Camera-move reset threshold (see WARM-4) |
| `ddgi/probeUpdatePass.ts` | 207 | `64 * 64` | Material buffer size: 64 materials × 64 bytes. Functional but not named |
| `walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts` | 365 | `3` | À-trous iteration count — could be `ATROUS_ITERATIONS = 3` |

The `EMITTER_STRIDE = 80` and `EMITTER_FLOATS = EMITTER_STRIDE / 4` constants in `restir/bvhCompute.ts` (lines 144–145) are the **positive example** — size math done once at the top, referenced by index throughout. Apply this pattern to the unnamed constants above.

---

## File-pair duplication

### `restir/bvhCompute.ts` vs `rc/bvhCompute.ts`

These files serve different roles and have minimal duplication:

- `restir/bvhCompute.ts` (704 LOC): delegates BVH build to `@vitrum/shared-bvh`, then does ReSTIR-specific packing — UV-into-position-w, RGBA8+texType into bvhIndex.w, Beer-Lambert color buffer, and emitter list construction.
- `rc/bvhCompute.ts` (126 LOC): delegates BVH build to `@vitrum/shared-bvh`, then does RC-specific packing — `MaterialEntry` flat-struct (16 floats, different layout from DDGI's 16-float layout), wrapped in `StorageBufferAttribute` for the Three.js WebGPU backend.

**What's shared**: Both call `buildSharedBVH` with `positionStride: 4`. Both receive the same `materials: THREE.Material[]` array and pack it into a flat Float32Array. The field ordering differs (RC packs `ior`, `attenuationDistance`, `roughness`, `metalness`, `thickness`; DDGI packs `ior`, `transmission`, `metalness`, `roughness`, `attenuationColor`, `flags`).

**The duplication is justified** because the two WGSL struct layouts are intentionally different — RC's `probeRayCast.wgsl` needs transmission + IOR + attenuation distance for Beer-Lambert; DDGI's `probeUpdateRays.wgsl` needs flags (isGlass bit) for branch control. Unifying them would require WGSL struct changes on both sides.

**One real concern**: `packBVHBeerColors` in `restir/bvhCompute.ts` and `_uploadMaterials` in `ddgi/probeUpdatePass.ts` both contain the `applyBeerLambert` logic — but only `restir/bvhCompute.ts` has the named `applyBeerLambert` function (lines 42–56). DDGI's `_uploadMaterials` does not apply Beer-Lambert at all (it uses raw `attenuationColor`). This is a design choice, not a bug — but it should be documented in `probeUpdatePass._uploadMaterials` to prevent a future contributor from "fixing" it by adding Beer-Lambert.

### `rc/bvhCompute.ts` vs `shared-bvh/bvhCommon.ts`

No overlap. `rc/bvhCompute.ts` is a thin adapter (126 LOC total, ~60 code) that calls `shared-bvh` and wraps the result in `StorageBufferAttribute`. The extraction plan worked correctly here.
