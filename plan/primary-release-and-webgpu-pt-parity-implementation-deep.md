# Deep Implementation Specification — Primary Release + WebGPU PT Parity

**Date:** 2026-05-26  
**Status:** Complete on `main` (2026-05-26). Signoff: `plan/PR-signoff-2026-05-26.md`, `plan/WG-signoff-2026-05-26.md`. CI: `npm run benchmark:gap-closure-mechanical` (six pt-webgpu baselines). Deferred by design: WG-7 BDPT, WG-9 svgf-real on pt-webgpu, PR-7/8 optional, pt-webgpu in `auto`.  
**Out of scope:** npm publish, release governance, cross-host verification policy.

---

## Table of contents

1. [How to read this document](#1-how-to-read-this-document)
2. [Program map and critical path](#2-program-map-and-critical-path)
3. [Shared foundation — `scenePack` + TLAS WGSL](#3-shared-foundation--scenepack--tlas-wgsl)
4. [Program PR — wave-by-wave deep spec](#4-program-pr--wave-by-wave-deep-spec)
   - [PR-0 Contract hygiene](#pr-0-contract-hygiene-week-1)
   - [PR-1 Material + emitter fast paths](#pr-1-material--emitter-fast-paths-weeks-2-3)
   - [PR-2 TLAS CPU pack](#pr-2-tlas-cpu-pack-weeks-4-6)
   - [PR-3 TLAS GPU (ReSTIR)](#pr-3-tlas-gpu-restir-weeks-7-11)
   - [PR-4 TLAS incremental refit](#pr-4-tlas-incremental-refit-weeks-12-13)
   - [PR-5 DDGI + RC alignment](#pr-5-ddgi--rc-alignment-weeks-12-15)
   - [PR-6 Scale + soak](#pr-6-scale--soak-weeks-14-16)
   - [PR-7 GPU skinning (optional)](#pr-7-gpu-skinning-optional)
   - [PR-8 pt-webgl incremental (optional)](#pr-8-pt-webgl-incremental-optional)
   - [PR-9 Signoff](#pr-9-primary-release-signoff)
5. [Program WG — wave-by-wave deep spec](#5-program-wg--wave-by-wave-deep-spec)
   - [WG-0 Visual harness](#wg-0-visual-harness-weeks-1-2)
   - [WG-1 OIDN-final](#wg-1-oidn-final-weeks-3-4)
   - [WG-2 Hero λ + CMF MIS](#wg-2-hero-λ--cmf-mis-weeks-5-10)
   - [WG-3 Cauchy dispersion](#wg-3-cauchy-dispersion)
   - [WG-4 Layered BSDF + transmission MIS](#wg-4-layered-bsdfs--transmission-mis)
   - [WG-5 SSS / translucent](#wg-5-sss--translucent)
   - [WG-6 Geometry incremental + scenePack migration](#wg-6-geometry-incremental)
   - [WG-7 BDPT (optional)](#wg-7-bdpt-optional)
   - [WG-8 createEngine opt-in](#wg-8-createengine-opt-in)
   - [WG-9 Extended denoisers (optional)](#wg-9-extended-denoisers-optional)
   - [WG-10 Signoff](#wg-10-webgpu-pt-signoff)
6. [Cross-cutting test matrix](#6-cross-cutting-test-matrix)
7. [Risk register and decision log](#7-risk-register-and-decision-log)

---

## 1. How to read this document

Each wave contains:

| Section | Purpose |
|---------|---------|
| **Metadata** | Duration, dependencies, owners, feature flags |
| **As-is** | Exact current code behavior with file:line anchors |
| **To-be** | Target behavior, invariants, and non-goals |
| **Data layouts** | Byte-level structs, GPU bindings, CPU arrays |
| **Implementation steps** | Numbered micro-tasks in commit order |
| **File manifest** | Every path touched (create / modify / delete) |
| **Tests** | New or extended tests with exact assertions |
| **Perf / memory** | Budgets and profiling commands |
| **Failure modes** | Degradation paths |
| **DoD** | Machine-checkable exit criteria |
| **Audit gate** | `/audit` scope after the wave |

Micro-task IDs are stable (`PR-2.7`, `WG-2.4`) for issue tracking.

---

## 2. Program map and critical path

### 2.1 Two programs

| Program | Packages | Exit artifact |
|---------|----------|----------------|
| **PR** | `walkaround-hybrid`, `pt-webgl`, `engine`, `shared-bvh`, `three-bindings` | `plan/PR-signoff-YYYY-MM-DD.md` + reference renders under `tools/reference-renders/PR-*` |
| **WG** | `pt-webgpu`, `shared-samplers`, `shared-denoisers`, `benchmark-runner` | Gap-closure JSON PASS + `pt-webgpu` README parity section |

### 2.2 Critical path (cannot parallelize)

```
PR-0 → PR-1 → PR-2 → PR-3 → PR-4 → PR-6 → PR-9
                    ↘ PR-5 (parallel after PR-3 CPU buffers exist)
WG-0 → WG-1 → WG-2 → WG-4 → WG-10
WG-0 → (parallel) WG-2
PR-2 → WG-6 (shared scenePack)
```

### 2.3 Parallel lanes (2 FTE)

| Lane A (Hybrid) | Lane B (pt-webgpu) |
|-----------------|---------------------|
| PR-0, PR-1, PR-2, PR-3, PR-4, PR-5, PR-6, PR-9 | WG-0, WG-1, WG-2, WG-3, WG-4, WG-5, WG-6, WG-8, WG-10 |
| Owns `shared-bvh/scenePack.ts` API design | Consumes scenePack; owns spectral/OIDN WGSL |

**Merge conflict hotspot:** `packages/shared-bvh/src/index.ts`, `packages/shared-bvh/src/scenePack.ts` — Lane A opens PR-2.1; Lane B only refactors pt-webgpu imports after PR-2.5 lands.

---

## 3. Shared foundation — `scenePack` + TLAS WGSL

This section is the **contract** both programs implement. No wave may fork a second packer.

### 3.1 New module: `packages/shared-bvh/src/scenePack.ts`

**Extract from:** `packages/pt-webgpu/src/scene/uploadSceneBuffers.ts` (`buildPackedScene`, `buildTlasFromInstances`, `rebuildTlasForSceneTransforms` logic).

#### 3.1.1 Public types

```ts
/** Per-primitive bookkeeping for transform-only TLAS refit (stable across frames). */
export interface PrimitiveTlasBinding {
  readonly primitiveId: string;
  readonly primitiveKind: 'mesh' | 'instanced-mesh' | 'skinned-mesh';
  readonly blasRoot: number;        // node index in concatenated bvhNodes
  readonly instanceCount: number;   // 1 for mesh/skinned; N for instanced
  readonly localAabbMin: readonly [number, number, number];
  readonly localAabbMax: readonly [number, number, number];
}

export interface ScenePackOptions {
  /** Build per-primitive local BLAS + TLAS. If false, caller uses merged path only. */
  readonly tlas?: boolean;
  /** Material id resolver: (primitiveId) => u32 mat slot */
  readonly resolveMaterialId: (primitiveId: string) => number;
}

export interface ScenePackResult {
  // Geometry (concatenated BLAS)
  readonly positions: Float32Array;      // vec4f stride 4, LOCAL space when tlas
  readonly normals: Float32Array;
  readonly indices: Uint32Array;          // vec4u stride 4
  readonly triMaterialIds: Uint32Array;
  readonly bvhNodes: Float32Array;        // 8 floats / node (32 B)
  readonly triangleCount: number;
  // TLAS (empty arrays when tlas:false)
  readonly tlasNodes: Uint32Array;
  readonly tlasInstanceIndices: Uint32Array;
  readonly tlasBlasRoots: Uint32Array;
  readonly tlasInstanceWorldToLocal: Float32Array; // 4 vec4f per instance
  readonly tlasInstanceLocalToWorld: Float32Array;
  readonly tlasNodeCount: number;
  readonly primitiveTlasBindings: readonly PrimitiveTlasBinding[];
  readonly warnings: readonly string[];
}
```

**API:**

- `packSceneFromCore(scene: Scene, opts: ScenePackOptions): ScenePackResult`
- `refitTlasTransforms(scene: Scene, bindings: readonly PrimitiveTlasBinding[], prev?: TlasGpuSnapshot): RefitTlasResult` — port of `rebuildTlasForSceneTransforms`

**Invariants:**

1. BLAS node layout = `shared-bvh` / three-mesh-bvh 32-byte layout (same as `tlas.ts` header).
2. Leaf triangle offset in concatenated BLAS is **absolute** in global index buffer (pt-webgpu lines 388–400 pattern).
3. `instanced-mesh`: one TLAS instance per matrix in `primitive.instances`.
4. `skinned-mesh`: treat as `mesh` with deformed positions supplied by host on CPU before pack (same as today’s `solveSkin` → positions patch).

#### 3.1.2 Algorithm: `packSceneFromCore` (pseudocode)

```
for each primitive in scene.primitives:
  if kind not in {mesh, skinned-mesh, instanced-mesh}: skip with warning
  build localPositions vec4 from primitive.positions (w=0)
  build localNormals vec4
  build localIndices vec4u from primitive.indices (w=0)
  localTriMatIds.fill(resolveMaterialId(id))
  localBvh = buildArrayBvh(localPositions, localIndices, localTriMatIds)
  nodeBase = bvhNodes.length / 8
  vertexBase = positions.length / 4
  triBase = triMaterialIds.length
  concat nodes with tri offset += triBase
  concat vertices/norms/indices with index += vertexBase
  localAabb = computeLocalAabb(positions)
  for each instance transform:
    push TlasInstance(blasRoot=nodeBase, world AABB, w2l, l2w)
  push PrimitiveTlasBinding(...)
if tlas: buildTlas(instances) else: empty TLAS arrays
return ScenePackResult
```

#### 3.1.3 Tests (`packages/shared-bvh/src/__tests__/scenePack.test.ts`)

| Test ID | Setup | Assert |
|---------|-------|--------|
| SP-1 | 2 boxes, static | `tlasNodeCount > 0`, `tlasIntersect` oracle = merged hit for same rays |
| SP-2 | 1 mesh, transform animation refit | `refitTlasTransforms` ok, AABB changes, topology unchanged |
| SP-3 | instanced 4× | `instanceCount === 4`, 4 TLAS leaves |
| SP-4 | remove primitive | refit returns `ok: false` with reason containing primitive id |
| SP-5 | 30k tris single mesh | pack < 200 ms CPU (soft budget, log only) |

### 3.2 New WGSL: `packages/shared-bvh/src/wgsl/tlasTraversal.wgsl.ts`

**Extract from:** `packages/pt-webgpu/src/wgsl/pathTrace/intersection.wgsl.ts` (`traceTlasClosest`, `traceTlasAny`, `traceMeshBvh` with `blasRoot`).

#### 3.2.1 New functions (exported string)

| Function | Signature | Semantics |
|----------|-----------|-----------|
| `traceMeshBvhAtRoot` | `(bvh_index, bvh_position, bvh, ray, tMin, tMax, closest, hit*, rootNode, decodeW)` | Same as canonical `bvhIntersectFirstHit` but stack starts at `rootNode` |
| `traceTlasClosest` | `(tlasNodes, tlasInstanceIndices, tlasBlasRoots, w2l, l2w, ..., hit*)` | Port pt-webgpu lines 427–497 |
| `traceTlasAny` | same + `tMax` | Port lines 499+ |
| `traceSceneClosest` | `(params: SceneTraceParams, ...)` | **Unified entry:** if `params.bvhMode == 0` → merged `bvhIntersectFirstHit`; if `1` → `traceTlasClosest` |

#### 3.2.2 Scene trace params (uniform extension)

Add to **Walkaround** `WalkaroundUBO` (PR-3) — requires layout bump:

| Field | Type | Offset (after `_padEnd`) | Values |
|-------|------|--------------------------|--------|
| `bvhMode` | `u32` | 336 | `0` = merged world BVH, `1` = TLAS+local BLAS |
| `tlasNodeCount` | `u32` | 340 | from CPU pack |
| `_tracePad` | `u32` × 2 | 344–352 | align to 352 bytes total UBO |

**Host packer:** `packages/walkaround-hybrid/src/pipeline/uboUpdater.ts` — extend `packWalkaroundUbo` with new fields; add test in `__tests__/uboLayout.test.ts` if exists, else create.

**WGSL modules to register in** `walkaround-hybrid/src/pipeline/wgslModules.ts` (or equivalent registry): `tlasTraversal` depends on `bvhIntersect`.

#### 3.2.3 Scene bind group extension (hybrid)

**Current** (`bindGroupLayouts.ts:88-101`): bindings 0–5.

**Target** (`getSceneBindGroupLayout`):

| Binding | Buffer | Visibility |
|---------|--------|------------|
| 0 | `bvhNodes` (concat BLAS) | compute |
| 1 | `bvhIndex` vec4u | compute |
| 2 | `bvhPositions` vec4f **local** when TLAS | compute |
| 3 | emitters | compute |
| 4 | emitterCdf | compute |
| 5 | bvhBeerColors | compute |
| 6 | `tlasNodes` | compute |
| 7 | `tlasInstanceIndices` | compute |
| 8 | `tlasBlasRoots` | compute |
| 9 | `tlasInstanceWorldToLocal` (vec4f×4 per instance) | compute |
| 10 | `tlasInstanceLocalToWorld` | compute |

**Pipeline impact:** Every pass using `@group(1)` scene BGL must re-create bind groups — `buildSceneBindGroup` in `bindGroupBuilders.ts` gains optional TLAS buffers (undefined → 16-byte dummy buffer for merged mode to avoid dual pipeline layouts, OR two scene BGL variants — **decision: dummy buffers in merged mode** to keep one layout).

**Dummy buffer contract:** 16-byte zeroed storage buffer bound to 6–10 when `bvhMode==0`; WGSL early-outs if `tlasNodeCount==0` (same as pt-webgpu).

### 3.3 pt-webgpu migration (PR-2.5 / WG-6)

| Step | Action |
|------|--------|
| WG-6.0 | Re-export `packSceneFromCore` from `pt-webgpu` `buildPackedScene` body = 10-line wrapper |
| WG-6.0 | Delete duplicated `buildTlasFromInstances` private copy after wrapper passes all `scenePack.test.ts` + `scenePack.emitters.test.ts` |

---

## 4. Program PR — wave-by-wave deep spec

---

### PR-0 Contract hygiene (Week 1)

**Metadata:** 5 person-days · depends on nothing · risk **Low**

#### As-is

| Issue | Location | Behavior |
|-------|----------|----------|
| `_lastScene` stale after geometry fast paths | `HybridEngine.ts:539-611` | `transformRefit` / `positionsRefit` update GPU + THREE mesh but not `_lastScene` |
| `updateEmitter` full reinit | `HybridEngine.ts:625-646` | Calls `setScene()` → `_teardownPipeline` + async reinit |
| Material patch full reinit | `HybridEngine.ts:598-605` | `setScene` on material-only |
| `updateEmitters` exists but unused publicly | `WalkaroundGPUPipeline.ts:614-619` | Only rebuilds emitter + CDF buffers |

#### To-be

1. `_lastScene` always reflects successful patches.
2. `updateEmitter` → CPU rebuild emitter list only + `pipeline.updateEmitters`.
3. Capabilities: `emitter: true` after PR-0.2; `material: false` until PR-1.

#### Implementation steps

**PR-0.1 — Sync `_lastScene` on geometry paths**

1. Add `applyPrimitivePatchToLastScene(scene, id, patch): Scene` in `HybridEnginePrimitiveUpdates.ts` (pure function).
2. In `transformRefit`, after step 7 (GPU upload), call it for `{ transform: patch.transform }` merged from current prim.
3. In `positionsRefit`, merge `positions`, `normals` if present.
4. In `topologyRebuild`, replace primitive in scene from patch before rebuild (already mutates THREE mesh — add vitrum scene sync).
5. `HybridEngine.updatePrimitive` assigns `this._lastScene = result.updatedScene` when helpers return it (extend `PrimitiveUpdateResult`).

```ts
export interface PrimitiveUpdateResult {
  readonly bvhBuffers: SceneBVHBuffers;
  readonly updatedScene?: Scene; // NEW
}
```

**PR-0.2 — Emitter fast path**

1. Add `rebuildEmitterBuffersFromScene(scene, lightDir, intensity): Pick<SceneBVHBuffers,'emitters'|'emitterCdf'|'emitterCount'|'totalEmissivePower'>` in `restir/bvhCompute.ts`:
   - Reuse `buildEmitterList(shared.mergedGeometry, ...)` **without** full BVH build.
   - Requires passing merged geometry OR re-walking THREE scene for emissive tris only.
   - **Implementation choice:** export `buildEmitterListFromRoots(sceneRoots, options)` from `emitterList.ts` (extract triangle walk from `buildReSTIRSceneBVH`).

2. `HybridEngine.updateEmitter`:
   ```
   patch emitter in _lastScene
   buffers = rebuildEmitterBuffersFromScene(...)
   _bvhBuffers.emitters = buffers (swap cpuData handles)
   _pipeline?.updateEmitters(buffers)
   _pipeline?.requestAccumReset()
   DDGI: optional invalidateProbeCache only (no full setScene)
   ```

3. Update `WalkaroundGPUPipeline` UBO fields `emitterCount`, `totalEmPower` on next `renderFrame` (already uploaded from engine state — verify `uboUpdater` reads `_bvhBuffers.emitterCount`).

**PR-0.3 — Capabilities**

```ts
incrementalPatchSupport: {
  transform: true,
  positions: true,
  material: false,
  emitter: true,  // changed
  topology: true,
},
```

Update `packages/core/src/engine/promiseLedger.ts` `walkaround-hybrid` entry.

**PR-0.4 — Docs**

- Archive banner on `plan/archive/animation-support-status.md`.
- Fix `AGENTS.md` / `CLAUDE.md` material bullet: "material fast path → PR-1".

**PR-0.5 — pt-webgl README**

Replace "Pre-alpha" with "Release-candidate track (see root README)".

#### File manifest

| Path | Action |
|------|--------|
| `walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` | modify |
| `walkaround-hybrid/src/HybridEngine.ts` | modify |
| `walkaround-hybrid/src/restir/emitterList.ts` | modify (export builder) |
| `walkaround-hybrid/src/restir/bvhCompute.ts` | modify |
| `walkaround-hybrid/__tests__/hybridEngineGeometryUpdate.test.ts` | extend |
| `walkaround-hybrid/__tests__/hybridEngineEmitterUpdate.test.ts` | **create** |
| `core/src/engine/promiseLedger.ts` | modify |
| `pt-webgl/README.md` | modify |

#### Tests (PR-0)

| ID | File | Assertion |
|----|------|-----------|
| T-0.1 | `hybridEngineGeometryUpdate.test.ts` | After `transformRefit`, `engine._lastScene.primitives[i].transform` equals patch |
| T-0.2 | `hybridEngineEmitterUpdate.test.ts` | Mock pipeline; `updateEmitter` calls `updateEmitters` once, never `setScene` |
| T-0.3 | `promiseLedger.test.ts` | `emitter: true` |

#### DoD (PR-0)

- [ ] T-0.1–T-0.3 green
- [ ] `npm run verify:mechanical`
- [ ] `/audit` HybridEngine + emitterList

---

### PR-1 Material + emitter fast paths (Weeks 2–3)

**Metadata:** 10 person-days · depends PR-0 · risk **Medium** (ReSTIR encoding)

#### As-is

Material change → `setScene` → full pipeline reinit (`HybridEngine.ts:598-605`).

ReSTIR does **not** use a separate materials texture for primary hits — material color is **baked into** `bvhIndex.w` (RGBA8) and `bvhBeerColors` per triangle via `packBVHIndexW` / `packBVHBeerColors` (`packingHelpers.ts`).

#### To-be

Material-only patch:

1. Update `@vitrum/core` primitive in `_lastScene`.
2. Update THREE `MeshPhysicalMaterial` on synthesized mesh (for emitter classification).
3. Re-pack **only** affected triangle slices in `bvhIndex` and `bvhBeerColors` CPU arrays.
4. `queue.writeBuffer` partial ranges on GPU.
5. `requestAccumReset()`; **no** DDGI full invalidation unless transmission/glass changes (see below).

#### Implementation steps

**PR-1.1 — `materialPatch(id, material, ctx)`**

1. Resolve `meshVertexRanges` entry + triangle count = `vertexCount/3` approx via index buffer length / 3.
2. Map `Scene.material` → THREE material via existing adapter in `vitrumSceneToThree` path OR inline `materialToThreePhysical` helper (grep `three-bindings`).
3. For each triangle in range `[triStart, triEnd)`:
   - Recompute `packBVHIndexW` single tri
   - Recompute `packBVHBeerColors` single tri
4. Upload:
   - `bvhIndex`: byte offset `triStart * 16`
   - `bvhBeerColors`: byte offset `triStart * 4`

**PR-1.2 — Transmission / glass flag**

If `material.transmission` crosses 0.01 threshold vs previous:

- Call `ddgi.invalidateProbeCache()` (glass changes DDGI response).
- Optionally rebuild emitter list if transmissive → emitter (`emitterList` rules).

**PR-1.3 — Router in `HybridEngine.updatePrimitive`**

```ts
if (hasMaterialChange && !hasTopologyChange && !hasPositionsChange && !hasTransformChange) {
  materialPatch(...);
  return;
}
```

**PR-1.4 — Capabilities `material: true`**

**PR-1.5 — hero-lighting-designer**

Replace `setScene` debounce with `updateEmitter` + `updatePrimitive({ material })`.

#### File manifest

| Path | Action |
|------|--------|
| `HybridEnginePrimitiveUpdates.ts` | add `materialPatch` |
| `HybridEngine.ts` | route |
| `restir/packingHelpers.ts` | export single-tri helpers if needed |
| `__tests__/hybridEngineMaterialPatch.test.ts` | create |
| `examples/hero-lighting-designer/` | modify |

#### Tests

| ID | Assertion |
|----|-----------|
| T-1.1 | Material patch does not increment `_initSeq` / pipeline compile counter |
| T-1.2 | `bvhIndex.w` byte changes for one tri |
| T-1.3 | Transmission 0→0.5 triggers `invalidateProbeCache` spy |

#### Perf budget

- 1 material / 10k tris mesh: < 2 ms CPU pack + < 0.5 ms GPU upload

#### DoD (PR-1)

- [ ] T-1.1–T-1.3
- [ ] hero-lighting-designer manual smoke
- [ ] `/audit`

---

### PR-2 TLAS CPU pack (Weeks 4–6)

**Metadata:** 15 person-days · depends PR-1 · risk **High**

#### As-is

- `buildReSTIRSceneBVH` → `buildSharedBVH` merges all meshes to **world space** (`bvhCommon.ts`).
- `SceneBVHBuffers` has no TLAS fields (`bvhCompute.ts:64-120`).
- pt-webgpu already packs TLAS separately.

#### To-be

- `SceneBVHBuffers` extended with TLAS GPU handles + `primitiveTlasBindings`.
- `buildReSTIRSceneBVH({ mode: 'merged' | 'tlas' })` — default **`tlas`** when `scene.primitives.filter(isMeshLike).length > 1` OR any `instanced-mesh`.
- Feature flag: `HybridEngineOptions.extensions['vitrum.hybrid.bvhMode']` overrides default.

#### Implementation steps

**PR-2.1** Implement §3.1 `scenePack.ts` in `shared-bvh`.

**PR-2.2** Extend `SceneBVHBuffers`:

```ts
export interface SceneBVHBuffers {
  // ... existing ...
  readonly bvhMode: 'merged' | 'tlas';
  readonly tlas?: {
    nodes: StorageBufferHandle;
    instanceIndices: StorageBufferHandle;
    blasRoots: StorageBufferHandle;
    worldToLocal: StorageBufferHandle;
    localToWorld: StorageBufferHandle;
    nodeCount: number;
  };
  readonly primitiveTlasBindings: readonly PrimitiveTlasBinding[];
}
```

**PR-2.3** `buildReSTIRSceneBVHTlas(sceneRoots, options)`:

1. Convert `THREE.Object3D[]` → temporary `Scene` **or** add `packSceneFromThreeRoots(roots)` that mirrors three mesh extraction (keeps emitter list path on THREE materials).

   **Decision (locked):** Add `packSceneFromThreeMeshes(meshes: MeshExtract[]): ScenePackResult` internal to walkaround; `buildReSTIRSceneBVH` calls it after `buildSharedBVH` mesh extraction refactor.

2. **ReSTIR-specific post-pass** still required on concatenated buffers:
   - `packUVIntoPositionW` on local positions (TLAS mode: UV in local `.w`, transform in TLAS)
   - `packBVHIndexW` / `packBVHBeerColors` — must use THREE materials from per-tri matId
   - `buildEmitterList` — still from merged **world** geometry for emissive discovery OR rebuild from THREE roots (keep current world emitter positions for 1:1 behavior in PR-2; optimize in PR-5)

**PR-2.4** `HybridEngineLifecycle` init: pass `bvhMode` into build.

**PR-2.5** Thin pt-webgpu wrapper (§3.3).

**PR-2.6** CPU oracle tests SP-1..SP-5.

#### File manifest (PR-2)

| Path | Action |
|------|--------|
| `shared-bvh/src/scenePack.ts` | **create** |
| `shared-bvh/src/index.ts` | export |
| `shared-bvh/src/__tests__/scenePack.test.ts` | **create** |
| `walkaround-hybrid/src/restir/bvhCompute.ts` | major modify |
| `walkaround-hybrid/src/HybridEngineLifecycle.ts` | modify |
| `walkaround-hybrid/src/HybridEngine.ts` | options |
| `pt-webgpu/src/scene/uploadSceneBuffers.ts` | thin |
| `pt-webgpu/src/__tests__/scenePack.test.ts` | unchanged if behavior same |

#### DoD (PR-2)

- [ ] SP-1..SP-5 pass
- [ ] Cornell 2-mesh: TLAS `nodeCount > 0`
- [ ] Single-mesh scene: auto merged OR tlas with 1 instance (document choice: **1 instance TLAS OK**)
- [ ] `/audit` scenePack < 500 LOC

---

### PR-3 TLAS GPU (ReSTIR) (Weeks 7–11)

**Metadata:** 20 person-days · depends PR-2 · risk **Critical**

#### Shader call sites (must migrate)

| File | Lines (approx) | Calls |
|------|----------------|-------|
| `shaders/ris.wgsl.ts` | 86, 163 | firstHit, any |
| `shaders/risGi.wgsl.ts` | 112, 150, 205 | firstHit, any |
| `shaders/shade.wgsl.ts` | 199, 438 | any, firstHit |
| `shaders/spatial.wgsl.ts` | grep | any/firstHit |
| `shaders/temporal.wgsl.ts` | grep | any/firstHit |
| `shaders/restirCastPrimary.wgsl.ts` | 50 | firstHit |
| `shaders/surfaceTextures.wgsl.ts` | visibility | any |

**Migration pattern:**

```wgsl
// BEFORE
let hit = bvhIntersectFirstHit(&bvh_index, &bvh_position, &bvh, primaryRay, ubo.triIntersectEpsilon);
// AFTER
var hit: IntersectionResult;
_ = traceSceneClosest(&hit, primaryRay, 0.0, ubo.triIntersectEpsilon);
```

Implement `traceSceneClosest` in `common.wgsl.ts` after including `TLAS_TRAVERSAL_WGSL`.

#### Implementation steps

**PR-3.1** Add §3.2 WGSL module to `shared-bvh`.

**PR-3.2** Register in `WGSL_MODULES` + `composeWgsl` deps.

**PR-3.3** Extend scene BGL + `buildSceneBindGroup` (bindings 6–10, dummy buffers).

**PR-3.4** Upload TLAS buffers in `WalkaroundGPUPipeline.initialize`.

**PR-3.5** Extend WalkaroundUBO + `uboUpdater` (352 bytes — verify `@group(2)` min binding size 352).

**PR-3.6** Migrate each shader file; run `sprint16-restirGiRis.test.ts` string pins updated.

**PR-3.7** GPU test `__tests__/hybridTlasPrimaryHit.gpu.test.ts`:
   - Scene: 2 boxes offset 2m
   - Move one box via `updatePrimitive` transform
   - Readback primary hit id / depth changes

**PR-3.8** Reference render `tools/reference-renders/PR-hybrid-tlas-on/`.

#### WebGPU limits check

```ts
// In initialize():
const maxStorage = device.limits.maxStorageBuffersPerShaderStage;
if (maxStorage < 11) { force bvhMode merged; warn once; }
```

Count storage buffers in worst pass (shade): frame(6+) + scene(11) + ubo — **inventory during PR-3.3** in `pipelineCompiler.ts` and document in README.

#### DoD (PR-3)

- [ ] All ReSTIR tests green
- [ ] GPU test passes with `VITRUM_GPU_TEST=1`
- [ ] Reference PNG hash stored
- [ ] No regression in merged mode (`bvhMode=0`)
- [ ] `/audit` pass registry + WGSL registry

---

### PR-4 TLAS incremental refit (Weeks 12–13)

**Metadata:** 8 person-days · depends PR-3 · risk **High**

#### Transform path rewrite

**Replace** `transformRefit` world-vertex delta (`HybridEnginePrimitiveUpdates.ts:150-177`) when `bvh.bvhMode === 'tlas'`:

```
1. Update THREE mesh matrix
2. refitTlasTransforms(_lastScene, primitiveTlasBindings)
3. writeBuffer tlas nodes + instance transforms only
4. NO bvhPositions wholesale rewrite
5. requestAccumReset()
```

**Merged mode:** keep existing delta path.

#### Positions path

When TLAS + positions patch:

1. Update local positions in `bvhPositions` slice (local space)
2. `refitBvhBounds` on BLAS node range `[blasRoot, ...)`
3. `refitTlas` for instance AABB
4. Upload BLAS nodes + TLAS nodes + position slice

#### Topology

- Rebuild single primitive BLAS via `scenePack` partial rebuild API (**add** `rebuildPrimitiveBlas(scene, primitiveId, bindings): ScenePackResult` in PR-4.3).

#### File manifest

| Path | Action |
|------|--------|
| `HybridEnginePrimitiveUpdates.ts` | branch on bvhMode |
| `shared-bvh/src/scenePack.ts` | `rebuildPrimitiveBlas` |
| `WalkaroundGPUPipeline.ts` | `refreshTlasRefit()` new method |

#### Tests

| ID | Scenario |
|----|----------|
| T-4.1 | TLAS transform: GPU primary hit moves, init counter unchanged |
| T-4.2 | Positions refit: same tri count, BLAS refit only |
| T-4.3 | Instance count change → falls back to topology rebuild |

#### DoD (PR-4)

- [ ] T-4.1–T-4.3
- [ ] Benchmark: transform refit < 2 ms @ 30k tris (log in test)

---

### PR-5 DDGI + RC alignment (Weeks 12–15)

**Metadata:** 12 person-days · parallel PR-4 · risk **Critical**

#### DDGI — Option A (recommended)

**Goal:** DDGI probe rays use same BLAS+TLAS as ReSTIR.

| Step | Detail |
|------|--------|
| PR-5.1 | `SceneBvh.update` stops calling `buildSceneBVH` merge; receives handles from `HybridEngine` shared pack |
| PR-5.2 | `probeUpdateRays.wgsl.ts`: switch `bvhIntersectFirstHitV3` → `traceSceneClosest` OR bind vec4 buffers |
| PR-5.3 | Dirty tracking: on transform refit, **do not** bump geometry version if only TLAS matrices changed — new `ddgi.markInstancesDirty()` |

**Fallback Option B:** Keep DDGI merged; document 2× BVH memory; schedule PR-5.1 for later.

#### RC

| Step | Detail |
|------|--------|
| PR-5.4 | Add `RCSubsystem.refitBounds(aabb)` without `setScene` |
| PR-5.5 | `transformRefit` calls `refitBounds` instead of `setScene(rcRoot)` |
| PR-5.6 | RC BVH still separate for cascade grid — **only** fix per-frame full rebuild on transform |

#### DoD (PR-5)

- [ ] DDGI probe update visual unchanged on Cornell (A/B hash)
- [ ] Transform animation: DDGI rebuild time not 50ms every frame (profile log)
- [ ] RC enabled: transform without `buildRCSceneBVH` in hot path (spy)

---

### PR-6 Scale + soak (Weeks 14–16)

#### Benchmark scenarios (add to `tools/benchmark-runner/`)

| Scenario ID | Engine | Scene | Metric |
|-------------|--------|-------|--------|
| `PR-hybrid-200k-static` | hybrid | generated 200k tris | p95 frame ms |
| `PR-hybrid-tlas-10-inst` | hybrid | 10 instanced chairs | p95 frame ms |
| `PR-hybrid-material-churn` | hybrid | Cornell | 100× material patch, 0 pipeline reinit |
| `PR-hybrid-emitter-churn` | hybrid | 3 lights | 100× emitter patch |

#### Lifecycle soak extensions (`run-lifecycle-soak.mjs`)

Add steps:

1. `updatePrimitive` material patch every 10 frames
2. `updateEmitter` intensity patch
3. Toggle `bvhMode` extension (if exposed in harness)

**Strict:** 8 iterations, 0 failures.

#### DoD (PR-6)

- [ ] JSON artifacts in `tools/benchmark-runner/results/`
- [ ] README perf table updated
- [ ] `hardening:wave4` green

---

### PR-7 GPU skinning (optional)

| Step | Detail |
|------|--------|
| PR-7.1 | Compute shader `skinLbs.wgsl.ts` — read bones SSBO, morph deltas, write positions/normals |
| PR-7.2 | `HybridEngine` option `gpuSkinning: true` |
| PR-7.3 | Dispatch before ReSTIR; chain to `positionsRefit` or direct BLAS refit |

**Normatives:** inverse-transpose 3×3 for normals when `boneMatrix` has non-uniform scale.

---

### PR-8 pt-webgl incremental (optional)

**Audit fork** `three-gpu-pathtracer` for `setGeometry` / matrix-only update.

If none: document in `pt-webgl/README.md` — patches call `setScene` by design; cost is fork BVH rebuild.

---

### PR-9 Primary release signoff

Execute PR-D1–D9 from summary plan +:

- [ ] Update `plan/backend-maturity-matrix-2026-05-26.md` hybrid row → **strong** all columns
- [ ] CHANGELOG entry
- [ ] Tag internal `pr-1.0.0-rc` git tag (only if user requests)

---

## 5. Program WG — wave-by-wave deep spec

---

### WG-0 Visual harness (Weeks 1–2)

**Blocks all fidelity promotions.**

#### As-is

- `tools/benchmark-runner/results/gap-closure-verification-2026-05-26.json` — scenarios **FAIL** (missing baselines).
- pt-webgpu tests are structural (WGSL strings, CPU pack), not image parity.

#### Implementation steps

**WG-0.1 — Capture adapter**

Create `tools/benchmark-runner/capturePtWebgpu.mjs`:

```
Inputs: scenarioId, seed, res, bounces, spp
Steps:
  1. Launch headless WebGPU (SwiftShader flags from vitest.gpu.config)
  2. createPTEngine_WebGPU({ ... })
  3. setScene(loadScenarioScene(scenarioId))
  4. Loop renderFrame until isConverged or spp cap
  5. readback HDR → PNG + sha256
Output: { scenarioId, hash, pngPath, msPerSample }
```

Reuse patterns from `examples/two-engines-one-scene`.

**WG-0.2 — Baseline directory**

```
tools/reference-renders/baseline/
  rfe03-layered-front-back.png
  rfe07-11-sss-mixed-panels.png
  rfe08-13-spectral-payload.png
  rfe14-thinfilm-angle-shift.png
  rfe09-bridge-global-cmf.png
  rfe05-caustic-strategy.png
  ptwgpu-parity-material-fields.png
```

**WG-0.3 — `run-gap-closure.mjs`**

Compare hash; write `gap-closure-verification-<ts>.json` with `passFail` per scenario.

**WG-0.4 — npm script**

Root `package.json`: `"benchmark:gap-closure": "npm run ... --workspace @vitrum/benchmark-runner"`

#### DoD (WG-0)

- [ ] At least `ptwgpu-parity-material-fields` PASS with committed baseline
- [ ] Document env vars in `benchmark-runner/README.md`

---

### WG-1 OIDN-final (Weeks 3–4)

#### As-is

- pt-webgpu: warns on any denoiser (`index.ts:1130-1133`)
- pt-webgpu **has** aux textures: `normalDepth`, `albedo`, `variance`, `motionVectors` on `FrameOutput`
- pt-webgl: `OIDNFinalDispatcher` color-only (no MRT)

#### To-be

pt-webgpu supports `denoiser: 'oidn-final'` with **full aux** (advantage over pt-webgl).

#### Implementation steps

**WG-1.1 — `packages/pt-webgpu/src/denoise/oidnFinal.ts`**

Mirror `pt-webgl/src/oidnFinalDispatcher.ts` but:

1. On `isConverged`, readback:
   - HDR accumulation texture
   - `normalDepth`, `albedo` from aux views (already allocated)
2. Pack tensors for `@vitrum/shared-denoisers` `denoiseFinal()`
3. Store latest denoised RGB in engine field
4. Expose on `FrameOutput` or separate `getDenoisedRgb()` for host

**WG-1.2 — Wire `renderFrame`**

After compute pass, if converged && denoiser==='oidn-final' → `kickIfReady()`.

**WG-1.3 — Capabilities**

```ts
supportedDenoisers: new Set(['none', 'oidn-final']),
```

**WG-1.4 — Tests**

`oidnFinalIntegration.test.ts` — mock `denoiseFinal` spy (pattern from pt-webgl).

#### DoD (WG-1)

- [ ] No console.warn on oidn-final
- [ ] Integration test green
- [ ] Manual: Cornell 64 spp → denoised output non-null

---

### WG-2 Hero λ + CMF MIS (Weeks 5–10)

#### As-is

- Fork: `forkUniformBridge.ts` uploads `uXCmfCdf`, `uYCmfCdf`, `uZCmfCdf`, integrals, hero strategy
- pt-webgpu: `kernel.wgsl.ts` lines 187-191 RGB λ probes for TMM

#### To-be

- WGSL hero sampling from `@vitrum/shared-samplers`
- Single-λ path for TMM + spectral μ

#### Implementation steps

**WG-2.1 — Export WGSL from shared-samplers**

Add `packages/shared-samplers/src/wgsl/heroWavelength.wgsl.ts`:

- `sampleHeroWavelengthMIS(uStrategy, uLambda) -> vec2f` (lambda, pdf)
- CMF CDF textures as **storage buffers** (not textures) to match pt-webgpu style

**WG-2.2 — FrameParams extension**

In `pathTrace/frameParams.wgsl.ts` + CPU writer (`index.ts`):

| Field | Type | Purpose |
|-------|------|---------|
| `heroStrategy` | u32 | 0=XYZ CMF MIS |
| `heroLambda` | f32 | current sample λ |
| `heroPdf` | f32 | MIS pdf |
| `cmfIntegralX/Y/Z` | f32 × 3 | |

**WG-2.3 — Kernel integration**

Replace RGB TMM block with:

```wgsl
let hero = sampleHeroWavelengthMIS(...);
let rt = thinFilmTmmRt(..., hero.x, ...);
```

**WG-2.4 — CPU oracle**

Extend `mcConvergence.test.ts` + `cpuTracer.ts` for hero path.

**WG-2.5 — Capture**

Run `rfe08-13-spectral-payload` → PASS vs baseline after visual review.

#### Tolerance

Document in `renderer-fidelity-matrix.md`: mean RGB Δ < 3% vs pt-webgl @ same SPP.

#### DoD (WG-2)

- [ ] `rfe08` scenario PASS
- [ ] `energyConservation.test.ts` green
- [ ] No regression `wgslContract.test.ts`

---

### WG-3 Cauchy dispersion

**WG-3.1** `materialPacking.ts`: pack Abbe number  
**WG-3.2** `material.wgsl.ts`: `iorCauchy(lambda, B, C)`  
**WG-3.3** Scenario row → experimental (not unsupported)

---

### WG-4 Layered BSDF + transmission MIS

**WG-4.1** Align `bsdf.wgsl.ts` `sampleNextBounceDirection` with fork layered front/back — read `external_requests/03-layered-front-back` + fork shader  
**WG-4.2** Fix `brdfDirectionalPdf` transmission branch (README calls out simplified)  
**WG-4.3** `rfe03-layered-front-back` PASS

---

### WG-5 SSS / translucent

**WG-5.1** Pack translucent flag in material struct (match fork `TRANSLUCENT_BIT`)  
**WG-5.2** Gate SSS in `kernel.wgsl.ts` on flag  
**WG-5.3** `rfe07-11-sss-mixed-panels` PASS

---

### WG-6 Geometry incremental

Depends on §3.1 `scenePack`.

**WG-6.1** `updatePrimitive({ positions })` → refit BLAS range + TLAS without full `setScene`  
**WG-6.2** Tests in `updatePrimitiveIncremental.test.ts`  
**WG-6.3** pt-webgpu uses shared `refitTlasTransforms`

---

### WG-7 BDPT (optional)

6–8 weeks — port `pt-webgl` BDPT uniforms + fork connections WGSL. **Defer** unless user sign-off.

---

### WG-8 createEngine opt-in

**WG-8.1** Extend `EnginePreference`:

```ts
type EnginePreference = 'realtime' | 'quality' | 'quality-webgpu' | 'auto';
```

**WG-8.2** `pickBackend`:

```ts
if (prefer === 'quality-webgpu') return hasWebGPU ? 'pt-webgpu' : 'pt-webgl';
```

**WG-8.3** `createEngine.ts` branch constructs `createPTEngine_WebGPU`.

**WG-8.4** Never add pt-webgpu to `auto`.

---

### WG-9 Extended denoisers (optional)

Wire `shared-denoisers` svgf-real using aux buffers — separate from pt-webgl parity.

---

### WG-10 WebGPU PT signoff

- [ ] WG-D1–D7
- [ ] All gap-closure scenarios PASS
- [ ] `pt-webgpu/README.md` rewrite limitations section
- [ ] Matrix: pt-webgpu deep integration → **strong**

---

## 6. Cross-cutting test matrix

| Layer | Command | When |
|-------|---------|------|
| Mechanical | `npm run verify:mechanical` | Every commit |
| Package | `npm test --workspace @vitrum/<pkg>` | Touching pkg |
| GPU hybrid | `VITRUM_GPU_TEST=1 npm test --workspace @vitrum/walkaround-hybrid -- vitest.gpu.config` | PR-3+ |
| GPU pt-webgpu | `VITRUM_GPU_TEST=1 npm test --workspace @vitrum/pt-webgpu -- vitest.gpu.config` | WG-* |
| Gap closure | `npm run benchmark:gap-closure` | WG-0+ |
| Lifecycle | `VITRUM_LIFECYCLE_SOAK_STRICT=1 npm run hardening:wave4` | PR-6, PR-9 |
| Shader CI | part of verify:mechanical | Always |

---

## 7. Risk register and decision log

| ID | Risk | Mitigation |
|----|------|------------|
| R1 | Storage buffer limit | merged fallback; capability probe |
| R2 | DDGI+TLAS convergence | Option B dual BVH |
| R3 | Emitter list vs local BLAS mismatch | PR-2 keep world emitter build; PR-5 unify |
| R4 | WalkaroundUBO size drift | Layout test pins offsets |
| R5 | Spectral parity | CPU oracle before GPU baseline |
| R6 | Scope creep BDPT | WG-7 explicit defer |

| Decision | Choice |
|----------|--------|
| D1 | Single `scenePack` in shared-bvh |
| D2 | Dummy TLAS buffers in merged mode (one BGL) |
| D3 | TLAS default multi-mesh |
| D4 | OIDN before spectral in WG |
| D5 | pt-webgpu not in auto |

---

## Appendix A — Complete file touch index (both programs)

### Creates

- `shared-bvh/src/scenePack.ts`
- `shared-bvh/src/wgsl/tlasTraversal.wgsl.ts`
- `shared-bvh/src/__tests__/scenePack.test.ts`
- `walkaround-hybrid/__tests__/hybridEngineEmitterUpdate.test.ts`
- `walkaround-hybrid/__tests__/hybridEngineMaterialPatch.test.ts`
- `walkaround-hybrid/__tests__/hybridTlasPrimaryHit.gpu.test.ts`
- `pt-webgpu/src/denoise/oidnFinal.ts`
- `shared-samplers/src/wgsl/heroWavelength.wgsl.ts`
- `tools/benchmark-runner/capturePtWebgpu.mjs`
- `tools/benchmark-runner/run-gap-closure.mjs`

### Major modifies

- `walkaround-hybrid/src/restir/bvhCompute.ts`
- `walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts`
- `walkaround-hybrid/src/HybridEngine.ts`
- `walkaround-hybrid/src/pipeline/bindGroupLayouts.ts`
- `walkaround-hybrid/src/pipeline/bindGroupBuilders.ts`
- `walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts`
- `walkaround-hybrid/src/shaders/common.wgsl.ts`
- `walkaround-hybrid/src/shaders/{ris,risGi,shade,spatial,temporal,restirCastPrimary,surfaceTextures}.wgsl.ts`
- `walkaround-hybrid/src/ddgi/*` (PR-5)
- `pt-webgpu/src/scene/uploadSceneBuffers.ts`
- `pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts`
- `pt-webgpu/src/index.ts`
- `engine/src/createEngineScale.ts`
- `engine/src/createEngine.ts`

---

## Appendix B — PR/WG checklist (execution tracker)

Copy into issue tracker; mark per micro-task.

### PR (42 + optional 8)

PR-0.1 … PR-0.5 · PR-1.1 … PR-1.5 · PR-2.1 … PR-2.6 · PR-3.1 … PR-3.8 · PR-4.1 … PR-4.4 · PR-5.1 … PR-5.6 · PR-6.1 … PR-6.4 · PR-7.* · PR-8.* · PR-D1–D9

### WG (38 + optional)

WG-0.1 … WG-0.4 · WG-1.1 … WG-1.4 · WG-2.1 … WG-2.5 · WG-3.* · WG-4.* · WG-5.* · WG-6.* · WG-7 · WG-8.* · WG-9 · WG-D1–D7

---

## Appendix C — WalkaroundUBO layout (current + PR-3 extension)

**Source of truth today:** `packages/walkaround-hybrid/src/pipeline/uboUpdater.ts` (336 bytes) + `common.wgsl.ts` struct.

| Byte offset | f32 index | Field | Type | Writer |
|-------------|-----------|-------|------|--------|
| 0 | 0–15 | viewMatrix | mat4 | `updateUBO` |
| 64 | 16–31 | projMatrix | mat4 | |
| 128 | 32–47 | prevViewMatrix | mat4 | |
| 192 | 48–50 | cameraPos | vec3 | |
| 204 | 51 | frameSeed | u32 | |
| 208 | 52–53 | screenSize | vec2u | |
| 216 | 54 | emitterCount | u32 | |
| 220 | 55 | totalEmPower | f32 | |
| 224 | 56–58 | sunDirection | vec3 | |
| 236 | 59 | sunIntensity | f32 | |
| 240 | 60–62 | skyTint | vec3 | |
| 252 | 63 | skyIrradiance | f32 | |
| 256–332 | 64–83 | tunables + indirectFireflyClamp | mixed | |

**PR-3 extension (352 bytes total):**

| Byte offset | f32 index | Field | Type | Default |
|-------------|-----------|-------|------|---------|
| 336 | 84 | bvhMode | u32 | 1 (TLAS) |
| 340 | 85 | tlasNodeCount | u32 | from pack |
| 344 | 86 | _tracePad0 | u32 | 0 |
| 348 | 87 | _tracePad1 | u32 | 0 |

**Required code changes:**

1. `uboUpdater.ts`: `WALKAROUND_UBO_SIZE_BYTES = 352`; write indices 84–85.
2. `resourceManager.ts`: duplicate literal `336` → `352` (comment cites uboUpdater).
3. `common.wgsl.ts`: extend struct; bump header comment "Bump documentation if layout changes".
4. **New test** `walkaround-hybrid/__tests__/walkaroundUboLayout.test.ts`:
   - Import WGSL struct via regex or duplicate offset table
   - Assert `352 % 16 === 0`
   - Assert `bvhMode` offset === 336

---

## Appendix D — Scene BVH / TLAS buffer byte contracts (hybrid)

### D.1 Merged mode (today)

| Buffer | WGSL type | Stride | Notes |
|--------|-----------|--------|-------|
| bvhNodes | `array<BVHNode>` | 32 B | world-space tree |
| bvhIndex | `array<vec4u>` | 16 B/tri | `.w` = packed material |
| bvhPositions | `array<vec4f>` | 16 B/vert | `.xyz` world, `.w` UV pack |
| bvhBeerColors | `array<u32>` | 4 B/tri | visible Beer-Lambert color |
| emitters | `array<EmitterTri>` | 80 B | see D.3 |
| emitterCdf | `array<f32>` | 4 B | same length as emitters |

### D.2 TLAS mode (PR-2/3)

| Buffer | Content |
|--------|---------|
| bvhNodes | **Concatenated local BLAS** (same 32 B node layout) |
| bvhPositions | **Local** positions (TLAS transforms ray to local) |
| tlasNodes | `TlasData.nodes` from `buildTlas` |
| tlasInstanceIndices | permutation array |
| tlasBlasRoots | `blasId` per original instance index |
| tlasInstanceWorldToLocal | 4×vec4f per instance (column-major) |
| tlasInstanceLocalToWorld | 4×vec4f per instance |

**Critical invariant:** `bvhIndex.w` decoding in shade must still work — packing runs **after** local positions are finalized, using THREE materials from the mesh that owns the triangle (map global tri index → primitive id via `meshVertexRanges` / tri base offsets).

### D.3 EmitterTri layout (80 bytes)

From `emitterList.ts` header:

| Offset | Field | Size |
|--------|-------|------|
| 0 | vertexA | 12 |
| 12 | vertexB | 12 |
| 24 | vertexC | 12 |
| 36 | normal | 12 |
| 48 | area | 4 |
| 52 | color | 12 |
| 64 | intensity | 4 |
| 68–79 | pad | 12 |

**PR-0.2 emitter rebuild** must preserve this layout when rebuilding only emitters.

---

## Appendix E — Storage buffer inventory (PR-3 risk R1)

`WalkaroundGPUPipeline` requests `maxStorageBuffersPerShaderStage: 16` at device init (`WalkaroundGPUPipeline.ts` ~line 128).

**Per-pass inventory (must complete in PR-3.3 before merging):**

| Pass | @group | Storage buffers (count) | Notes |
|------|--------|---------------------------|-------|
| ris | 0 frame | bindings 5,6,7,11 | reservoirs |
| ris | 1 scene | **11 after PR-3** (was 6) | |
| ris | 2 ubo | 1 uniform (not storage) | |
| shade | 0 frame | + hdr outs | worst case |
| shade | 1 scene | 11 | |
| ddgi probe | separate pipeline | uses V3 buffers today | PR-5 |

**Action item PR-3.3a:** Add script or test `__tests__/storageBufferCount.test.ts` that parses `bindGroupLayouts.ts` + each pass's `@group` annotations and asserts ≤ `device.limits.maxStorageBuffersPerShaderStage` on default adapter.

**Fallback:** If shade exceeds limit on SwiftShader (10): split scene BGL into `@group(1)` BLAS (0–5) + `@group(4)` TLAS (6–10) — requires pipeline layout change across all passes (higher cost). Prefer dummy-buffer single layout first.

---

## Appendix F — PR-3 shader migration checklist (every call site)

For each file, replace and verify `skipGlass` preserved on Any-hit:

| # | File | Function context | Old call | `skipGlass` |
|---|------|------------------|----------|-------------|
| F1 | `ris.wgsl.ts` | primary | `bvhIntersectFirstHit` | n/a |
| F2 | `ris.wgsl.ts` | shadow | `bvhIntersectAny(..., true)` | **true** |
| F3 | `risGi.wgsl.ts` | primary | firstHit | n/a |
| F4 | `risGi.wgsl.ts` | bounce | firstHit | n/a |
| F5 | `risGi.wgsl.ts` | occluder | any | **true** |
| F6 | `shade.wgsl.ts` | NEE occ | any | **true** |
| F7 | `shade.wgsl.ts` | primary | firstHit | n/a |
| F8 | `spatial.wgsl.ts` | (grep) | any/firstHit | per callsite |
| F9 | `temporal.wgsl.ts` | (grep) | any/firstHit | per callsite |
| F10 | `restirCastPrimary.wgsl.ts` | cast | firstHit | n/a |
| F11 | `surfaceTextures.wgsl.ts` | visibility | any | **true** |

**Wrapper signature (common.wgsl.ts):**

```wgsl
fn traceSceneClosest(hit: ptr<function, IntersectionResult>, ray: Ray, tMin: f32, triEps: f32) -> bool {
  if (ubo.bvhMode == 0u) {
    *hit = bvhIntersectFirstHit(&bvh_index, &bvh_position, &bvh, ray, triEps);
    return hit.didHit;
  }
  return traceTlasClosest(ray, tMin, 1e20, hit); // uses module bindings
}
```

**Regression strings:** Update `sprint16-restirGiRis.test.ts` only if it pins removed symbol names; prefer pinning `traceSceneClosest` exists.

---

## Appendix G — WG-2 fork → pt-webgpu uniform / buffer mapping

| Fork uniform (`forkUniformBridge.ts`) | pt-webgpu destination | Phase |
|---------------------------------------|----------------------|-------|
| `uSpectralRendering` | `FrameParams.spectralEnabled` u32 | WG-2 |
| `uXCmfCdf`, `uYCmfCdf`, `uZCmfCdf` | storage buffers `cmfCdfX/Y/Z` | WG-2 |
| `uXCmfIntegral`, etc. | `FrameParams` f32×3 | WG-2 |
| Hero strategy uniform | `FrameParams.heroStrategy` | WG-2 |
| `vitrumSpectralAttenuation` | already in `materialPacking` 32-bin | verify WG-2.4 |
| `vitrumThinFilmStack` | 8 layers in material vec4s | WG-2.3 per-λ TMM |
| `vitrumDispersionAbbeNumber` | **missing** | WG-3 |
| `vitrumFrontLayer` / `vitrumBackLayer` | layered fields in pack | WG-4 |
| `TRANSLUCENT_BIT` | **missing** | WG-5 |
| BDPT textures | **missing** | WG-7 |

**WG-2 work order inside kernel:**

1. Sample hero λ + pdf at path start (or per bounce — match fork).
2. Use λ for `thinFilmTmmRt` and `sampleMaterialSpectralMu`.
3. Accumulate radiance with CMF weight = MIS pdf denominator.

**CPU pack verification:** Extend `scenePack.materials.test.ts` to assert spectral bins non-zero when `Material.extensions.spectral` present.

---

## Appendix H — PR-0.2 emitter-only rebuild algorithm

**New function:** `rebuildEmitterBuffersFromThreeRoots(roots, options): EmitterBuffers`

```
1. mergedGeo = extractMergedGeometryForEmitters(roots)
   // Option A: lightweight traverse — collect all Mesh triangles world positions
   // Option B: reuse existing mergedGeometry from _bvhBuffers if topology unchanged
2. (emitters, cdf, count, power) = buildEmitterList(mergedGeo, primaryLightDir, intensity)
3. return { emitters: cpuData, emitterCdf, emitterCount, totalEmissivePower }
```

**HybridEngine.updateEmitter after patch:**

```
_lastScene.emitters[idx] = merged
buffers = rebuildEmitterBuffersFromThreeRoots(_ensureThreeSceneRoot(), { dir, intensity })
_bvhBuffers = { ..._bvhBuffers, ...buffers }  // swap emitter handles only
_pipeline.updateEmitters(buffers)
_pipeline.requestAccumReset()
// Do NOT call ddgi.invalidateProbeCache unless emissive panel / transmission classification changes
```

**Edge case:** emitter patch changes mesh emissive material → may need to rebuild emitter list from THREE material luminance — `buildEmitterList` already reads materials.

**Edge case:** patch changes `kind` → throw, force `setScene`.

---

## Appendix I — Gap-closure scenario implementation notes

| Scenario | Scene fixture location | pt-webgpu focus | Baseline capture SPP |
|----------|------------------------|-----------------|----------------------|
| `rfe03-layered-front-back` | fork test scene / build in runner | front/back asymmetry | 512 |
| `rfe07-11-sss-mixed-panels` | mixed translucent flags | WG-5 | 512 |
| `rfe08-13-spectral-payload` | spectral materials | WG-2 | 1024 |
| `rfe14-thinfilm-angle-shift` | 35-layer stack | WG-2.3 | 1024 |
| `rfe09-bridge-global-cmf` | CMF tables | WG-2 | 256 |
| `rfe05-caustic-strategy` | caustic modes | existing WGSL modes | 1024 |
| `ptwgpu-parity-material-fields` | packed field torture test | materialPacking | 512 |

**Each scenario JSON artifact must include:** `scenarioId`, `seed`, `resolution`, `bounces`, `spp`, `beforeImageHash`, `afterImageHash`, `deltaSummary`, `perfBaselineMsPerSample`, `perfCandidateMsPerSample`, `passFail` (per `gap-closure-acceptance-matrix.md`).

---

## Appendix J — `materialPatch` tri-range resolution (PR-1 detail)

**Problem:** `meshVertexRanges` is per-vertex, not per-triangle.

**Algorithm:**

```
range = meshVertexRanges.find(name === id)
if (!range) throw
// Triangle count from bvhIndex.count (vec4u per tri)
triStart = inferTriStartFromVertexStart(range.vertexStart)  
// Implementation: store triStart in meshVertexRanges at build time (PR-1.0)
// OR scan bvhIndex for first tri whose indices all >= vertexStart (slow — avoid)
```

**PR-1.0 prerequisite:** Extend `meshVertexRanges` in `bvhCompute.ts`:

```ts
readonly triStart: number;
readonly triCount: number;
```

Populate during `buildReSTIRSceneBVH` from merged geometry index buffer.

**materialPatch loop:**

```ts
for (let t = range.triStart; t < range.triStart + range.triCount; t++) {
  packBVHIndexWTri(t, mat, ...);
  packBVHBeerTri(t, mat, ...);
}
```

---

## Appendix K — Document map

| Document | Role |
|----------|------|
| `plan/primary-release-and-webgpu-pt-parity-implementation-deep.md` | **This file** — execution spec |
| `plan/primary-release-and-webgpu-pt-parity-2026-05-26.md` | Executive summary + checklist |
| `plan/backend-maturity-matrix-2026-05-26.md` | Signoff targets |
| `plan/archive/gap-closure-acceptance-matrix.md` | Scenario IDs |
| `plan/renderer-fidelity-matrix.md` | Feature rows |

---

*End of deep specification.*
