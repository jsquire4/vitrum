# Sweep Remediation Plan — Pass 2

**Source:** Phase 11 complexity sweep, 2026-05-11
**Verification:** all findings cross-checked against current code
**Total verified findings:** 19 (1 regression-class, others pre-existing or stale comment)

The order below is dependency-driven: regression hazards first, then dead-code removal, then refactors / docs / style.

---

## Pass-2 Phase 1 — Regression hazards

### P2-1.1 Delete dead `_bvhNormalBuffer` / `_bvhUvBuffer` allocations

**Files:**
- `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts:179,180,277,278,796,797`
- `packages/walkaround-hybrid/src/restir/bvhCompute.ts:72,74,234-239`

**Bug:** Both buffers are allocated with real vertex data + destroyed in dispose but never bound to any compute pass. The data is already encoded into other buffers (UV into `bvh_position.w`, normals computed in-shader).

**Fix:** Remove the two fields, the two `uploadBuffer` calls, and the two `?.destroy()` calls. Remove `bvhNormals` and `bvhUvs` from the `SceneBVHBuffers` interface in restir/bvhCompute.ts (and the construction sites at 234-239). Cross-check `packages/walkaround-hybrid/src/restir/emitterList.ts` and `packingHelpers.ts` for any cpuData reads — these are CPU-side and may stay if used.

---

## Pass-2 Phase 2 — Dead code / unused exports

### P2-2.1 Delete `PT_BOUNCES` deprecated alias

**File:** `packages/pt-webgl/src/constants.ts:27-28`

No in-repo readers. Backwards compatibility is moot pre-publish.

**Fix:** delete the `PT_BOUNCES` export.

### P2-2.2 Migrate `isHardwareGpu` reader and drop the deprecation

**Files:**
- `packages/core/src/wgpuSupport.ts:15-22` — `WgpuProbeResult.isHardwareGpu`
- `packages/core/src/gpuDetection.ts:34-39` — `GpuDetection.isHardwareGpu`
- `packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts:137` — sole remaining reader

**Fix:**
1. Update probeUpdatePass.ts:137 to `if (gpu.isWebGPU && gpu.adapterKind === 'swiftshader')`.
2. Remove `isHardwareGpu` from both interfaces and from the implementation in `wgpuSupport.ts` (lines ~141–152 set the field).
3. Remove from JSDocs.

### P2-2.3 Resolve `cellPower` orphan buffer

**File:** `packages/walkaround-hybrid/src/restir/bvhCompute.ts:264-267`

The `cellPower` buffer is built every BVH rebuild but no consumer reads it.

**Fix (choose one):**
- Option A (remove): drop the field from `SceneBVHBuffers`, the build site in `emitterList.ts`, and the population in bvhCompute.ts. Reclaim the CPU memory.
- Option B (defer-mark): keep but rename to `_deferredCellPower` and add a single-source-of-truth comment pointing at the eventual consumer.

Lean Option A — Sprint 9/10 integration will rebuild this purposefully.

---

## Pass-2 Phase 3 — Stale comments

### P2-3.1 Fix `ProbeGrid.buildUniformData()` "32 bytes" docstring

**File:** `packages/walkaround-hybrid/src/ddgi/probeGrid.ts:43-55`

Doc claims 32 bytes; actual is 64 bytes (Float32Array(16)).

**Fix:** Update docstring to say "64 bytes" and list slots 12..15 as zero-padding.

### P2-3.2 Document the byte-aliasing in `_uploadFrameParams`

**File:** `packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts:451-462`

The function aliases a `Float32Array` with a `Uint32Array` view over the same backing buffer. The randomRotation/frameIndex/probeCount layout is correct but the aliasing isn't documented.

**Fix:** Add a slot-by-slot comment showing offsets 0..3 (rotation xyz + frameIndex), 4 (probeCount), 5 (probesPerFrame).

### P2-3.3 Rename `uniformBuf` → `cascadeParamsBuf` in `cascadeDispatch.ts`

**File:** `packages/walkaround-hybrid/src/rc/cascadeDispatch.ts:429-444,489-498`

Field name + comment say "uniform buffer" but the buffer is created with `GPUBufferUsage.STORAGE` and bound as `'read-only-storage'`. The size (160 bytes) exceeds the default `maxUniformBufferBindingSize` on low-end adapters — that's the real reason it's a storage buffer.

**Fix:** Rename `uniformBuf` → `cascadeParamsBuf`, `uniformRaw` → `cascadeParamsRaw` for both cast and merge passes. Update the comment to explain the storage-vs-uniform choice.

### P2-3.4 Trim `HybridEngine.ts` JSDoc planning prose

**File:** `packages/walkaround-hybrid/src/HybridEngine.ts:14-19`

A paragraph of plan-document framing inside the class JSDoc.

**Fix:** Reduce to one line referencing `plan/walkaround-without-three.md`.

### P2-3.5 Resolve `three-bindings/environment.ts` ProceduralSkyEnvironment TODO

**File:** `packages/three-bindings/src/environment.ts:22-26`

TODO is honest but there's no host that uses THREE.Sky yet.

**Fix:** Convert the TODO into a docstring on `resolveEnvironment` describing the asymmetry (vitrumSceneToThree handles the procedural-sky kind with a warning; the reverse direction does not). Move to a one-line note at the function body.

---

## Pass-2 Phase 4 — Refactors / extractions

### P2-4.1 Single canonical DDGI grid-params packer

**Files:**
- `packages/walkaround-hybrid/src/ddgi/probeGrid.ts:43-67` (`buildUniformData`)
- `packages/walkaround-hybrid/src/pipeline/resourceManager.ts:287-330` (`packDDGIGridParams`)

Two implementations of the same 64-byte layout.

**Fix:** Delete `ProbeGrid.buildUniformData()`. Update its sole call site (`probeUpdatePass._uploadGridParams` line 447) to call `packDDGIGridParams(this._grid.params)` from resourceManager. The packer already accepts the same shape ProbeGrid exposes.

### P2-4.2 Collapse the 7× UboRef duplication in WalkaroundGPUPipeline

**File:** `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts:225-238,327-338,801-807`

Seven near-identical `{ buf: GPUBuffer | undefined }` fields, all allocated in `initialize()` and destroyed in `dispose()`.

**Fix:** Introduce `UboBundle = Record<string, GPUBuffer>` keyed by name. `initialize()` runs a schedule (name → size → mode-gate). `dispose()` iterates. Cuts ~30 lines.

### P2-4.3 Extract `classifyTriangleEmitter` helper in `emitterList.ts`

**File:** `packages/walkaround-hybrid/src/restir/emitterList.ts:109-145`

The inline classification block does 3 things in one nested if.

**Fix:** Extract `classifyTriangleEmitter(mat, lightDir, intensity): { color, intensity } | null`. Outer loop becomes "for each triangle, classify → if not null, accumulate."

### P2-4.4 Split `uploadSceneBuffers.ts` god file

**File:** `packages/pt-webgpu/src/scene/uploadSceneBuffers.ts` (1058 lines, 8 concerns)

**Fix (multi-step):**
1. Extract material packing (lines 181-289) → `materialPacking.ts`.
2. Extract environment params (lines 528-665) → `environmentPacking.ts`.
3. Extract the four `firstXLight` resolvers + the four `packX` array loops into a single `extractLights(scene)` in `emitterPacking.ts` that returns `{ singletonSlots, arrays }`. Collapses 4× near-identical loops into one generic.
4. Keep `buildPackedScene` and `uploadPackedScene` in the original file as thin orchestrators.

### P2-4.5 Generalize `packEmitterArray<K>` helper

**File:** `packages/pt-webgpu/src/scene/uploadSceneBuffers.ts:679-812`

Four near-identical emitter array loops.

**Fix:** `packEmitterArray<K extends SceneEmitter['kind']>(scene, kind, max, stride, writer): { count, data, warnings }`. Each kind supplies its writer. ~80 line reduction. (Subsumed by P2-4.4 step 3.)

### P2-4.6 Replace `passIdx: number` with named pass IDs in WalkaroundGPUPipeline

**Files:**
- `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts` (multiple call sites of `computeDesc`)
- `packages/walkaround-hybrid/src/pipeline/timestampQueries.ts` (consumer)

The `denoiseBase + 2 + iter` arithmetic is brittle.

**Fix:** Replace `passIdx: number` with `passId: string` literal union, let `timestampQueries.ts` map names → slots. Eliminates the offset math.

---

## Pass-2 Phase 5 — Style / naming

### P2-5.1 Move `WebGLPathTracerCompat` interface to types section in pt-webgl

**File:** `packages/pt-webgl/src/index.ts:22-36`

Declared mid-file between imports. Should be near other interface declarations.

**Fix:** Move alongside `PTEngineWebGL2Init` (line 317).

### P2-5.2 Move `HybridEngine._fingerprintRebuildKey` static helper to end of class

**File:** `packages/walkaround-hybrid/src/HybridEngine.ts:172-179`

Private static placed at top of class body.

**Fix:** Move to end of class. Style only.

---

## Pass-2 Phase 6 — Cross-engine material packing (large)

### P2-6.1 Extract shared `getMaterialPacked` helper

**Files (current duplications):**
- `packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts:347-388` (DDGIMaterial pack)
- `packages/walkaround-hybrid/src/restir/packingHelpers.ts:79-92` (resolveTriColor)
- `packages/pt-webgpu/src/scene/uploadSceneBuffers.ts:196-271` (materialToPackedVec4s)

Each engine re-reads `THREE.MeshPhysicalMaterial` fields with slightly different fallback values.

**Fix:** Define `extractCommonMaterialFields(mat): { baseColor, emissive, roughness, metallic, transmission, ior, attenuationColor, thickness, scattering, ... }` in `@vitrum/three-bindings`. Each engine's packer consumes this canonical struct then packs into its WGSL-specific layout. Reduces drift hazard.

**Risk:** Touching three engines + DDGI material upload format simultaneously. Substantial validation work. **Optional** — flag as deferred to a future remediation pass if time-bounded.

---

## Pass-2 Phase 7 — Coupling cleanups

### P2-7.1 Replace `window.__WGPU__` debug mutations with engine.debug getter

**Files:** `HybridEngine.ts:513-522`, `DDGI.updateFrame:199-205`, `probeUpdatePass.runFrame:316-322`

Library code writes to host-owned globals.

**Fix:** Expose `engine.debug: { lastFrameMs: number; framesDispatched: number; ... }` getter. Host owns the global publication. Library stops crossing the boundary.

### P2-7.2 Drop the last `three/webgpu` import in walkaround DDGI

**File:** `packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts:20-21,93-94`

Only `StorageTexture` type import remains. Used as a WeakMap cache key + a way to read .image.width/height.

**Fix:** Replace `StorageTexture` with a structural interface `{ image: { width: number; height: number } }`. Keys the cache on the same shape without the package dep. Also touches `probeGrid.allocateAtlases()` lines 117-142.

**Risk:** Need to verify that the ProbeGrid's atlas-texture creation in `allocateAtlases` doesn't need the actual `StorageTexture` constructor at runtime. If it does, this is a partial fix.

### P2-7.3 Disposable singleton: `_sharedDispatcher` in cascadeDispatch.ts

**File:** `packages/walkaround-hybrid/src/rc/cascadeDispatch.ts:528-537`

Functional API holds a module singleton with no public dispose.

**Fix:** Export `disposeSharedDispatcher()` for host teardown. Document on the functional API's JSDoc.
