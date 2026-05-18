# Vitrum Complexity Sweep — Pass 2 Raw Findings

**Date:** 2026-05-11
**Scope:** Post-remediation second pass on ~35K lines across 8 packages, 3 examples, 3 tools.
**Skipped:** `node_modules/`, `_staging/legacy-source/`, `plan/`, `dist/`, the four documented Phase-4 deferred items.

## Pre-flight signals

- TODOs/FIXMEs across non-test code: only **4** total. All four are reviewed below — three are linked to documented deferred work, one is stale/orphan.
- Deprecation markers: **3** (`@deprecated isHardwareGpu` on `GpuDetection`, `@deprecated isHardwareGpu` on `WgpuProbeResult`, `@deprecated PT_BOUNCES` alias). Each names a removal target (Phase 7 / Sprint 1) but that sprint has not been opened yet, so the deprecations have been carried for ≥1 sprint without action.
- Files over 600 lines:
  - `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts` — 1664 lines (mostly inherent for path-tracer kernel; specific issues below)
  - `packages/pt-webgpu/src/scene/uploadSceneBuffers.ts` — 1058 lines (mixed concerns + repeated patterns; see findings)
  - `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts` — 846 lines (slightly above threshold, see specific findings)
  - `packages/walkaround-hybrid/src/HybridEngine.ts` — 846 lines
  - `packages/pt-webgl/src/index.ts` — 830 lines (the engine + 5 large helper functions; see findings)
  - `packages/walkaround-hybrid/src/shaders/common.wgsl.ts` — 691 lines (inherent: WGSL helpers)
  - `packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts` — 650 lines (god-file risk; see findings)
  - `packages/walkaround-hybrid/src/pipeline/resourceManager.ts` — 600 lines (acceptable, single responsibility)

---

## 1. Correctness / runtime hazards

### FINDING: packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts:177-180,277-278,796-797

**CATEGORY:** Dead Code (GPU allocation)
**WHAT:** `_bvhNormalBuffer` and `_bvhUvBuffer` are private fields, uploaded with real vertex data in `initialize()`, and destroyed in `dispose()`, but **never bound to any compute pass**. They consume GPU memory (~`vertexCount × 16` bytes each at stride-4) for every frame the pipeline is alive, for nothing.
**EVIDENCE:**

```
private _bvhNormalBuffer!: GPUBuffer;            // line 179
private _bvhUvBuffer!: GPUBuffer;                // line 180
...
this._bvhNormalBuffer = uploadBuffer(d, bvhBuffers.bvhNormals.cpuData, GPUBufferUsage.STORAGE);  // line 277
this._bvhUvBuffer     = uploadBuffer(d, bvhBuffers.bvhUvs.cpuData,     GPUBufferUsage.STORAGE);  // line 278
...
this._bvhNormalBuffer?.destroy();                // line 796
this._bvhUvBuffer?.destroy();                    // line 797
```

A `grep -rn` over the package finds zero bind-group references to either buffer. Vertex normals are encoded into `bvhIndex.w` (or read indirectly via face normals reconstructed in WGSL); UVs are packed into `bvh_position.w` as documented in `bvhCompute.ts` lines 169-177.
**FIX:** Delete both fields, the two `uploadBuffer` calls, and the two `?.destroy()` calls. Also remove `bvhNormals` / `bvhUvs` from `SceneBVHBuffers` if no other consumer reads them on the CPU side — `restir/bvhCompute.ts` lines 234-239 still build them, so check there too. Tag: this regression-introduced waste survived the first sweep.

### FINDING: packages/walkaround-hybrid/src/ddgi/probeGrid.ts:158-183

**CATEGORY:** Stale Comment + Coupling
**WHAT:** The docstring above `buildUniformData()` claims `Layout (std140 compatible, 32 bytes)` and lists only ~24 bytes of fields, but the function allocates `new Float32Array(16)` = 64 bytes and writes fields up to `buf[11]`. Slots `buf[12..15]` remain implicitly zero. The matching WGSL `ProbeGridParams` struct in `probeUpdateRays.wgsl.ts` lines 110-118 is 64 bytes wide (origin/spacing 16 + dims/pad 16 + atlasDims 16 + padding 16). Future reader trusting the comment would think the buffer is half its actual size.
**EVIDENCE:**

```
/**
 * Build the raw Float32Array for the ProbeGridParams uniform.
 * Layout (std140 compatible, 32 bytes):  ← WRONG
 *   vec3f origin (12 bytes + 4 pad = 16)
 *   f32  spacing (4)
 *   vec3u dims   (12 + 4 pad = 16 — stored as 3 floats)
 *   f32  _pad
 *   f32  irrW, irrH, visW, visH
 */
buildUniformData(): Float32Array {
  ...
  const buf = new Float32Array(16);  // 64 bytes, not 32
```

**FIX:** Update docstring to "64 bytes" and list the 4 trailing zero-padded slots `buf[12..15]` so the WGSL-side ProbeGridParams struct alignment is documented. Also worth noting that the canonical packer `packDDGIGridParams` in `pipeline/resourceManager.ts:305-330` writes the exact same 64-byte layout — these two helpers should share a single packer (see Finding under "Refactor candidates" below).

### FINDING: packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts:451-462

**CATEGORY:** Stale Comment
**WHAT:** `_uploadFrameParams` comment says `randomRotation: random vec3f for per-frame probe direction rotation` but writes 3 independent `Math.random() * 2π` into `data[0..2]`, then bumps slots `u32[3..5]` with frameIndex / probeCount / probeCount/4. The data slot 3 is overwritten by `u32[3] = frameIndex` (sharing the buffer), but the comment makes the layout ambiguous to future readers.
**EVIDENCE:**

```
private _uploadFrameParams(device: GPUDevice): void {
  const data = new Float32Array(8);
  const u32 = new Uint32Array(data.buffer);
  // randomRotation: random vec3f for per-frame probe direction rotation.
  data[0] = Math.random() * Math.PI * 2;
  data[1] = Math.random() * Math.PI * 2;
  data[2] = Math.random() * Math.PI * 2;
  u32[3] = this._frameIndex;    // overwrites data[3]
  u32[4] = this._grid.probeCount;
  u32[5] = Math.ceil(this._grid.probeCount / 4);
```

The WGSL `FrameParams` struct (probeUpdateRays.wgsl.ts:121-127) confirms `randomRotation: vec3f` (offsets 0,4,8) followed by `frameIndex: u32` (offset 12) — so the TS code is correct, but the byte aliasing of `data` and `u32` views into the same buffer is not documented in the comment, only in the field names.
**FIX:** Add `// 8 floats / 32 bytes; aliased u32 view shares the storage:` to the docstring. Inline-document each slot offset like the `_uploadGridParams` peer or like the params buffer in `pt-webgpu/src/index.ts:155-274`.

### FINDING: packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:199-213,262-275

**CATEGORY:** Leaky Abstraction
**WHAT:** HDRI environment-map dimensions are packed into `params.meshAreaTriB.w` and `params.meshAreaTriC.w` (slot 71 / 75 in the f32 view of the params buffer) — a deliberate space-saving trick documented in a long comment. This creates an invariant that any future packer touching the params buffer must preserve. The packer side (index.ts:257,262) is correctly commented but the convention is documented in two places and easy to break.
**EVIDENCE:**

```
// pathTraceBruteforce.wgsl.ts:199-213
// NOTE: HDRI environment dimensions are packed into the .w lanes of the
// meshAreaTriB / meshAreaTriC vec4 slots in the params UBO ...
// Any host-side packer that touches the params buffer MUST preserve this
// convention. Future cleanup: lift to a dedicated params.environmentDims
// (vec2u) field once the params struct is repacked.
fn environmentDimensions() -> vec2u {
  return vec2u(u32(params.meshAreaTriB.w), u32(params.meshAreaTriC.w));
}
```

The WGSL exposes a single function `environmentDimensions()` but the TS packer at index.ts:257 / 262 uses bare numeric indices. Comment already calls this out as "Future cleanup."
**FIX:** Either (a) defer until the params UBO is repacked at the next layout bump (low risk today), or (b) extract the slot indices `MESH_AREA_TRI_B_W_ENV_WIDTH = 71` etc. into named constants in `pt-webgpu/src/scene/uploadSceneBuffers.ts` so the next person who edits the packer can't silently break the convention.

### FINDING: packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:992-994

**CATEGORY:** Leaky Abstraction
**WHAT:** Caustic mode is decoded from `params.spotLightRadiance.w` rather than a dedicated field. Like the env-dims slot, this hijacks a "free" w-lane in a vec4 used for something else. The packer side (index.ts:216-221) writes the strategy code to `paramsF32[39]`, which is the `.w` of the spot-light-radiance vec4 starting at index 36.
**EVIDENCE:**

```
fn causticMode() -> u32 {
  return u32(max(params.spotLightRadiance.w, 0.0));
}
```

The host code:

```
paramsF32[36..38] = sb.spotLightRadiance.xyz;
paramsF32[39] =
  this.#causticStrategy === 'manifold-nee'
    ? 1
    : this.#causticStrategy === 'photon-map'
      ? 2
      : 0;
```

No code-level comment links the two. The struct member name `spotLightRadiance` carries no hint that `.w` holds the caustic strategy.
**FIX:** Same as previous finding — either add a dedicated `params.causticStrategy: u32` (likely the cleanest), or stamp the convention in both the WGSL `FrameParams` struct comment and the TS `#buildParamsBuffer` slot-71 comment.

### FINDING: packages/walkaround-hybrid/src/HybridEngine.ts:708-806

**CATEGORY:** Deep Nesting + Error Handling
**WHAT:** `_initPipeline` runs an async IIFE inside its outer function, then inside that runs a `while` loop polling scene readiness with a 50ms `setTimeout`, then a try/catch builds BVH + pipeline. This is 3+ levels of nesting with cancellation logic interleaved (`if (this._disposed) return;` checks at multiple points). The pattern works but is the densest part of the file.
**EVIDENCE:** Lines 708-806 form a 100-line IIFE with three early-return cancellation checks and the actual init code. A reader has to mentally track `_disposed`, `_pollIters`, `_state`, and the eventual `_pipeline` assignment.
**FIX:** Extract the inner IIFE body into a private `async _runInitPipelineAsync()` method. Reduces nesting depth by one level and makes the IIFE → method call boundary the natural place to consolidate cancellation. Inherent complexity of the "fire-and-forget async init with cancel checkpoints" pattern, but the structure can be flatter.

### FINDING: packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts:347-388

**CATEGORY:** Deep Nesting + Stale Comment
**WHAT:** `_uploadMaterials` has 3-level nesting (matsToUse.forEach → conditional emissive/attenuation reads → u32 vs float writes via aliased views). The function is 42 lines and the std140 layout comment is correct, but the `instanceof THREE.MeshPhysicalMaterial` cast is inline rather than via the same `isPhysical()` helper used in three-bindings/src/material.ts.
**EVIDENCE:** Lines 347-388 mix the std140 layout doc with the float vs u32 write protocol (the `u32view[base + 15] = isGlass ? 1 : 0;` is critical and easy to miss). Duplicated logic across packages: the same `m.transmission > 0` → glass-flag decision is also encoded in `restir/packingHelpers.ts:79-92` and `pt-webgpu/scene/uploadSceneBuffers.ts:196-271`.
**FIX:** Extract a shared `getMaterialPacked` helper into `@vitrum/three-bindings` that resolves baseColor / emissive / transmission / ior / scattering RGB into a stable struct, then have each engine's packer consume it. Today each engine reinvents the same THREE.MeshPhysicalMaterial reads with slightly different fallback values. Material packing is one of the highest-duplication concentrations in the codebase.

---

## 2. Coupling / boundary issues

### FINDING: packages/walkaround-hybrid/src/HybridEngine.ts:513-522

**CATEGORY:** Coupling (window global)
**WHAT:** When `debug` is set, HybridEngine writes per-frame timing into `window.__WGPU__.walkaround.frameTimings`. The host (not the engine) owns this global; the engine reaches in to mutate it. This is the same pattern flagged in `_uploadLights` and `DDGI` — debug instrumentation crosses the library/host boundary.
**EVIDENCE:**

```
if (this._debug && typeof window !== 'undefined') {
  const w = window as unknown as { __WGPU__?: { walkaround?: { frameTimings: unknown } } };
  if (w.__WGPU__?.walkaround) {
    const ft = w.__WGPU__.walkaround.frameTimings as Array<{ t: number; ms: number }>;
    if (Array.isArray(ft)) {
      ft.push({ t: now, ms: dt });
      if (ft.length > 240) ft.shift();
    }
  }
}
```

Library code is reading and pushing to a host-owned array. The cast pyramid `as unknown as { __WGPU__?: ... }` is the smell. The pattern is consistent with `DDGI.updateFrame` lines 199-205 and `probeUpdatePass.runFrame` lines 316-322, all gated on `_debug`.
**FIX:** Engine should expose a `debug?: { framesDispatched: number; lastFrameMs: number; ... }` getter the host can poll; let the host write its own globals. Today the host is forced to be `__WGPU__`-shaped to receive the data.

### FINDING: packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts:20-21,93-94

**CATEGORY:** Coupling (three/webgpu peer dep)
**WHAT:** The class imports `StorageTexture` from `three/webgpu`. The class JSDoc calls this out as `RISK-3: three.js WebGPU coupling` (line 16-18). Per the project's CLAUDE.md and library architecture, the goal is for walkaround-hybrid to consume only `@vitrum/core` + raw WebGPU. This is the one remaining `three/webgpu` import on the DDGI path; the rest of the package now uses raw GPU buffers.
**EVIDENCE:**

```
import type { StorageTexture } from 'three/webgpu';
...
private _textureCache = new WeakMap<StorageTexture, GPUTexture>();
```

The `_getOrCreateAtlasTexture` helper (lines 475-496) creates a real `GPUTexture` from `StorageTexture.image.width/height`. The `StorageTexture` is only used as a cache key + a way to read/store dimensions.
**FIX:** Drop the `StorageTexture` type. The probeGrid could own the dimensions directly as numbers (it already has `irradianceAtlasW/H`). The cache could key on the grid's read/write phase index (0/1 of the ping-pong) rather than the StorageTexture object identity. The StorageTexture allocation in `probeGrid.allocateAtlases()` lines 117-142 (which also imports from `three/webgpu`) would also be removed in the same change.

### FINDING: packages/walkaround-hybrid/src/rc/cascadeDispatch.ts:429-444

**CATEGORY:** Stale Comment
**WHAT:** The "uniformBuf" name and comment in the cast-pass setup describe a "uniform buffer for CascadeUniforms" but the buffer is allocated with `GPUBufferUsage.STORAGE` and the bind-group-layout entry binds it as `'read-only-storage'` (line 305). The buffer is correctly used as a read-only storage buffer, but the field name `uniformBuf` and the comment `Per-pass uniform buffer for CascadeUniforms` are misleading.
**EVIDENCE:**

```
// Per-pass uniform buffer for CascadeUniforms (40 floats = 160 bytes).
...
const uniformBuf = device.createBuffer({
  label:  `rc-cast-C${k}-uniforms`,
  size:   uniformRaw.byteLength,
  usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,    // ← STORAGE, not UNIFORM
  ...
```

And BGL line 305:

```
{ binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // CascadeUniforms
```

**FIX:** Rename `uniformBuf` / `uniformRaw` to `cascadeParamsBuf` / `cascadeParamsRaw` and update the comment to say "read-only-storage buffer for CascadeParams." Same goes for the merge-pass equivalents (lines 489-498). 160 bytes > 64 = max-uniform-binding-size-on-low-end-adapters might be why it's storage to begin with; if so, that should be the comment.

### FINDING: packages/walkaround-hybrid/src/rc/cascadeDispatch.ts:528-537

**CATEGORY:** Module-Level Singleton
**WHAT:** `_sharedDispatcher` is a module-level `RCDispatcher` instance used by the functional `dispatchCascadePasses` wrapper. The JSDoc says "Single-canvas-scoped by design" and warns multi-canvas hosts to instantiate the class directly. This works but the singleton lives forever once the module is loaded — there is no way for the host to dispose it from the functional API.
**EVIDENCE:**

```
// Single-canvas-scoped by design: this functional API serves the legacy /
// single-host call sites where there is exactly one RC canvas per page.
const _sharedDispatcher = new RCDispatcher();
export async function dispatchCascadePasses(opts: RCDispatchOpts): Promise<void> {
  return _sharedDispatcher.dispatchFrame(opts);
}
```

Memory cost of an unused `RCDispatcher` is low (no GPU resources until `dispatchFrame` is called), but on hot module reload in dev, the shared dispatcher may hold stale shader modules.
**FIX:** Either deprecate `dispatchCascadePasses` (the comment already nudges callers to the class) or add a `disposeSharedDispatcher()` export so hosts can release the singleton on teardown.

### FINDING: packages/pt-webgl/src/iblBaker.ts:53,71-80

**CATEGORY:** Module-Level Singleton (cross-context risk)
**WHAT:** The IBL bake cache is a module-level `Map<string, CachedBake>` shared across all `WebGLRenderer` instances in the process. A baked DataTexture is GL-context-specific (carries a `WebGLTexture` handle once uploaded). The comment correctly warns about this, but the safety guard is "hosts always have exactly one renderer."
**EVIDENCE:**

```
// WARNING: Module-level singleton — shared across all WebGLRenderer instances
// in the same JS process. A baked sky texture is bound to the GL context that
// produced it ... consumers that create multiple renderers concurrently MUST
// not share this cache between them.
const cache = new Map<string, CachedBake>();
```

This is documented but not enforced — multi-renderer hosts (the `two-engines-one-scene` example creates one WebGL + one WebGPU device, but not two WebGLs) would silently miscolor reflections.
**FIX:** Make the cache per-renderer by keying the Map on `renderer.id` (or a WeakMap with `renderer` as the key). The cost is a small refactor of `bakeSkyEquirect(renderer, params)` to look up the per-renderer sub-cache. Optional fix — current users do not actually trigger the hazard.

---

## 3. Refactor candidates (extraction / dedup)

### FINDING: packages/pt-webgpu/src/scene/uploadSceneBuffers.ts (god file)

**CATEGORY:** Mixed Concerns / God File
**WHAT:** This 1058-line file does everything: material packing (lines 181-289), analytic-shape header building (lines 291-313), per-emitter resolution (lines 315-444 — five `firstXLight` functions), env-map params (lines 528-665), emitter-array packing (lines 667-825), the public `buildPackedScene` (lines 827-999), and `uploadPackedScene` (lines 1001-1058). Eight distinct subsystems in one file.
**EVIDENCE:** 1058 lines, 31 top-level functions. The flat list of "firstX/firstY/firstZ" resolvers is repeated structure:

- `firstPointLight` (338-360)
- `firstSpotLight` (362-394)
- `firstRectAreaLight` (396-444)
- `firstMeshAreaLight` (446-526)
  These are all "find the first emitter of kind K in scene.emitters; if none, return zeros." Then `packEmitterArrays` (667-825) does the same loop over `scene.emitters` four times to fill `pointLightsData / spotLightsData / rectAreaLightsData / meshAreaLightsData`.
  **FIX:** Split into `materialPacking.ts`, `emitterPacking.ts`, `environmentPacking.ts`, and a slim `buildPackedScene.ts` that orchestrates them. The `firstXLight` helpers are a vestige from when the pipeline supported only one light per kind — now there's a multi-light array path too, so the `firstX` helpers do double work (also write the singleton fields in the params UBO). Either delete the `firstXLight` helpers and have the params UBO read `pointLightsData[0]`, etc., or wrap both passes into one `extractLights(scene)` that returns both the singleton slot data + the array data.

### FINDING: packages/pt-webgpu/src/scene/uploadSceneBuffers.ts:679-812 (4× near-identical emitter loops)

**CATEGORY:** Duplication
**WHAT:** `packEmitterArrays` runs four nearly-identical loops (point, spot, rect-area+disc, mesh-area), each with the same:

1. allocate buffer of `MAX_K × K_FLOAT_STRIDE`
2. iterate scene.emitters, skip non-matching kinds
3. break with warning at MAX_K
4. write fields at fixed offsets within the per-light stride
   The four loops differ only in field count and offsets.
   **EVIDENCE:** Lines 679-697 (point), 699-725 (spot), 727-781 (rect/disc — slightly different), 783-812 (mesh-area). Four near-clones.
   **FIX:** Extract `packEmitterArray<T, K extends SceneEmitter['kind']>(scene, kind, max, stride, writer: (entry, dst, offset) => void): { count, data, warnings }`. Each call provides its writer. Saves ~80 lines.

### FINDING: packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts:438-571 (renderFrame compute dispatch loop)

**CATEGORY:** Mixed Concerns / God Method
**WHAT:** `renderFrame` is 270 lines (lines 403-671). Inside, it:

1. updates UBO
2. builds 3 bind groups
3. computes `denoiseBase` index
4. clears PPG buffers if enabled
5. dispatches 5 fixed compute passes (RIS, temporal, 2× spatial, shade)
6. optionally dispatches PPG update
7. computes camera-move flag for accumulator reset
8. dispatches SVGF or atrous-legacy denoise chain (via private helpers)
9. dispatches temporal accumulator
10. dispatches composite render pass
11. resolves timestamps and queues readback
12. swaps reservoir ping-pong via a second command encoder + submit
13. returns

The dispatch counter (`denoiseBase`, `accumPassIdx`) is the load-bearing mechanism for tying each pass to its timestamp query slot, but the offset math is brittle.
**EVIDENCE:** Line 436: `const denoiseBase = this._ppgEnabled ? 6 : 5;`. Line 606-608: `const accumPassIdx = this._denoiserMode === 'svgf' ? denoiseBase + 2 + SVGF_DEFAULT_ATROUS_ITERATIONS : denoiseBase + 3;`. The timestamp slot indices are passed manually to every `computeDesc(label, passIdx)` call.
**FIX:** Replace `passIdx: number` with a `passId: string` literal type (e.g. `'ris' | 'temporal' | 'spatial-1' | ...`), and let `timestampQueries.ts` map names → slots. Eliminates the brittle "+ 2 + iterations" arithmetic. Lower-risk alternative: make `computeDesc` auto-increment an internal counter and only validate that the labels match a known schedule.

### FINDING: packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts (god file risk)

**CATEGORY:** Mixed Concerns
**WHAT:** 650 lines doing: GPU resource setup (init), BVH/buffer rebuilds, material upload, light upload, grid+frame+blend uniform packing, three compute passes, atlas-texture caching, and disposal. The class is cohesive but each subsystem could be teased out.
**EVIDENCE:** 15 private methods, three of which (`_rebuildBvhBuffers`, `_uploadMaterials`, `_uploadLights`) are responsible for the bulk of the per-frame data motion. The `GPUResources` interface (lines 53-75) bundles 13 GPU buffers + 1 sampler + 3 pipelines into one struct.
**FIX:** Optional split — extract `DDGIBuffers` (the BVH + material + light + grid + frame + blend buffers) into its own class with a single `upload(device, ...)` method. Leaves probeUpdatePass with pipeline state + the three compute dispatch helpers. Estimated 200-line reduction.

### FINDING: packages/walkaround-hybrid/src/restir/emitterList.ts:82-148 (selection state machine)

**CATEGORY:** Mixed Concerns
**WHAT:** The triangle classification block (lines 109-145) does three things in one nested if:

- normal accumulation (lines 97-107)
- material kind check + emissive vs transmissive branching (lines 109-144)
- power filter (lines 147-148)
  Adding a new emitter selection rule (e.g. mesh-area emitters) would require modifying this block in-place. The `emitterFlags` user-data hook (`skipEmitter`) is a single-bit escape.
  **EVIDENCE:** The inner block:

```
if (emissiveLum > 0 && meshMat.emissiveIntensity && meshMat.emissiveIntensity > 0) {
  cr = meshMat.emissive.r * meshMat.emissiveIntensity;
  ...
} else {
  const physMat = mat as THREE.MeshPhysicalMaterial;
  if (physMat.transmission && physMat.transmission > 0.1) {
    const skipEmitter = (mat.userData as { skipEmitter?: boolean } | undefined)?.skipEmitter === true;
    if (skipEmitter) continue;
    ...
```

**FIX:** Extract `classifyTriangleEmitter(mat, lightDir, intensity)` returning `{ color, intensity } | null`. Then the outer loop is "for each triangle, classify → if not null, accumulate." Easier to extend later.

### FINDING: packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts:225-238

**CATEGORY:** Duplication
**WHAT:** Seven `UboRef` wrappers (`_atrousUboRef`, `_accumUboRef`, `_welfordUboRef`, `_svgfVarianceUboRef`, `_svgfAtrousUboRef`, `_ppgUpdateUboRef`, `_ppgShadeMetaUboRef`) each follow the same shape: `{ buf: GPUBuffer | undefined }` allocated eagerly in `initialize()`, written each frame via `queue.writeBuffer`, destroyed in `dispose()`.
**EVIDENCE:** Initialize 327-338 (7 buffers), dispose 801-807 (7 destroys). Each is a 16-byte UBO except SVGF (16 bytes both). The lifecycle ceremony is duplicated 7 times.
**FIX:** Wrap into `UboBundle` — an array of `{ key: string, size: number, buf: GPUBuffer }`. Initialize iterates a schedule, dispose iterates and destroys. Reduces field count + dispose lines.

### FINDING: packages/walkaround-hybrid/src/ddgi/probeGrid.ts:158-183 vs packages/walkaround-hybrid/src/pipeline/resourceManager.ts:287-330

**CATEGORY:** Duplication (two definitions of the same uniform layout)
**WHAT:** `ProbeGrid.buildUniformData()` writes the same 64-byte DDGI grid-params layout that `packDDGIGridParams()` writes. The two should not drift, but today they're two separate implementations across two files.
**EVIDENCE:** `ProbeGrid.buildUniformData()` returns `Float32Array(16)`. `packDDGIGridParams` returns `ArrayBuffer(64)`. Both fill: origin xyz, spacing, dims xyz, pad, atlas dims. The packer is invoked from `HybridEngine.renderFrame` (line 469); the builder is invoked from `probeUpdatePass._uploadGridParams` (line 447).
**FIX:** Single canonical `packDDGIGridParams(grid: ProbeGrid)` exported from `pipeline/resourceManager.ts`. ProbeGrid would expose `params` (which it already does) and the packer pulls from that.

---

## 4. Stale code / unused exports

### FINDING: packages/shared-samplers/src/index.ts:24-45 (BDPT exports flagged as deferred)

**CATEGORY:** Dead Code (deferred exports)
**WHAT:** The Sprint 10c BDPT exports are intentionally exposed in the public API but documented as "DEFERRED" with the trigger criterion still unmet (Sprint 7 floor-caustic noise threshold). They compile, have tests, but no engine integration. The audit note (L-3 from 2026-05-09) already flags this.
**EVIDENCE:**

```
// ── Sprint 10c (BDPT) — DEFERRED ─────────────────────────────────────────
// Trigger criterion: Sprint 7 hero-render floor-caustic noise exceeds threshold.
// AUDIT NOTE L-3 (2026-05-09): Exports appear in the public API before
// integration testing is complete.
export { BDPT_KIND_LIGHT, ... } from './bdptVertex.js';
export { bdptConnectionMIS, buildBDPTStrategyPDFs } from './bdptMIS.js';
```

**FIX:** Either remove the exports from `src/index.ts` (keep them buildable via deep import for tests) or leave as-is and accept the documented L-3 risk. The audit note is honest enough that this is a "remediation choice" not a finding to fix.

### FINDING: packages/shared-samplers/src/jakobHanika.ts:49-56 (legitimate TODO)

**CATEGORY:** Stale Pattern (deferred but worth tracking)
**WHAT:** The single `TODO (post-Sprint-12)` in shared-samplers points to integrating the full Jakob+Hanika precomputed table. The placeholder is documented in detail (lines 13-67). The TODO is real, tracked in `plan/phase-6-status.md`, and the public API uses an alias (`rgbToSpectralCoefficients`) that is forward-compatible with the swap.
**FIX:** Keep. This is exemplary stale-pattern documentation (TODO + rationale + plan reference + forward-compatible API alias). The Sprint 12 swap is the actual fix.

### FINDING: packages/three-bindings/src/environment.ts:22-26 (orphan TODO)

**CATEGORY:** Stale Comment
**WHAT:** A `TODO: ProceduralSkyEnvironment is not handled here` comment in the `resolveEnvironment` function. The core type `ProceduralSkyEnvironment` exists in `@vitrum/core/scene.ts`, but the THREE → vitrum reverse-binding hasn't been implemented. This is the inverse asymmetry of the `vitrumSceneToThree` path which DOES handle procedural sky (with a warning).
**EVIDENCE:**

```
// TODO: ProceduralSkyEnvironment is not handled here. A THREE.Sky object
// with uniforms { turbidity, mieCoefficient, mieDirectionalG, rayleigh }
// would feed this branch. See core/scene.ts ProceduralSkyEnvironment for
// the expected fields and how the resolved environment is consumed
// downstream.
```

The function currently returns `{ kind: 'none' }` for any non-HDRI scene.
**FIX:** Either implement THREE.Sky → ProceduralSkyEnvironment (the comment shows the mapping is well-understood) or remove the TODO and call out the asymmetry in the package README. Two-line implementation if the user has a THREE.Sky object — but if the project doesn't have a host yet that uses THREE.Sky, the TODO will rot.

### FINDING: packages/walkaround-hybrid/**tests**/sprint11-ppg.test.ts:192 (orphan TODO in test)

**CATEGORY:** Stale Pattern (test scaffold)
**WHAT:** A `TODO(Sprint-11-integration): @group(2) is a placeholder until the PPG` comment in the test. This is in a test file (technically outside the scan scope but I'm flagging because it's the only TODO in any walkaround code). The integration is documented but not done.
**FIX:** Leave; reasonable test-scaffold comment.

### FINDING: packages/core/src/wgpuSupport.ts:15-22, packages/core/src/gpuDetection.ts:34-39

**CATEGORY:** Stale Pattern (deprecated fields)
**WHAT:** Both `WgpuProbeResult.isHardwareGpu` and `GpuDetection.isHardwareGpu` are marked `@deprecated ... Scheduled for removal in Phase 7 / Sprint 1` but Phase 7 has not been formally opened. Code today still reads them (see `probeUpdatePass.ts:137` — `if (gpu.isWebGPU && !gpu.isHardwareGpu)`). The deprecation is honored at the implementation level (callers should migrate to `adapterKind !== 'swiftshader'`) but not enforced.
**EVIDENCE:** The single remaining reader in `probeUpdatePass.ts:137` uses `gpu.isHardwareGpu`. Migrating it to `gpu.adapterKind !== 'swiftshader'` is 1 line.
**FIX:** Migrate the one caller now, then drop the deprecated fields. Don't carry the deprecation indefinitely.

### FINDING: packages/pt-webgl/src/constants.ts:27-28 (deprecated alias)

**CATEGORY:** Stale Pattern (deprecated alias)
**WHAT:** `PT_BOUNCES = PT_PREVIEW_BOUNCES` marked `@deprecated. Kept for backward compatibility`. Used nowhere in this repo per `grep -rn "PT_BOUNCES" --include="*.ts" packages/ examples/`.
**EVIDENCE:**

```
/** @deprecated Use PT_PREVIEW_BOUNCES. Kept for backward compatibility. */
export const PT_BOUNCES = PT_PREVIEW_BOUNCES;
```

`grep` confirms zero in-repo readers.
**FIX:** Delete. No internal users; "backward compatibility" only matters if there's an external consumer (no npm publish per CLAUDE.md). Cheap cleanup.

### FINDING: packages/walkaround-hybrid/src/restir/bvhCompute.ts:264-267 (deferred buffer)

**CATEGORY:** Stale Pattern (export with no consumer)
**WHAT:** `SceneBVHBuffers.cellPower` is built (`emitterList.ts:197-200`) but documented as "Not yet consumed by any WGSL shader — GPU-side consumption is deferred pending walkaround dispatch integration (Sprint 9/10)."
**EVIDENCE:**

```
// Per-emitter radiant flux (f32[], same length as emitters).
// Sprint 3 light tree (shared-samplers buildLightTreeCDF) uses this as its
// `powers` input. Not yet consumed by any WGSL shader — GPU dispatch deferred.
cellPower: { cpuData: cellPowerArray.buffer as ArrayBuffer, ... },
```

The CPU consumer is the light-tree builder, but the light-tree CDF doesn't currently feed the walkaround pipeline either. So the data is built every frame's BVH rebuild for nothing.
**FIX:** Either wire it to `buildLightTree` and pack it into a GPU buffer the shaders read, or strip it from `SceneBVHBuffers` until Sprint 9/10 actually needs it. Today it's CPU memory + a doc claim that the data exists.

### FINDING: packages/walkaround-hybrid/src/HybridEngine.ts:14-19 (planning prose in code)

**CATEGORY:** Stale Comment (verbose planning notes)
**WHAT:** The class JSDoc contains an entire paragraph about "RC re-composition is tracked design work — see plan/walkaround-without-three.md ... not a one-line TODO." This is plan-document content embedded in source.
**EVIDENCE:** Lines 14-19 — a paragraph of plan-document framing for the RC subsystem's relationship to the engine.
**FIX:** Trim to one line: "RC subsystem: see plan/walkaround-without-three.md for re-integration plan."

---

## 5. Style / naming

### FINDING: packages/walkaround-hybrid/src/HybridEngine.ts:172-179 (private static method placement)

**CATEGORY:** Style
**WHAT:** `HybridEngine._fingerprintRebuildKey` is a private static method placed at the top of the class body, before the instance fields. Conventionally static helpers go at the bottom. Minor but inconsistent with the rest of the file.
**FIX:** Move to the end of the class. Style only.

### FINDING: packages/pt-webgl/src/index.ts:22-36 (interface declared mid-file)

**CATEGORY:** Style
**WHAT:** `WebGLPathTracerCompat` interface is declared inline at line 22 (top of file body, between imports and the rest of the imports). Cosmetic; would be cleaner alongside `PTEngineWebGL2Init` (line 317).
**FIX:** Move to the types section near `PTEngineWebGL2Init` and `DeviceLimits`. Style only.

---

## Domain group summary

**A. @vitrum/core** — Clean. The deprecation pattern on `isHardwareGpu` is the only flag, and it's well-documented. The contract types are stable and the structure is appropriate for a library entry-point.

**B. three-bindings** — Mostly clean. One orphan TODO in `environment.ts`. The material packing logic (`material.ts:103-176`) reads userData keys for RFE-06/07/08/03, which is appropriate for the contract escape hatch. The discAreaEmitter rect-conversion logging warning fires every call — could memoize to avoid log spam but it's a designed behavior.

**C. shared-bvh + shared-samplers** — shared-bvh is healthy. shared-samplers has the documented Sprint-12 placeholder and the Sprint-10c deferred-export concerns flagged above. The light-tree implementation is honestly the cleanest non-trivial module in the codebase.

**D. shared-denoisers** — Clean. `disposeSharedWebGPUDevice` is intentionally not in the public index but is consumed by tests via deep import; OK. svgfWebGPU.ts is appropriately structured around the `uploadTexture2D` generic helper which already collapsed six format-specific clones.

**E. pt-webgl** — `index.ts` at 830 lines is on the edge of god-file territory. The state-slot, scheduler, telemetry, and engine class all live in one file. The scheduler options handler `defaultSchedulerOptions` could move to its own file along with the quality-mode types.

**F. pt-webgpu** — The two flagged hot spots: `pathTraceBruteforce.wgsl.ts` has the env-dims-in-meshTriB.w packing convention that two files have to agree on; `uploadSceneBuffers.ts` is a god file with 4× near-identical emitter-loop duplication.

**G. walkaround-hybrid Engine/Pipeline/Shaders/DDGI** — The two regression-grade issues are here: dead `_bvhNormalBuffer`/`_bvhUvBuffer` GPU allocations, and the stale "32 bytes" docstring on `ProbeGrid.buildUniformData()`. Also identified the `cellPower` orphan buffer, double-implementation of DDGI grid params packing, and 7× UboRef duplication in WalkaroundGPUPipeline.

**H. walkaround-hybrid RC/ReSTIR/PPG/Neural** — RC cascadeDispatch has misleading `uniformBuf` naming + module singleton concern. ReSTIR's emitterList.ts could extract the classification state machine. Neural InferenceGraph is structurally sound but the "tensors registered with elementCount=0, buffer=null" contract is fragile (line 219-221) — host must populate via `outputs` map at run time, otherwise dispatch throws. PPG buffer handling looks clean.

**I. Examples/Tools** — The cornell-box main.ts is 721 lines and dense (URL param parsing + scene building + denoise mode switching + capture harness). Could split URL-parsing into its own file, but most of the density is mechanical. two-engines-one-scene wires pt-webgpu via import + `console.debug` so tree-shaking doesn't drop the import — that's a workable integration anchor but a real UI toggle would be cleaner. benchmark-runner is fine.

---

## Synthesis

Most concerning class (regression-introduced):

1. **Dead BVH normal/UV GPU buffers in WalkaroundGPUPipeline** — confirmed by grep-and-read. Either restore the bindings (if the shaders should consume them) or delete the allocations + the corresponding `SceneBVHBuffers` fields if no consumer remains.

Most concerning class (pre-existing): 2. **`uploadSceneBuffers.ts` god file** — 1058 lines, 4 near-identical emitter packing loops, mixed concerns. Documented as flagged in pass 1. 3. **DDGI grid params packed in two places** (probeGrid.buildUniformData vs packDDGIGridParams) — small drift risk; one canonical implementation should win.

Stale-comment regression class: 4. **`ProbeGrid.buildUniformData()` claims "32 bytes" but is 64** — misleading on read. 5. **`cascadeDispatch.ts` `uniformBuf` is actually a STORAGE buffer** — misleading naming.

Deprecation hygiene: 6. **`isHardwareGpu` and `PT_BOUNCES`** scheduled for removal but Phase 7 has not opened. Either open Phase 7 and complete the deprecation, or drop the deprecation markers since they're carrying technical debt for no near-term benefit. (CLAUDE.md says "no upstream PRs yet, no npm publish yet," so backwards compatibility is purely internal.)

The remediation pass clearly succeeded at the level it tackled — comment-vs-code drift has not regressed broadly. The remaining issues are mostly accidental complexity (god files, duplication) plus the regression-introduced GPU buffer waste.
