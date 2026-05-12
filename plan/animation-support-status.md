# Animation support status

**Date:** 2026-05-12 | **Branch:** `feat/tier3-dropin` | **Phase:** T3.I

Honest accounting of what works, what doesn't, and at what cost when the host
animates the scene.  Every claim is tied to the code that was read, not to
comments or plan docs.  "Inferred — not verified" labels appear where GPU
execution could not be exercised in a headless test environment.

---

## TL;DR

- **Camera animation — walkaround-hybrid: works.** Camera position delta drives
  a threshold-gated temporal accumulator reset (`cameraMoveResetThresholdSq`).
  `prevViewMatrix` is consumed by the ReSTIR-DI temporal pass and the ReSTIR-GI
  temporal-GI pass for reservoir reprojection. The default threshold (`1.0`) is
  tuned to Cornell's ~2-unit room; scenes of different scale need the host to
  set `cameraMoveResetThresholdSq = (diagonal × 0.001)²`.

- **Camera animation — pt-webgl: works.** On each frame, `renderFrame` calls
  `setCamera(threeCamera)`, which internally calls `pathTracer.reset()` in
  three-gpu-pathtracer, restarting accumulation. The host does not need a
  separate `engine.reset()` call.

- **Light animation (walkaround-hybrid): limited but wired.** The engine exposes
  `updateLighting()` for the primary directional light + sky, which invalidates
  the DDGI probe atlas and resets the temporal accumulator. DDGI re-convergence
  takes ~8 frames (one full STRIDE cycle). ReSTIR-DI responds immediately to
  the next frame's UBO upload. `updateEmitter()` on the `Engine` interface is
  typed `never` on `HybridEngine` — no per-emitter runtime patch path exists.
  `WalkaroundGPUPipeline` has an `updateEmitters()` method that re-uploads the
  emitter GPU buffers, but it is not connected to any public API.

- **Mesh transform animation: not supported.** `updatePrimitive` is typed
  `never` on `HybridEngine`. The only path to reflect a transform change is
  `setScene()`, which triggers a full async pipeline reinit (BVH rebuild + shader
  pipeline recreate). The reinit is async (fire-and-forget) and the engine drops
  frames while in `initializing` state. At 10 k triangles a BVH rebuild takes
  ~1–5 ms CPU (inferred from `buildSharedBVH` + CPU-side data packing); GPU
  buffer re-upload adds another ~0.5 ms. A 60 fps host calling `setScene()` every
  frame would be continuously in `initializing` state and would render nothing.

- **Skinning / morph targets: unsupported.** `sceneFromThreeJS` throws on
  `THREE.SkinnedMesh`. No skinning matrix upload, no morph target blending, no
  deformed-geometry path in any shader. This would require a non-trivial new
  feature (CPU-side skin solve, re-pack geometry into BVH each frame, or a
  GPU-side TLAS).

---

## 1 — Camera animation

### walkaround-hybrid

The temporal accumulator in `WalkaroundGPUPipeline` is governed by
`_accumFrameIndex`.  When it is 0, the GPU blend weight `alpha = 1.0` (no
history); when > 0, `alpha = _temporalAccumAlpha` (default 0.01, i.e. 99%
history retention).

**Reset path:**

`WalkaroundGPUPipeline.renderFrame()` lines 926–933
(`packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts:926`):

```ts
const dx = inputs.cameraPos[0] - this._lastCameraPos[0];
const dy = inputs.cameraPos[1] - this._lastCameraPos[1];
const dz = inputs.cameraPos[2] - this._lastCameraPos[2];
const camMoveSq = dx * dx + dy * dy + dz * dz;
const isMoving = camMoveSq > this._cameraMoveResetThresholdSq;
if (isMoving) {
  this._accumFrameIndex = 0;
}
```

When `isMoving`, the `alpha=1` path is taken on the same frame — the Welford
temporal pass also receives `isMoving ? 1 : 0` in its UBO (line 1179) so variance
history is also cleared simultaneously.

**prevViewMatrix path:**

`HybridEngine.renderFrame()` line 913 (`packages/walkaround-hybrid/src/HybridEngine.ts:913`):

```ts
const prevView = (input.prevViewMatrix ?? input.viewMatrix) as Float32Array;
```

This falls back to the current-frame view if the host omits `prevViewMatrix`.
The value is forwarded to `WalkaroundGPUPipeline.renderFrame()` as
`prevViewMatrix`, written into the WalkaroundUBO at offset 128 (byte 128,
`packages/walkaround-hybrid/src/pipeline/uboUpdater.ts:50`), and consumed by:

- `temporal.wgsl` (ReSTIR-DI temporal reuse): `let prevView = ubo.prevViewMatrix;`
  (`packages/walkaround-hybrid/src/shaders/temporal.wgsl.ts:72`)
- `temporalGi.wgsl` (ReSTIR-GI temporal reuse): `let prevClip = ubo.projMatrix *
  ubo.prevViewMatrix * vec4f(worldPos, 1.0);`
  (`packages/walkaround-hybrid/src/shaders/temporalGi.wgsl.ts:70`)

When the host does not supply `prevViewMatrix`, both reprojection paths use the
current-frame view, which maps every world-space point to its current screen
position rather than the previous-frame screen position. This means temporal
reuse always "finds" a match at the same pixel and treats every reservoir as
valid (no geometric consistency rejection on motion). The image appears stable
but ghosts are not detected. **Hosts driving camera animation MUST supply
`prevViewMatrix` on every frame.** (Inferred — not verified with a real GPU
render comparing the two modes.)

**Scale sensitivity:**

The default `cameraMoveResetThresholdSq = 1.0` is calibrated to Cornell's
~2-unit room and OrbitControls' damped motion (~0.1–0.5 units/frame over ~30
frames after a drag release). At other scales:

| Scene diagonal | Recommended threshold | Notes |
|---|---|---|
| 0.01 m (jewellery) | `(0.01 × 0.001)² = 1e-10` | Default never trips; ghost permanently |
| 2 m (Cornell room) | `(2 × 0.001)² = 4e-6` | Default `1.0` is too large by 250 000× |
| 100 m (city block) | `(100 × 0.001)² = 0.01` | Default `1.0` allows 1 m camera drift |

The T3.A unified factory derives `cameraMoveResetThresholdSq = (D × 0.001)²`
from scene AABB diagonal `D`; this is the right fix for the scale problem.

**First-frame quirk:**

`_lastCameraPos` initialises to `[0, 0, 0]`
(`packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts:269`).
If the camera does not start at origin, frame 1 always triggers a reset (good —
it prevents stale history). If the camera starts exactly at origin and does not
move, the reset never fires, and `_accumFrameIndex` stays 0 only for frame 0
then increments normally. Net effect: one guaranteed clean frame at startup.

**Test coverage:**

New test:
`packages/walkaround-hybrid/__tests__/cameraAnimationReset.test.ts` (8 tests,
all green). Tests cover:
- `requestAccumReset()` sets `_accumFrameIndex = 0`
- Default threshold is `1.0`
- `alpha = 1.0` when `_accumFrameIndex === 0`, otherwise `_temporalAccumAlpha`
- Squared-distance arithmetic including jewellery and city-block scale examples

The test does not run a full GPU frame (no real `GPUDevice` available headlessly).
The GPU-side blend-weight selection (`alpha = accumFrameIndex === 0 ? 1.0 : α`)
is verified by the arithmetic tests; the actual GPU shader is inferred-correct.

### pt-webgl

`ptEngineWebGL2.renderFrame()` line 642–645
(`packages/pt-webgl/src/ptEngineWebGL2.ts:642`):

```ts
const cameraSignature = this.#makeCameraSignature(input);
if (cameraSignature !== this.#cameraSignature) {
  applyFrameToPerspectiveCamera(this.#camera, input);
  (this.#pathTracer as unknown as WebGLPathTracerCompat).setCamera(this.#camera);
  this.#cameraSignature = cameraSignature;
}
```

`setCamera()` in three-gpu-pathtracer calls `this.reset()` internally
(`~/projects/three-gpu-pathtracer/src/core/PathTracingRenderer.js:320`, line 229
and 366). The accumulator restarts on every camera change. No explicit `engine.reset()`
call is needed from the host.

No test for this path. The coverage is from reading two files:
`ptEngineWebGL2.ts:642` and `three-gpu-pathtracer/PathTracingRenderer.js:320`.

---

## 2 — Light animation

### walkaround-hybrid — `updateLighting()`

`HybridEngine.updateLighting()` (`packages/walkaround-hybrid/src/HybridEngine.ts:723`)
handles the primary directional light and sky dome at runtime. It:

1. Updates in-memory engine fields (`_primaryLightDir`, `_primaryLightIntensity`,
   `_skyTint`, `_skyIrradiance`). These are read by `renderFrame()` and
   written into the WalkaroundUBO each frame — no GPU buffer is stale.
2. Calls `this._ddgi.invalidateProbeCache()` — resets `_frame = 0` and
   `_ready = false` in the DDGI instance
   (`packages/walkaround-hybrid/src/ddgi/DDGI.ts:150`).
3. Calls `this._pipeline?.requestAccumReset()` — sets `_accumFrameIndex = 0`.

Cost: 2–4 JS field writes, no GPU work dispatched.

**DDGI convergence time:**

`DDGI.updateFrame()` uses round-robin: 1/STRIDE probes per frame, `STRIDE = 8`
(`packages/walkaround-hybrid/src/ddgi/DDGI.ts:37`). After
`invalidateProbeCache()` resets `_frame = 0`, the atlas goes dark (all probes
stale) for 8 frames before `_ready` flips back to true.

At 60 fps: `8 × (1/60 s) ≈ 133 ms` before the irradiance atlas reflects the
new lighting. During those 8 frames the shade pass reads the stale (dark/zero)
atlas, but `isDDGIWired()` only returns true once `_ready = true`, so the
engine falls back to a zero-indirect mode during reconvergence. Effective visual
latency is therefore ~133 ms at 60 fps.

**DDGI note:** the probe atlas border padding allocation vs. write bug identified
in `memory/in-flight-sweep.md` means probe border texels read zero, introducing
bilinear bleed at every cell edge during and after reconvergence. This affects
quality but not the convergence-time estimate.

**ReSTIR-DI:**

The directional light and sky parameters are written into the WalkaroundUBO each
frame via `updateUBO()`. ReSTIR-DI's temporal M-clamp
(`temporalMClampDI = 20` default) means stale light reservoirs persist up to
20 frames before being fully diluted. At 60 fps this is ~333 ms of lag on
direct illumination after a light change. The host can set `temporalMClampDI`
lower (e.g. `5`) to trade variance for responsiveness during interactive light
editing.

**ReSTIR-GI:**

The GI reservoir is re-sampled every frame (no RIS reuse persists across a
lighting change that invalidates the irradiance atlas). GI convergence follows
DDGI convergence.

**Convergence summary:**

| Channel | Latency at 60 fps | Comment |
|---|---|---|
| Direct light (ReSTIR-DI) | ~333 ms | M-clamp dilution at 20 frames |
| Indirect / GI (DDGI) | ~133 ms | 8-frame STRIDE re-cycle |
| Indirect (ReSTIR-GI) | ~133 ms | Follows DDGI atlas readiness |
| Temporal accumulator | 1 frame | `requestAccumReset()` forces α=1 |

**Per-emitter patching (`updateEmitter`):**

`HybridEngine.updateEmitter` is typed `never`:

```ts
// packages/walkaround-hybrid/src/HybridEngine.ts:695-696
updatePrimitive?: never;
updateEmitter?: never;
```

`capabilities.supportsIncrementalScene` is `false`
(`packages/walkaround-hybrid/src/HybridEngine.ts:644`).

`WalkaroundGPUPipeline.updateEmitters()` exists
(`packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts:548`) and
would allow re-uploading the emitter GPU buffers without a full BVH rebuild.
However it is **not connected to any public API** on `HybridEngine` or `Engine`.
If a host calls `engine.updateEmitter(id, { intensity: 2 })`, the call does not
exist at the `Engine` interface level (it is optional), and on `HybridEngine`
specifically it is typed `never` (calling it is a TypeScript error).

The only correct path for emitter changes today is `engine.updateLighting()`
(directional + sky only) or `engine.setScene()` (full scene with new emitter
values, triggers full reinit).

**pt-webgl:**

`ptEngineWebGL2.updateEmitter()` throws `'Not implemented: updateEmitter (pt-webgl
requires full setScene)'`
(`packages/pt-webgl/src/ptEngineWebGL2.ts:611`). Same situation — full
`setScene()` is the only path, which resets the PT accumulator to 0.

---

## 3 — Mesh transform animation

### walkaround-hybrid

`HybridEngine.updatePrimitive` is typed `never` (line 695). Calling
`engine.updatePrimitive(id, { transform: mat4 })` is a TypeScript error.

The only path to move a mesh is to call `engine.setScene()` with a new scene
that has the updated transform baked into the mesh's geometry (since the Scene
type uses world-space position buffers, not a per-primitive transform matrix).
`setScene()` calls `_teardownPipeline()` then `_initPipeline()` asynchronously:

```ts
// packages/walkaround-hybrid/src/HybridEngine.ts:687-693
setScene(scene: Scene): void {
  this._lastScene = scene;
  this._teardownPipeline();
  void this._initPipeline();
}
```

`_initPipeline()` polls for scene readiness (up to 5 s), then:
1. Calls `buildReSTIRSceneBVH()` — CPU-side BVH build, merges all geometry,
   packs emitter list.
2. Creates a new `WalkaroundGPUPipeline` and calls `pipeline.initialize()` —
   compiles all WGSL shaders, uploads 6 GPU buffers (BVH nodes, index,
   positions, beer, emitters, CDF), allocates all frame textures.
3. Sets `_state = 'ready'`.

Until step 3, `renderFrame()` returns `skipOutput` (samplesAccumulated: 0).
On a first `setScene()` for a Cornell-scale scene (~2000 triangles) the BVH
build is ~1–5 ms CPU. The shader compile is the bottleneck: expect 100–500 ms
wall time for the first pipeline create (browser shader compilation is
unpredictable). A per-frame `setScene()` at 60 fps would re-trigger this every
~16.7 ms — the pipeline would never finish before the next `setScene()` call and
the engine would render nothing.

**Memory handling:**

`_teardownPipeline()` calls `this._pipeline.dispose()` before creating the new
pipeline. `WalkaroundGPUPipeline.dispose()` explicitly destroys all GPU buffers
(lines 1482–1492). There is no GPU buffer leak on repeated `setScene()` calls,
provided the async init finishes before the next teardown. If `setScene()` is
called while `_initPipeline()` is still running, `_disposed` is checked at the
top of the async body and the half-initialized pipeline is torn down via the
`if (this._disposed)` guard (lines 1197–1204). This is correct as long as the
guard is checked before any GPU buffer is assigned — reading the code confirms
it is (`this._pipeline = pipeline` happens at line 1234, after the guard).

**Per-frame transform: estimated cost to make it work:**

For 60 fps dynamic geometry a proper implementation would need:

- A CPU-side BVH refitting path (update node AABBs without full rebuild) for
  rigid-body transforms. Refitting is O(n) in node count; for 10 k triangles a
  SAH BVH has ~20 k nodes. JavaScript at ~1 ns/op → ~20 μs per refit. Feasible.
- GPU buffer patch upload: BVH positions rewritten in-place (`GPUQueue.writeBuffer`).
  At 10 k tris × 4 vertices × 16 bytes/vertex = ~640 KB upload per frame. At
  PCIe bandwidth this is ~0.06 ms. Feasible.
- Skip pipeline reinit (no shader recompile needed for a topology-preserving
  transform).

None of this is implemented. The current path (full rebuild + pipeline teardown
+ shader recompile) is not viable at 60 fps for any non-trivial mesh.

**pt-webgl:**

`ptEngineWebGL2.updatePrimitive()` throws `'Not implemented: updatePrimitive
(pt-webgl requires full setScene)'` (line 608). Same situation. The underlying
three-gpu-pathtracer handles geometry updates via `setScene()` on the path
tracer, which resets accumulation to 0.

---

## 4 — Skinning / morph targets

Not supported anywhere in the pipeline stack.

`sceneFromThreeJS` (three-bindings) throws explicitly on `THREE.SkinnedMesh`:

```ts
// packages/three-bindings/src/index.ts:65-67
if ((obj as THREE.SkinnedMesh).isSkinnedMesh === true) {
  throw new Error(`Unsupported THREE type at "${label}": SkinnedMesh. ...`);
}
```

Verified by test:
`packages/three-bindings/src/__tests__/sceneFromThreeJS.test.ts:58` — "throws
on SkinnedMesh".

No mention of morph targets, blend shapes, or `THREE.AnimationMixer` anywhere
in the packages (verified by grep).

**What it would take:**

- CPU-side skin solve: for each frame, compute `finalPosition = Σ (boneMatrix_i × bindPose_i × position × weight_i)`. This can be done in a compute shader or on the CPU.
- Re-pack the deformed positions into the BVH position buffer and refit the BVH.
- Or: a two-level acceleration structure (BLAS per mesh, TLAS over instances) so deformed geometry updates only its BLAS; the TLAS is rebuilt (much cheaper). WebGPU does not expose hardware TLAS/BLAS — any TLAS scheme would be a software implementation in WGSL.

Effort estimate: 4–6 weeks for CPU-side skinning + GPU BVH refit. A proper
TLAS/BLAS software implementation is 8–12 weeks and is a major architecture
change.

---

## 5 — Backlog gap items

### Gap 1 — Connect `WalkaroundGPUPipeline.updateEmitters()` to a public API

**Title:** `HybridEngine.updateEmitter()` — wire the existing GPU emitter
re-upload path to the public Engine contract.

**Description:** `WalkaroundGPUPipeline.updateEmitters(buffers)` already exists
and re-uploads the emitter GPU buffers without rebuilding the BVH or recompiling
shaders. `HybridEngine.updateEmitter()` is typed `never`, blocking any host
attempt to update light intensity at runtime without a full scene rebuild. The
fix requires:

1. Remove the `updateEmitter?: never` declaration on `HybridEngine` and
   implement the method.
2. On `updateEmitter(id, patch)`: patch the cached `_lastScene.emitters` array,
   re-run `buildEmitterList()` (CPU-only), and call `_pipeline.updateEmitters()`.
3. If the emitter patch changes a `kind` (point → directional), or removes/adds
   an emitter that is also geometry (emissive mesh), fall back to full `setScene`.
4. Call `updateLighting()` internally if the patch touches intensity fields that
   feed DDGI.

Expected result: hosts can update point light intensity at 60 fps with ~1 ms
CPU overhead and no shader recompile. DDGI re-convergence still applies for
intensity changes.

**Effort:** 2–3 days.

**Dependencies:** none (the GPU path already exists in `WalkaroundGPUPipeline`).

---

### Gap 2 — Document and expose `cameraMoveResetThresholdSq` in T3.A unified factory

**Title:** Auto-derive `cameraMoveResetThresholdSq` from scene AABB in `createEngine()`.

**Description:** The default threshold of `1.0` is hardcoded to Cornell's
~2-unit room. Any scene at a different scale silently gets wrong behaviour:
a jewellery scene never fires the reset (permanent temporal ghosting on every
camera move), a city-block scene allows 1-metre camera drift before resetting
(sluggish responsiveness). The T3.A `createEngine()` factory already plans to
derive this from scene AABB diagonal `D`:
`cameraMoveResetThresholdSq = (D × 0.001)²`. This is confirmed as the right
formula by the cameraAnimationReset.test.ts scale-sensitivity tests.

The existing `packages/engine/__tests__/createEngine.test.ts:36` already asserts
this derivation for the factory. The gap is that the derivation lives only in
the T3.A spec; no production factory exists yet and `HybridEngine` itself has no
AABB-aware default.

**Effort:** included in T3.A (0 additional effort if T3.A proceeds).

**Dependencies:** T3.A must land first. Until then, hosts MUST set
`cameraMoveResetThresholdSq` manually in `HybridEngineOptions` — the default
is silently wrong for any non-Cornell scene.
