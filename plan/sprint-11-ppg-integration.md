# Sprint 11 — PPG Integration Spec (Deferred)

**Status**: Deferred — GPU verification not available.
**Created**: 2026-05-09
**Applies to**: `@vitrum/walkaround-hybrid` only (walkaround-only per the Sprint 11 DoD).

## BLOCKING CONDITIONS — must resolve before wiring dispatch

> **These are hard blockers. Do not wire the PPG dispatch (Steps 1–6 below) until all are resolved.**

### BLOCK-1 — `ppgFindCellIndex` is O(N) brute force — unacceptable for live dispatch

`ppgSample.wgsl.ts:ppgFindCellIndex` performs a linear scan over all `PPG_MAX_SPATIAL_CELLS`
(up to 10,000) cells per shader invocation. At walkaround resolution (1920×1080, checkerboard
= 50% pixels, 2 bounces), this is approximately **10,000 × 1,080 × 960 = 10.4 billion
comparisons per frame**. At 8 ns/comparison (GPU SIMD), worst-case cost is ~83 seconds per
frame — clearly unsuitable.

**Required fix:** Replace the brute-force scan in `ppgFindCellIndex` with a proper kd-tree
binary descent (O(log N), ~13 comparisons for 10K cells — a 769× reduction). See §kd-tree
storage layout below for the design decision record.

The current implementation is authored for structural correctness only and is explicitly tagged
with a TODO comment in the WGSL source. Do not remove that comment until the kd-tree is in place.

## What was built in Sprint 11

| Artifact | Location | Status |
|---|---|---|
| `PPGDirectionalBin`, `PPGQuadTreeNode`, `PPGSpatialCell` types | `src/ppg/types.ts` | Shipped |
| `PPG_MAX_SPATIAL_CELLS`, `PPG_DIRECTIONS`, byte-stride constants | `src/ppg/types.ts` | Shipped |
| `PPG_SAMPLE_WGSL` fragment | `src/ppg/wgsl/ppgSample.wgsl.ts` | Authored; not dispatched |
| `PPG_UPDATE_WGSL` compute kernel | `src/ppg/wgsl/ppgUpdate.wgsl.ts` | Authored; not dispatched |
| `createPPGBuffers` / `destroyPPGBuffers` helpers | `src/pipeline/resourceManager.ts` | Exported; opt-in via `ppgEnabled` |
| `PPGBuffers` interface + `FrameResources.ppgBuffers` field | `src/pipeline/resourceManager.ts` | Optional field; undefined when disabled |
| `HybridEngineOptions.ppgEnabled` | `src/HybridEngine.ts` | Construction-time opt-in |
| `HybridEngine.setPPGEnabled()` | `src/HybridEngine.ts` | Toggle method; no-op for dispatch in Sprint 11 |
| `HybridEngine.ppgEnabled` getter | `src/HybridEngine.ts` | Reflects current toggle state |

## What integration requires

When GPU verification is available, integrate in this order.

### 1. Add PPG shader compilation in `pipelineCompiler.ts`

Two new `GPUComputePipeline` objects are needed:

```ts
// ppgUpdate — runs after shade pass on even frames.
// Includes COMMON_WGSL (for luminance() and vec3f utilities) + PPG_UPDATE_WGSL.
const ppgUpdateModule = device.createShaderModule({
  label: 'ppg-update',
  code: COMMON_WGSL + PPG_UPDATE_WGSL,
});
const ppgUpdatePipeline = device.createComputePipeline({
  label: 'ppg-update',
  layout: 'auto',
  compute: { module: ppgUpdateModule, entryPoint: 'ppgUpdateKernel' },
});

// ppgSample — included into the shade.wgsl module as a fragment.
// PPG_SAMPLE_WGSL declares its bindings at @group(2); shade.wgsl must
// be updated to include the fragment and forward the group-2 bindings.
// (Alternatively, compile shade.wgsl as: COMMON_WGSL + PPG_SAMPLE_WGSL + SHADE_WGSL.)
```

### 2. Add PPG bind group layouts in `bindGroupLayouts.ts`

**PPG Update BGL** (group 0, 4 entries):
| Binding | Resource | Type |
|---|---|---|
| 0 | `PPGUpdateUniforms` UBO | `uniform` |
| 1 | `ppgSamples` (PPGPathSample array, read) | `storage` |
| 2 | `ppgCells` (PPGSpatialCell array, read) | `storage` |
| 3 | `ppgLeafData` (atomic u32 array, read_write) | `storage` |

**PPG Sample BGL** (added to shade.wgsl's bind group, group 2):
| Binding | Resource | Type |
|---|---|---|
| 0 | `ppgCells` (PPGSpatialCell array, read) | `storage` |
| 1 | `ppgLeaves` (PPGDirectionalLeaf array, read) | `storage` |

Note: `ppgLeaves` (read by ppgSample) and `ppgLeafData` (atomic write by ppgUpdate)
are the same underlying `leafBuffer` with different access modes. Use two `GPUBindGroup`
objects bound to the same `GPUBuffer` — the WebGPU spec permits this.

### 3. New GPU resources needed in `FrameResources`

Add to `FrameResources` (already present as optional `ppgBuffers` field):

| Field | Format | Usage | Notes |
|---|---|---|---|
| `ppgBuffers.cellBuffer` | raw (PPGSpatialCell array) | `STORAGE \| COPY_DST \| COPY_SRC` | Allocated when `ppgEnabled: true` |
| `ppgBuffers.leafBuffer` | raw (atomic u32 array) | `STORAGE \| COPY_DST \| COPY_SRC` | 256 bytes/leaf; 128 used atomically |
| `ppgBuffers.sampleBuffer` | raw (PPGPathSample array) | `STORAGE \| COPY_DST \| COPY_SRC` | Written by shade pass; consumed by ppgUpdate |

No new textures — PPG data lives entirely in storage buffers.

### 4. Add PPGUpdateUniforms UBO in `WalkaroundGPUPipeline`

A small (16-byte) UBO per frame:

```ts
// Packed PPGUpdateUniforms (16 bytes = 4 × u32):
//   u32[0] = sampleCount  (number of completed paths this frame)
//   u32[1] = frameParity  (frameIndex & 1 — 0=update, 1=skip)
//   u32[2] = cellCount    (active cells in ppgCells)
//   u32[3] = _pad
```

Write this UBO each frame in `renderFrame()` after the shade pass.

### 5. New render-frame dispatch order in `WalkaroundGPUPipeline.renderFrame()`

Current dispatch order (7 passes, Sprint 9 additions still deferred):
1. RIS
2. Temporal reuse
3. Spatial reuse × 2
4. Shade + GI → writes radiance to `hdrColorTexture`; **NEW: also writes PPGPathSample records to `ppgBuffers.sampleBuffer`**
5. À-trous denoiser × 3 iterations
6. Temporal accumulation (accum)
7. Composite blit

New dispatch with PPG update (8 passes):
1. RIS
2. Temporal reuse
3. Spatial reuse × 2
4. Shade + GI → writes radiance + PPGPathSample records
5. **NEW** PPG update pass (even frames only — gate on `frameParity == 0`) → reads `sampleBuffer`, atomically updates `leafBuffer`
6. À-trous denoiser × 3 iterations
7. Temporal accumulation (accum)
8. Composite blit

The PPG update pass dispatches at `@workgroup_size(64, 1, 1)` with
`ceil(sampleCount / 64)` workgroups. `sampleCount` is bounded by `maxCells`
(10,000) — at most 157 workgroups.

### 6. Shade pass changes for PPG integration

Two changes to `shade.wgsl`:

#### 6a. Include PPG_SAMPLE_WGSL fragment

Add `PPG_SAMPLE_WGSL` as a preamble to `shade.wgsl` (or compile as a concatenated module).
The fragment declares `@group(2) @binding(0)` for `ppgCells` and `@group(2) @binding(1)`
for `ppgLeaves`. These bindings must be populated in the shade pass's bind group builder.

Gate via a WGSL override constant so the shader compiles without PPG when `ppgEnabled = false`:

```wgsl
override ppgEnabled: bool = false;

// In the indirect-bounce sampling loop:
var indirectDir: vec3f;
if (ppgEnabled) {
  indirectDir = ppgSampleDirection(hitPos, hitNormal, rand_f32(rng), rand_f32(rng), &rng);
} else {
  indirectDir = sampleCosineHemisphere(hitNormal, &rng);
}
```

#### 6b. Write PPGPathSample records

After computing each indirect-bounce radiance estimate, append a `PPGPathSample`
to `ppgSampleBuffer`. Track the write index via an atomic counter (separate
`ppgSampleCountBuffer` — one u32).

```wgsl
// At each indirect bounce that produces a non-zero radiance estimate:
let sampleIdx = atomicAdd(&ppgSampleCount[0], 1u);
if (sampleIdx < arrayLength(&ppgSamples)) {
  ppgSamples[sampleIdx].worldPos    = hitPos;
  ppgSamples[sampleIdx].incidentDir = bounceDir;
  ppgSamples[sampleIdx].radiance    = estimatedRadiance;
}
```

Reset `ppgSampleCountBuffer` to 0 each frame before the shade dispatch.

### 7. Trigger gates (per Sprint 11 roadmap)

- **Bail-out criterion**: if walkaround framerate drops below 30 fps after enabling PPG,
  call `engine.setPPGEnabled(false)` and `engine.reset()` to deallocate PPG buffers.
  The 30 fps floor is monitored via `HybridEngine._dbg.framesDispatched` over a 5s window.
- **Cold-start**: the first ~100 frames after PPG enable will use cosine-weighted sampling
  (leaf bins all zero). The learned PDF builds up gradually — visible improvement is
  expected after 200–500 frames in a static scene.

### 8. Definition of done (Sprint 11 roadmap)

Per `plan/phase-6-roadmap.md` Sprint 11:
- [ ] PPG kd-tree allocated as WebGPU storage buffer (sparse, capped at ~10K cells) ✓ (structure authored)
- [ ] Each cell holds a quad-tree of directional bins (16-direction discretisation) ✓ (authored)
- [ ] Per-frame: collect path-completion samples into the structure; ping-pong update ✓ (authored; dispatch deferred)
- [ ] Path-tracing dispatch reads the cell at each indirect bounce, samples from learned PDF (deferred)
- [ ] Visual A/B: indirect-only-lit scene converges at 30 samples vs. 90 samples baseline (deferred — requires GPU)

### 9. Convergence target

Sprint 11 DoD: indirect-only convergence at 30 vs. 90 samples baseline.

Measurement protocol:
1. Load the reference stained-glass room scene (all panels lit, no direct sun).
2. Disable DDGI so indirect lighting is from the ReSTIR GI bounce only.
3. Capture a 30-sample frame from the baseline (cosine-weighted) pipeline.
4. Enable PPG (`ppgEnabled: true`, let it warm up for 500 frames).
5. Capture a 30-sample frame from the PPG pipeline.
6. Compare floor/wall per-pixel stddev via the reference-render harness.
7. Target: PPG 30-sample noise ≤ baseline 90-sample noise (3× sample efficiency).

Reference renders captured in `tools/reference-renders/sprint-11-ppg-baseline/`.

## Ping-pong frame parity convention

PPG update runs on even frames (`frameParity = frameIndex & 1 == 0`). This halves the
update bandwidth and mirrors the Sprint 9 checkerboard resolve convention (which also uses
`frameParity`). Both are driven by the same per-frame uniform `frameSeed % 2`.

When the two features compose (Sprint 9 integration + Sprint 11):
- Even frames: shade → PPG update → sample-budget pass → resolve
- Odd frames: shade → (skip PPG update) → sample-budget pass → resolve

## kd-tree storage layout — design decision record

**Decision: dense linear array with brute-force nearest-cell lookup.**

Rationale:
- The kd-tree binary descent in `ppgSample.wgsl` uses a brute-force O(N) linear scan
  over `ppgCells` in Sprint 11. At 10K cells, this is 10K vec3f distance comparisons
  per indirect-bounce shader invocation. At 8 ns/comparison (GPU SIMD), this costs
  ~80 µs per invocation in the worst case — acceptable for the structural-prep phase.
- A proper kd-tree index (halving N at each level, O(log N) per lookup) reduces this
  to ~13 comparisons — a 769× reduction. Post-Sprint-11 optimisation priority.
- Dense allocation (all `PPG_MAX_SPATIAL_CELLS` slots allocated upfront) avoids GPU-side
  dynamic allocation, which WebGPU does not support. Unused cells have `position = (0,0,0)`
  and empty bins; the nearest-cell logic still converges because every scene point will
  find its actual nearest cell before the origin.

## Atomic update strategy — design decision record

**Decision: fixed-point u32 atomics.**

WebGPU does not support `atomicAdd` on f32 storage buffers. The PPG update kernel stores
radiance sums as `round(value × 65536)` in `atomic<u32>` fields. The fixed-point scale
(65536 = 2¹⁶) gives ~0.0015% precision (sufficient for HDR values in the 0–1000 nit
range before hitting u32 overflow at 65,536 total accumulated radiance).

The `ppgSample.wgsl` fragment reads the bins as `vec2f` (x=radianceSum, y=sampleCount)
and reconstructs floating-point radiance as `f32(storedU32) / 65536.0`. This division
is a single reciprocal multiply — no performance concern.

Periodic reset: to prevent u32 overflow accumulation, the host may periodically zero-fill
`leafBuffer` (e.g. every 10,000 frames or on scene change). Sprint 11 does not implement
automatic decay — a future sprint may add exponential radiance decay per frame.
