# Sprint 9 — Walkaround Adaptive Sampling Integration Spec

**Status**: Integration deferred — GPU verification not available.
**Created**: 2026-05-09
**Applies to**: `@vitrum/walkaround-hybrid` only.

## What was built in Sprint 9

| Artifact                            | Location                           | Status                      |
| ----------------------------------- | ---------------------------------- | --------------------------- |
| `WelfordVariance` struct + helpers  | `src/shaders/common.wgsl.ts`       | Shipped in `COMMON_WGSL`    |
| `sampleBudgetKernel` compute shader | `src/shaders/sampleBudget.wgsl.ts` | Complete, not dispatched    |
| `resolveKernel` compute shader      | `src/shaders/resolve.wgsl.ts`      | Complete, not dispatched    |
| `createVarianceBuffer` helper       | `src/pipeline/resourceManager.ts`  | Exported; allocates at init |
| Variance buffer in `FrameResources` | `src/pipeline/resourceManager.ts`  | Allocated, not written      |

## What integration requires

When GPU verification is available, integrate in this order.

### 1. Export new shaders from the package index

`src/shaders/index.ts` (or wherever shaders are re-exported) should add:

```ts
export { SAMPLE_BUDGET_WGSL } from './sampleBudget.wgsl.js';
export { RESOLVE_WGSL } from './resolve.wgsl.js';
```

### 2. Compile new pipelines in `pipelineCompiler.ts`

The pipeline compiler (`src/pipeline/pipelineCompiler.ts`) currently compiles 7
pipelines (ris, temporal, spatial, shade, atrous, accum, composite). Add two more:

```ts
// Shader source: COMMON_WGSL + SAMPLE_BUDGET_WGSL (concatenated, or include via
// the existing pattern of passing COMMON_WGSL as a preamble).
const sampleBudgetModule = device.createShaderModule({
  label: 'sample-budget',
  code: COMMON_WGSL + SAMPLE_BUDGET_WGSL,
});
const sampleBudgetPipeline = device.createComputePipeline({
  label: 'sample-budget',
  layout: 'auto',
  compute: { module: sampleBudgetModule, entryPoint: 'sampleBudgetKernel' },
});

// Shader source: RESOLVE_WGSL (resolve.wgsl declares its own structs;
// it does NOT need COMMON_WGSL as a preamble unless welfordVariance is
// called inline — in Sprint 9 the shader has inline comments for standalone
// validation; the inline [INLINE-COPY] stubs must be replaced with real
// WGSL declarations before concatenation).
const resolveModule = device.createShaderModule({
  label: 'resolve',
  code: COMMON_WGSL + RESOLVE_WGSL,
});
const resolvePipeline = device.createComputePipeline({
  label: 'resolve',
  layout: 'auto',
  compute: { module: resolveModule, entryPoint: 'resolveKernel' },
});
```

### 3. Build new bind group layouts in `bindGroupLayouts.ts`

**Sample-budget BGL** (group 0, 4 entries):
| Binding | Resource | Type |
|---|---|---|
| 0 | `SampleBudgetUniforms` UBO | `uniform` |
| 1 | `t_variance` (rg32float, read) | `storage-texture`, access: `read-only` |
| 2 | `t_tier_out` (r32uint, write) | `storage-texture`, access: `write-only` |
| 3 | `u_sampleCount` UBO | `uniform` |

> Note: bindings 0 and 3 are both uniforms. They may be merged into one struct if desired — the shader can be refactored to put sampleCount inside `SampleBudgetUniforms`.

**Resolve BGL** (group 0, 5 entries):
| Binding | Resource | Type |
|---|---|---|
| 0 | `ResolveUniforms` UBO | `uniform` |
| 1 | `t_current_radiance` (rgba16float, read) | `storage-texture`, access: `read-only` |
| 2 | `t_prev_radiance` (rgba16float, read) | `storage-texture`, access: `read-only` |
| 3 | `t_motion_vectors` (rg32float, read) | `storage-texture`, access: `read-only` |
| 4 | `t_resolved_out` (rgba16float, write) | `storage-texture`, access: `write-only` |

### 4. New GPU textures needed in `FrameResources`

Add to `FrameResources` (in `resourceManager.ts`):

| Field                     | Format        | Usage                                | Notes                                                                                                                                         |
| ------------------------- | ------------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `tierTexture`             | `r32uint`     | `STORAGE_BINDING`                    | Per-pixel tier byte (1/2/4). Written by sample-budget, read by next-frame RIS.                                                                |
| `resolvedRadianceTexture` | `rgba16float` | `STORAGE_BINDING \| TEXTURE_BINDING` | Output of resolve pass; replaces composite's current input.                                                                                   |
| `motionVectorTexture`     | `rg32float`   | `STORAGE_BINDING`                    | Written by a new G-buffer motion-vector pass (or host-provided). If not available, pass 1×1 zero texture (resolve falls back to zero-motion). |

The `varianceBuffer` field is already in `FrameResources` from Sprint 9.

### 5. New render-frame dispatch order in `WalkaroundGPUPipeline.renderFrame()`

Current dispatch order (7 passes):

1. RIS
2. Temporal reuse
3. Spatial reuse × 2
4. Shade + GI
5. À-trous denoiser × 3 iterations
6. Temporal accumulation (accum)
7. Composite blit

New dispatch order with adaptive sampling (10 passes):

1. RIS (reads prev-frame `tierTexture` to dispatch tier-appropriate rays per pixel)
2. Temporal reuse
3. Spatial reuse × 2
4. Shade + GI → writes radiance to `hdrColorTexture`, writes variance update to `varianceBuffer`
5. À-trous denoiser × 3 iterations
6. Temporal accumulation (accum) → writes to `accumTextureA/B`
7. **NEW** Sample-budget pass → reads `varianceBuffer`, writes `tierTexture` for next frame
8. **NEW** Resolve pass → reads `hdrColorTexture` (current) + `accumTextureA/B` (prev) + `motionVectorTexture`, writes `resolvedRadianceTexture`
9. Composite blit → reads `resolvedRadianceTexture` instead of accumTexture directly

> The variance write in step 4 requires a WGSL change to the shade pass or a new
> dedicated "variance-update" compute pass after shade. The simplest approach:
> after shading, run a tiny 1-thread-per-pixel pass that reads `hdrColorTexture`
> luminance and calls `welfordUpdate` on the existing `varianceBuffer` texel,
> then stores back. This pass is not in Sprint 9 — it's the first piece of Sprint 10a.

### 6. Variance buffer write path (Sprint 10a prerequisite)

The `varianceBuffer` is allocated but never written in Sprint 9. Sprint 10a SVGF
will add the write path as part of its temporal accumulation upgrade. The Sprint 10a
implementation should:

1. Add a `varianceUpdateKernel` compute pass (or inline into the SVGF temporal pass).
2. For each pixel, call `welfordUpdate(prevState, luminance(shade_output), sampleCount)`.
3. Store the updated `WelfordVariance` back to `varianceBuffer`.

The `WelfordVariance` struct layout in `common.wgsl.ts` is pinned at `@version 1`
(Decision 13). Do not change the field order or add fields without bumping the
version comment and auditing all consumers.

### 7. RIS shader changes for adaptive dispatch (deferred)

When the per-pixel `tierTexture` is available, the RIS shader needs to read it
and conditionally skip ray casting for high-confidence pixels (tier=1):

```wgsl
// In ris.wgsl — at the top of the main kernel, after computing (px, py):
let tier = textureLoad(t_tier, vec2<u32>(px, py)).r;
if (tier == 1u) {
  // Pixel is converged — skip RIS, preserve existing reservoir.
  return;
}
// tier == 2 → cast 2 candidate rays (existing path, current candidate count)
// tier == 4 → cast 4 candidate rays (increase M in the RIS loop)
```

This change is NOT applied in Sprint 9. The RIS shader currently dispatches
uniformly across all pixels regardless of variance.

## Motion vector availability

Motion vectors are not yet wired in the walkaround pipeline. The `resolveKernel`
handles this gracefully: if `t_motion_vectors` is a 1×1 texture, the shader
detects this via `textureDimensions(t_motion_vectors)` and falls back to
zero-motion reprojection (gap pixels copy from the same screen position in the
previous frame).

When motion vectors become available (e.g. from a G-buffer velocity pass), bind
the real texture at group 0 binding 3. The shader automatically uses them without
any conditional change at the WGSL level.

## Test coverage added in Sprint 9

See `__tests__/sprint9-welford.test.ts` for:

- WelfordVariance struct presence in `COMMON_WGSL`
- `welfordUpdate` and `welfordVariance` function presence
- RG32Float layout assertion (8 bytes per texel)
- Sample-budget shader entry point and binding presence
- Resolve shader entry point and binding presence
- `createVarianceBuffer` return-type and format assertion (mocked GPUDevice)
- `FrameResources.varianceBuffer` presence and typing

## Decision 13 reminder

> "Layout pinned here. All future sprints that need per-pixel variance MUST
> import this struct from COMMON_WGSL rather than declaring their own."

Sprint 10a SVGF, Sprint 11 PPG, and Sprint 13 neural denoiser all need variance
state. All three should read `WelfordVariance` from `COMMON_WGSL`. If any sprint
needs to extend the struct (e.g. add a sample count field), the extension must be
proposed as a layout change with a version bump, not a new parallel struct.
