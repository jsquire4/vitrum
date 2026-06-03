/**
 * UBO struct layouts for the small per-pass uniform buffers.
 *
 * Theme E (Task 2.1, #6) — GTAOPass / ResolvePass / SampleBudgetPass each
 * inlined a `defineUbo([...])` literal that mirrors the matching `*.wgsl.ts`
 * uniform struct (kept in sync by a comment). These are hoisted here so the
 * single TS definition is shared by the pass + readable next to the other
 * layouts, eliminating the per-pass copy-paste.
 *
 * NOTE ON CO-LOCATION: the ideal home for each `defineUbo` would be its own
 * shader module (`src/shaders/*.wgsl.ts`), so the TS layout sits beside the
 * WGSL struct it mirrors. Until that move is made, the layouts are
 * consolidated here under `passes/`. Each layout's doc comment names its
 * source WGSL struct + the byte size both must agree on.
 */

import { defineUbo } from '@vitrum/shared-samplers';

/**
 * `GTAOUniforms` (mirrors `src/shaders/gtao.wgsl.ts` +
 * `src/shaders/gtaoUpsample.wgsl.ts`): 6 active f32 + 2 explicit f32 pad =
 * 32 bytes. `gtaoDownscale` (2 = half-res, 4 = quarter-res) was the former
 * inert `_pad0`; both gtao + gtaoUpsample read it to map between the AO grid
 * and full-res coords. The 32-byte size is what `resourceManager` allocates
 * and both shaders re-bind.
 */
export const GTAO_UBO = defineUbo([
  { name: 'tanFovHalf',          type: 'f32' },
  { name: 'radiusPx',            type: 'f32' },
  { name: 'intensity',           type: 'f32' },
  { name: 'depthThresh',         type: 'f32' },
  { name: 'bilateralDepthSigma', type: 'f32' },
  { name: 'gtaoDownscale',       type: 'f32' },
  { name: '_pad1',               type: 'f32' },
  { name: '_pad2',               type: 'f32' },
] as const);

/**
 * `ResolveUniforms` (mirrors `src/shaders/resolve.wgsl.ts`): 4×u32 = 16 B.
 * screenW, screenH, frameParity, checkerboardOn. frameParity high-bit-flips
 * the chroma kernel offset; checkerboardOn=0 means full-density passthrough.
 */
export const RESOLVE_UBO = defineUbo([
  { name: 'screenW',        type: 'u32' },
  { name: 'screenH',        type: 'u32' },
  { name: 'frameParity',    type: 'u32' },
  { name: 'checkerboardOn', type: 'u32' },
] as const);

/**
 * `SampleBudgetUniforms` (mirrors `src/shaders/sampleBudget.wgsl.ts`):
 * 2×f32 + 2×u32 = 16 B. Adaptive-sampling thresholds + screen extent for
 * tier classification (f32 @0/4, u32 @8/12).
 */
export const SAMPLE_BUDGET_UBO = defineUbo([
  { name: 'thresholdLow',  type: 'f32' },
  { name: 'thresholdHigh', type: 'f32' },
  { name: 'screenW',       type: 'u32' },
  { name: 'screenH',       type: 'u32' },
] as const);

/**
 * `SampleCountUniforms` (mirrors `src/shaders/sampleBudget.wgsl.ts`):
 * 1×u32 + 3 trailing pad u32 = 16 B floor. Per-frame 1-based sample count;
 * `defineUbo` zero-fills bytes 4..15 to match the prior `[count, 0, 0, 0]`
 * write.
 */
export const SAMPLE_COUNT_UBO = defineUbo([
  { name: 'sampleCount', type: 'u32' },
] as const);
