/**
 * temporalAccumBindings.ts — TypeScript UBO helper for the temporal
 * accumulation pass.
 *
 * Mirrors `AccumUBO` in `wgsl/temporalAccum.wgsl.ts`:
 *
 *   struct AccumUBO {
 *     alpha: f32,  // [0, 1] blend weight on the current frame
 *     _pad1: f32,
 *     _pad2: f32,
 *     _pad3: f32,
 *   };
 *
 * Layout (byte-identical to the WGSL struct):
 *   offset  0 — alpha : f32  (4 bytes)
 *   offset  4 — _pad1 : f32  (4 bytes, zero-filled by pack)
 *   offset  8 — _pad2 : f32  (4 bytes, zero-filled by pack)
 *   offset 12 — _pad3 : f32  (4 bytes, zero-filled by pack)
 * Total: 16 bytes (the single active scalar rounds up to the WebGPU 16-byte
 * minimum uniform-binding size; `defineUbo.pack` zero-fills the pad slots).
 *
 * Pattern follows `atrousVarianceBindings.ts` (W2-C13: `defineUbo` replaces
 * any hand-rolled DataView packing for small UBOs).
 *
 * Note for wiring: `walkaround-hybrid`'s `bindGroupBuilders.ts` already
 * packs this UBO via an inline `defineUbo([{ name:'alpha', type:'f32' }])`
 * called `ACCUM_UBO`. That consumer can be switched to import
 * `TEMPORAL_ACCUM_UBO_SIZE_BYTES` + `packTemporalAccumUniforms` from here
 * once the two are verified byte-identical (they are — both are single f32
 * padded to 16 bytes by `defineUbo`). The switch is a separate task; this
 * file provides the canonical helper for that migration.
 */

import { defineUbo } from '@vitrum/shared-samplers';

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/**
 * Uniforms for the temporal accumulation pass.
 *
 * `alpha` is the EMA blend weight applied to the current frame:
 *   accum = current × alpha + clamped_prev × (1 − alpha)
 *
 * A value of 1.0 discards history entirely (used on large camera motion).
 * Typical runtime values: 0.02–0.1 for stable convergence.
 */
export interface TemporalAccumUniforms {
  /**
   * EMA blend weight on the current frame, in [0, 1].
   * 1.0 = discard all history; 0.0 = pure history (no new contribution).
   */
  readonly alpha: number;
}

// ────────────────────────────────────────────────────────────────────────────
// Codegen
// ────────────────────────────────────────────────────────────────────────────

/**
 * `AccumUBO` codegen — single source of truth for size, layout, and pack.
 *
 * Layout (byte-identical to the WGSL struct):
 *   offset 0  — alpha : f32  (4 bytes)
 *   offset 4  — _pad1 : f32  (zero-filled)
 *   offset 8  — _pad2 : f32  (zero-filled)
 *   offset 12 — _pad3 : f32  (zero-filled)
 * Total: 16 bytes.
 */
const TEMPORAL_ACCUM_UBO = defineUbo([
  { name: 'alpha', type: 'f32' },
] as const);

/**
 * Byte size of the AccumUBO struct (16 bytes — the WebGPU minimum uniform-
 * binding size; the single f32 is padded by `defineUbo` to this floor).
 */
export const TEMPORAL_ACCUM_UBO_SIZE_BYTES = TEMPORAL_ACCUM_UBO.sizeBytes;

/**
 * Pack TemporalAccumUniforms into an ArrayBuffer at the given byte offset.
 *
 * The three trailing pad slots are deterministically zero-filled by
 * `defineUbo.pack`, so callers need not pre-zero the destination region.
 *
 * @param u       - Uniform values to pack.
 * @param target  - Destination ArrayBuffer (must be ≥ offset + TEMPORAL_ACCUM_UBO_SIZE_BYTES).
 * @param offset  - Byte offset into target (default: 0).
 */
export function packTemporalAccumUniforms(
  u: TemporalAccumUniforms,
  target: ArrayBuffer,
  offset = 0,
): void {
  TEMPORAL_ACCUM_UBO.pack(new DataView(target), offset, { alpha: u.alpha });
}
