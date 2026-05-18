/**
 * svgfRealBindings.ts — TypeScript helpers for the real Schied 2017 SVGF pipeline.
 *
 * Provides typed UBO structs and ArrayBuffer packers for the three new SVGF
 * compute passes (reprojection, variance-from-moments, 7×7 spatial fallback).
 *
 * std140 layout rules apply: each field is 4-byte aligned; structs are
 * padded to a multiple of 16 bytes.
 */

import {
  SVGF_REAL_DEFAULT_ALPHA_MIN,
  SVGF_REAL_DEFAULT_SIGMA_DEPTH,
  SVGF_REAL_DEFAULT_SIGMA_NORMAL,
} from './svgfRealConstants.js';

// ============================================================
// SVGFReprojUBO — reprojection + EMA tuning
// ============================================================

export interface SVGFReprojUniforms {
  /** σ_z: max relative depth deviation for disocclusion test (Schied Eq. 2). Default 0.10. */
  readonly sigmaDepth: number;
  /** σ_n: minimum normal dot-product for acceptance (Schied Eq. 2). Default 0.95. */
  readonly sigmaNormal: number;
  /** α_min: minimum EMA weight (Schied Eq. 4). Default 0.05. */
  readonly alphaMin: number;
}

/**
 * SVGFReprojUBO byte size.
 *
 * Layout:
 *   offset 0  — sigmaDepth  : f32  (4 bytes)
 *   offset 4  — sigmaNormal : f32  (4 bytes)
 *   offset 8  — alphaMin    : f32  (4 bytes)
 *   offset 12 — _pad        : u32  (4 bytes — alignment)
 * Total: 16 bytes.
 */
export const SVGF_REPROJ_UNIFORMS_SIZE_BYTES = 16 as const;

export const SVGF_REPROJ_DEFAULT_UNIFORMS: SVGFReprojUniforms = {
  sigmaDepth: SVGF_REAL_DEFAULT_SIGMA_DEPTH,
  sigmaNormal: SVGF_REAL_DEFAULT_SIGMA_NORMAL,
  alphaMin: SVGF_REAL_DEFAULT_ALPHA_MIN,
} as const;

/**
 * Pack SVGFReprojUniforms into an ArrayBuffer.
 *
 * @param u      - Uniform values to pack.
 * @param target - Destination ArrayBuffer (must be ≥ offset + SVGF_REPROJ_UNIFORMS_SIZE_BYTES).
 * @param offset - Byte offset into target (default: 0).
 */
export function packSVGFReprojUniforms(
  u: SVGFReprojUniforms,
  target: ArrayBuffer,
  offset = 0,
): void {
  const view = new DataView(target, offset, SVGF_REPROJ_UNIFORMS_SIZE_BYTES);
  view.setFloat32(0, u.sigmaDepth, true);
  view.setFloat32(4, u.sigmaNormal, true);
  view.setFloat32(8, u.alphaMin, true);
  view.setUint32(12, 0, true); // _pad
}
