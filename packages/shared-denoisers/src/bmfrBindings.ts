/**
 * bmfrBindings.ts — UBO struct + packer for the BMFR per-block regression kernel.
 *
 * BMFR = Koskela et al. 2019, "Blockwise Multi-Order Feature Regression for
 * Real-Time Path-Tracing Reconstruction" (ACM TOG 38(5)).
 *
 * Generated via `defineUbo` (single source of truth for size + std140 layout
 * + pack), matching the convention established by `svgfRealBindings.ts`.
 */

import { defineUbo } from '@vitrum/shared-samplers';
import {
  BMFR_BLOCK_SIZE,
} from './bmfrRegression.js';
import {
  BMFR_DEFAULT_BLOCK_STRIDE,
  BMFR_DEFAULT_POSITION_SCALE,
  BMFR_DEFAULT_TEMPORAL_ALPHA,
} from './bmfrConstants.js';

export interface BmfrUniforms {
  /** Square block edge length in pixels (default BMFR_BLOCK_SIZE = 32). */
  readonly blockSize: number;
  /**
   * Block grid stride in pixels. With stride < blockSize the blocks overlap;
   * each fit writes a private coefficient record and the resolve pass averages
   * every covering block deterministically. Default = half a block.
   */
  readonly blockStride: number;
  /** World-space normalisation scale for the squared position features. */
  readonly positionScale: number;
  /** Temporal-accumulation EMA weight on the current reconstructed frame. */
  readonly temporalAlpha: number;
  /** Tikhonov loading represented as sqrt(lambda) identity rows in direct QR. */
  readonly regularisation: number;
  /**
   * 0 = first frame / history invalid (output = reconstructed only),
   * 1 = blend with history texture. Packed as f32 (WGSL reads `> 0.5`).
   */
  readonly hasHistory: number;
  /**
   * Position-feature source:
   *   0 = world position (worldPos texture .xyz, .w = validity) — one-shot host.
   *   1 = screen-space proxy: position = (pixelX, pixelY, depth); the host binds
   *       a depth texture at the worldPos slot and the kernel reads depth from
   *       .w (= gNormalDepth depth channel in the walkaround pipeline). Lets the
   *       realtime pipeline skip a dedicated world-position G-buffer.
   * Default 0.
   */
  readonly positionMode: number;
}

const BMFR_UBO = defineUbo([
  { name: 'blockSize',      type: 'u32' },
  { name: 'blockStride',    type: 'u32' },
  { name: 'positionScale',  type: 'f32' },
  { name: 'temporalAlpha',  type: 'f32' },
  { name: 'regularisation', type: 'f32' },
  { name: 'hasHistory',     type: 'f32' },
  { name: 'positionMode',   type: 'u32' },
] as const);

/** BMFR UBO byte size (std140-padded to the 16-byte WebGPU minimum-binding floor). */
export const BMFR_UNIFORMS_SIZE_BYTES = BMFR_UBO.sizeBytes;

export const BMFR_DEFAULT_UNIFORMS: BmfrUniforms = {
  blockSize: BMFR_BLOCK_SIZE,
  blockStride: BMFR_DEFAULT_BLOCK_STRIDE,
  positionScale: BMFR_DEFAULT_POSITION_SCALE,
  temporalAlpha: BMFR_DEFAULT_TEMPORAL_ALPHA,
  regularisation: 1e-3,
  hasHistory: 0,
  positionMode: 0,
} as const;

/** Pack BmfrUniforms into an ArrayBuffer (defaults zero-filled by `pack`). */
export function packBmfrUniforms(u: BmfrUniforms, target: ArrayBuffer, offset = 0): void {
  const view = new DataView(target);
  BMFR_UBO.pack(view, offset, {
    blockSize: u.blockSize >>> 0,
    blockStride: u.blockStride >>> 0,
    positionScale: u.positionScale,
    temporalAlpha: u.temporalAlpha,
    regularisation: u.regularisation,
    hasHistory: u.hasHistory,
    positionMode: u.positionMode >>> 0,
  });
}
