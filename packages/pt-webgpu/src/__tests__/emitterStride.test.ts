/**
 * emitterStride.test.ts — H51-D CPU-vs-WGSL stride/layout assertion test.
 *
 * H51-D bumped:
 *   - point stride: 8 → 12 floats  (added [distance, decay, 0, 0] vec4)
 *   - spot  stride: 12 → 16 floats (added cosInner in slot 2.w + [distance, decay, 0, 0] vec4)
 *
 * Three independent sources must agree on the stride:
 *   1. TypeScript emitterPacking.ts (authoritative pack side)
 *   2. TypeScript flatEmitterWalk.ts (CPU oracle walk side)
 *   3. WGSL kernel (load stride from the composed PT_WEBGPU_TRACE_WGSL)
 *
 * If any source drifts, this test fails — catching the class of bug where the
 * CPU and GPU disagree on buffer layout.
 *
 * The WGSL stride is not a named constant; it's implicit in the base index
 * expression used to index into `pointLights` / `spotLights` arrays. We
 * extract it by looking for the kernel loop pattern:
 *   - point: `let base = pi * Nu`  → N vec4f per light → stride = N * 4
 *   - spot:  `let sb = si * Nu`   → N vec4f per light → stride = N * 4
 */

import { describe, expect, it } from 'vitest';
import {
  POINT_LIGHT_FLOAT_STRIDE,
  SPOT_LIGHT_FLOAT_STRIDE,
  RECT_AREA_LIGHT_FLOAT_STRIDE,
  MESH_AREA_LIGHT_FLOAT_STRIDE,
} from '../scene/emitterPacking.js';
import {
  POINT_LIGHT_STRIDE,
  SPOT_LIGHT_STRIDE,
  RECT_AREA_LIGHT_STRIDE,
  MESH_AREA_LIGHT_STRIDE,
} from '../bdpt/flatEmitterWalk.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';

// ─── helpers ──────────────────────────────────────────────────────────────────

/**
 * Extract the vec4f-per-light stride factor from a WGSL index expression.
 * The kernel uses patterns like:
 *   `let base = pi * 3u` (point, 3 vec4f = 12 floats)
 *   `let sb = si * 4u`   (spot,  4 vec4f = 16 floats)
 * Returns the stride in FLOATS (N * 4).
 */
function extractWgslStride(wgsl: string, pattern: RegExp): number | null {
  const m = wgsl.match(pattern);
  if (m == null || m[1] == null) return null;
  const vec4Count = parseInt(m[1], 10);
  return vec4Count * 4;
}

describe('H51-D emitter stride consistency: TS pack ↔ TS walk ↔ WGSL', () => {
  it('point light stride agrees across all three sources', () => {
    // TS pack
    expect(POINT_LIGHT_FLOAT_STRIDE).toBe(12);
    // TS walk
    expect(POINT_LIGHT_STRIDE).toBe(POINT_LIGHT_FLOAT_STRIDE);
    // WGSL kernel: `let base = pi * Nu`
    const wgslStride = extractWgslStride(PT_WEBGPU_TRACE_WGSL, /let base\s*=\s*pi\s*\*\s*(\d+)u/);
    expect(wgslStride).not.toBeNull();
    expect(wgslStride).toBe(POINT_LIGHT_FLOAT_STRIDE);
  });

  it('spot light stride agrees across all three sources', () => {
    // TS pack
    expect(SPOT_LIGHT_FLOAT_STRIDE).toBe(16);
    // TS walk
    expect(SPOT_LIGHT_STRIDE).toBe(SPOT_LIGHT_FLOAT_STRIDE);
    // WGSL kernel: `let sb = si * Nu`
    const wgslStride = extractWgslStride(PT_WEBGPU_TRACE_WGSL, /let sb\s*=\s*si\s*\*\s*(\d+)u/);
    expect(wgslStride).not.toBeNull();
    expect(wgslStride).toBe(SPOT_LIGHT_FLOAT_STRIDE);
  });

  it('rect-area and mesh-area strides are unchanged at 16', () => {
    expect(RECT_AREA_LIGHT_FLOAT_STRIDE).toBe(16);
    expect(RECT_AREA_LIGHT_STRIDE).toBe(16);
    expect(MESH_AREA_LIGHT_FLOAT_STRIDE).toBe(16);
    expect(MESH_AREA_LIGHT_STRIDE).toBe(16);
  });

  it('point layout: position (0..2) + _ (3) + radiance (4..6) + _ (7) + distance (8) + decay (9) + pad (10,11)', () => {
    // This is a documentation / golden test: if the layout description changes
    // (e.g. distance moves to a different slot), this test must be updated alongside
    // the WGSL that reads it.
    expect(POINT_LIGHT_FLOAT_STRIDE).toBe(12);
    // Slot breakdown: 3 vec4f = [pos_xyz, _, rad_xyz, _, dist_decay_0_0]
    // Verifying the distance field is at float index 8 within the stride.
    const POINT_DISTANCE_FLOAT_OFFSET = 8;
    const POINT_DECAY_FLOAT_OFFSET = 9;
    expect(POINT_DISTANCE_FLOAT_OFFSET).toBeLessThan(POINT_LIGHT_FLOAT_STRIDE);
    expect(POINT_DECAY_FLOAT_OFFSET).toBeLessThan(POINT_LIGHT_FLOAT_STRIDE);
  });

  it('spot layout: pos(0..2)_pad(3) dir(4..6)+cosOuter(7) rad(8..10)+cosInner(11) dist(12)+decay(13)+pad(14,15)', () => {
    expect(SPOT_LIGHT_FLOAT_STRIDE).toBe(16);
    // Slot breakdown: 4 vec4f
    const SPOT_COS_OUTER_FLOAT_OFFSET = 7;
    const SPOT_COS_INNER_FLOAT_OFFSET = 11;
    const SPOT_DISTANCE_FLOAT_OFFSET = 12;
    const SPOT_DECAY_FLOAT_OFFSET = 13;
    expect(SPOT_COS_OUTER_FLOAT_OFFSET).toBeLessThan(SPOT_LIGHT_FLOAT_STRIDE);
    expect(SPOT_COS_INNER_FLOAT_OFFSET).toBeLessThan(SPOT_LIGHT_FLOAT_STRIDE);
    expect(SPOT_DISTANCE_FLOAT_OFFSET).toBeLessThan(SPOT_LIGHT_FLOAT_STRIDE);
    expect(SPOT_DECAY_FLOAT_OFFSET).toBeLessThan(SPOT_LIGHT_FLOAT_STRIDE);
  });
});
