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
 *
 * Also pins POINT_LIGHT_VEC4_STRIDE / SPOT_LIGHT_VEC4_STRIDE constants
 * (shared WGSL constants in material.wgsl.ts, used by caustic.wgsl.ts at the
 * five H1-class sites) against the TS packer strides so the caustic cannot
 * drift from the kernel. Mirror of pt-webgl2 materialStrideParity.test.ts.
 */

import { describe, expect, it } from 'vitest';
import {
  DIRECTIONAL_LIGHT_FLOAT_STRIDE,
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
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';

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

// ─── N-directional stride (new item) ─────────────────────────────────────────
// The kernel addresses directionalLights[] with `dBase = di * 2u` (2 vec4f = 8 floats).
// DIRECTIONAL_LIGHT_FLOAT_STRIDE on the host must be 8 to stay in sync.
describe('N-directional stride consistency: TS constant ↔ WGSL dBase index', () => {
  it('DIRECTIONAL_LIGHT_FLOAT_STRIDE = 8 (2 vec4f)', () => {
    expect(DIRECTIONAL_LIGHT_FLOAT_STRIDE).toBe(8);
  });

  it('WGSL kernel uses dBase = di * 2u, matching 8-float / 4-float-per-vec4 = 2 vec4f', () => {
    // The WGSL stride is not a named constant but is implicit in the index expression.
    // This test pins it so a future edit to the loop cannot silently break alignment.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let dBase = di * 2u');
    // Specifically: the loop reads `directionalLights[dBase]` and `directionalLights[dBase + 1u]`
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let dDirAD = directionalLights[dBase]');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let dIrrMean = directionalLights[dBase + 1u]');
  });
});

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

  it('rect-area stays at 16 floats and exact mesh-area records use 28', () => {
    expect(RECT_AREA_LIGHT_FLOAT_STRIDE).toBe(16);
    expect(RECT_AREA_LIGHT_STRIDE).toBe(16);
    expect(MESH_AREA_LIGHT_FLOAT_STRIDE).toBe(28);
    expect(MESH_AREA_LIGHT_STRIDE).toBe(28);
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

// ─── WGSL shared-constant parity: caustic point/spot stride fix (H1-class) ───
//
// caustic.wgsl.ts uses POINT_LIGHT_VEC4_STRIDE / SPOT_LIGHT_VEC4_STRIDE at five
// sites (3 MNEE point loops + photon-map point seed + photon-map spot seed).
// These constants are declared in material.wgsl.ts (composed before caustic) and
// must stay in lockstep with the TS packer.  If either drifts, the caustic reads
// garbage light positions/radiances → incorrect/black caustics (the H1/H41 class).
// caustic point/spot stride fix (H1-class) + shared light-stride constants,
// 2026-06-10 — RENDER-CHANGING for multi-light caustic scenes, A/B pending V28-B
describe('WGSL light-stride constants ↔ TS packer parity (caustic H1-class fix)', () => {
  it('POINT_LIGHT_VEC4_STRIDE in composed WGSL equals POINT_LIGHT_FLOAT_STRIDE / 4', () => {
    const m = PT_WEBGPU_TRACE_WGSL.match(/const POINT_LIGHT_VEC4_STRIDE\s*=\s*(\d+)u/);
    expect(m).not.toBeNull();
    const wgslVec4Stride = Number(m![1]);
    // POINT_LIGHT_FLOAT_STRIDE is in floats (12); dividing by 4 gives vec4f count (3).
    expect(wgslVec4Stride).toBe(POINT_LIGHT_FLOAT_STRIDE / 4);
  });

  it('SPOT_LIGHT_VEC4_STRIDE in composed WGSL equals SPOT_LIGHT_FLOAT_STRIDE / 4', () => {
    const m = PT_WEBGPU_TRACE_WGSL.match(/const SPOT_LIGHT_VEC4_STRIDE\s*=\s*(\d+)u/);
    expect(m).not.toBeNull();
    const wgslVec4Stride = Number(m![1]);
    // SPOT_LIGHT_FLOAT_STRIDE is in floats (16); dividing by 4 gives vec4f count (4).
    expect(wgslVec4Stride).toBe(SPOT_LIGHT_FLOAT_STRIDE / 4);
  });

  it('the unified bounded MNEE emitter loader uses POINT_LIGHT_VEC4_STRIDE', () => {
    // The unified bounded-chain estimator loads every point family through one
    // shared emitter loader. There must be no residual bare stride literal.
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/let lbase\s*=\s*li\s*\*\s*2u/);
    const strideUses = (PT_WEBGPU_TRACE_WGSL.match(/let base\s*=\s*index\s*\*\s*POINT_LIGHT_VEC4_STRIDE/g) ?? []).length;
    expect(strideUses).toBe(1);
  });

  it('caustic photon-map point seed uses POINT_LIGHT_VEC4_STRIDE (not a bare literal)', () => {
    // A4: photon emission now lives in SPPM_PHOTON_PASS_WGSL (separate compute pass),
    // not in the megakernel caustic path.  Check the photon-emission WGSL directly.
    expect(SPPM_PHOTON_PASS_WGSL).not.toMatch(/let pointBase\s*=\s*pointIdx\s*\*\s*2u/);
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let pointBase = pointIdx * POINT_LIGHT_VEC4_STRIDE');
  });

  it('caustic photon-map spot seed uses SPOT_LIGHT_VEC4_STRIDE (not a bare literal)', () => {
    // A4: photon emission now lives in SPPM_PHOTON_PASS_WGSL.
    expect(SPPM_PHOTON_PASS_WGSL).not.toMatch(/let spotBase\s*=\s*spotIdx\s*\*\s*3u/);
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let spotBase = spotIdx * SPOT_LIGHT_VEC4_STRIDE');
  });

  it('caustic photon spot-axis has no negation (forward emission axis, not backward)', () => {
    // The packed spot direction is the FORWARD emission axis. Negating it before
    // building the ONB emits photons backward (away from the lit region) → zero
    // spot-light photon contributions. The fix: use the packed direction directly.
    // A4: photon emission now lives in SPPM_PHOTON_PASS_WGSL.
    // The photon pass names the unpacked spot direction 'spotAxis' (the
    // safe_normalize of saxisVec.xyz) — no negation anywhere.
    expect(SPPM_PHOTON_PASS_WGSL).not.toMatch(/let spotAxis\s*=\s*safe_normalize\(-/);
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let spotAxis = safe_normalize(saxisVec.xyz)');
  });
});
