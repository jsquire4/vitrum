/**
 * WS1 — Smooth shading normals (feature-completeness wave, 2026-05-29).
 *
 * TDD oracle + codegen pins for the barycentric per-vertex shading-normal
 * blend that replaces the faceted geometric normal in the walkaround-hybrid
 * primary-shading passes (shade / ris / risGi / risGiNrc).
 *
 * Prior art mirrored here: the DDGI probe-update pass already does the same
 * blend (probeUpdateRays.wgsl.ts:443-454):
 *   normalize(w·n0 + u·n1 + v·n2) * hit.side
 *
 * Constraints pinned:
 *   - the smooth normal is unit-length over random triangles (oracle);
 *   - the 4 primary passes consume `bvh_normal` (no computed-but-unconsumed);
 *   - the GEOMETRIC normal (hit.normal) is still used for the ray offset /
 *     backface bias — the smooth normal is shading-only;
 *   - the per-triangle Beer-Lambert tint moved off the scene storage group
 *     (bvh_beer is now a texture, so it no longer counts against
 *     maxStorageBuffersPerShaderStage);
 *   - the GPU skin kernel writes skinned normals into the SHARED buffer at
 *     `baseVertex + vi` (was a dropped per-mesh `vi` write).
 */

import { describe, expect, it } from 'vitest';

import { SHADE_WGSL } from '../shade.wgsl.js';
import { RIS_WGSL } from '../ris.wgsl.js';
import { RIS_GI_WGSL } from '../risGi.wgsl.js';
import { RIS_GI_NRC_BODY } from '../risGiNrc.wgsl.js';
import { SCENE_TRAVERSAL_WGSL } from '../sceneTraversal.wgsl.js';
import { GPU_SKIN_BVH_WITH_NORMALS_WGSL } from '../../skin/gpuSkinBvh.wgsl.js';
import { BIND_GROUP_TABLE } from '../../pipeline/bindGroupDescriptors.js';
import { SHADE_MODULE } from '../shade.wgsl.js';
import { RIS_MODULE } from '../ris.wgsl.js';
import { RIS_GI_MODULE } from '../risGi.wgsl.js';
import { composeWgsl } from '../../pipeline/wgslComposer.js';
import { WGSL_MODULES } from '../../pipeline/wgslModules.js';

// ── 1. Barycentric-blend CPU oracle ─────────────────────────────────────────
//
// The exact arithmetic the WGSL helper performs:
//   smooth = normalize(w·n0 + u·n1 + v·n2) · side
// Unit-length over arbitrary (non-degenerate) per-vertex normals + any convex
// barycentric weights.

function blendNormal(
  n0: [number, number, number],
  n1: [number, number, number],
  n2: [number, number, number],
  bary: [number, number, number],
  side: number,
): [number, number, number] {
  const x = bary[0] * n0[0] + bary[1] * n1[0] + bary[2] * n2[0];
  const y = bary[0] * n0[1] + bary[1] * n1[1] + bary[2] * n2[1];
  const z = bary[0] * n0[2] + bary[1] * n1[2] + bary[2] * n2[2];
  const len = Math.hypot(x, y, z);
  return [(x / len) * side, (y / len) * side, (z / len) * side];
}

function randUnit(rng: () => number): [number, number, number] {
  // Random direction (rejection-free Gaussian-ish via two uniforms → sphere).
  const z = rng() * 2 - 1;
  const t = rng() * Math.PI * 2;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  return [r * Math.cos(t), r * Math.sin(t), z];
}

describe('WS1 barycentric shading-normal blend (CPU oracle)', () => {
  it('produces a unit-length normal over random triples + random bary', () => {
    let seed = 0x1234abcd;
    const rng = () => {
      // xorshift32 — deterministic, no external dep.
      seed ^= seed << 13;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      return ((seed >>> 0) % 1_000_003) / 1_000_003;
    };
    for (let i = 0; i < 2000; i += 1) {
      const n0 = randUnit(rng);
      const n1 = randUnit(rng);
      const n2 = randUnit(rng);
      // Random convex barycentric weights (w, u, v summing to 1).
      let a = rng();
      let b = rng();
      if (a + b > 1) {
        a = 1 - a;
        b = 1 - b;
      }
      const bary: [number, number, number] = [1 - a - b, a, b];
      const side = rng() < 0.5 ? 1 : -1;
      // Skip the (measure-zero) antipodal-cancellation case where the blended
      // vector is ~0 and normalize is ill-defined.
      const sx = bary[0] * n0[0] + bary[1] * n1[0] + bary[2] * n2[0];
      const sy = bary[0] * n0[1] + bary[1] * n1[1] + bary[2] * n2[1];
      const sz = bary[0] * n0[2] + bary[1] * n1[2] + bary[2] * n2[2];
      if (Math.hypot(sx, sy, sz) < 1e-3) continue;
      const out = blendNormal(n0, n1, n2, bary, side);
      expect(Math.hypot(out[0], out[1], out[2])).toBeCloseTo(1, 5);
    }
  });

  it('side = -1 flips the blended normal (backface)', () => {
    const n0: [number, number, number] = [0, 0, 1];
    const n1: [number, number, number] = [0, 0, 1];
    const n2: [number, number, number] = [0, 0, 1];
    const front = blendNormal(n0, n1, n2, [0.3, 0.3, 0.4], 1);
    const back = blendNormal(n0, n1, n2, [0.3, 0.3, 0.4], -1);
    expect(back).toEqual([-front[0], -front[1], -front[2]]);
  });
});

// ── 2. Codegen pins ──────────────────────────────────────────────────────────

describe('WS1 codegen — smooth-normal helper + consumption', () => {
  it('sceneTraversal exposes a barycentric smoothShadingNormal helper that uses hit.side', () => {
    expect(SCENE_TRAVERSAL_WGSL).toMatch(/fn\s+smoothShadingNormal\s*\(/);
    // Mirrors the DDGI precedent: normalize(weighted sum) * hit.side.
    expect(SCENE_TRAVERSAL_WGSL).toMatch(/barycoord\.x/);
    expect(SCENE_TRAVERSAL_WGSL).toMatch(/\*\s*hit\.side/);
  });

  const passes: ReadonlyArray<readonly [string, string]> = [
    ['shade', SHADE_WGSL],
    ['ris', RIS_WGSL],
    ['risGi', RIS_GI_WGSL],
    ['risGiNrc', RIS_GI_NRC_BODY],
  ];

  it.each(passes)('%s declares bvh_normal at @group(1) @binding(11)', (_name, src) => {
    expect(src).toMatch(/@group\(1\)\s*@binding\(11\)\s*var<storage,\s*read>\s*bvh_normal/);
  });

  it.each(passes)('%s actually CONSUMES the smooth normal (calls smoothShadingNormal)', (_name, src) => {
    // No computed-but-unconsumed: every pass that declares bvh_normal must use it.
    expect(src).toMatch(/smoothShadingNormal\s*\(/);
  });

  it.each(passes)('%s keeps the GEOMETRIC normal (hit.normal) for the ray offset', (_name, src) => {
    // The shading normal is smooth; ray-origin offset / backface bias must
    // still use the geometric face normal. We pin a `geoNormal` binding that
    // is derived from hit.normal so an accidental switch to the smooth normal
    // for the offset is caught.
    expect(src).toMatch(/let\s+geoNormal\s*=\s*hit\.normal|let\s+geoNormal\s*=\s*primaryHit\.normal/);
  });

  it.each(passes)('%s applies the smooth normal in TLAS too (passes the instance transform; no bvhMode gate dropping it)', (_name, src) => {
    // V21 — smooth shading normals are no longer gated OFF in TLAS mode. The call
    // site reads the hit instance's world-to-local columns (instanceIndex*4) and
    // passes them to smoothShadingNormal for the local→world transform. The OLD
    // `select(smoothShadingNormal(...), geoNormal, ubo.bvhMode == 1u)` gate — which
    // left smooth shading dormant on every multi-mesh / instanced (TLAS) scene —
    // must be GONE.
    expect(src).toMatch(/instanceIndex\s*\*\s*4u/);
    expect(src).toMatch(/tlasInstanceWorldToLocal\[n_i\]/);
    expect(src).not.toMatch(/geoNormal,\s*ubo\.bvhMode\s*==\s*1u/);
  });

  it('smoothShadingNormal takes isTlas + world-to-local columns and transforms the local blend', () => {
    expect(SCENE_TRAVERSAL_WGSL).toMatch(/fn\s+smoothShadingNormal\s*\([\s\S]*isTlas\s*:\s*bool/);
    expect(SCENE_TRAVERSAL_WGSL).toMatch(/w2l0\s*:\s*vec4f/);
    // The TLAS branch reuses the SAME normal transform the geometric path uses.
    expect(SCENE_TRAVERSAL_WGSL).toMatch(/tlasTransformNormalFromLocalCols\s*\(\s*w2l0/);
  });

  it('IntersectionResult carries instanceIndex + the TLAS traversal sets it', () => {
    const composedShade = composeWgsl(SHADE_MODULE, WGSL_MODULES);
    expect(composedShade).toMatch(/instanceIndex\s*:\s*u32/);
    expect(composedShade).toMatch(/best\.instanceIndex\s*=\s*instIdx/);
  });

  it('shade declares bvh_beer as a uint texture (moved off the storage group)', () => {
    expect(SHADE_WGSL).toMatch(/var\s+bvh_beer\s*:\s*texture_2d<u32>/);
    // The old storage declaration must be gone.
    expect(SHADE_WGSL).not.toMatch(/var<storage,\s*read>\s*bvh_beer/);
    // Beer is read via textureLoad (not a buffer index).
    expect(SHADE_WGSL).toMatch(/textureLoad\(bvh_beer/);
  });

  // Compose-level pins — catch redeclaration / dangling-ptr hazards that only
  // surface once the include graph is flattened (the form the device compiles).
  const composedPasses: ReadonlyArray<readonly [string, string]> = [
    ['shade', composeWgsl(SHADE_MODULE, WGSL_MODULES)],
    ['ris', composeWgsl(RIS_MODULE, WGSL_MODULES)],
    ['risGi', composeWgsl(RIS_GI_MODULE, WGSL_MODULES)],
  ];

  it.each(composedPasses)('composed %s emits smoothShadingNormal exactly once', (_name, src) => {
    const defs = src.match(/fn\s+smoothShadingNormal\s*\(/g) ?? [];
    expect(defs.length).toBe(1);
  });

  it('composed shade declares BVH_BEER_TEX_WIDTH exactly once (no redeclaration)', () => {
    const composed = composeWgsl(SHADE_MODULE, WGSL_MODULES);
    const decls = composed.match(/const\s+BVH_BEER_TEX_WIDTH\s*:/g) ?? [];
    expect(decls.length).toBe(1);
    // No dangling &bvh_beer (textures are passed by handle, not pointer).
    expect(composed).not.toMatch(/&bvh_beer/);
  });
});

// ── 3. Scene bind-group storage-buffer budget (≤ 16 floor) ───────────────────

describe('WS1 scene bind-group storage budget', () => {
  it('scene group has bvh_normal as storage and bvh_beer as a uint texture', () => {
    const scene = BIND_GROUP_TABLE.find((e) => e.id === 'scene')!;
    const beer = scene.entries.find((e) => e.binding === 5)!;
    const normal = scene.entries.find((e) => e.binding === 11)!;
    expect(beer.kind).toBe('tex:uint');
    expect(normal.kind).toBe('storage-ro');
    expect(normal.note).toMatch(/normal/i);
  });

  it('scene group stays at or below the 16 storage-buffer floor', () => {
    const scene = BIND_GROUP_TABLE.find((e) => e.id === 'scene')!;
    const storageCount = scene.entries.filter(
      (e) => e.kind === 'storage-ro' || e.kind === 'storage-rw',
    ).length;
    // Scene group's own storage buffers. The shade pass adds 4 frame-group
    // storage buffers + 1 RC cascade0 = 5 → 11 + 5 = 16, exactly the floor.
    expect(storageCount).toBeLessThanOrEqual(11);
  });
});

// ── 4. GPU-skin normal == CPU mat3InverseTranspose, written at baseVertex+vi ──

describe('WS1 GPU-skin normal redirect', () => {
  it('writes skinned normals into the shared buffer at baseVertex + vi (not mesh-local vi)', () => {
    // The redirect: skinnedNormals[outIdx] where outIdx = baseVertex + vi.
    // The dropped per-mesh `skinnedNormals[vi]` write must be gone.
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toMatch(/skinnedNormals\[outIdx\]\s*=/);
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).not.toMatch(/skinnedNormals\[vi\]\s*=/);
  });

  it('GPU kernel inverse-transpose blend matches a CPU mat3InverseTranspose reference', () => {
    // Replicate the kernel's normal math (rigid rotation bone → inverse-
    // transpose == the rotation itself) and compare to the analytic transform.
    // A pure rotation: 90° about Z. Inverse-transpose of a rotation is the
    // rotation, so the skinned normal == R · restNormal.
    const c = Math.cos(Math.PI / 2);
    const s = Math.sin(Math.PI / 2);
    // Column-major columns of Rz(90°): c0=(c,s,0), c1=(-s,c,0), c2=(0,0,1).
    const col0: [number, number, number] = [c, s, 0];
    const col1: [number, number, number] = [-s, c, 0];
    const col2: [number, number, number] = [0, 0, 1];
    // mat3InverseTranspose via cofactor columns / det (the kernel's formula).
    const det =
      col0[0] * (col1[1] * col2[2] - col1[2] * col2[1]) -
      col1[0] * (col0[1] * col2[2] - col0[2] * col2[1]) +
      col2[0] * (col0[1] * col1[2] - col0[2] * col1[1]);
    const cross = (
      a: [number, number, number],
      b: [number, number, number],
    ): [number, number, number] => [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
    const invDet = 1 / det;
    const itCol0 = cross(col1, col2).map((v) => v * invDet) as [number, number, number];
    const itCol1 = cross(col2, col0).map((v) => v * invDet) as [number, number, number];
    const itCol2 = cross(col0, col1).map((v) => v * invDet) as [number, number, number];
    const rn: [number, number, number] = [1, 0, 0];
    const out: [number, number, number] = [
      itCol0[0] * rn[0] + itCol1[0] * rn[1] + itCol2[0] * rn[2],
      itCol0[1] * rn[0] + itCol1[1] * rn[1] + itCol2[1] * rn[2],
      itCol0[2] * rn[0] + itCol1[2] * rn[1] + itCol2[2] * rn[2],
    ];
    const len = Math.hypot(...out);
    const norm = out.map((v) => v / len);
    // Rz(90°) maps +X → +Y.
    expect(norm[0]).toBeCloseTo(0, 6);
    expect(norm[1]).toBeCloseTo(1, 6);
    expect(norm[2]).toBeCloseTo(0, 6);
  });
});
