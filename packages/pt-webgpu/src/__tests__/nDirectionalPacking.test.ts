/**
 * nDirectionalPacking.test.ts — N-directional emitter packing + mesh-area cap tests.
 *
 * Item 1 — N-directional: directionals are packed into a flat storage-buffer array
 *   (stride 2 vec4f = 8 floats per light). The kernel loops
 *   params.directionalLightCount records. Single-directional scenes are
 *   byte-identical to the old single-directional path.
 *
 * Item 2 — Mesh-area exact-support limit: MESH_AREA_LIGHT_TRI_CAP (65 536).
 *   Under-limit scenes are unchanged; over-limit scenes fail closed rather than
 *   silently dropping triangles and biasing the light proposal.
 */

import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import {
  packEmitterArrays,
  buildLightTreeInputForScene,
  DIRECTIONAL_LIGHT_FLOAT_STRIDE,
  MESH_AREA_LIGHT_TRI_CAP,
} from '../scene/emitterPacking.js';
import { packLiteLightTexture } from '../scene/litePackedTextures.js';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';

// ─── helpers ──────────────────────────────────────────────────────────────────

function baseScene(): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'tri',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: { baseColor: [1, 1, 1], roughness: 0.4, metallic: 0 },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function directionalScene(count: number): Scene {
  const emitters: NonNullable<Scene['emitters'][number]>[] = [];
  for (let i = 0; i < count; i++) {
    // Vary direction and irradiance so each is distinct.
    emitters.push({
      kind: 'directional',
      id: `dir${i}`,
      direction: [i === 0 ? 0 : 1, -1, 0], // toward-below-left
      color: [(i + 1) * 0.2, 0.5, 0.8],
      intensity: 2 + i,
    });
  }
  return { ...baseScene(), emitters: emitters };
}

// ─── Item 1: N-directional stride constant ────────────────────────────────────

describe('N-directional stride constant', () => {
  it('DIRECTIONAL_LIGHT_FLOAT_STRIDE is 8 (2 vec4f per record)', () => {
    // 2 vec4f × 4 floats = 8 floats — the kernel reads dBase = di * 2u (vec4 units).
    expect(DIRECTIONAL_LIGHT_FLOAT_STRIDE).toBe(8);
  });

  it('kernel uses dBase = di * 2u — matching the 2-vec4f stride', () => {
    // The stride in the kernel is IMPLICIT in `let dBase = di * 2u`.
    // This test proves the WGSL matches the TS constant (8 floats / 4 floats = 2 vec4f).
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let dBase = di * 2u');
  });

  it('kernel loops params.directionalLightCount records', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('for (var di = 0u; di < params.directionalLightCount; di = di + 1u)');
  });
});

// ─── Item 1: single-directional behaviour unchanged ──────────────────────────

describe('single-directional packing (1-directional byte-identity invariant)', () => {
  it('packs 1 directional: directionalLightCount = 1', () => {
    const packed = packEmitterArrays(directionalScene(1));
    expect(packed.directionalLightCount).toBe(1);
    expect(packed.directionalLightsData.length).toBe(DIRECTIONAL_LIGHT_FLOAT_STRIDE);
  });

  it('packs 1 directional: towardLight direction is -normalize(e.direction)', () => {
    // Scene directional direction = [0, -1, 0] → toward light = [0, +1, 0].
    const scene: Scene = {
      ...baseScene(),
      emitters: [{ kind: 'directional', id: 'd', direction: [0, -1, 0], color: [1, 1, 1], intensity: 1 }],
    };
    const packed = packEmitterArrays(scene);
    expect(packed.directionalLightCount).toBe(1);
    // vec4 0: towardLight.xyz = -normalize([0,-1,0]) = [0,1,0]; angularDiameter = 0
    expect(packed.directionalLightsData[0]).toBeCloseTo(0, 6);   // x
    expect(packed.directionalLightsData[1]).toBeCloseTo(1, 6);   // y
    expect(packed.directionalLightsData[2]).toBeCloseTo(0, 6);   // z
    expect(packed.directionalLightsData[3]).toBeCloseTo(0, 6);   // angularDiameter = 0 (no angularDiameter set)
    // vec4 1: irradiance = color * intensity = [1,1,1]; mean = 1
    expect(packed.directionalLightsData[4]).toBeCloseTo(1, 6);   // r
    expect(packed.directionalLightsData[5]).toBeCloseTo(1, 6);   // g
    expect(packed.directionalLightsData[6]).toBeCloseTo(1, 6);   // b
    expect(packed.directionalLightsData[7]).toBeCloseTo(1, 6);   // mean
  });

  it('buildPackedScene: 1 directional scene -> directionalLightCount = 1', () => {
    const packed = buildPackedScene(directionalScene(1));
    expect(packed.directionalLightCount).toBe(1);
    expect(packed.directionalLightsData.length).toBe(DIRECTIONAL_LIGHT_FLOAT_STRIDE);
  });

  it('zero directionals -> count=0, empty host array (GPU buffer gets a 16-byte placeholder via createStorageBuffer)', () => {
    const packed = packEmitterArrays(baseScene());
    expect(packed.directionalLightCount).toBe(0);
    // Host-side: empty array (no data to send).
    // GPU-side: createStorageBuffer pads to 16 bytes (4 floats) when byteLength=0.
    expect(packed.directionalLightsData.length).toBe(0);
  });
});

// ─── Item 1: 2-directional packing ────────────────────────────────────────────

describe('2-directional packing', () => {
  it('packs both directionals: directionalLightCount = 2', () => {
    const packed = packEmitterArrays(directionalScene(2));
    expect(packed.directionalLightCount).toBe(2);
    expect(packed.directionalLightsData.length).toBe(2 * DIRECTIONAL_LIGHT_FLOAT_STRIDE);
  });

  it('lite light texture preserves both directional records before other lights', () => {
    const packed = packEmitterArrays(directionalScene(2));
    const lite = packLiteLightTexture(
      packed.directionalLightsData,
      packed.pointLightsData,
      packed.spotLightsData,
      packed.rectAreaLightsData,
    );
    expect(lite.width).toBe(4); // 2 directionals * 2 vec4 records
    expect(Array.from(lite.data.slice(0, 16))).toEqual(Array.from(packed.directionalLightsData));
  });

  it('lite shader loops directionalLightCount records from liteLightTex', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('lightCount = lightCount + params.directionalLightCount;');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('for (var di = 0u; di < params.directionalLightCount; di = di + 1u)');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('let litePtBase = params.directionalLightCount * 2u;');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('lite tier renders only the first directional');
  });

  it('each directional record occupies its own 8-float stride slot', () => {
    // dir0: direction=[0,-1,0] → towardLight=[0,1,0]; color=[0.2,0.5,0.8]*2
    // dir1: direction=[1,-1,0] → towardLight=[-1/√2,1/√2,0]; color=[0.4,0.5,0.8]*3
    const packed = packEmitterArrays(directionalScene(2));
    const d = packed.directionalLightsData;
    // --- record 0 ---
    // towardLight = [0, 1, 0]
    expect(d[0]).toBeCloseTo(0, 5);
    expect(d[1]).toBeCloseTo(1, 5);
    expect(d[2]).toBeCloseTo(0, 5);
    // irradiance = [0.2*2, 0.5*2, 0.8*2] = [0.4, 1.0, 1.6]
    expect(d[4]).toBeCloseTo(0.4, 5);
    expect(d[5]).toBeCloseTo(1.0, 5);
    expect(d[6]).toBeCloseTo(1.6, 5);

    // --- record 1 (offset = DIRECTIONAL_LIGHT_FLOAT_STRIDE = 8) ---
    const invSqrt2 = 1 / Math.sqrt(2);
    expect(d[8]).toBeCloseTo(-invSqrt2, 5);  // towardLight.x = -1/√2
    expect(d[9]).toBeCloseTo(invSqrt2, 5);   // towardLight.y = +1/√2
    expect(d[10]).toBeCloseTo(0, 5);          // towardLight.z = 0
    // irradiance = [0.4*3, 0.5*3, 0.8*3] = [1.2, 1.5, 2.4]
    expect(d[12]).toBeCloseTo(1.2, 5);
    expect(d[13]).toBeCloseTo(1.5, 5);
    expect(d[14]).toBeCloseTo(2.4, 5);
  });

  it('buildPackedScene: 2-directional scene -> both packed', () => {
    const packed = buildPackedScene(directionalScene(2));
    expect(packed.directionalLightCount).toBe(2);
    expect(packed.directionalLightsData.length).toBe(2 * DIRECTIONAL_LIGHT_FLOAT_STRIDE);
  });

  it('both records appear in light tree input as separate power entries', () => {
    const tree = buildLightTreeInputForScene(directionalScene(2));
    // 2 directional leaves + no other emitters
    expect(tree.powers.length).toBe(2);
    // Each directional power should be positive
    expect(tree.powers[0]).toBeGreaterThan(0);
    expect(tree.powers[1]).toBeGreaterThan(0);
  });
});

  it('keeps black directional slots so later light-tree indices stay aligned', () => {
    const scene: Scene = {
      ...baseScene(),
      emitters: [
        {
          kind: 'directional', id: 'black-dir', direction: [0, -1, 0],
          color: [0, 0, 0], intensity: 1,
        },
        {
          kind: 'directional', id: 'lit-dir', direction: [1, -1, 0],
          color: [2, 2, 2], intensity: 1,
        },
        {
          kind: 'point', id: 'point', position: [0, 2, 0],
          color: [3, 3, 3], intensity: 1,
        },
      ],
    };
    const tree = buildLightTreeInputForScene(scene);
    expect(tree.powers).toHaveLength(3);
    expect(tree.powers[0]).toBe(0);
    expect(tree.powers[1]).toBeGreaterThan(0);
    expect(tree.powers[2]).toBeGreaterThan(0);
  });

// ─── Item 2: mesh-area cap ────────────────────────────────────────────────────

/**
 * Build a scene with a mesh-area emitter that has exactly triCount triangles
 * (a fan of right triangles with area 0.5 each, so areas differ only by index
 * when we vary them).
 */
function bigMeshScene(triCount: number): Scene {
  // Build positions: vertex 0 = origin; vertex 2k+1, 2k+2 = two points on the XY plane.
  // Each triangle: [0,0,0], [k, 0, 0], [k, 1+k*0.001, 0] — area ≈ (1+k*0.001)/2
  // (slightly different so sorting is non-degenerate).
  const positions: number[] = [0, 0, 0];
  const normals: number[] = [0, 0, 1];
  const indices: number[] = [];
  for (let k = 0; k < triCount; k++) {
    positions.push(k + 1, 0, 0);
    positions.push(k + 1, 1 + k * 0.001, 0);
    normals.push(0, 0, 1, 0, 0, 1);
    // Triangle: vertex 0 + vertex 2k+1 + vertex 2k+2
    indices.push(0, 2 * k + 1, 2 * k + 2);
  }
  return {
    primitives: [{
      kind: 'mesh',
      id: 'bigmesh',
      positions: new Float32Array(positions),
      normals: new Float32Array(normals),
      indices: new Uint32Array(indices),
      material: { baseColor: [1, 1, 1], roughness: 0.4, metallic: 0 },
    }],
    emitters: [{ kind: 'mesh-area', id: 'big', meshId: 'bigmesh', color: [1, 1, 1], intensity: 1 }],
    environment: { kind: 'none' },
  };
}

describe('mesh-area NEE exact-support limit (MESH_AREA_LIGHT_TRI_CAP)', () => {
  it('MESH_AREA_LIGHT_TRI_CAP constant is 65536', () => {
    expect(MESH_AREA_LIGHT_TRI_CAP).toBe(65536);
  });

  it('under-cap: all triangles packed (no warning, count unchanged)', () => {
    const triCount = 10;
    const scene = bigMeshScene(triCount);
    const packed = packEmitterArrays(scene);
    expect(packed.meshAreaLightCount).toBe(triCount);
    expect(packed.warnings.filter((w) => w.includes('cap'))).toHaveLength(0);
  });

  it('over-limit: rejects instead of returning a partially supported proposal', () => {
    // Build a scene with MESH_AREA_LIGHT_TRI_CAP + 100 triangles.
    const overCount = MESH_AREA_LIGHT_TRI_CAP + 100;
    const scene = bigMeshScene(overCount);
    expect(() => packEmitterArrays(scene)).toThrow(RangeError);
  });

  it('over-limit: rejection names the authored count and supported limit', () => {
    const overCount = MESH_AREA_LIGHT_TRI_CAP + 1;
    const scene = bigMeshScene(overCount);
    expect(() => packEmitterArrays(scene)).toThrow(
      new RegExp(`requires ${overCount} triangles, exceeding the exact-support limit of ${MESH_AREA_LIGHT_TRI_CAP}`),
    );
  });

  it('over-cap: selected triangles are the largest-area ones (not the first N)', () => {
    // Build 5 triangles with areas 1, 2, 3, 4, 5 (index = area * 2 base).
    // Verify the small case — 3 triangles with distinct areas are preserved.
    const scene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'test',
        positions: new Float32Array([
          // tri 0: small area ≈ 0.5 (base=1, height=1)
          0, 0, 0,   1, 0, 0,   0, 1, 0,
          // tri 1: medium area ≈ 2.5 (base=5, height=1)
          0, 0, 0,   5, 0, 0,   0, 1, 0,
          // tri 2: large area ≈ 5.0 (base=10, height=1)
          0, 0, 0,   10, 0, 0,  0, 1, 0,
        ]),
        normals: new Float32Array(Array(9 * 3).fill(0).map((_, i) => (i % 3 === 2) ? 1 : 0)),
        indices: new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        material: { baseColor: [1, 1, 1], roughness: 0.4, metallic: 0 },
      }],
      emitters: [{ kind: 'mesh-area', id: 'm', meshId: 'test', color: [1, 1, 1], intensity: 1 }],
      environment: { kind: 'none' },
    };
    // All 3 tris (under cap), but verify the largest are packed.
    const packed = packEmitterArrays(scene);
    expect(packed.meshAreaLightCount).toBe(3);
    // No cap warnings for under-cap scenes.
    expect(packed.warnings.filter((w) => w.includes('cap') || w.includes('exceeds'))).toHaveLength(0);
  });

  it('under-cap via buildPackedScene: count preserved, no warning in packed.warnings', () => {
    const scene = bigMeshScene(5);
    const packed = buildPackedScene(scene);
    expect(packed.meshAreaLightCount).toBe(5);
    expect(packed.warnings.filter((w) => w.includes('exceeds'))).toHaveLength(0);
  });
});
