/**
 * Sprint 2 (Phase 6) — cellPower foundation tests.
 *
 * Verifies that `SceneBVHBuffers.cellPower` is correctly populated by
 * `buildReSTIRSceneBVH` in `restir/bvhCompute.ts`.
 *
 * Test strategy: construct synthetic THREE.Mesh objects with emissive
 * materials (so they contribute emitters), call buildReSTIRSceneBVH, and assert
 * on the cellPower array. No real WebGPU device is required — these are
 * CPU-side unit tests on the Float32Array produced before GPU upload.
 *
 * Definition-of-done item from Sprint 2:
 *   "Round-trip test: setting Le[i]=2× doubles cellPower[i]."
 * This is implemented in the `cellPower scales linearly with emissive intensity` test.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { buildReSTIRSceneBVH } from '../src/restir/bvhCompute.js';

// ── Geometry helpers ──────────────────────────────────────────────────────────

/**
 * Build a simple flat square mesh of known area.
 * Side length `s` → area = s².  Triangulated into 2 right-triangle tris,
 * each with area = s²/2.
 *
 *   v0 --- v1
 *   |  \   |
 *   v3 --- v2
 *
 * Tris: [v0,v1,v2] and [v0,v2,v3]
 */
function makeSquareMesh(s: number, material: THREE.Material): THREE.Mesh {
  const positions = new Float32Array([
    0, 0, 0,   // v0
    s, 0, 0,   // v1
    s, s, 0,   // v2
    0, s, 0,   // v3
  ]);
  const normals = new Float32Array([
    0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1,
  ]);
  const uvs = new Float32Array([
    0, 0,  1, 0,  1, 1,  0, 1,
  ]);
  const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('normal',   new THREE.BufferAttribute(normals,   3));
  geo.setAttribute('uv',       new THREE.BufferAttribute(uvs,       2));
  geo.setIndex(new THREE.BufferAttribute(indices, 1));

  const mesh = new THREE.Mesh(geo, material);
  mesh.updateMatrixWorld(true);
  return mesh;
}

// Standard photometric luminance (Rec. 709).
function luminance(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('Sprint 2 — SceneBVHBuffers.cellPower structure', () => {
  it('cellPower field is present on the returned SceneBVHBuffers object', () => {
    const mat = new THREE.MeshStandardMaterial({
      emissive: new THREE.Color(1, 1, 1),
      emissiveIntensity: 1,
    });
    const scene = new THREE.Scene();
    scene.add(makeSquareMesh(2, mat));
    scene.updateMatrixWorld(true);

    const buffers = buildReSTIRSceneBVH([scene]);
    expect(buffers).toHaveProperty('cellPower');
  });

  it('cellPower is a StorageBufferHandle with cpuData, byteLength, count', () => {
    const mat = new THREE.MeshStandardMaterial({
      emissive: new THREE.Color(1, 0, 0),
      emissiveIntensity: 2,
    });
    const scene = new THREE.Scene();
    scene.add(makeSquareMesh(1, mat));
    scene.updateMatrixWorld(true);

    const buffers = buildReSTIRSceneBVH([scene]);
    const cp = buffers.cellPower;
    expect(cp).toHaveProperty('cpuData');
    expect(cp).toHaveProperty('byteLength');
    expect(cp).toHaveProperty('count');
    expect(cp.count).toBe(buffers.emitterCount);
  });

  it('cellPower buffer has same entry count as emitterCdf buffer', () => {
    const mat = new THREE.MeshStandardMaterial({
      emissive: new THREE.Color(0, 1, 0),
      emissiveIntensity: 3,
    });
    const scene = new THREE.Scene();
    scene.add(makeSquareMesh(3, mat));
    scene.updateMatrixWorld(true);

    const buffers = buildReSTIRSceneBVH([scene]);
    expect(buffers.cellPower.count).toBe(buffers.emitterCdf.count);
  });

  it('cellPower values are non-negative', () => {
    const mat = new THREE.MeshStandardMaterial({
      emissive: new THREE.Color(1, 0.5, 0.2),
      emissiveIntensity: 4,
    });
    const scene = new THREE.Scene();
    scene.add(makeSquareMesh(2, mat));
    scene.updateMatrixWorld(true);

    const buffers = buildReSTIRSceneBVH([scene]);
    const view = new Float32Array(buffers.cellPower.cpuData);
    for (let i = 0; i < buffers.emitterCount; i++) {
      expect(view[i]).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('Sprint 2 — cellPower linear scaling (DoD round-trip test)', () => {
  /**
   * DoD: "Setting Le[i]=2× doubles cellPower[i]."
   *
   * Build two scenes with identical geometry but Le factor 1× and 2×.
   * Assert cellPower sum (over all emitter triangles) doubles.
   *
   * We use the sum rather than per-triangle comparison because the BVH
   * may produce a different number of emitter triangles across builds
   * (glass panels near the threshold could cross the sunDot>0.05 gate
   * differently). Using an emissive material avoids the sunDot dependency.
   */
  it('doubling emissiveIntensity doubles total cellPower', () => {
    function buildScene(intensity: number): ReturnType<typeof buildReSTIRSceneBVH> {
      const mat = new THREE.MeshStandardMaterial({
        emissive: new THREE.Color(1, 1, 1),
        emissiveIntensity: intensity,
      });
      const scene = new THREE.Scene();
      scene.add(makeSquareMesh(2, mat));
      scene.updateMatrixWorld(true);
      return buildReSTIRSceneBVH([scene]);
    }

    const b1 = buildScene(1);
    const b2 = buildScene(2);

    const view1 = new Float32Array(b1.cellPower.cpuData);
    const view2 = new Float32Array(b2.cellPower.cpuData);

    // Sum all emitter powers (both scenes have the same geometry so
    // the same set of triangles qualifies as emitters).
    const sum1 = Array.from({ length: b1.emitterCount }, (_, i) => view1[i] ?? 0)
      .reduce((a, x) => a + x, 0);
    const sum2 = Array.from({ length: b2.emitterCount }, (_, i) => view2[i] ?? 0)
      .reduce((a, x) => a + x, 0);

    // Allow 0.01% floating-point tolerance.
    expect(sum2).toBeCloseTo(sum1 * 2, 3);
  });

  it('halving emissive color halves total cellPower', () => {
    function buildScene(emissive: THREE.Color): ReturnType<typeof buildReSTIRSceneBVH> {
      const mat = new THREE.MeshStandardMaterial({
        emissive,
        emissiveIntensity: 2,
      });
      const scene = new THREE.Scene();
      scene.add(makeSquareMesh(2, mat));
      scene.updateMatrixWorld(true);
      return buildReSTIRSceneBVH([scene]);
    }

    const b1 = buildScene(new THREE.Color(1, 1, 1));   // luminance ≈ 1.0
    const b2 = buildScene(new THREE.Color(0.5, 0.5, 0.5)); // luminance ≈ 0.5

    const view1 = new Float32Array(b1.cellPower.cpuData);
    const view2 = new Float32Array(b2.cellPower.cpuData);

    const sum1 = Array.from({ length: b1.emitterCount }, (_, i) => view1[i] ?? 0)
      .reduce((a, x) => a + x, 0);
    const sum2 = Array.from({ length: b2.emitterCount }, (_, i) => view2[i] ?? 0)
      .reduce((a, x) => a + x, 0);

    expect(sum2).toBeCloseTo(sum1 * 0.5, 3);
  });
});

describe('Sprint 2 — cellPower values match luminance × area formula', () => {
  it('single 2×2 square emitter: cellPower ≈ luminance(Le) × area_per_tri × tri_count', () => {
    // Square 2×2 = area 4. Divided into 2 triangles, each area 2.
    // The emitter list includes each triangle separately.
    // Total power = luminance(Le) × 4.
    const R = 0.8, G = 0.6, B = 0.4;
    const intensity = 3;
    const mat = new THREE.MeshStandardMaterial({
      emissive: new THREE.Color(R, G, B),
      emissiveIntensity: intensity,
    });
    const scene = new THREE.Scene();
    scene.add(makeSquareMesh(2, mat));
    scene.updateMatrixWorld(true);

    const buffers = buildReSTIRSceneBVH([scene]);
    const view = new Float32Array(buffers.cellPower.cpuData);

    // Total expected power = luminance(R*intensity, G*intensity, B*intensity) × area
    const Le_r = R * intensity;
    const Le_g = G * intensity;
    const Le_b = B * intensity;
    const expectedTotalPower = luminance(Le_r, Le_g, Le_b) * 4; // area = 2×2 = 4

    const actualTotalPower = Array.from({ length: buffers.emitterCount }, (_, i) => view[i] ?? 0)
      .reduce((a, x) => a + x, 0);

    // 1% tolerance for floating-point area computation
    expect(actualTotalPower).toBeCloseTo(expectedTotalPower, 1);
  });
});

describe('Sprint 2 — cellPower with no emitters (dummy emitter path)', () => {
  it('non-emissive scene still returns a valid cellPower buffer with count=1 (dummy)', () => {
    // A plain opaque mesh with no emissive and no transmission — the
    // emitter list inserts a dummy sentinel so GPU buffers are non-empty.
    const mat = new THREE.MeshStandardMaterial({ color: 0x808080 });
    const scene = new THREE.Scene();
    scene.add(makeSquareMesh(1, mat));
    scene.updateMatrixWorld(true);

    const buffers = buildReSTIRSceneBVH([scene]);
    // Dummy sentinel is always inserted; count = 1.
    expect(buffers.emitterCount).toBe(1);
    expect(buffers.cellPower.count).toBe(1);
    expect(buffers.cellPower.byteLength).toBeGreaterThan(0);
  });
});
