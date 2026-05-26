import { describe, expect, it } from 'vitest';
import type { Mat4, Scene, Vec3 } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import {
  computeWorldAabbForBindings,
  packSceneFromCore,
  rebuildPrimitiveBlas,
  refitTlasTransforms,
} from '../scenePack.js';
import { tlasIntersect } from '../tlas.js';

function unitTriMesh(id: string, transform?: Mat4): Scene['primitives'][number] {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.4, metallic: 0 },
    ...(transform != null ? { transform } : {}),
  };
}

function boxMesh(id: string, min: Vec3, max: Vec3, transform?: Mat4): Scene['primitives'][number] {
  const [x0, y0, z0] = min;
  const [x1, y1, z1] = max;
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([
      x0, y0, z0,
      x1, y0, z0,
      x0, y1, z0,
      x0, y0, z1,
      x1, y1, z1,
      x1, y0, z1,
      x0, y1, z1,
      x1, y1, z1,
    ]),
    normals: new Float32Array(24).fill(0).map((_, i) => (i % 3 === 2 ? 1 : 0)),
    indices: new Uint32Array([
      0, 1, 2, 4, 1, 2,
      1, 5, 6, 5, 4, 6,
      0, 2, 3, 2, 6, 7,
      0, 1, 3, 1, 5, 3,
      3, 5, 7, 5, 6, 7,
      0, 4, 3, 4, 6, 7,
    ]),
    material: { baseColor: [0.6, 0.6, 0.6], roughness: 0.5, metallic: 0 },
    ...(transform != null ? { transform } : {}),
  };
}

describe('packSceneFromCore (SP-*)', () => {
  it('SP-1: two static boxes build TLAS and nearest instance matches oracle', () => {
    const scene: Scene = {
      primitives: [
        boxMesh('box-a', [0, 0, 0], [1, 1, 1]),
        boxMesh('box-b', [0, 0, 0], [1, 1, 1], asMat4(new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          3, 0, 0, 1,
        ]))),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    expect(packed.tlasNodeCount).toBeGreaterThan(0);
    expect(packed.primitiveTlasBindings).toHaveLength(2);
    expect(packed.tlasBlasRoots.length).toBe(2);

    const tlasData = {
      nodes: packed.tlasNodes,
      nodeCount: packed.tlasNodeCount,
      instanceIndices: packed.tlasInstanceIndices,
      blasRoots: packed.tlasBlasRoots,
      instanceTransforms: packed.tlasInstanceWorldToLocal,
    };
    const hitsNearA = tlasIntersect(tlasData, [-1, 0.5, 0.5], [1, 0, 0], 2);
    const hitsTowardB = tlasIntersect(tlasData, [1.5, 0.5, 0.5], [1, 0, 0], 2);
    expect(hitsNearA).toContain(0);
    expect(hitsTowardB).toContain(1);
  });

  it('SP-2: transform-only refit updates TLAS matrices and bounds', () => {
    const base = unitTriMesh('tri');
    if (base.kind !== 'mesh') throw new Error('test setup');
    const packed = packSceneFromCore(
      { primitives: [base], emitters: [], environment: { kind: 'none' } },
      { tlas: true, resolveMaterialId: () => 0 },
    );
    const moved: Scene = {
      primitives: [{
        kind: 'mesh',
        id: base.id,
        positions: base.positions,
        normals: base.normals,
        material: base.material,
        transform: asMat4(new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          2, 0, 0, 1,
        ])),
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const rebuilt = refitTlasTransforms(
      moved,
      packed.primitiveTlasBindings,
      {
        tlasNodes: packed.tlasNodes,
        tlasInstanceIndices: packed.tlasInstanceIndices,
        tlasBlasRoots: packed.tlasBlasRoots,
        tlasInstanceWorldToLocal: packed.tlasInstanceWorldToLocal,
      },
    );
    expect(rebuilt.ok).toBe(true);
    if (!rebuilt.ok) return;
    expect(rebuilt.tlasNodes.length).toBe(packed.tlasNodes.length);
    expect(rebuilt.tlasInstanceLocalToWorld[12]).toBeCloseTo(2, 5);
    expect(rebuilt.tlasInstanceWorldToLocal[12]).toBeCloseTo(-2, 5);
    expect(rebuilt.tlasNodes[0]).not.toBe(packed.tlasNodes[0]);
  });

  it('SP-3: instanced mesh produces four TLAS instances', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'instanced-mesh',
        id: 'inst',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        instances: [
          asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
          asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1])),
          asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1])),
          asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1])),
        ],
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    expect(packed.primitiveTlasBindings[0]?.instanceCount).toBe(4);
    expect(packed.tlasBlasRoots.length).toBe(4);
    expect(packed.tlasInstanceLocalToWorld.length).toBe(64);
  });

  it('T-4.3: instance count change fails refit (forces topology rebuild)', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'instanced-mesh',
        id: 'inst',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        instances: [
          asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
          asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1])),
        ],
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    const extraInstance = asMat4(new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      2, 0, 0, 1,
    ]));
    const instPrim = scene.primitives[0]!;
    if (instPrim.kind !== 'instanced-mesh') {
      throw new Error('expected instanced-mesh');
    }
    const sceneMore: Scene = {
      ...scene,
      primitives: [{
        kind: 'instanced-mesh',
        id: instPrim.id,
        positions: instPrim.positions,
        normals: instPrim.normals,
        material: instPrim.material,
        instances: [...instPrim.instances, extraInstance],
      }],
    };
    const rebuilt = refitTlasTransforms(
      sceneMore,
      packed.primitiveTlasBindings,
      {
        tlasNodes: packed.tlasNodes,
        tlasInstanceIndices: packed.tlasInstanceIndices,
        tlasBlasRoots: packed.tlasBlasRoots,
        tlasInstanceWorldToLocal: packed.tlasInstanceWorldToLocal,
      },
    );
    expect(rebuilt.ok).toBe(false);
    if (rebuilt.ok) return;
    expect(rebuilt.reason).toMatch(/instance/i);
  });

  it('computeWorldAabbForBindings unions instance world bounds', () => {
    const scene: Scene = {
      primitives: [
        boxMesh('box-a', [0, 0, 0], [1, 1, 1]),
        boxMesh('box-b', [0, 0, 0], [1, 1, 1], asMat4(new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          5, 0, 0, 1,
        ]))),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    const bounds = computeWorldAabbForBindings(scene, packed.primitiveTlasBindings);
    expect(bounds).not.toBeNull();
    expect(bounds!.min[0]).toBeLessThanOrEqual(0);
    expect(bounds!.max[0]).toBeGreaterThanOrEqual(5);
  });

  it('rebuildPrimitiveBlas splices in-place when topology size is unchanged', () => {
    const scene: Scene = {
      primitives: [
        boxMesh('box-a', [0, 0, 0], [0.5, 0.5, 0.5]),
        boxMesh('box-b', [0, 0, 0], [0.4, 0.4, 0.4]),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    const bindingB = packed.primitiveTlasBindings.find((b) => b.primitiveId === 'box-b');
    expect(bindingB).toBeDefined();
    const anchorIdx = (bindingB!.vertexStart + 1) * 4 + 1;
    const anchorBefore = packed.positions[anchorIdx]!;

    const movedB = boxMesh('box-b', [0.1, 0.1, 0.1], [0.5, 0.5, 0.5]);
    const rebuilt = rebuildPrimitiveBlas(
      { primitives: [scene.primitives[0]!, movedB], emitters: [], environment: { kind: 'none' } },
      'box-b',
      packed,
      { tlas: true, resolveMaterialId: () => 0 },
    );
    expect(rebuilt.ok).toBe(true);
    if (!rebuilt.ok) return;
    expect(rebuilt.pack.triangleCount).toBe(packed.triangleCount);
    expect(rebuilt.pack.positions.length).toBe(packed.positions.length);
    expect(rebuilt.pack.positions[anchorIdx]).not.toBe(anchorBefore);
    const bindingA = rebuilt.pack.primitiveTlasBindings.find((b) => b.primitiveId === 'box-a');
    expect(bindingA?.vertexStart).toBe(0);
    expect(rebuilt.pack.positions[0]).toBe(packed.positions[0]);
    if (rebuilt.ok) expect(rebuilt.strategy).toBe('splice');
  });

  it('rebuildPrimitiveBlas splice beats full packSceneFromCore on two-box scene', () => {
    const scene: Scene = {
      primitives: [
        boxMesh('box-a', [0, 0, 0], [0.5, 0.5, 0.5]),
        boxMesh('box-b', [0, 0, 0], [0.4, 0.4, 0.4]),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    const movedB = boxMesh('box-b', [0.05, 0, 0], [0.45, 0.45, 0.45]);
    const nextScene: Scene = {
      primitives: [scene.primitives[0]!, movedB],
      emitters: [],
      environment: { kind: 'none' },
    };
    const opts = { tlas: true, resolveMaterialId: () => 0 };

    const t0 = performance.now();
    const spliced = rebuildPrimitiveBlas(nextScene, 'box-b', packed, opts);
    const spliceMs = performance.now() - t0;

    const t1 = performance.now();
    packSceneFromCore(nextScene, opts);
    const fullMs = performance.now() - t1;

    expect(spliced.ok).toBe(true);
    if (spliced.ok) expect(spliced.strategy).toBe('splice');
    expect(spliceMs).toBeLessThan(fullMs);
  });

  it('rebuildPrimitiveBlas full-repacks when triangle count changes', () => {
    const packed = packSceneFromCore(
      { primitives: [unitTriMesh('shape')], emitters: [], environment: { kind: 'none' } },
      { tlas: true, resolveMaterialId: () => 0 },
    );
    const rebuilt = rebuildPrimitiveBlas(
      {
        primitives: [boxMesh('shape', [0, 0, 0], [1, 1, 1])],
        emitters: [],
        environment: { kind: 'none' },
      },
      'shape',
      packed,
      { tlas: true, resolveMaterialId: () => 0 },
    );
    expect(rebuilt.ok).toBe(true);
    if (!rebuilt.ok) return;
    expect(rebuilt.pack.triangleCount).toBeGreaterThan(packed.triangleCount);
    if (rebuilt.ok) expect(rebuilt.strategy).toBe('full');
  });

  it('rebuildPrimitiveBlas fails when primitive was not in previous pack', () => {
    const packed = packSceneFromCore(
      { primitives: [unitTriMesh('a')], emitters: [], environment: { kind: 'none' } },
      { tlas: true, resolveMaterialId: () => 0 },
    );
    const rebuilt = rebuildPrimitiveBlas(
      { primitives: [unitTriMesh('b')], emitters: [], environment: { kind: 'none' } },
      'b',
      packed,
      { tlas: true, resolveMaterialId: () => 0 },
    );
    expect(rebuilt.ok).toBe(false);
    if (rebuilt.ok) return;
    expect(rebuilt.reason).toContain('b');
  });

  it('SP-4: removed primitive fails refit with explicit id', () => {
    const packed = packSceneFromCore(
      { primitives: [unitTriMesh('gone')], emitters: [], environment: { kind: 'none' } },
      { tlas: true, resolveMaterialId: () => 0 },
    );
    const rebuilt = refitTlasTransforms(
      { primitives: [], emitters: [], environment: { kind: 'none' } },
      packed.primitiveTlasBindings,
      {
        tlasNodes: packed.tlasNodes,
        tlasInstanceIndices: packed.tlasInstanceIndices,
        tlasBlasRoots: packed.tlasBlasRoots,
        tlasInstanceWorldToLocal: packed.tlasInstanceWorldToLocal,
      },
    );
    expect(rebuilt.ok).toBe(false);
    if (rebuilt.ok) return;
    expect(rebuilt.reason).toContain('gone');
  });

  it('SP-5: large single mesh packs within CI ceiling (200ms soft budget logged)', () => {
    const triCount = 10_000;
    const vertCount = triCount * 3;
    const positions = new Float32Array(vertCount * 3);
    const normals = new Float32Array(vertCount * 3);
    const indices = new Uint32Array(triCount * 3);
    for (let i = 0; i < vertCount; i += 1) {
      positions[i * 3] = (i % 100) * 0.01;
      positions[i * 3 + 1] = Math.floor(i / 100) * 0.01;
      positions[i * 3 + 2] = 0;
      normals[i * 3 + 2] = 1;
    }
    for (let t = 0; t < triCount; t += 1) {
      indices[t * 3] = t * 3;
      indices[t * 3 + 1] = t * 3 + 1;
      indices[t * 3 + 2] = t * 3 + 2;
    }
    const scene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'big',
        positions,
        normals,
        indices,
        material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const t0 = performance.now();
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    const ms = performance.now() - t0;
    if (ms > 200) {
      console.warn(`[scenePack SP-5] pack took ${ms.toFixed(1)}ms (> 200ms soft budget)`);
    }
    expect(packed.triangleCount).toBe(triCount);
    expect(ms).toBeLessThan(2000);
  });
});
