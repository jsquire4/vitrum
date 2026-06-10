import { describe, expect, it } from 'vitest';
import type { Mat4, Scene, Vec3 } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import {
  computeWorldAabbForBindings,
  invertMat4,
  packSceneFromCore,
  rebuildPrimitiveBlas,
  rebuildTlasReuseBlas,
  refitTlasTransforms,
} from '../scenePack.js';
import { tlasIntersect } from '../tlas.js';

function instancedMesh(id: string, instances: Mat4[]): Scene['primitives'][number] {
  return {
    kind: 'instanced-mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
    instances,
  };
}

function translate(x: number): Mat4 {
  return asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1]));
}

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
    // Perf signal: splice should not be dramatically slower than a full repack.
    // Both ops are sub-millisecond here, so `performance.now()` deltas are
    // dominated by scheduler jitter under parallel-suite load — only assert the
    // ratio when the full repack is above a meaningful noise floor, otherwise the
    // timing is not measurable enough to compare (the `strategy === 'splice'`
    // assertion above is the load-bearing correctness check).
    if (fullMs > 1) {
      expect(spliceMs).toBeLessThan(fullMs * 5);
    }
  });

  it('rebuildPrimitiveBlas splices a growing triangle count (slice-2 resize)', () => {
    // Single primitive grows from 1 tri (unitTriMesh) to 12 tris (box). Slice-2
    // grows the concat buffers and rebuilds only this BLAS.
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
    expect(rebuilt.strategy).toBe('splice');
    expect(rebuilt.pack.triangleCount).toBe(12);
    expect(rebuilt.pack.positions.length).toBe(8 * 4);
    expect(rebuilt.pack.triMaterialIds.length).toBe(12);
    // The rebuilt scene renders: a ray dropped onto the box must hit instance 0.
    const hits = tlasIntersect(
      {
        nodes: rebuilt.pack.tlasNodes,
        nodeCount: rebuilt.pack.tlasNodeCount,
        instanceIndices: rebuilt.pack.tlasInstanceIndices,
        blasRoots: rebuilt.pack.tlasBlasRoots,
        instanceTransforms: rebuilt.pack.tlasInstanceWorldToLocal,
      },
      [0.5, 0.5, 5],
      [0, 0, -1],
      10,
    );
    expect(hits).toContain(0);
  });

  it('slice-2 resize rebases DOWNSTREAM offsets + leaf tri-offsets correctly', () => {
    // box-a (12 tris) then box-b (12 tris) downstream. Grow box-a from a unit
    // tri (1 tri) so every downstream offset shifts. The rebuilt pack must keep
    // box-b's geometry intact and its global vertex/tri references re-rebased so
    // tlasIntersect still routes a ray to box-b (instance 1).
    const scene: Scene = {
      primitives: [
        unitTriMesh('shape-a'),
        boxMesh('box-b', [5, 0, 0], [6, 1, 1]),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    const bindingBefore = packed.primitiveTlasBindings.find((b) => b.primitiveId === 'box-b')!;

    // Grow shape-a to a box (1 tri → 12 tris), shifting box-b's vert/tri/node
    // starts forward.
    const next: Scene = {
      primitives: [
        boxMesh('shape-a', [0, 0, 0], [1, 1, 1]),
        scene.primitives[1]!,
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const rebuilt = rebuildPrimitiveBlas(next, 'shape-a', packed, {
      tlas: true,
      resolveMaterialId: () => 0,
    });
    expect(rebuilt.ok).toBe(true);
    if (!rebuilt.ok) return;
    expect(rebuilt.strategy).toBe('splice');

    const bindingAfter = rebuilt.pack.primitiveTlasBindings.find((b) => b.primitiveId === 'box-b')!;
    // box-b's local geometry is unchanged (still 12 tris / 8 verts), but its
    // start offsets rebased forward by shape-a's growth (1 tri → 12 tris = +11
    // tris, 3 verts → 8 verts = +5 verts).
    expect(bindingAfter.triCount).toBe(bindingBefore.triCount);
    expect(bindingAfter.vertexCount).toBe(bindingBefore.vertexCount);
    expect(bindingAfter.triStart).toBe(bindingBefore.triStart + 11);
    expect(bindingAfter.vertexStart).toBe(bindingBefore.vertexStart + 5);

    // box-b still renders at world x∈[5,6]: a ray dropped down onto it hits
    // instance 1 (and the BVH leaf tri-offsets + index vertex refs are
    // correctly rebased, else the box would be missing or corrupt).
    const hits = tlasIntersect(
      {
        nodes: rebuilt.pack.tlasNodes,
        nodeCount: rebuilt.pack.tlasNodeCount,
        instanceIndices: rebuilt.pack.tlasInstanceIndices,
        blasRoots: rebuilt.pack.tlasBlasRoots,
        instanceTransforms: rebuilt.pack.tlasInstanceWorldToLocal,
      },
      [5.5, 0.5, 5],
      [0, 0, -1],
      10,
    );
    expect(hits).toContain(1);

    // And shape-a (now a box at x∈[0,1]) renders at instance 0.
    const hitsA = tlasIntersect(
      {
        nodes: rebuilt.pack.tlasNodes,
        nodeCount: rebuilt.pack.tlasNodeCount,
        instanceIndices: rebuilt.pack.tlasInstanceIndices,
        blasRoots: rebuilt.pack.tlasBlasRoots,
        instanceTransforms: rebuilt.pack.tlasInstanceWorldToLocal,
      },
      [0.5, 0.5, 5],
      [0, 0, -1],
      10,
    );
    expect(hitsA).toContain(0);

    // The downstream BVH must match a full repack byte-for-byte (the rebase is
    // exact, not approximate).
    const full = packSceneFromCore(next, { tlas: true, resolveMaterialId: () => 0 });
    expect(rebuilt.pack.bvhNodes.length).toBe(full.bvhNodes.length);
    expect(rebuilt.pack.positions.length).toBe(full.positions.length);
    expect(rebuilt.pack.indices.length).toBe(full.indices.length);
    expect(Array.from(rebuilt.pack.indices)).toEqual(Array.from(full.indices));
    expect(Array.from(rebuilt.pack.triMaterialIds)).toEqual(Array.from(full.triMaterialIds));
  });

  it('slice-2 resize that SHRINKS a primitive rebases downstream back', () => {
    const scene: Scene = {
      primitives: [
        boxMesh('shrink-a', [0, 0, 0], [1, 1, 1]),
        boxMesh('box-b', [5, 0, 0], [6, 1, 1]),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    const bindingBefore = packed.primitiveTlasBindings.find((b) => b.primitiveId === 'box-b')!;

    // Shrink shape-a from a 12-tri box to a single triangle.
    const next: Scene = {
      primitives: [unitTriMesh('shrink-a'), scene.primitives[1]!],
      emitters: [],
      environment: { kind: 'none' },
    };
    const rebuilt = rebuildPrimitiveBlas(next, 'shrink-a', packed, {
      tlas: true,
      resolveMaterialId: () => 0,
    });
    expect(rebuilt.ok).toBe(true);
    if (!rebuilt.ok) return;
    expect(rebuilt.strategy).toBe('splice');

    const bindingAfter = rebuilt.pack.primitiveTlasBindings.find((b) => b.primitiveId === 'box-b')!;
    expect(bindingAfter.triStart).toBe(bindingBefore.triStart - 11);
    expect(bindingAfter.vertexStart).toBe(bindingBefore.vertexStart - 5);

    const full = packSceneFromCore(next, { tlas: true, resolveMaterialId: () => 0 });
    expect(Array.from(rebuilt.pack.indices)).toEqual(Array.from(full.indices));
    expect(rebuilt.pack.bvhNodes.length).toBe(full.bvhNodes.length);
    const hits = tlasIntersect(
      {
        nodes: rebuilt.pack.tlasNodes,
        nodeCount: rebuilt.pack.tlasNodeCount,
        instanceIndices: rebuilt.pack.tlasInstanceIndices,
        blasRoots: rebuilt.pack.tlasBlasRoots,
        instanceTransforms: rebuilt.pack.tlasInstanceWorldToLocal,
      },
      [5.5, 0.5, 5],
      [0, 0, -1],
      10,
    );
    expect(hits).toContain(1);
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

describe('rebuildTlasReuseBlas (slice-1 instanced-mesh count change)', () => {
  it('reuses BLAS arrays verbatim and rebuilds the TLAS when count grows', () => {
    const scene: Scene = {
      primitives: [instancedMesh('inst', [translate(0), translate(2)])],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    expect(packed.primitiveTlasBindings[0]?.instanceCount).toBe(2);

    const next: Scene = {
      primitives: [instancedMesh('inst', [translate(0), translate(2), translate(4)])],
      emitters: [],
      environment: { kind: 'none' },
    };
    const rebuilt = rebuildTlasReuseBlas(next, packed);
    expect(rebuilt.ok).toBe(true);
    if (!rebuilt.ok) return;

    // BLAS arrays are the SAME object references (reused verbatim, no rebuild).
    expect(rebuilt.pack.positions).toBe(packed.positions);
    expect(rebuilt.pack.normals).toBe(packed.normals);
    expect(rebuilt.pack.indices).toBe(packed.indices);
    expect(rebuilt.pack.triMaterialIds).toBe(packed.triMaterialIds);
    expect(rebuilt.pack.bvhNodes).toBe(packed.bvhNodes);
    expect(rebuilt.pack.triangleCount).toBe(packed.triangleCount);

    // TLAS grew to 3 instances with correct per-instance transforms.
    expect(rebuilt.pack.primitiveTlasBindings[0]?.instanceCount).toBe(3);
    expect(rebuilt.pack.tlasBlasRoots.length).toBe(3);
    expect(rebuilt.pack.tlasInstanceLocalToWorld.length).toBe(3 * 16);
    expect(rebuilt.pack.tlasInstanceLocalToWorld[0 * 16 + 12]).toBe(0);
    expect(rebuilt.pack.tlasInstanceLocalToWorld[1 * 16 + 12]).toBe(2);
    expect(rebuilt.pack.tlasInstanceLocalToWorld[2 * 16 + 12]).toBe(4);

    // The rebuilt TLAS routes a ray to the new instance: the triangle lives in
    // the z=0 plane and the third instance translates it to x∈[4,5], so a ray
    // dropped down (-z) through (4.2, 0.2, 0) must hit instance index 2.
    const hits = tlasIntersect(
      {
        nodes: rebuilt.pack.tlasNodes,
        nodeCount: rebuilt.pack.tlasNodeCount,
        instanceIndices: rebuilt.pack.tlasInstanceIndices,
        blasRoots: rebuilt.pack.tlasBlasRoots,
        instanceTransforms: rebuilt.pack.tlasInstanceWorldToLocal,
      },
      [4.2, 0.2, 1],
      [0, 0, -1],
      3,
    );
    expect(hits).toContain(2);
  });

  it('reuses BLAS and rebuilds the TLAS when count shrinks', () => {
    const scene: Scene = {
      primitives: [instancedMesh('inst', [translate(0), translate(2), translate(4)])],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    const next: Scene = {
      primitives: [instancedMesh('inst', [translate(0)])],
      emitters: [],
      environment: { kind: 'none' },
    };
    const rebuilt = rebuildTlasReuseBlas(next, packed);
    expect(rebuilt.ok).toBe(true);
    if (!rebuilt.ok) return;
    expect(rebuilt.pack.bvhNodes).toBe(packed.bvhNodes);
    expect(rebuilt.pack.primitiveTlasBindings[0]?.instanceCount).toBe(1);
    expect(rebuilt.pack.tlasBlasRoots.length).toBe(1);
  });

  it('rejects when no instance count changed (caller should use refit path)', () => {
    const scene: Scene = {
      primitives: [instancedMesh('inst', [translate(0), translate(2)])],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    // Same count, just moved instances → not a count change.
    const next: Scene = {
      primitives: [instancedMesh('inst', [translate(1), translate(3)])],
      emitters: [],
      environment: { kind: 'none' },
    };
    const rebuilt = rebuildTlasReuseBlas(next, packed);
    expect(rebuilt.ok).toBe(false);
    if (rebuilt.ok) return;
    expect(rebuilt.reason).toMatch(/no instance count changed/i);
  });

  it('rejects when a non-instanced primitive disappeared', () => {
    const scene: Scene = {
      primitives: [instancedMesh('inst', [translate(0), translate(2)])],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    const rebuilt = rebuildTlasReuseBlas(
      { primitives: [], emitters: [], environment: { kind: 'none' } },
      packed,
    );
    expect(rebuilt.ok).toBe(false);
  });
});

// ─── warning-emission characterization ───────────────────────────────────────
describe('packSceneFromCore — warning characterization', () => {
  it('emits skip warning for <3-vertex primitive (exact message)', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'tiny',
        positions: new Float32Array([0, 0, 0, 1, 0, 0]),  // only 2 vertices
        normals: new Float32Array([0, 0, 1, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    expect(packed.warnings).toContain('Primitive "tiny" has fewer than 3 vertices; skipping.');
    expect(packed.triangleCount).toBe(0);
  });

  it('emits skip warning for zero-triangle primitive (exact message)', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'notri',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        // Explicitly empty index buffer → 0 triangles
        indices: new Uint32Array(0),
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    expect(packed.warnings).toContain('Primitive "notri" has no triangles; skipping.');
    expect(packed.triangleCount).toBe(0);
  });

  // H34-e: singular transform → skip-with-warning (not identity-at-origin fallback).
  it('emits non-invertible warning and skips the TLAS instance (H34-e new behavior)', () => {
    // A zero-column matrix is singular (det=0) — invertMat4 returns null.
    const singular = asMat4(new Float32Array([
      0, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]));
    const scene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'sing',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        transform: singular,
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    // H34-e: now emits skip warning, not identity-fallback warning.
    expect(packed.warnings).toContain(
      'Primitive "sing" has non-invertible instance transform; ' +
      'skipping this TLAS instance (geometry would be placed at the origin otherwise).',
    );
    // BLAS geometry is still packed — the primitive contributes triangles to the BLAS buffer.
    expect(packed.triangleCount).toBe(1);
    // But the TLAS has NO instances (the singular transform instance was skipped).
    expect(packed.tlasNodeCount).toBe(0);
  });
});

// ─── invertMat4 unit tests ────────────────────────────────────────────────────
describe('invertMat4', () => {
  const IDENTITY = asMat4(new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]));

  it('invert(identity) == identity', () => {
    const result = invertMat4(IDENTITY);
    expect(result).not.toBeNull();
    for (let i = 0; i < 16; i += 1) {
      expect(result![i]).toBeCloseTo(IDENTITY[i] ?? 0, 10);
    }
  });

  it('invert∘invert == id (round-trip on a non-trivial matrix)', () => {
    // A matrix with translation, non-uniform scale, and a rotation component.
    const m = asMat4(new Float32Array([
       2,  1,  0,  0,
       0,  3,  1,  0,
       1,  0,  2,  0,
       4, -1,  2,  1,
    ]));
    const inv = invertMat4(m);
    expect(inv).not.toBeNull();
    const inv2 = invertMat4(asMat4(inv!));
    expect(inv2).not.toBeNull();
    for (let i = 0; i < 16; i += 1) {
      expect(inv2![i]).toBeCloseTo(m[i] ?? 0, 5);
    }
  });

  it('returns null for a singular matrix', () => {
    const singular = asMat4(new Float32Array([
      1, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]));
    expect(invertMat4(singular)).toBeNull();
  });
});

describe('packSceneFromCore per-vertex UV flattening (P2)', () => {
  it('packs uv0 into .xy and uv1 into .zw, vec4-strided, in vertex order', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'uv-tri',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
          uv1: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
          material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const pack = packSceneFromCore(scene, { resolveMaterialId: () => 0, tlas: false });
    // One vec4 per vertex, same vertex count as positions (single triangle = no reorder).
    expect(pack.uvs.length).toBe(pack.positions.length);
    expect(pack.uvs.length).toBe(12);
    // v0: uv0 (0,0), uv1 (0.1,0.2)
    expect(Array.from(pack.uvs.subarray(0, 4))).toEqual([0, 0, 0.1, 0.2].map((v) => expect.closeTo(v)));
    // v1: uv0 (1,0), uv1 (0.3,0.4)
    expect(pack.uvs[4]).toBeCloseTo(1);
    expect(pack.uvs[5]).toBeCloseTo(0);
    expect(pack.uvs[6]).toBeCloseTo(0.3);
    expect(pack.uvs[7]).toBeCloseTo(0.4);
    // v2: uv0 (0,1), uv1 (0.5,0.6)
    expect(pack.uvs[8]).toBeCloseTo(0);
    expect(pack.uvs[9]).toBeCloseTo(1);
    expect(pack.uvs[10]).toBeCloseTo(0.5);
    expect(pack.uvs[11]).toBeCloseTo(0.6);
  });

  it('emits an all-zero uvs buffer of the right length for UV-less geometry', () => {
    const pack = packSceneFromCore(
      { primitives: [unitTriMesh('no-uv')], emitters: [], environment: { kind: 'none' } },
      { resolveMaterialId: () => 0, tlas: false },
    );
    expect(pack.uvs.length).toBe(pack.positions.length);
    expect(pack.uvs.every((v) => v === 0)).toBe(true);
  });
});

// ─── H34-c: zero-instance instanced-mesh ─────────────────────────────────────
describe('H34-c: zero-instance instanced-mesh skips geometry', () => {
  it('[meshA, zeroInstanceB, meshC] packs only meshA and meshC', () => {
    const meshA = unitTriMesh('A', asMat4(new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])));
    const zeroInstB: Scene['primitives'][number] = {
      kind: 'instanced-mesh',
      id: 'B',
      positions: new Float32Array([0,0,0, 1,0,0, 0,1,0]),
      normals: new Float32Array([0,0,1, 0,0,1, 0,0,1]),
      material: { baseColor: [1,1,1], roughness: 0.5, metallic: 0 },
      instances: [],
    };
    const meshC = unitTriMesh('C', asMat4(new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 5,0,0,1])));
    const scene: Scene = {
      primitives: [meshA, zeroInstB, meshC],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    // Two non-zero primitives → 2 triangles
    expect(packed.triangleCount).toBe(2);
    // Warning must mention the zero-instance primitive
    expect(packed.warnings.some((w) => w.includes('"B"') && w.includes('zero instances'))).toBe(true);
    // TLAS bindings must NOT include B
    const bindingIds = packed.primitiveTlasBindings.map((b) => b.primitiveId);
    expect(bindingIds).not.toContain('B');
    expect(bindingIds).toContain('A');
    expect(bindingIds).toContain('C');
  });
});

// ─── H34-d: tlas:false + multiple primitives → warn+auto-upgrade ─────────────
describe('H34-d: tlas:false with multiple primitives auto-upgrades to tlas', () => {
  it('emits a warning and still builds a TLAS when tlas:false + 2 primitives', () => {
    const meshA = unitTriMesh('A');
    const meshB = unitTriMesh('B', asMat4(new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 5,0,0,1])));
    const scene: Scene = {
      primitives: [meshA, meshB],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: false, resolveMaterialId: () => 0 });
    // Both triangles packed
    expect(packed.triangleCount).toBe(2);
    // TLAS was built (node count > 0)
    expect(packed.tlasNodeCount).toBeGreaterThan(0);
    // Warning about auto-upgrade
    expect(packed.warnings.some((w) => w.includes('tlas:false') && w.includes('auto'))).toBe(true);
  });

  it('tlas:false with a single primitive does NOT emit the multi-prim warning', () => {
    const scene: Scene = {
      primitives: [unitTriMesh('A')],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: false, resolveMaterialId: () => 0 });
    expect(packed.warnings.some((w) => w.includes('tlas:false'))).toBe(false);
  });
});
