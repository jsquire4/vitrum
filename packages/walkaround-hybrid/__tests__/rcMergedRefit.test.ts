/**
 * Theme 3 — RC merged-mode moving-instance refit WITHOUT pipeline teardown.
 *
 * Mirrors the TLAS refit test (rcRestirTlas.test.ts): a moved instance must
 * refresh the RC BVH geometry via writeBuffer + in-place node refit, NOT via
 * a full SAH rebuild + buffer realloc + dispatcher recreation.
 *
 * The merged BVH is built from a `@vitrum/core` `Scene` via
 * `setSceneFromCore()`, so these fixtures are core `MeshPrimitive`s. The refit
 * logic under test (`refitMergedInstance`) operates on the merged
 * positions/nodes after scene packing.
 */

import { describe, expect, it, vi } from 'vitest';
import type { Scene, MeshPrimitive, MaterialSpec } from '@vitrum/core';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';

installWebGPUPolyfills();

type MockBuffer = GPUBuffer & {
  readonly label?: string;
  readonly mappedBytes: ArrayBuffer;
};

/** Mock GPUDevice recording createBuffer + writeBuffer; buffers are tagged by
 *  their label so we can assert which got re-uploaded. */
function makeMockDevice(): {
  device: GPUDevice;
  createBuffer: ReturnType<typeof vi.fn>;
  writeBuffer: ReturnType<typeof vi.fn>;
} {
  const createBuffer = vi.fn((desc: { label?: string; size?: number }): unknown => {
    const mappedBytes = new ArrayBuffer(Math.max(16, desc.size ?? 16));
    return {
      label: desc.label,
      size: desc.size,
      mappedBytes,
      getMappedRange: () => mappedBytes,
      unmap: vi.fn(),
      destroy: vi.fn(),
    };
  });
  const writeBuffer = vi.fn();
  const device = {
    createBuffer,
    queue: { writeBuffer },
  } as unknown as GPUDevice;
  return { device, createBuffer, writeBuffer };
}

const GREY: MaterialSpec = { baseColor: [0.5, 0.5, 0.5], roughness: 1, metallic: 0 };
const GLASS: MaterialSpec = {
  baseColor: [0.8, 0.95, 1.0],
  roughness: 0,
  metallic: 0,
  transmission: 0.8,
};

/** One quad (two tris) as a core MeshPrimitive (winding mirrors the RC
 *  core-equivalence fixtures). */
function quad(
  id: string,
  verts: [number, number, number][],
  normal: [number, number, number],
  material: MaterialSpec = GREY,
): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([...verts[0]!, ...verts[1]!, ...verts[2]!, ...verts[3]!]),
    normals: new Float32Array([...normal, ...normal, ...normal, ...normal]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 2, 1, 2, 0, 3]),
    material,
  };
}

/** Six faces of a unit cube spanning [-0.5, 0.5]^3 — 12 tris ⇒ a real multi-node
 *  merged BVH so refit has nodes to recompute. `off` shifts the whole cube so
 *  two cubes can coexist with distinct world positions. */
function unitCubePrimitives(
  idPrefix: string,
  off: [number, number, number] = [0, 0, 0],
  material: MaterialSpec = GREY,
): MeshPrimitive[] {
  const n = -0.5;
  const p = 0.5;
  const [ox, oy, oz] = off;
  const v = (x: number, y: number, z: number): [number, number, number] => [x + ox, y + oy, z + oz];
  return [
    quad(`${idPrefix}-zneg`, [v(n, n, n), v(p, n, n), v(p, p, n), v(n, p, n)], [0, 0, -1], material),
    quad(`${idPrefix}-zpos`, [v(n, n, p), v(p, n, p), v(p, p, p), v(n, p, p)], [0, 0, 1], material),
    quad(`${idPrefix}-xneg`, [v(n, n, n), v(n, n, p), v(n, p, p), v(n, p, n)], [-1, 0, 0], material),
    quad(`${idPrefix}-xpos`, [v(p, n, n), v(p, n, p), v(p, p, p), v(p, p, n)], [1, 0, 0], material),
    quad(`${idPrefix}-yneg`, [v(n, n, n), v(p, n, n), v(p, n, p), v(n, n, p)], [0, -1, 0], material),
    quad(`${idPrefix}-ypos`, [v(n, p, n), v(p, p, n), v(p, p, p), v(n, p, p)], [0, 1, 0], material),
  ];
}

/** A core Scene with one unit cube at the origin — replaces the old THREE
 *  `boxScene()`. */
function cubeScene(): Scene {
  return { primitives: unitCubePrimitives('box-0'), emitters: [], environment: { kind: 'none' } };
}

describe('RCSubsystem merged-mode moving-instance refit (PR-5.3)', () => {
  it('packs transmission into merged RC bvhIndex.w so skipGlass works outside TLAS mode', async () => {
    const { RCSubsystem } = await import('../src/HybridEngineRC.js');
    const { device, createBuffer } = makeMockDevice();
    const rc = new RCSubsystem(device);

    rc.setSceneFromCore({
      primitives: [quad('glass-pane', [
        [-0.5, -0.5, 0],
        [0.5, -0.5, 0],
        [0.5, 0.5, 0],
        [-0.5, 0.5, 0],
      ], [0, 0, 1], GLASS)],
      emitters: [],
      environment: { kind: 'none' },
    });

    const indexBuffer = createBuffer.mock.results
      .map((result) => result.value as MockBuffer)
      .find((buffer) => buffer.label === 'rc-bvh-indices');
    expect(indexBuffer).toBeDefined();

    const words = new Uint32Array(indexBuffer!.mappedBytes);
    expect(words.length).toBeGreaterThanOrEqual(8);
    for (const triPayload of [words[3]!, words[7]!]) {
      const trans4 = (triPayload >> 4) & 0xF;
      expect(trans4).toBeGreaterThan(4);
    }
  });

  it('transactionally replaces refit positions + nodes without recreating dispatcher', async () => {
    const { RCSubsystem } = await import('../src/HybridEngineRC.js');
    const { device, createBuffer, writeBuffer } = makeMockDevice();
    const rc = new RCSubsystem(device);

    rc.setSceneFromCore(cubeScene());
    const createCallsAfterBuild = createBuffer.mock.calls.length;
    expect(createCallsAfterBuild).toBeGreaterThan(0);
    const dispatcherBefore = (rc as unknown as { _dispatcher: unknown })._dispatcher;
    expect(dispatcherBefore).not.toBeNull();
    const invalidateSpy = vi.spyOn(
      dispatcherBefore as { invalidateBindings: () => void },
      'invalidateBindings',
    );

    // Grab the build-time merged positions mirror and shift every vertex +5 X
    // (simulating a moved instance), then refit.
    const positions = (rc as unknown as { _mergedPositionsStride4: Float32Array | null })
      ._mergedPositionsStride4;
    expect(positions).not.toBeNull();
    const moved = new Float32Array(positions!);
    for (let i = 0; i < moved.length; i += 4) moved[i] = moved[i]! + 5; // X lane

    const writeCallsBefore = writeBuffer.mock.calls.length;
    const ok = rc.refitMergedInstance(moved, [4.5, -0.5, -0.5], [5.5, 0.5, 0.5]);
    expect(ok).toBe(true);

    // Positions + nodes are candidate allocations; no live buffer is partially written.
    expect(writeBuffer.mock.calls.length).toBe(writeCallsBefore);
    expect(createBuffer.mock.calls.length).toBe(createCallsAfterBuild + 2);
    // Dispatcher is the SAME instance (not recreated).
    expect((rc as unknown as { _dispatcher: unknown })._dispatcher).toBe(dispatcherBefore);
    expect(invalidateSpy).toHaveBeenCalledTimes(1);

    // Cascade probe bounds updated to the moved AABB.
    const geo = rc.getCascadeGeometry();
    expect(geo).not.toBeNull();
    expect(geo!.probeOriginWorld[0]).toBeCloseTo(4.5, 5);
  });



  it('transactionally replaces a merged RC scene across every allocation stage', async () => {
    const { RCSubsystem } = await import('../src/HybridEngineRC.js');
    const { device, createBuffer } = makeMockDevice();
    const rc = new RCSubsystem(device);
    const scene: Scene = {
      primitives: [quad('transaction-tri', [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
      ], [0, 0, 1])],
      emitters: [],
      environment: { kind: 'none' },
    };
    rc.setSceneFromCore(scene);
    const allocationsPerScene = createBuffer.mock.calls.length;

    const internal = rc as unknown as {
      _bvhBuffers: Record<string, GPUBuffer> | null;
      _cascadeBufs: GPUBuffer[] | null;
      _dispatcher: unknown;
      _mergedNodesCpu: Float32Array | null;
      _mergedPositionsStride4: Float32Array | null;
    };
    const previousBvh = internal._bvhBuffers;
    const previousCascades = internal._cascadeBufs;
    const previousDispatcher = internal._dispatcher;
    const previousNodes = internal._mergedNodesCpu;
    const previousPositions = internal._mergedPositionsStride4;
    const previousDestroySpies = [
      ...Object.values(previousBvh ?? {}).map(
        (buffer) => buffer.destroy as ReturnType<typeof vi.fn>,
      ),
      ...(previousCascades ?? []).map(
        (buffer) => buffer.destroy as ReturnType<typeof vi.fn>,
      ),
    ];

    const original = createBuffer.getMockImplementation();
    if (original == null) throw new Error('expected createBuffer mock implementation');
    for (let failAt = 1; failAt <= allocationsPerScene; failAt += 1) {
      let calls = 0;
      const resultStart = createBuffer.mock.results.length;
      createBuffer.mockImplementation((desc: { label?: string; size?: number }) => {
        calls += 1;
        if (calls === failAt) throw new Error(`merged scene allocation fault ${failAt}`);
        return original(desc);
      });
      try {
        expect(() => rc.setSceneFromCore(scene))
          .toThrow(`merged scene allocation fault ${failAt}`);
      } finally {
        createBuffer.mockImplementation(original);
      }

      expect(internal._bvhBuffers).toBe(previousBvh);
      expect(internal._cascadeBufs).toBe(previousCascades);
      expect(internal._dispatcher).toBe(previousDispatcher);
      expect(internal._mergedNodesCpu).toBe(previousNodes);
      expect(internal._mergedPositionsStride4).toBe(previousPositions);
      for (const destroy of previousDestroySpies) expect(destroy).not.toHaveBeenCalled();
      for (const result of createBuffer.mock.results.slice(resultStart)) {
        if (result.type === 'return') {
          const buffer = result.value as GPUBuffer & { destroy: ReturnType<typeof vi.fn> };
          expect(buffer.destroy).toHaveBeenCalledTimes(1);
        }
      }
    }

    rc.setSceneFromCore(scene);
    expect(internal._bvhBuffers).not.toBe(previousBvh);
    for (const destroy of previousDestroySpies) expect(destroy).toHaveBeenCalledTimes(1);
    rc.setSceneFromCore(scene);
    expect(() => rc.dispose()).not.toThrow();
    expect(() => rc.dispose()).not.toThrow();
  });
  it('preserves merged RC state when either refit allocation fails', async () => {
    const { RCSubsystem } = await import('../src/HybridEngineRC.js');
    const { device, createBuffer } = makeMockDevice();
    const rc = new RCSubsystem(device);
    rc.setSceneFromCore(cubeScene());

    const internal = rc as unknown as {
      _bvhBuffers: { bvhNodesBuf: GPUBuffer; bvhPositionsBuf: GPUBuffer } | null;
      _mergedNodesCpu: Float32Array | null;
      _mergedPositionsStride4: Float32Array | null;
    };
    const previousBvh = internal._bvhBuffers;
    const previousNodes = internal._mergedNodesCpu;
    const previousPositions = internal._mergedPositionsStride4;
    const previousGeometry = rc.getCascadeGeometry();
    const oldNodeDestroy = previousBvh!.bvhNodesBuf.destroy as ReturnType<typeof vi.fn>;
    const oldPositionDestroy = previousBvh!.bvhPositionsBuf.destroy as ReturnType<typeof vi.fn>;
    const moved = new Float32Array(previousPositions!);
    for (let i = 0; i < moved.length; i += 4) moved[i] = moved[i]! + 3;

    const original = createBuffer.getMockImplementation();
    if (original == null) throw new Error('expected createBuffer mock implementation');
    for (let failAt = 1; failAt <= 2; failAt += 1) {
      let calls = 0;
      const resultStart = createBuffer.mock.results.length;
      createBuffer.mockImplementation((desc: { label?: string; size?: number }) => {
        calls += 1;
        if (calls === failAt) throw new Error(`merged refit allocation fault ${failAt}`);
        return original(desc);
      });
      try {
        expect(() => rc.refitMergedInstance(moved, [2.5, -0.5, -0.5], [3.5, 0.5, 0.5]))
          .toThrow(`merged refit allocation fault ${failAt}`);
      } finally {
        createBuffer.mockImplementation(original);
      }

      expect(internal._bvhBuffers).toBe(previousBvh);
      expect(internal._mergedNodesCpu).toBe(previousNodes);
      expect(internal._mergedPositionsStride4).toBe(previousPositions);
      expect(rc.getCascadeGeometry()).toEqual(previousGeometry);
      expect(oldNodeDestroy).not.toHaveBeenCalled();
      expect(oldPositionDestroy).not.toHaveBeenCalled();
      for (const result of createBuffer.mock.results.slice(resultStart)) {
        if (result.type === 'return') {
          const buffer = result.value as GPUBuffer & { destroy: ReturnType<typeof vi.fn> };
          expect(buffer.destroy).toHaveBeenCalledTimes(1);
        }
      }
    }

    expect(rc.refitMergedInstance(moved, [2.5, -0.5, -0.5], [3.5, 0.5, 0.5])).toBe(true);
    expect(rc.refitMergedInstance(previousPositions!, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]))
      .toBe(true);
    expect(() => rc.dispose()).not.toThrow();
    expect(() => rc.dispose()).not.toThrow();
  });
  it('refit recomputes node AABBs so the moved bounds are reflected (bound-sanity)', async () => {
    const { RCSubsystem } = await import('../src/HybridEngineRC.js');
    const { device } = makeMockDevice();
    const rc = new RCSubsystem(device);
    rc.setSceneFromCore(cubeScene());

    const nodes = (rc as unknown as { _mergedNodesCpu: Float32Array | null })._mergedNodesCpu;
    const positions = (rc as unknown as { _mergedPositionsStride4: Float32Array | null })
      ._mergedPositionsStride4;
    expect(nodes).not.toBeNull();
    // Root node bounds (slots 0..5 = min xyz, max xyz) before the move.
    const rootMinXBefore = nodes![0]!;
    const rootMaxXBefore = nodes![3]!;

    const moved = new Float32Array(positions!);
    for (let i = 0; i < moved.length; i += 4) moved[i] = moved[i]! + 10; // +10 X
    rc.refitMergedInstance(moved, [9.5, -0.5, -0.5], [10.5, 0.5, 0.5]);

    const nodesAfter = (rc as unknown as { _mergedNodesCpu: Float32Array | null })._mergedNodesCpu;
    const rootMinXAfter = nodesAfter![0]!;
    const rootMaxXAfter = nodesAfter![3]!;
    // The root AABB must have shifted by ~+10 on X after refit.
    expect(rootMinXAfter).toBeCloseTo(rootMinXBefore + 10, 3);
    expect(rootMaxXAfter).toBeCloseTo(rootMaxXBefore + 10, 3);
  });

  it('merged BVH includes ALL core mesh primitives (vertex-set parity with ReSTIR)', async () => {
    // refitMergedInstance adopts ReSTIR's `bvhPositions.cpuData` directly into
    // RC's position mirror, then refits RC's own nodes/indices against it. That
    // is correct ONLY when both built the SAME vertex layout. In the old THREE
    // path this was a real hazard: ReSTIR's merged build filtered `isMesh`
    // (ALL meshes) while `buildRCSceneBVH`'s DEFAULT filter accepted only
    // MeshStandard/MeshPhysical — a non-PBR mesh would diverge the vertex sets.
    // The decoupled core path has NO material-class filter: `setSceneFromCore`
    // → `buildRCSceneBVHFromCore` consumes every core MeshPrimitive, so the
    // vertex sets match ReSTIR's by construction. Pin it: a two-cube scene must
    // contribute exactly 2x a one-cube scene's vertices (no spurious dropping).
    const { RCSubsystem } = await import('../src/HybridEngineRC.js');
    const { device } = makeMockDevice();

    const oneCube: Scene = {
      primitives: unitCubePrimitives('a'),
      emitters: [],
      environment: { kind: 'none' },
    };
    const rcOne = new RCSubsystem(device);
    rcOne.setSceneFromCore(oneCube);
    const oneVerts =
      ((rcOne as unknown as { _mergedPositionsStride4: Float32Array | null })
        ._mergedPositionsStride4?.length ?? 0) / 4;
    expect(oneVerts).toBeGreaterThan(0);

    // Two cubes at distinct world positions (offset so they can't coalesce).
    const twoCubes: Scene = {
      primitives: [...unitCubePrimitives('a'), ...unitCubePrimitives('b', [3, 0, 0])],
      emitters: [],
      environment: { kind: 'none' },
    };
    const rcTwo = new RCSubsystem(device);
    rcTwo.setSceneFromCore(twoCubes);
    const twoVerts =
      ((rcTwo as unknown as { _mergedPositionsStride4: Float32Array | null })
        ._mergedPositionsStride4?.length ?? 0) / 4;

    // Both cubes contributed — proving no spurious primitive filtering.
    expect(twoVerts).toBe(oneVerts * 2);
  });

  it('returns false (caller falls back to setScene) when in TLAS mode', async () => {
    const { RCSubsystem } = await import('../src/HybridEngineRC.js');
    const { device } = makeMockDevice();
    const rc = new RCSubsystem(device);
    // Never built merged → bvhMode defaults merged but no mirrors → false.
    const out = rc.refitMergedInstance(new Float32Array(16), [0, 0, 0], [1, 1, 1]);
    expect(out).toBe(false);
  });

  it('returns false on a vertex-count change (topology change needs a rebuild)', async () => {
    const { RCSubsystem } = await import('../src/HybridEngineRC.js');
    const { device } = makeMockDevice();
    const rc = new RCSubsystem(device);
    rc.setSceneFromCore(cubeScene());
    // Wrong-length positions buffer → cannot apply the fast path.
    const out = rc.refitMergedInstance(new Float32Array(8), [0, 0, 0], [1, 1, 1]);
    expect(out).toBe(false);
  });
});
