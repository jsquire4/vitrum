/**
 * Theme 3 — RC merged-mode moving-instance refit WITHOUT pipeline teardown.
 *
 * Mirrors the TLAS refit test (rcRestirTlas.test.ts): a moved instance must
 * refresh the RC BVH geometry via writeBuffer + in-place node refit, NOT via
 * a full SAH rebuild + buffer realloc + dispatcher recreation.
 *
 * THREE-decouple (2026-06-08): `RCSubsystem.setScene(rawThreeScene)` was removed
 * (it now throws — see HybridEngineRC.ts). The merged BVH is built from a
 * `@vitrum/core` `Scene` via `setSceneFromCore()`, so these fixtures are core
 * `MeshPrimitive`s instead of `THREE.Mesh`es. The refit logic under test
 * (`refitMergedInstance`) is unchanged — it operates on the merged
 * positions/nodes regardless of how they were built.
 */

import { describe, expect, it, vi } from 'vitest';
import type { Scene, MeshPrimitive, MaterialSpec } from '@vitrum/core';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';

installWebGPUPolyfills();

/** Mock GPUDevice recording createBuffer + writeBuffer; buffers are tagged by
 *  their label so we can assert which got re-uploaded. */
function makeMockDevice(): {
  device: GPUDevice;
  createBuffer: ReturnType<typeof vi.fn>;
  writeBuffer: ReturnType<typeof vi.fn>;
} {
  const createBuffer = vi.fn((desc: { label?: string; size?: number }): unknown => ({
    label: desc.label,
    size: desc.size,
    getMappedRange: () => new ArrayBuffer(Math.max(16, desc.size ?? 16)),
    unmap: vi.fn(),
    destroy: vi.fn(),
  }));
  const writeBuffer = vi.fn();
  const device = {
    createBuffer,
    queue: { writeBuffer },
  } as unknown as GPUDevice;
  return { device, createBuffer, writeBuffer };
}

const GREY: MaterialSpec = { baseColor: [0.5, 0.5, 0.5], roughness: 1, metallic: 0 };

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

/** Six faces of a unit cube spanning [-0.5, 0.5]^3 (the core analogue of the old
 *  `THREE.BoxGeometry(1, 1, 1)`) — 12 tris ⇒ a real multi-node merged BVH so
 *  refit has nodes to recompute. `off` shifts the whole cube so two cubes can
 *  coexist with distinct world positions. */
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
  it('refits positions + nodes via writeBuffer without realloc or dispatcher recreation', async () => {
    const { RCSubsystem } = await import('../src/HybridEngineRC.js');
    const { device, createBuffer, writeBuffer } = makeMockDevice();
    const rc = new RCSubsystem(device);

    rc.setSceneFromCore(cubeScene());
    const createCallsAfterBuild = createBuffer.mock.calls.length;
    expect(createCallsAfterBuild).toBeGreaterThan(0);
    const dispatcherBefore = (rc as unknown as { _dispatcher: unknown })._dispatcher;
    expect(dispatcherBefore).not.toBeNull();

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

    // Positions + nodes were re-uploaded (2 new writeBuffer calls at least).
    expect(writeBuffer.mock.calls.length).toBeGreaterThanOrEqual(writeCallsBefore + 2);
    // NO new buffers allocated (no teardown / realloc).
    expect(createBuffer.mock.calls.length).toBe(createCallsAfterBuild);
    // Dispatcher is the SAME instance (not recreated).
    expect((rc as unknown as { _dispatcher: unknown })._dispatcher).toBe(dispatcherBefore);

    // Cascade probe bounds updated to the moved AABB.
    const geo = rc.getCascadeGeometry();
    expect(geo).not.toBeNull();
    expect(geo!.probeOriginWorld[0]).toBeCloseTo(4.5, 5);
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

    const rootMinXAfter = nodes![0]!;
    const rootMaxXAfter = nodes![3]!;
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
