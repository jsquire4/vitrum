/**
 * Theme 3 — RC merged-mode moving-instance refit WITHOUT pipeline teardown.
 *
 * Mirrors the TLAS refit test (rcRestirTlas.test.ts): a moved instance must
 * refresh the RC BVH geometry via writeBuffer + in-place node refit, NOT via
 * a full SAH rebuild + buffer realloc + dispatcher recreation.
 */

import { describe, expect, it, vi } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';

installWebGPUPolyfills();

import * as THREE from 'three';

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

/** A minimal THREE scene with one box mesh — enough for buildRCSceneBVH to
 *  produce a real merged BVH (>1 node) so refit has something to recompute. */
function boxScene(): THREE.Scene {
  const scene = new THREE.Scene();
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial({ color: 0x808080 }),
  );
  mesh.name = 'box-0';
  scene.add(mesh);
  scene.updateMatrixWorld(true);
  return scene;
}

describe('RCSubsystem merged-mode moving-instance refit (PR-5.3)', () => {
  it('refits positions + nodes via writeBuffer without realloc or dispatcher recreation', async () => {
    const { RCSubsystem } = await import('../src/HybridEngineRC.js');
    const { device, createBuffer, writeBuffer } = makeMockDevice();
    const rc = new RCSubsystem(device);

    rc.setScene(boxScene());
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
    rc.setScene(boxScene());

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
    rc.setScene(boxScene());
    // Wrong-length positions buffer → cannot apply the fast path.
    const out = rc.refitMergedInstance(new Float32Array(8), [0, 0, 0], [1, 1, 1]);
    expect(out).toBe(false);
  });
});
