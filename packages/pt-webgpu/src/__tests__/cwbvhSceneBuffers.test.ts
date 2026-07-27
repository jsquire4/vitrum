import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { CWBVH_CHILD_COUNT_INVALID } from '@vitrum/shared-bvh';
import {
  CWBVH_ROOT_PAIR_MAGIC,
  buildPackedScene,
  packCwbvhRootPair,
  isValidCwbvhRootPair,
} from '../scene/uploadSceneBuffers.js';

function fiveTriangleMesh(id: string): Scene['primitives'][number] {
  const positions: number[] = [];
  const normals: number[] = [];
  for (let tri = 0; tri < 5; tri += 1) {
    const x = tri * 2;
    positions.push(
      x, 0, 0,
      x + 1, 0, 0,
      x, 1, 0,
    );
    normals.push(
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    );
  }
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    material: { baseColor: [0.6, 0.6, 0.6], roughness: 0.5, metallic: 0 },
  };
}

function oneTriangleMesh(id: string): Scene['primitives'][number] {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([20, 0, 0, 21, 0, 0, 20, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [0.4, 0.4, 0.4], roughness: 0.5, metallic: 0 },
  };
}

describe('pt-webgpu CWBVH scene-buffer pack', () => {
  it('packs a CWBVH forest and remaps TLAS BLAS roots away from binary node indices', () => {
    const scene: Scene = {
      primitives: [
        fiveTriangleMesh('wide-a'),
        oneTriangleMesh('single-b'),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const packed = buildPackedScene(scene);
    expect(packed.tlasBlasRoots.length).toBe(2);
    expect(packed.cwbvhTlasBlasRoots.length).toBe(8);
    expect(packed.cwbvhNodeCount).toBeGreaterThan(0);
    expect(packed.cwbvhNodeBounds.length).toBe(packed.cwbvhNodeCount * 6);
    expect(packed.cwbvhChildBoundsPacked.length).toBe(packed.cwbvhNodeCount * 8 * 3);
    expect(packed.cwbvhChildMeta.length).toBe(packed.cwbvhNodeCount * 8 * 3);
    expect(packed.cwbvhChildCount.length).toBe(packed.cwbvhNodeCount);

    const secondBinaryRoot = packed.tlasBlasRoots[1]!;
    const secondRecord = packed.cwbvhTlasBlasRoots.subarray(4, 8);
    expect(Array.from(secondRecord)).toEqual(Array.from(packCwbvhRootPair(secondBinaryRoot, secondRecord[2]!)));
    expect(secondRecord[0]).toBe(CWBVH_ROOT_PAIR_MAGIC);
    expect(secondRecord[1]).toBe(secondBinaryRoot);
    const secondWideRoot = secondRecord[2]!;
    expect(secondBinaryRoot).toBeGreaterThan(1);
    expect(secondWideRoot).toBeLessThan(secondBinaryRoot);
    expect(packed.cwbvhBinaryRootToWideRoot[secondBinaryRoot]).toBe(secondWideRoot);
  });

  it('uses an asymmetric integrity word and reserves the invalid-root sentinel', () => {
    const maxValid = CWBVH_CHILD_COUNT_INVALID - 1;
    const maxPair = packCwbvhRootPair(maxValid, maxValid - 1);
    expect(Array.from(maxPair.slice(0, 3))).toEqual([
      CWBVH_ROOT_PAIR_MAGIC,
      maxValid,
      maxValid - 1,
    ]);

    const forward = packCwbvhRootPair(17, 29);
    const swapped = packCwbvhRootPair(29, 17);
    expect(isValidCwbvhRootPair(forward)).toBe(true);
    // A hostile/stale record can be rechecksummed and therefore pass record-local
    // integrity. The traversal must additionally compare `.y` with the live
    // tlasBlasRoots entry and range-check both roots before accepting it.
    const rechecksummedStale = packCwbvhRootPair(29, 29);
    expect(isValidCwbvhRootPair(rechecksummedStale)).toBe(true);
    expect(rechecksummedStale[1]).not.toBe(forward[1]);
    const corruptChecksum = new Uint32Array(forward);
    corruptChecksum[3] = (corruptChecksum[3]! ^ 1) >>> 0;
    expect(isValidCwbvhRootPair(corruptChecksum)).toBe(false);
    const swappedInPlace = new Uint32Array(forward);
    [swappedInPlace[1], swappedInPlace[2]] = [swappedInPlace[2]!, swappedInPlace[1]!];
    expect(isValidCwbvhRootPair(swappedInPlace)).toBe(false);
    const sentinelBinary = new Uint32Array(forward);
    sentinelBinary[1] = CWBVH_CHILD_COUNT_INVALID;
    expect(isValidCwbvhRootPair(sentinelBinary)).toBe(false);
    expect(isValidCwbvhRootPair(new Uint32Array(3))).toBe(false);

    expect(forward[3]).not.toBe(swapped[3]);
    expect(() => packCwbvhRootPair(CWBVH_CHILD_COUNT_INVALID, 0)).toThrow(/invalid sentinel/);
    expect(() => packCwbvhRootPair(0, CWBVH_CHILD_COUNT_INVALID)).toThrow(/invalid sentinel/);
  });

  it('keeps an empty scene valid and publishes no instance-root records', () => {
    const packed = buildPackedScene({
      primitives: [],
      emitters: [],
      environment: { kind: 'none' },
    });
    expect(packed.triangleCount).toBe(0);
    expect(packed.cwbvhTlasBlasRoots).toHaveLength(0);
    expect(packed.cwbvhNodeCount).toBe(0);
    expect(packed.cwbvhChildCount).toHaveLength(0);
  });
});
