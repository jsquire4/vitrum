import { describe, expect, it } from 'vitest';
import type { Scene, Vec3 } from '@vitrum/core';
import { packSceneFromCore } from '../scenePack.js';

const UINT32_PER_NODE = 8;
const LEAFNODE_HIGH = 0xffff;

/**
 * Box with 12 triangles (two per face). With `DEFAULT_MAX_LEAF_TRIANGLES = 4`
 * the SAH builder must split this into several leaves, so the concatenated
 * BLAS exercises the leaf triangle-offset rebasing path (node word n+6).
 */
function boxMesh(id: string, min: Vec3, max: Vec3): Scene['primitives'][number] {
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
  };
}

describe('packSceneFromCore — concatenated BLAS leaf triangle-offset rebasing', () => {
  it('rebases the SECOND primitive leaf triangle offsets into the global triangle array', () => {
    const scene: Scene = {
      primitives: [
        boxMesh('box-a', [0, 0, 0], [1, 1, 1]),
        boxMesh('box-b', [3, 0, 0], [4, 1, 1]),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });

    const bindingA = packed.primitiveTlasBindings.find((b) => b.primitiveId === 'box-a');
    const bindingB = packed.primitiveTlasBindings.find((b) => b.primitiveId === 'box-b');
    expect(bindingA).toBeDefined();
    expect(bindingB).toBeDefined();

    // triBase for the SECOND primitive == running triangle total before it,
    // which is exactly the first primitive's triangle count.
    const triBaseB = bindingB!.triStart;
    expect(triBaseB).toBe(bindingA!.triCount);
    expect(triBaseB).toBeGreaterThan(0);

    const nodeWords = new Uint32Array(
      packed.bvhNodes.buffer,
      packed.bvhNodes.byteOffset,
      packed.bvhNodes.length,
    );
    const totalNodes = Math.floor(nodeWords.length / UINT32_PER_NODE);

    // The second BLAS's nodes start at bindingB.blasRoot and run until the
    // end of the concatenated node array (it is the last primitive here).
    const blasStart = bindingB!.blasRoot;
    expect(blasStart).toBeGreaterThan(0);

    let leafCount = 0;
    for (let n = blasStart; n < totalNodes; n += 1) {
      const base = n * UINT32_PER_NODE;
      const splitOrCount = nodeWords[base + 7]!;
      const isLeaf = (splitOrCount >>> 16) === LEAFNODE_HIGH;
      if (!isLeaf) continue;
      leafCount += 1;
      const triOffset = nodeWords[base + 6]!;
      const triCount = splitOrCount & 0xffff;
      // A leaf belonging to the second primitive must reference triangles
      // that live AFTER the first primitive's triangles in the global array.
      expect(triOffset).toBeGreaterThanOrEqual(triBaseB);
      // ...and stay within the second primitive's triangle span.
      expect(triOffset + triCount).toBeLessThanOrEqual(triBaseB + bindingB!.triCount);
    }

    // Sanity: the second BLAS must actually contain at least one leaf, else
    // the assertions above are vacuous.
    expect(leafCount).toBeGreaterThan(0);
  });
});
