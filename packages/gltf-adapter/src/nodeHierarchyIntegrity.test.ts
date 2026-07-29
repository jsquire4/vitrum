import { describe, expect, it } from 'vitest';
import type { GltfJson } from './gltfTypes.js';
import { gltfToScene } from './gltfToScene.js';
import { buildWorldTransforms } from './transforms.js';

describe('selected-scene node hierarchy integrity', () => {
  it('rejects multiple parents instead of choosing the first DFS transform', async () => {
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0, 1] }],
      nodes: [
        { translation: [1, 0, 0], children: [2] },
        { translation: [10, 0, 0], children: [2] },
        { translation: [0, 2, 0] },
      ],
    };

    await expect(gltfToScene(gltf)).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'invalid-node-hierarchy',
        path: 'nodes[1].children[0]',
      })],
    });
  });

  it('rejects duplicate roots and root reuse as a child', () => {
    expect(() => buildWorldTransforms({
      asset: { version: '2.0' },
      nodes: [{}],
    }, [0, 0])).toThrow(/Node 0 is referenced more than once/);

    expect(() => buildWorldTransforms({
      asset: { version: '2.0' },
      nodes: [{ children: [1] }, { children: [0] }],
    }, [0])).toThrow(/Node 0 is reused or participates in a cycle/);
  });

  it('rejects missing child nodes with the exact authored path', () => {
    expect(() => buildWorldTransforms({
      asset: { version: '2.0' },
      nodes: [{ children: [4] }],
    }, [0])).toThrow(/nodes\[0\]\.children\[0\] references missing node 4/);
  });

  it.each([null, 42, []])(
    'rejects malformed node records through the structured import boundary (%j)',
    async (invalidNode) => {
      const gltf = {
        asset: { version: '2.0' },
        scene: 0,
        scenes: [{ nodes: [0] }],
        nodes: [invalidNode],
      } as unknown as GltfJson;

      await expect(gltfToScene(gltf)).rejects.toMatchObject({
        name: 'GltfImportError',
        diagnostics: [expect.objectContaining({
          severity: 'error',
          code: 'invalid-node-hierarchy',
          path: 'nodes[0]',
        })],
      });
    },
  );

  it.each([42, []])(
    'rejects malformed selected-scene records through the structured import boundary (%j)',
    async (invalidScene) => {
      const gltf = {
        asset: { version: '2.0' },
        scene: 0,
        scenes: [invalidScene],
      } as unknown as GltfJson;

      await expect(gltfToScene(gltf)).rejects.toMatchObject({
        name: 'GltfImportError',
        diagnostics: [expect.objectContaining({
          severity: 'error',
          code: 'invalid-node-hierarchy',
          path: 'scenes[0]',
        })],
      });
    },
  );

  it('rejects non-array node and child containers at their authored paths', async () => {
    expect(() => buildWorldTransforms({
      asset: { version: '2.0' },
      nodes: {},
    } as unknown as GltfJson, [0])).toThrow(/nodes must be an array/);

    const invalidRoots = {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: { 0: 0, length: 1 } }],
      nodes: [{}],
    } as unknown as GltfJson;
    await expect(gltfToScene(invalidRoots)).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'invalid-node-hierarchy',
        path: 'scenes[0].nodes',
      })],
    });

    const gltf = {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ children: { 0: 1, length: 1 } }],
    } as unknown as GltfJson;
    await expect(gltfToScene(gltf)).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'invalid-node-hierarchy',
        path: 'nodes[0].children',
      })],
    });
  });

  it('retains deterministic world transforms for a valid tree', () => {
    const world = buildWorldTransforms({
      asset: { version: '2.0' },
      nodes: [
        { translation: [2, 0, 0], children: [1, 2] },
        { translation: [0, 3, 0] },
        { translation: [0, 0, 4] },
      ],
    }, [0]);

    expect(Array.from(world.get(1)!.slice(12, 15))).toEqual([2, 3, 0]);
    expect(Array.from(world.get(2)!.slice(12, 15))).toEqual([2, 0, 4]);
  });
});
