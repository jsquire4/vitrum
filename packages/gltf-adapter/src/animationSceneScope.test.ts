import { describe, expect, it } from 'vitest';
import { convertAnimations } from './animations.js';
import type { GltfJson } from './gltfTypes.js';

describe('convertAnimations selected-scene scoping', () => {
  it('ignores an unsupported standard channel path when its target node is outside the selected scene', () => {
    const gltf = {
      asset: { version: '2.0' },
      nodes: [{}, {}],
      animations: [{
        samplers: [{ input: 99, output: 100 }],
        channels: [{
          sampler: 0,
          target: {
            node: 1,
            path: 'unsupported-outside-scene-path',
          },
        }],
      }],
    } as unknown as GltfJson;
    const warnings: string[] = [];

    expect(convertAnimations(
      gltf,
      new Map(),
      warnings,
      undefined,
      undefined,
      { reachableNodeIndices: new Set([0]) },
    )).toEqual([]);
    expect(warnings).toEqual([]);
  });
});
