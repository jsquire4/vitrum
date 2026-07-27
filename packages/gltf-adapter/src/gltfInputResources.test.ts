import { runInNewContext } from 'node:vm';
import { describe, expect, it } from 'vitest';
import { loadGltfAsset } from './assetLoader.js';
import { gltfToScene, type GltfToSceneOptions } from './gltfToScene.js';
import type { GltfJson } from './gltfTypes.js';
import { GltfResourceLimitError } from './importResourceBudget.js';

function emptyGltf(): GltfJson {
  return {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [] }],
    nodes: [],
    buffers: [{ byteLength: 4 }],
  };
}

function emptyGlbWithBin(): ArrayBuffer {
  const json = new TextEncoder().encode(JSON.stringify(emptyGltf()));
  const jsonLength = Math.ceil(json.byteLength / 4) * 4;
  const totalLength = 12 + 8 + jsonLength + 8 + 4;
  const buffer = new ArrayBuffer(totalLength);
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, totalLength, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, 0x4e4f534a, true);
  bytes.fill(0x20, 20, 20 + jsonLength);
  bytes.set(json, 20);
  const binOffset = 20 + jsonLength;
  view.setUint32(binOffset, 4, true);
  view.setUint32(binOffset + 4, 0x004e4942, true);
  return buffer;
}

describe('low-level glTF input resource normalization', () => {
  it('accepts a cross-realm ReadonlyMap and cross-realm ArrayBuffer', async () => {
    const foreignMap = runInNewContext(
      'new Map([[0, new ArrayBuffer(4)]])',
    ) as ReadonlyMap<number, ArrayBuffer>;

    const result = await gltfToScene(emptyGltf(), {
      buffers: foreignMap,
      resourceLimits: {
        maxEncodedResourceBytes: 4,
        maxTotalEncodedBytes: 4,
      },
    });
    expect(result.scene.primitives).toEqual([]);
  });

  it('accepts a conforming custom ReadonlyMap implementation', async () => {
    const backing = new Map<number, ArrayBuffer>([[0, new ArrayBuffer(4)]]);
    const mapLike: ReadonlyMap<number, ArrayBuffer> = {
      get size() {
        return backing.size;
      },
      entries: () => backing.entries(),
      get: (key) => backing.get(key),
      has: (key) => backing.has(key),
      keys: () => backing.keys(),
      values: () => backing.values(),
      forEach: (callback, thisArg) =>
        backing.forEach((value, key) => callback.call(thisArg, value, key, mapLike)),
      [Symbol.iterator]: () => backing[Symbol.iterator](),
    };

    const result = await gltfToScene(emptyGltf(), { buffers: mapLike });
    expect(result.scene.primitives).toEqual([]);
  });

  it('rejects non-canonical record keys and invalid map keys before charging', async () => {
    const invalidRecord = {
      NaN: new ArrayBuffer(4),
    } as unknown as NonNullable<GltfToSceneOptions['buffers']>;
    await expect(gltfToScene(emptyGltf(), {
      buffers: invalidRecord,
    })).rejects.toThrow(/key "NaN" must be a canonical non-negative integer/);

    const invalidMap = new Map<unknown, ArrayBuffer>([
      [-1, new ArrayBuffer(4)],
    ]);
    await expect(gltfToScene(emptyGltf(), {
      buffers: invalidMap as unknown as ReadonlyMap<number, ArrayBuffer>,
    })).rejects.toThrow(/key must be a non-negative safe integer/);
  });

  it('rejects SharedArrayBuffer and typed-array inputs explicitly', async () => {
    if (typeof SharedArrayBuffer !== 'undefined') {
      await expect(gltfToScene(
        new SharedArrayBuffer(4) as unknown as ArrayBuffer,
      )).rejects.toThrow(/must not be a SharedArrayBuffer/);
    }
    await expect(gltfToScene(
      new Uint8Array(4) as unknown as ArrayBuffer,
    )).rejects.toThrow(/ArrayBuffer view/);
  });

  it('accounts for an unused GLB BIN copy when options.buffers overrides buffer zero', async () => {
    const input = emptyGlbWithBin();
    await expect(gltfToScene(input, {
      buffers: new Map([[0, new ArrayBuffer(4)]]),
      resourceLimits: {
        maxEncodedResourceBytes: input.byteLength,
        maxTotalEncodedBytes: input.byteLength + 4,
      },
    })).rejects.toMatchObject({
      constructor: GltfResourceLimitError,
      limitKind: 'total-encoded-bytes',
      resourceKey: 'buffer:0:unused-glb-copy',
    });
  });

  it('rejects malformed UTF-8 in raw JSON byte inputs on both loader paths', async () => {
    const input = new TextEncoder().encode(
      JSON.stringify({
        asset: { version: '2.0' },
        scenes: [{ name: 'valid' }],
      }),
    );
    const marker = input.lastIndexOf(0x76);
    expect(marker).toBeGreaterThanOrEqual(0);
    input[marker] = 0xff;

    await expect(gltfToScene(input.buffer)).rejects.toMatchObject({
      reason: 'json-parse-failed',
    });
    await expect(loadGltfAsset(input.buffer)).rejects.toMatchObject({
      reason: 'json-parse-failed',
    });
  });
});
