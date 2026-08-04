import { describe, expect, it, vi } from 'vitest';
import { gltfToScene } from './gltfToScene.js';
import { GltfResourceLimitError, ImportResourceLedger } from './importResourceBudget.js';
import { resolveCompression, type DracoDecodeFn, type MeshoptDecodeFn } from './compression.js';
import { createImportBuiltinCompressionDecoders } from './builtinCompressionDecoders.js';
import type { GltfJson } from './gltfTypes.js';
import { MeshoptEncoder } from 'meshoptimizer';
import type { MeshPrimitive } from '@vitrum/core';

function float32Bytes(values: readonly number[]): Uint8Array {
  return new Uint8Array(new Float32Array(values).buffer);
}

function uint16Bytes(values: readonly number[]): Uint8Array {
  return new Uint8Array(new Uint16Array(values).buffer);
}

describe('compressed geometry import resource accounting', () => {
  it('shares one aggregate budget across synthetic decode buffers and accessors', async () => {
    const compressed = new Uint8Array([0xa0, 0xc0, 0xde, 0x01, 0xe1, 0xc0, 0xde, 0x01]).buffer;
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [
        {
          primitives: [
            {
              attributes: { POSITION: 0 },
              indices: 1,
            },
          ],
        },
      ],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
      ],
      bufferViews: [
        {
          buffer: 0,
          byteOffset: 0,
          byteLength: 36,
          byteStride: 12,
          extensions: {
            EXT_meshopt_compression: {
              buffer: 1,
              byteOffset: 0,
              byteLength: 4,
              byteStride: 12,
              count: 3,
              mode: 'ATTRIBUTES',
            },
          },
        },
        {
          buffer: 0,
          byteOffset: 36,
          byteLength: 6,
          extensions: {
            EXT_meshopt_compression: {
              buffer: 1,
              byteOffset: 4,
              byteLength: 4,
              byteStride: 2,
              count: 3,
              mode: 'TRIANGLES',
            },
          },
        },
      ],
      buffers: [
        {
          byteLength: 42,
          extensions: { EXT_meshopt_compression: { fallback: true } },
        },
        { byteLength: 8 },
      ],
      extensionsUsed: ['EXT_meshopt_compression'],
    };
    const decode: MeshoptDecodeFn = (_source, count, byteStride) => {
      if (count === 3 && byteStride === 12) {
        return float32Bytes([0, 0, 0, 1, 0, 0, 0, 1, 0]);
      }
      if (count === 3 && byteStride === 2) {
        return uint16Bytes([0, 1, 2]);
      }
      throw new Error(`unexpected meshopt layout ${count}x${byteStride}`);
    };

    const importAttempt = gltfToScene(gltf, {
      buffers: new Map([[1, compressed]]),
      meshoptDecode: decode,
      // 42 hook-output bytes + 42 retained synthetic bytes + 128 node
      // local/world matrix bytes + 36 POSITION accessor bytes = 248.
      resourceLimits: { maxDecodedGeometryBytes: 247 },
    });
    await expect(importAttempt).rejects.toBeInstanceOf(GltfResourceLimitError);
    await expect(importAttempt).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      limitKind: 'decoded-geometry-bytes',
      limit: 247,
      actual: 248,
      path: 'accessors[0]',
    });
  });

  function meshoptPreflightAsset(): {
    readonly gltf: GltfJson;
    readonly buffers: Map<number, ArrayBuffer>;
  } {
    const compressed = new Uint8Array([0xa0, 0xc0, 0xde, 0x01]).buffer;
    return {
      gltf: {
        asset: { version: '2.0' },
        scene: 0,
        scenes: [{ nodes: [0] }],
        nodes: [{ mesh: 0 }],
        meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
        accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
        bufferViews: [
          {
            buffer: 0,
            byteOffset: 0,
            byteLength: 36,
            byteStride: 12,
            extensions: {
              EXT_meshopt_compression: {
                buffer: 1,
                byteOffset: 0,
                byteLength: compressed.byteLength,
                byteStride: 12,
                count: 3,
                mode: 'ATTRIBUTES',
              },
            },
          },
        ],
        buffers: [
          {
            byteLength: 36,
            extensions: { EXT_meshopt_compression: { fallback: true } },
          },
          { byteLength: compressed.byteLength },
        ],
        extensionsUsed: ['EXT_meshopt_compression'],
      },
      buffers: new Map([[1, compressed]]),
    };
  }

  it('rejects host meshopt output before invoking the decoder or publishing buffers', async () => {
    const { gltf, buffers } = meshoptPreflightAsset();
    const decodeSpy = vi.fn(() => new Uint8Array(36));
    const decode: MeshoptDecodeFn = decodeSpy;
    const importAttempt = gltfToScene(gltf, {
      buffers,
      meshoptDecode: decode,
      resourceLimits: { maxDecodedGeometryBytes: 71 },
    });

    await expect(importAttempt).rejects.toBeInstanceOf(GltfResourceLimitError);
    await expect(importAttempt).rejects.toMatchObject({
      limitKind: 'decoded-geometry-bytes',
      actual: 72,
      path: 'bufferViews[0].extensions.EXT_meshopt_compression decoded output and retained copy',
    });
    expect(decodeSpy).not.toHaveBeenCalled();
    expect(buffers.size).toBe(1);
    expect(gltf.bufferViews?.[0]?.extensions?.EXT_meshopt_compression).toBeDefined();
  });

  it('applies the same meshopt preflight to the built-in decoder path', async () => {
    const { gltf, buffers } = meshoptPreflightAsset();
    const importAttempt = gltfToScene(gltf, {
      buffers,
      resourceLimits: { maxDecodedGeometryBytes: 71 },
    });

    await expect(importAttempt).rejects.toBeInstanceOf(GltfResourceLimitError);
    await expect(importAttempt).rejects.toMatchObject({
      limitKind: 'decoded-geometry-bytes',
      actual: 72,
    });
  });

  it('plumbs the public typed ledger into built-in pre-allocation checks', async () => {
    const ledger = new ImportResourceLedger({
      maxDecodedGeometryBytes: 3,
    });
    const { meshoptDecode } = createImportBuiltinCompressionDecoders(ledger);
    const decodeAttempt = meshoptDecode(new Uint8Array(), 1, 4, 'ATTRIBUTES', 'NONE');

    await expect(decodeAttempt).rejects.toBeInstanceOf(GltfResourceLimitError);
    await expect(decodeAttempt).rejects.toMatchObject({
      limitKind: 'decoded-geometry-bytes',
      limit: 3,
      actual: 4,
      path: 'built-in meshopt decoded buffer',
    });
  });

  it('retains repeated failed built-in meshopt target allocations in the public ledger', async () => {
    const ledger = new ImportResourceLedger({
      maxDecodedGeometryBytes: 0,
    });
    const { meshoptDecode } = createImportBuiltinCompressionDecoders(ledger);
    const malformed = new Uint8Array([0xa0, 0xc0, 0xde, 0x01]);

    await expect(
      meshoptDecode(malformed, 3, 12, 'ATTRIBUTES', 'NONE'),
    ).rejects.toThrow();
    await expect(
      meshoptDecode(malformed, 3, 12, 'ATTRIBUTES', 'NONE'),
    ).rejects.toThrow();

    expect(ledger.decodedGeometryBytes).toBe(72);
  });

  it('keeps a failed built-in meshopt target charged across optional fallback', async () => {
    const { gltf, buffers } = meshoptPreflightAsset();
    buffers.set(
      0,
      new Uint8Array(float32Bytes([0, 0, 0, 1, 0, 0, 0, 1, 0])).buffer,
    );

    const importAttempt = gltfToScene(gltf, {
      buffers,
      // Failed built-in target (36) + fallback validation (36) + node
      // matrices (128) reaches 200. Generated normals then preflight both the
      // Float32 result (36) and Float64 accumulator (72), so the next attempted
      // allocation reaches 308 rather than silently omitting scratch storage.
      resourceLimits: { maxDecodedGeometryBytes: 200 },
    });

    await expect(importAttempt).rejects.toBeInstanceOf(GltfResourceLimitError);
    await expect(importAttempt).rejects.toMatchObject({
      limitKind: 'decoded-geometry-bytes',
      limit: 200,
      actual: 308,
      path: 'meshes[0].primitives[0] generated normals',
    });
  });

  it('does not double-charge a successful import-provided meshopt output', async () => {
    await MeshoptEncoder.ready;
    const source = float32Bytes([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const compressed = MeshoptEncoder.encodeGltfBuffer(
      source,
      3,
      12,
      'ATTRIBUTES',
      0,
    );
    const { gltf, buffers } = meshoptPreflightAsset();
    const extension = gltf.bufferViews![0]!.extensions!.EXT_meshopt_compression as {
      byteLength: number;
    };
    extension.byteLength = compressed.byteLength;
    gltf.buffers![1]!.byteLength = compressed.byteLength;
    buffers.set(1, new Uint8Array(compressed).buffer);

    const result = await gltfToScene(gltf, {
      buffers,
      // Built-in output (36) + retained synthetic copy (36) + node matrices
      // (128) + POSITION decode (36) + generated normals/result scratch
      // (36 + 72) = 344 exactly.
      resourceLimits: { maxDecodedGeometryBytes: 344 },
    });

    expect(result.scene.primitives).toHaveLength(1);
    expect(Array.from((result.scene.primitives[0] as MeshPrimitive).positions)).toEqual([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]);
  });

  it('lets an explicit zero geometry ceiling reach a host meshopt decoder past the legacy cap', async () => {
    const { gltf, buffers } = meshoptPreflightAsset();
    const count = Math.floor((512 * 1024 * 1024) / 12) + 1;
    const decodedByteLength = count * 12;
    gltf.bufferViews![0]!.byteLength = decodedByteLength;
    gltf.bufferViews![0]!.byteStride = 12;
    gltf.buffers![0]!.byteLength = decodedByteLength;
    const extension = gltf.bufferViews![0]!.extensions!.EXT_meshopt_compression as {
      count: number;
      byteStride: number;
    };
    extension.count = count;
    extension.byteStride = 12;
    gltf.extensionsRequired = ['EXT_meshopt_compression'];

    const decodeSpy = vi.fn(() => {
      throw new Error('meshopt-zero-cap-hook-reached');
    });
    const importAttempt = gltfToScene(gltf, {
      buffers,
      meshoptDecode: decodeSpy,
      resourceLimits: { maxDecodedGeometryBytes: 0 },
    });

    await expect(importAttempt).rejects.toThrow('meshopt-zero-cap-hook-reached');
    expect(decodeSpy).toHaveBeenCalledOnce();
  });

  it('charges meshopt hook output and its retained copy exactly once each', async () => {
    const { gltf, buffers } = meshoptPreflightAsset();
    const ledger = new ImportResourceLedger({
      maxDecodedGeometryBytes: 72,
    });
    await resolveCompression(
      gltf,
      buffers,
      { meshoptDecode: () => new Uint8Array(36) },
      [],
      undefined,
      { resourceLedger: ledger },
    );

    expect(ledger.decodedGeometryBytes).toBe(72);
    expect(buffers.size).toBe(2);
  });

  function dracoPreflightAsset(): {
    readonly gltf: GltfJson;
    readonly buffers: Map<number, ArrayBuffer>;
  } {
    const compressed = new Uint8Array([0x44, 0x52, 0x41, 0x43]).buffer;
    return {
      gltf: {
        asset: { version: '2.0' },
        scene: 0,
        scenes: [{ nodes: [0] }],
        nodes: [{ mesh: 0 }],
        meshes: [
          {
            primitives: [
              {
                attributes: { POSITION: 0 },
                indices: 1,
                extensions: {
                  KHR_draco_mesh_compression: {
                    bufferView: 0,
                    attributes: { POSITION: 0 },
                  },
                },
              },
            ],
          },
        ],
        accessors: [
          { componentType: 5126, count: 3, type: 'VEC3' },
          { componentType: 5123, count: 3, type: 'SCALAR' },
        ],
        bufferViews: [
          {
            buffer: 0,
            byteOffset: 0,
            byteLength: compressed.byteLength,
          },
        ],
        buffers: [{ byteLength: compressed.byteLength }],
        extensionsUsed: ['KHR_draco_mesh_compression'],
        extensionsRequired: ['KHR_draco_mesh_compression'],
      },
      buffers: new Map([[0, compressed]]),
    };
  }

  it('rejects structurally known Draco output before invoking a host decoder', async () => {
    const { gltf, buffers } = dracoPreflightAsset();
    const decodeSpy = vi.fn(() => ({
      attributes: {
        POSITION: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      },
      indices: new Uint16Array([0, 1, 2]),
    }));
    const decode: DracoDecodeFn = decodeSpy;
    const importAttempt = gltfToScene(gltf, {
      buffers,
      dracoDecode: decode,
      resourceLimits: { maxDecodedGeometryBytes: 125 },
    });

    await expect(importAttempt).rejects.toBeInstanceOf(GltfResourceLimitError);
    await expect(importAttempt).rejects.toMatchObject({
      limitKind: 'decoded-geometry-bytes',
      actual: 126,
      path: 'meshes[0].primitives[0].extensions.KHR_draco_mesh_compression declared decoded output and adapter copies',
    });
    expect(decodeSpy).not.toHaveBeenCalled();
    expect(buffers.size).toBe(1);
  });

  it('applies the same Draco preflight to the built-in decoder path', async () => {
    const { gltf, buffers } = dracoPreflightAsset();
    const importAttempt = gltfToScene(gltf, {
      buffers,
      resourceLimits: { maxDecodedGeometryBytes: 125 },
    });

    await expect(importAttempt).rejects.toBeInstanceOf(GltfResourceLimitError);
    await expect(importAttempt).rejects.toMatchObject({
      limitKind: 'decoded-geometry-bytes',
      actual: 126,
    });
  });

  it('does not reimpose the legacy Draco input cap when public byte ceilings are zero', async () => {
    const { gltf, buffers } = dracoPreflightAsset();
    const declaredByteLength = 512 * 1024 * 1024 + 1;
    gltf.bufferViews![0]!.byteLength = declaredByteLength;
    gltf.buffers![0]!.byteLength = declaredByteLength;
    const decodeSpy = vi.fn(() => {
      throw new Error('decoder must not run for a short backing buffer');
    });
    const decode: DracoDecodeFn = decodeSpy;

    const importAttempt = gltfToScene(gltf, {
      buffers,
      dracoDecode: decode,
      resourceLimits: {
        maxDecodedGeometryBytes: 0,
        maxEncodedResourceBytes: 0,
        maxTotalEncodedBytes: 0,
      },
    });

    await expect(importAttempt).rejects.toThrow(/range \[0, 536870913\).*buffer length 4/);
    await expect(importAttempt).rejects.not.toThrow(/compression input budget/);
    expect(decodeSpy).not.toHaveBeenCalled();
  });

  it('preserves the typed limit error through animation sampler decoding', async () => {
    const times = new Float32Array([0, 1]);
    const translations = new Float32Array([0, 0, 0, 1, 0, 0]);
    const bytes = new Uint8Array(times.byteLength + translations.byteLength);
    bytes.set(new Uint8Array(times.buffer), 0);
    bytes.set(new Uint8Array(translations.buffer), times.byteLength);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{}],
      animations: [
        {
          samplers: [{ input: 0, output: 1 }],
          channels: [
            {
              sampler: 0,
              target: { node: 0, path: 'translation' },
            },
          ],
        },
      ],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 2, type: 'SCALAR' },
        { bufferView: 1, componentType: 5126, count: 2, type: 'VEC3' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: times.byteLength },
        {
          buffer: 0,
          byteOffset: times.byteLength,
          byteLength: translations.byteLength,
        },
      ],
      buffers: [{ byteLength: bytes.byteLength }],
    };

    const importAttempt = gltfToScene(gltf, {
      buffers: new Map([[0, bytes.buffer]]),
      // One root node consumes 128 bytes for its local/world matrices. The
      // animation input accessor is the next 8-byte decoded allocation.
      resourceLimits: { maxDecodedGeometryBytes: 128 },
    });
    await expect(importAttempt).rejects.toBeInstanceOf(GltfResourceLimitError);
    await expect(importAttempt).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      limitKind: 'decoded-geometry-bytes',
      limit: 128,
      actual: 136,
      path: 'accessors[0]',
    });
  });
});
