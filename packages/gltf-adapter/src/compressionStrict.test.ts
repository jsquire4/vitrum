import { describe, expect, it } from 'vitest';
import {
  resolveCompression,
  type DracoDecodeFn,
  type GltfCompressionDiagnostic,
  type MeshoptDecodeFn,
} from './compression.js';
import type { GltfAccessor, GltfJson } from './gltfTypes.js';
import {
  BUILTIN_COMPRESSION_DECODE_BUDGET_BYTES,
  builtinMeshoptDecode,
  createBuiltinMeshoptDecode,
} from './builtinCompressionDecoders.js';
import {
  chargeCompressionHookOutput,
  checkedCompressionProduct,
  compressionTypedArrayInfo,
  CompressionAllocationLedger,
  validateCompressionInputBudget,
} from './compressionLimits.js';

const COMPRESSED = [0xa0, 0xc0, 0xde, 0x01] as const;
const POSITIONS = [0, 0, 0, 1, 0, 0, 0, 1, 0] as const;
const NORMALS = [0, 0, 1, 0, 0, 1, 0, 0, 1] as const;

function f32Buffer(values: readonly number[]): ArrayBuffer {
  return new Float32Array(values).buffer;
}

function u32Buffer(values: readonly number[]): ArrayBuffer {
  return new Uint32Array(values).buffer;
}

function makeMeshoptAsset(realFallback = false): {
  gltf: GltfJson;
  buffers: Map<number, ArrayBuffer>;
} {
  const fallback = new ArrayBuffer(36);
  const compressed = new Uint8Array(COMPRESSED).buffer;
  return {
    gltf: {
      asset: { version: '2.0' },
      extensionsUsed: ['EXT_meshopt_compression'],
      buffers: [
        realFallback
          ? { byteLength: fallback.byteLength }
          : {
              byteLength: fallback.byteLength,
              extensions: { EXT_meshopt_compression: { fallback: true } },
            },
        { byteLength: compressed.byteLength },
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
              byteLength: compressed.byteLength,
              byteStride: 12,
              count: 3,
              mode: 'ATTRIBUTES',
            },
          },
        },
      ],
    },
    buffers: new Map([[1, compressed], ...(realFallback ? [[0, fallback] as const] : [])]),
  };
}

function meshoptExtension(gltf: GltfJson): Record<string | symbol, unknown> {
  return gltf.bufferViews![0]!.extensions!.EXT_meshopt_compression as Record<
    string | symbol,
    unknown
  >;
}

function makeDracoAsset(withFallback: boolean): {
  gltf: GltfJson;
  buffers: Map<number, ArrayBuffer>;
} {
  const compressed = new Uint8Array(COMPRESSED).buffer;
  const positionFallback = f32Buffer(POSITIONS);
  const normalFallback = f32Buffer(NORMALS);
  const indexFallback = u32Buffer([0, 1, 2]);
  const accessors: GltfAccessor[] = [
    { componentType: 5126, count: 3, type: 'VEC3', ...(withFallback ? { bufferView: 1 } : {}) },
    { componentType: 5126, count: 3, type: 'VEC3', ...(withFallback ? { bufferView: 2 } : {}) },
    { componentType: 5125, count: 3, type: 'SCALAR', ...(withFallback ? { bufferView: 3 } : {}) },
  ];
  const gltf: GltfJson = {
    asset: { version: '2.0' },
    extensionsUsed: ['KHR_draco_mesh_compression'],
    meshes: [
      {
        primitives: [
          {
            attributes: { POSITION: 0, NORMAL: 1 },
            indices: 2,
            extensions: {
              KHR_draco_mesh_compression: {
                bufferView: 0,
                attributes: { POSITION: 10, NORMAL: 11 },
              },
            },
          },
        ],
      },
    ],
    accessors,
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: compressed.byteLength },
      ...(withFallback
        ? [
            { buffer: 1, byteOffset: 0, byteLength: positionFallback.byteLength },
            { buffer: 2, byteOffset: 0, byteLength: normalFallback.byteLength },
            { buffer: 3, byteOffset: 0, byteLength: indexFallback.byteLength },
          ]
        : []),
    ],
    buffers: [
      { byteLength: compressed.byteLength },
      ...(withFallback
        ? [
            { byteLength: positionFallback.byteLength },
            { byteLength: normalFallback.byteLength },
            { byteLength: indexFallback.byteLength },
          ]
        : []),
    ],
  };
  return {
    gltf,
    buffers: new Map([
      [0, compressed],
      ...(withFallback
        ? [
            [1, positionFallback] as const,
            [2, normalFallback] as const,
            [3, indexFallback] as const,
          ]
        : []),
    ]),
  };
}

function exactAttributes(): Record<string, Float32Array> {
  return {
    POSITION: new Float32Array(POSITIONS),
    NORMAL: new Float32Array(NORMALS),
  };
}

const exactDraco: DracoDecodeFn = () => ({
  attributes: exactAttributes(),
  indices: new Uint32Array([0, 1, 2]),
});

describe('strict transactional compression resolution', () => {
  it('does not publish earlier meshopt buffers, warnings, or diagnostics after a late Draco failure', async () => {
    const { gltf, buffers } = makeMeshoptAsset(false);
    gltf.accessors = [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }];
    gltf.meshes = [
      {
        primitives: [
          {
            attributes: { POSITION: 0 },
            extensions: {
              KHR_draco_mesh_compression: {
                bufferView: 0,
                attributes: { POSITION: 10 },
              },
            },
          },
        ],
      },
    ];
    gltf.extensionsUsed = ['EXT_meshopt_compression', 'KHR_draco_mesh_compression'];
    gltf.extensionsRequired = ['KHR_draco_mesh_compression'];
    const originalBuffer = buffers.get(1)!;
    const originalBytes = Array.from(new Uint8Array(originalBuffer));
    const warnings = ['existing'];
    const diagnostics: GltfCompressionDiagnostic[] = [];

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          meshoptDecode: () => new Uint8Array(f32Buffer(POSITIONS)),
          dracoDecode: () => {
            throw new Error('late Draco failure');
          },
        },
        warnings,
        (diagnostic) => diagnostics.push(diagnostic),
      ),
    ).rejects.toThrow(/extensionsRequired.*late Draco failure/);

    expect([...buffers.keys()]).toEqual([1]);
    expect(buffers.get(1)).toBe(originalBuffer);
    expect(Array.from(new Uint8Array(originalBuffer))).toEqual(originalBytes);
    expect(warnings).toEqual(['existing']);
    expect(diagnostics).toEqual([]);
    expect(gltf.bufferViews![0]!.extensions!.EXT_meshopt_compression).toBeDefined();
  });

  const invalidMeshoptCases: ReadonlyArray<{
    readonly name: string;
    readonly mutate: (extension: Record<string | symbol, unknown>) => void;
    readonly error: RegExp;
  }> = [
    {
      name: 'non-integer buffer',
      mutate: (ext) => {
        ext.buffer = Number.NaN;
      },
      error: /\.buffer must be a safe integer/,
    },
    {
      name: 'negative byteOffset',
      mutate: (ext) => {
        ext.byteOffset = -1;
      },
      error: /\.byteOffset must be a safe integer/,
    },
    {
      name: 'zero byteLength',
      mutate: (ext) => {
        ext.byteLength = 0;
      },
      error: /\.byteLength must be a safe integer/,
    },
    {
      name: 'fractional stride',
      mutate: (ext) => {
        ext.byteStride = 1.5;
      },
      error: /\.byteStride must be a safe integer/,
    },
    {
      name: 'non-four-byte attribute stride',
      mutate: (ext) => {
        ext.byteStride = 6;
      },
      error: /multiple of 4 for ATTRIBUTES/,
    },
    {
      name: 'oversize stride',
      mutate: (ext) => {
        ext.byteStride = 257;
      },
      error: /byteStride must be <= 256/,
    },
    {
      name: 'zero count',
      mutate: (ext) => {
        ext.count = 0;
      },
      error: /\.count must be a safe integer/,
    },
    {
      name: 'unsafe decoded product',
      mutate: (ext) => {
        ext.count = Number.MAX_SAFE_INTEGER;
      },
      error: /count \* byteStride is not a safe integer/,
    },
    {
      name: 'unknown mode',
      mutate: (ext) => {
        ext.mode = 'POINTS';
      },
      error: /\.mode must be ATTRIBUTES/,
    },
    {
      name: 'unknown filter',
      mutate: (ext) => {
        ext.filter = 'MAGIC';
      },
      error: /\.filter must be NONE/,
    },
    {
      name: 'null filter',
      mutate: (ext) => {
        ext.filter = null;
      },
      error: /\.filter must be NONE/,
    },
    {
      name: 'array glTFProperty extensions',
      mutate: (ext) => {
        ext.extensions = [];
      },
      error: /\.extensions must be an object/,
    },
    {
      name: 'unknown enumerable field',
      mutate: (ext) => {
        ext.mystery = 1;
      },
      error: /unsupported enumerable field "mystery"/,
    },
    {
      name: 'unknown enumerable symbol',
      mutate: (ext) => {
        Object.defineProperty(ext, Symbol('mystery'), { enumerable: true, value: 1 });
      },
      error: /unsupported enumerable field Symbol\(mystery\)/,
    },
    {
      name: 'out-of-range source slice',
      mutate: (ext) => {
        ext.byteOffset = 2;
        ext.byteLength = 4;
      },
      error: /source range.*exceeds declared buffers\[1\]\.byteLength/,
    },
  ];

  for (const testCase of invalidMeshoptCases) {
    it(`rejects meshopt ${testCase.name} before invoking the decoder`, async () => {
      const { gltf, buffers } = makeMeshoptAsset(false);
      testCase.mutate(meshoptExtension(gltf));
      let calls = 0;
      await expect(
        resolveCompression(
          gltf,
          buffers,
          {
            meshoptDecode: () => {
              calls += 1;
              return new Uint8Array(36);
            },
          },
          [],
        ),
      ).rejects.toThrow(testCase.error);
      expect(calls).toBe(0);
      expect([...buffers.keys()]).toEqual([1]);
    });
  }

  it('rejects inherited required meshopt extension fields', async () => {
    const { gltf, buffers } = makeMeshoptAsset(false);
    const inherited = meshoptExtension(gltf);
    gltf.bufferViews![0]!.extensions!.EXT_meshopt_compression = Object.create(inherited);
    let calls = 0;

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          meshoptDecode: () => {
            calls += 1;
            return new Uint8Array(36);
          },
        },
        [],
      ),
    ).rejects.toThrow(/\.buffer must be an own property/);
    expect(calls).toBe(0);
  });

  it('does not activate a meshopt extension inherited through the bufferView', async () => {
    const { gltf, buffers } = makeMeshoptAsset(false);
    const view = gltf.bufferViews![0]!;
    const inheritedExtensions = view.extensions;
    delete view.extensions;
    Object.setPrototypeOf(view, { extensions: inheritedExtensions });
    let calls = 0;

    const out = await resolveCompression(
      gltf,
      buffers,
      {
        meshoptDecode: () => {
          calls += 1;
          return new Uint8Array(36);
        },
      },
      [],
    );

    expect(out).toBe(gltf);
    expect(calls).toBe(0);
  });

  it('validates parent meshopt layout before invoking the decoder', async () => {
    for (const testCase of [
      {
        mutate: (gltf: GltfJson) => {
          gltf.bufferViews![0]!.byteLength = 32;
        },
        error: /byteLength must equal decoded count \* byteStride \(36\)/,
      },
      {
        mutate: (gltf: GltfJson) => {
          gltf.bufferViews![0]!.byteStride = 16;
        },
        error: /byteStride must equal 12 for EXT_meshopt_compression/,
      },
      {
        mutate: (gltf: GltfJson) => {
          gltf.bufferViews![0]!.byteStride = 2;
        },
        error: /byteStride must be a safe integer >= 4/,
      },
    ]) {
      const { gltf, buffers } = makeMeshoptAsset(false);
      testCase.mutate(gltf);
      let calls = 0;
      await expect(
        resolveCompression(
          gltf,
          buffers,
          {
            meshoptDecode: () => {
              calls += 1;
              return new Uint8Array(36);
            },
          },
          [],
        ),
      ).rejects.toThrow(testCase.error);
      expect(calls).toBe(0);
    }
  });

  it('validates meshopt ranges against declared buffer lengths before decoding', async () => {
    {
      const { gltf, buffers } = makeMeshoptAsset(true);
      gltf.buffers![0]!.byteLength = 35;
      await expect(resolveCompression(gltf, buffers, {}, [])).rejects.toThrow(
        /bufferViews\[0\] range \[0, 36\).*declared buffers\[0\]\.byteLength 35/,
      );
    }
    {
      const { gltf, buffers } = makeMeshoptAsset(false);
      gltf.buffers![1]!.byteLength = 3;
      let calls = 0;
      await expect(
        resolveCompression(
          gltf,
          buffers,
          {
            meshoptDecode: () => {
              calls += 1;
              return new Uint8Array(36);
            },
          },
          [],
        ),
      ).rejects.toThrow(/source range \[0, 4\).*declared buffers\[1\]\.byteLength 3/);
      expect(calls).toBe(0);
    }
  });

  it('rejects missing and invalid meshopt buffer descriptors before decoding', async () => {
    for (const mutate of [
      (gltf: GltfJson) => {
        gltf.buffers!.splice(1, 1);
      },
      (gltf: GltfJson) => {
        gltf.buffers![1]!.byteLength = -1;
      },
    ]) {
      const { gltf, buffers } = makeMeshoptAsset(false);
      mutate(gltf);
      let calls = 0;
      await expect(
        resolveCompression(
          gltf,
          buffers,
          {
            meshoptDecode: () => {
              calls += 1;
              return new Uint8Array(36);
            },
          },
          [],
        ),
      ).rejects.toThrow(
        /missing buffer descriptor 1|buffers\[1\]\.byteLength must be a safe integer/,
      );
      expect(calls).toBe(0);
    }
  });

  it('accepts glTFProperty fields and preserves a distinct KHR parent stride', async () => {
    const { gltf, buffers } = makeMeshoptAsset(false);
    const bufferView = gltf.bufferViews![0]!;
    const extension = bufferView.extensions!.EXT_meshopt_compression as Record<string, unknown>;
    delete bufferView.extensions!.EXT_meshopt_compression;
    extension.extras = { source: 'test' };
    extension.extensions = { VENDOR_metadata: { enabled: true } };
    bufferView.extensions!.KHR_meshopt_compression = extension;
    gltf.buffers![0]!.extensions = {
      KHR_meshopt_compression: { fallback: true },
    };
    bufferView.byteStride = 16;
    gltf.extensionsUsed = ['KHR_meshopt_compression'];

    const out = await resolveCompression(
      gltf,
      buffers,
      {
        meshoptDecode: () => new Uint8Array(36),
      },
      [],
    );

    expect(out.bufferViews![0]!.byteStride).toBe(16);
    expect(out.bufferViews![0]!.extensions?.KHR_meshopt_compression).toBeUndefined();
    expect(extension.extras).toEqual({ source: 'test' });
  });

  it('applies base glTF stride constraints to a distinct KHR parent stride', async () => {
    const { gltf, buffers } = makeMeshoptAsset(false);
    const bufferView = gltf.bufferViews![0]!;
    const extension = bufferView.extensions!.EXT_meshopt_compression;
    delete bufferView.extensions!.EXT_meshopt_compression;
    bufferView.extensions!.KHR_meshopt_compression = extension;
    gltf.buffers![0]!.extensions = {
      KHR_meshopt_compression: { fallback: true },
    };
    bufferView.byteStride = 14;
    gltf.extensionsUsed = ['KHR_meshopt_compression'];
    let calls = 0;

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          meshoptDecode: () => {
            calls += 1;
            return new Uint8Array(36);
          },
        },
        [],
      ),
    ).rejects.toThrow(/byteStride must be a multiple of 4 between 4 and 252/);
    expect(calls).toBe(0);
  });

  it('rejects EXT and KHR meshopt declarations that coexist on a view or buffer', async () => {
    {
      const { gltf, buffers } = makeMeshoptAsset(false);
      gltf.bufferViews![0]!.extensions!.KHR_meshopt_compression = {
        ...(gltf.bufferViews![0]!.extensions!.EXT_meshopt_compression as object),
      };
      await expect(resolveCompression(gltf, buffers, {}, [])).rejects.toThrow(
        /must not coexist on one bufferView/,
      );
    }
    {
      const { gltf, buffers } = makeMeshoptAsset(false);
      gltf.buffers![0]!.extensions!.KHR_meshopt_compression = { fallback: true };
      await expect(resolveCompression(gltf, buffers, {}, [])).rejects.toThrow(
        /must not coexist on buffer 0/,
      );
    }
  });

  it('rejects a meshopt buffer marker whose extension name differs from its view', async () => {
    const { gltf, buffers } = makeMeshoptAsset(false);
    gltf.buffers![0]!.extensions = {
      KHR_meshopt_compression: { fallback: true },
    };
    await expect(resolveCompression(gltf, buffers, {}, [])).rejects.toThrow(
      /buffer 0 is marked KHR_meshopt_compression fallback:true.*bufferViews\[0\].*carries EXT_meshopt_compression/,
    );
  });

  it('rejects any ordinary bufferView that aliases a fallback:true buffer', async () => {
    const { gltf, buffers } = makeMeshoptAsset(false);
    gltf.bufferViews!.push({ buffer: 0, byteOffset: 0, byteLength: 4 });

    await expect(resolveCompression(gltf, buffers, {}, [])).rejects.toThrow(
      /buffer 0 is marked EXT_meshopt_compression fallback:true.*bufferViews\[1\].*must carry EXT_meshopt_compression/,
    );
  });

  it('uses real fallback:true bytes, clears the marker, and retains glTFProperty metadata', async () => {
    const { gltf, buffers } = makeMeshoptAsset(true);
    gltf.buffers![0]!.extensions = {
      EXT_meshopt_compression: {
        fallback: true,
        extras: { retained: true },
        extensions: { VENDOR_metadata: { enabled: true } },
      },
    };

    const out = await resolveCompression(gltf, buffers, {}, []);
    const marker = out.buffers![0]!.extensions!.EXT_meshopt_compression as Record<string, unknown>;
    expect(marker.fallback).toBeUndefined();
    expect(marker.extras).toEqual({ retained: true });
    expect(marker.extensions).toEqual({ VENDOR_metadata: { enabled: true } });
    expect(out.bufferViews![0]!.extensions?.EXT_meshopt_compression).toBeUndefined();
    expect(gltf.buffers?.[0]?.extensions?.EXT_meshopt_compression).toEqual(
      expect.objectContaining({ fallback: true }),
    );
  });

  it('materializes a scoped fallback view without clearing a shared fallback marker', async () => {
    const { gltf, buffers } = makeMeshoptAsset(true);
    const fallback = new ArrayBuffer(72);
    const fallbackBytes = new Uint8Array(fallback);
    for (let index = 0; index < fallbackBytes.length; index += 1) {
      fallbackBytes[index] = index + 1;
    }
    buffers.set(0, fallback);
    gltf.buffers![0] = {
      byteLength: fallback.byteLength,
      extensions: {
        EXT_meshopt_compression: {
          fallback: true,
          extras: { retained: true },
          extensions: { VENDOR_metadata: { enabled: true } },
        },
      },
    };
    gltf.bufferViews!.push({
      buffer: 0,
      byteOffset: 36,
      byteLength: 36,
      byteStride: 12,
      extensions: {
        EXT_meshopt_compression: {
          buffer: 1,
          byteOffset: 0,
          byteLength: COMPRESSED.length,
          byteStride: 12,
          count: 3,
          mode: 'ATTRIBUTES',
        },
      },
    });

    const out = await resolveCompression(gltf, buffers, {}, [], undefined, {
      bufferViewIndices: new Set([0]),
    });

    const resolved = out.bufferViews![0]!;
    const unresolved = out.bufferViews![1]!;
    expect(resolved.extensions?.EXT_meshopt_compression).toBeUndefined();
    expect(resolved.buffer).not.toBe(0);
    expect(resolved.byteOffset).toBe(0);
    expect(Array.from(new Uint8Array(buffers.get(resolved.buffer)!))).toEqual(
      Array.from(fallbackBytes.subarray(0, 36)),
    );
    expect(unresolved.buffer).toBe(0);
    expect(unresolved.extensions?.EXT_meshopt_compression).toBeDefined();
    expect(out.buffers![0]!.extensions?.EXT_meshopt_compression).toEqual({
      fallback: true,
      extras: { retained: true },
      extensions: { VENDOR_metadata: { enabled: true } },
    });
    expect(gltf.buffers?.[0]?.extensions?.EXT_meshopt_compression).toEqual(
      expect.objectContaining({ fallback: true }),
    );
  });

  it('rejects a compressed source marked fallback-only before invoking the decoder', async () => {
    const { gltf, buffers } = makeMeshoptAsset(true);
    gltf.buffers![1]!.extensions = {
      EXT_meshopt_compression: { fallback: true },
    };
    let calls = 0;
    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          meshoptDecode: () => {
            calls += 1;
            return new Uint8Array(36);
          },
        },
        [],
      ),
    ).rejects.toThrow(
      /references buffer 1, which is marked EXT_meshopt_compression fallback:true.*cannot be used as compressed source data/,
    );

    expect(calls).toBe(0);
  });

  it('validates meshopt bitstream headers by extension and mode before hooks', async () => {
    const cases = [
      {
        name: 'EXT_meshopt_compression' as const,
        mode: 'ATTRIBUTES',
        header: 0xff,
        error: /EXT_meshopt_compression ATTRIBUTES data has invalid header/,
      },
      {
        name: 'KHR_meshopt_compression' as const,
        mode: 'ATTRIBUTES',
        header: 0xff,
        error: /KHR_meshopt_compression ATTRIBUTES data has invalid header/,
      },
      {
        name: 'EXT_meshopt_compression' as const,
        mode: 'TRIANGLES',
        header: 0xa0,
        error: /TRIANGLES data has invalid header.*expected 0xe1/,
      },
      {
        name: 'KHR_meshopt_compression' as const,
        mode: 'INDICES',
        header: 0xe1,
        error: /INDICES data has invalid header.*expected 0xd1/,
      },
    ];

    for (const testCase of cases) {
      const { gltf, buffers } = makeMeshoptAsset(false);
      const view = gltf.bufferViews![0]!;
      const extension = view.extensions!.EXT_meshopt_compression as Record<string, unknown>;
      if (testCase.name === 'KHR_meshopt_compression') {
        delete view.extensions!.EXT_meshopt_compression;
        view.extensions!.KHR_meshopt_compression = extension;
        gltf.buffers![0]!.extensions = {
          KHR_meshopt_compression: { fallback: true },
        };
      }
      if (testCase.mode !== 'ATTRIBUTES') {
        extension.mode = testCase.mode;
        extension.byteStride = 2;
        extension.count = 3;
        view.byteLength = 6;
        delete view.byteStride;
      }
      gltf.extensionsUsed = [testCase.name];
      gltf.extensionsRequired = [testCase.name];
      buffers.set(1, new Uint8Array([testCase.header, 0, 0, 0]).buffer);
      let calls = 0;

      await expect(
        resolveCompression(
          gltf,
          buffers,
          {
            meshoptDecode: () => {
              calls += 1;
              return new Uint8Array(view.byteLength);
            },
          },
          [],
        ),
      ).rejects.toThrow(testCase.error);
      expect(calls).toBe(0);
    }
  });

  it('reports an invalid meshopt header as degraded only after using an exact fallback', async () => {
    const { gltf, buffers } = makeMeshoptAsset(true);
    buffers.set(1, new Uint8Array([0xff, 0, 0, 0]).buffer);
    const diagnostics: GltfCompressionDiagnostic[] = [];
    let calls = 0;

    const out = await resolveCompression(
      gltf,
      buffers,
      {
        meshoptDecode: () => {
          calls += 1;
          return new Uint8Array(36);
        },
      },
      [],
      (diagnostic) => diagnostics.push(diagnostic),
    );

    expect(calls).toBe(0);
    expect(out.bufferViews![0]!.extensions?.EXT_meshopt_compression).toBeUndefined();
    expect(diagnostics).toEqual([
      expect.objectContaining({
        code: 'meshopt-invalid-bitstream-header',
        bufferViewIndex: 0,
      }),
    ]);
  });

  it('accepts the v0 ATTRIBUTES header under KHR_meshopt_compression', async () => {
    const { gltf, buffers } = makeMeshoptAsset(false);
    const view = gltf.bufferViews![0]!;
    const extension = view.extensions!.EXT_meshopt_compression;
    delete view.extensions!.EXT_meshopt_compression;
    view.extensions!.KHR_meshopt_compression = extension;
    gltf.buffers![0]!.extensions = {
      KHR_meshopt_compression: { fallback: true },
    };
    gltf.extensionsUsed = ['KHR_meshopt_compression'];
    gltf.extensionsRequired = ['KHR_meshopt_compression'];
    let calls = 0;

    const out = await resolveCompression(
      gltf,
      buffers,
      {
        meshoptDecode: () => {
          calls += 1;
          return new Uint8Array(36);
        },
      },
      [],
    );

    expect(calls).toBe(1);
    expect(out.bufferViews![0]!.extensions?.KHR_meshopt_compression).toBeUndefined();
  });

  it('uses a fully validated meshopt fallback for a malformed decoder result', async () => {
    const { gltf, buffers } = makeMeshoptAsset(true);
    const diagnostics: GltfCompressionDiagnostic[] = [];
    const malformed = (() => new Uint16Array(18)) as unknown as MeshoptDecodeFn;

    const out = await resolveCompression(
      gltf,
      buffers,
      { meshoptDecode: malformed },
      [],
      (diagnostic) => diagnostics.push(diagnostic),
    );

    expect(out.bufferViews![0]!.extensions?.EXT_meshopt_compression).toBeUndefined();
    expect(diagnostics).toEqual([
      expect.objectContaining({ code: 'meshopt-decode-hook-failed', bufferViewIndex: 0 }),
    ]);
  });

  it('uses intrinsic meshopt result size instead of shadowable instance fields', async () => {
    const { gltf, buffers } = makeMeshoptAsset(true);
    const short = new Uint8Array(1);
    Object.defineProperty(short, 'byteLength', { value: 36 });
    const diagnostics: GltfCompressionDiagnostic[] = [];

    const out = await resolveCompression(
      gltf,
      buffers,
      { meshoptDecode: () => short },
      [],
      (diagnostic) => diagnostics.push(diagnostic),
    );

    expect(out.bufferViews![0]!.buffer).toBe(0);
    expect(diagnostics).toEqual([
      expect.objectContaining({ code: 'meshopt-decoded-byte-length-mismatch' }),
    ]);
  });

  it('uses intrinsic meshopt source size instead of a shadowable ArrayBuffer byteLength', async () => {
    const { gltf, buffers } = makeMeshoptAsset(false);
    const source = buffers.get(1)!;
    Object.defineProperty(source, 'byteLength', { value: 0 });

    const out = await resolveCompression(
      gltf,
      buffers,
      { meshoptDecode: () => new Uint8Array(36) },
      [],
    );

    expect(out.bufferViews![0]!.extensions?.EXT_meshopt_compression).toBeUndefined();
  });

  it('accepts a genuine cross-realm Uint8Array meshopt result', async () => {
    const { runInNewContext } = await import('node:vm');
    const { gltf, buffers } = makeMeshoptAsset(false);
    const crossRealm = runInNewContext('new Uint8Array(36)') as Uint8Array;

    const out = await resolveCompression(gltf, buffers, { meshoptDecode: () => crossRealm }, []);

    expect(out.bufferViews![0]!.extensions?.EXT_meshopt_compression).toBeUndefined();
    expect(out.bufferViews![0]!.byteLength).toBe(36);
  });

  it('rejects shared meshopt output and uses an exact fallback', async () => {
    const { gltf, buffers } = makeMeshoptAsset(true);
    const shared = new Uint8Array(new SharedArrayBuffer(36));
    const diagnostics: GltfCompressionDiagnostic[] = [];

    const out = await resolveCompression(
      gltf,
      buffers,
      { meshoptDecode: () => shared },
      [],
      (diagnostic) => diagnostics.push(diagnostic),
    );

    expect(out.bufferViews![0]!.buffer).toBe(0);
    expect(diagnostics).toEqual([expect.objectContaining({ code: 'meshopt-decode-hook-failed' })]);
  });

  it('routes hostile meshopt result branding through an exact fallback', async () => {
    const { gltf, buffers } = makeMeshoptAsset(true);
    const hostile = Object.create(null) as Record<PropertyKey, unknown>;
    Object.defineProperty(hostile, Symbol.toStringTag, {
      get: () => {
        throw new Error('do not inspect me');
      },
    });

    const out = await resolveCompression(
      gltf,
      buffers,
      { meshoptDecode: () => hostile as unknown as Uint8Array },
      [],
    );

    expect(out.bufferViews![0]!.buffer).toBe(0);
  });

  it('rejects a meshopt fallback whose byte length is not exactly count × stride', async () => {
    const { gltf, buffers } = makeMeshoptAsset(true);
    gltf.bufferViews![0]!.byteLength = 35;
    await expect(resolveCompression(gltf, buffers, {}, [])).rejects.toThrow(
      /byteLength must equal decoded count \* byteStride \(36\); received 35/,
    );
  });

  it('keeps caller-owned compressed bytes immutable even when a meshopt hook mutates its input', async () => {
    const { gltf, buffers } = makeMeshoptAsset(false);
    const source = buffers.get(1)!;
    await resolveCompression(
      gltf,
      buffers,
      {
        meshoptDecode: (compressed) => {
          compressed.fill(0);
          return new Uint8Array(36);
        },
      },
      [],
    );
    expect(Array.from(new Uint8Array(source))).toEqual(COMPRESSED);
  });

  it('publishes no adapter buffers when a hook claims a staged synthetic index', async () => {
    const { gltf, buffers } = makeMeshoptAsset(false);
    const claimed = new ArrayBuffer(7);
    const warnings = ['existing'];

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          meshoptDecode: () => {
            buffers.set(2, claimed);
            return new Uint8Array(36);
          },
        },
        warnings,
      ),
    ).rejects.toThrow(/claimed synthetic buffer index 2.*no adapter-owned buffers were published/);

    expect(buffers.get(2)).toBe(claimed);
    expect([...buffers.keys()]).toEqual([1, 2]);
    expect(warnings).toEqual(['existing']);
  });

  it('ignores a huge unrelated host buffer key when publishing meshopt output', async () => {
    const { gltf, buffers } = makeMeshoptAsset(false);
    buffers.set(Number.MAX_SAFE_INTEGER, new ArrayBuffer(0));

    const out = await resolveCompression(
      gltf,
      buffers,
      {
        meshoptDecode: () => new Uint8Array(f32Buffer(POSITIONS)),
      },
      [],
    );

    expect(out.buffers).toHaveLength(3);
    expect(out.bufferViews![0]!.buffer).toBe(2);
    expect(buffers.has(2)).toBe(true);
    expect(buffers.has(Number.MAX_SAFE_INTEGER)).toBe(true);
  });

  it('charges meshopt allocations only after source and header validation', async () => {
    const oversized = BUILTIN_COMPRESSION_DECODE_BUDGET_BYTES + 1;

    {
      const { gltf, buffers } = makeMeshoptAsset(true);
      const extension = meshoptExtension(gltf);
      extension.byteLength = oversized;
      gltf.buffers![1]!.byteLength = oversized;
      const out = await resolveCompression(gltf, buffers, {}, []);
      expect(out.bufferViews![0]!.extensions?.EXT_meshopt_compression).toBeUndefined();
    }

    {
      const { gltf, buffers } = makeMeshoptAsset(true);
      const extension = meshoptExtension(gltf);
      extension.byteLength = oversized;
      gltf.buffers![1]!.byteLength = oversized;
      const diagnostics: GltfCompressionDiagnostic[] = [];
      let calls = 0;
      const out = await resolveCompression(
        gltf,
        buffers,
        {
          meshoptDecode: () => {
            calls += 1;
            return new Uint8Array(36);
          },
        },
        [],
        (diagnostic) => diagnostics.push(diagnostic),
      );
      expect(calls).toBe(0);
      expect(out.bufferViews![0]!.extensions?.EXT_meshopt_compression).toBeUndefined();
      expect(diagnostics).toEqual([
        expect.objectContaining({
          code: 'meshopt-buffer-unavailable',
        }),
      ]);
    }

    {
      const { gltf, buffers } = makeMeshoptAsset(true);
      const extension = meshoptExtension(gltf);
      extension.byteLength = oversized;
      gltf.buffers![1]!.byteLength = oversized;
      gltf.extensionsRequired = ['EXT_meshopt_compression'];
      let calls = 0;
      await expect(
        resolveCompression(
          gltf,
          buffers,
          {
            meshoptDecode: () => {
              calls += 1;
              return new Uint8Array(36);
            },
          },
          [],
        ),
      ).rejects.toThrow(/source range.*exceeds buffer length/);
      expect(calls).toBe(0);
    }

    {
      const { gltf, buffers } = makeMeshoptAsset(false);
      const extension = meshoptExtension(gltf);
      const count = Math.floor(BUILTIN_COMPRESSION_DECODE_BUDGET_BYTES / 4) + 1;
      extension.byteStride = 4;
      extension.count = count;
      gltf.bufferViews![0]!.byteLength = count * 4;
      gltf.bufferViews![0]!.byteStride = 4;
      gltf.buffers![0]!.byteLength = count * 4;
      gltf.extensionsRequired = ['EXT_meshopt_compression'];
      let calls = 0;
      await expect(
        resolveCompression(
          gltf,
          buffers,
          {
            meshoptDecode: () => {
              calls += 1;
              return new Uint8Array();
            },
          },
          [],
        ),
      ).rejects.toThrow(/cumulative allocation.*exceeds the compression budget/);
      expect(calls).toBe(0);
    }
  });
});

describe('built-in compression allocation guards', () => {
  it('checks cumulative allocation arithmetic without allocating the declared size', () => {
    const ledger = new CompressionAllocationLedger();
    ledger.charge(BUILTIN_COMPRESSION_DECODE_BUDGET_BYTES - 1, 'boundary');
    ledger.charge(1, 'boundary');
    expect(ledger.chargedBytes).toBe(BUILTIN_COMPRESSION_DECODE_BUDGET_BYTES);
    expect(() => ledger.charge(1, 'overflow')).toThrow(
      /cumulative allocation.*exceeds the compression budget/,
    );
    expect(() => checkedCompressionProduct(Number.MAX_SAFE_INTEGER, 2, 'unsafe product')).toThrow(
      /exceeds the safe integer range/,
    );
    expect(() => checkedCompressionProduct(-1, 0, 'negative product')).toThrow(/invalid factor/);
  });

  it('keeps safe arithmetic while zero disables duplicate compression ceilings', async () => {
    const oversized = BUILTIN_COMPRESSION_DECODE_BUDGET_BYTES + 1;
    const ledger = new CompressionAllocationLedger(0);
    ledger.charge(oversized, 'zero-opt-out');
    expect(ledger.chargedBytes).toBe(oversized);
    expect(() => validateCompressionInputBudget(oversized, 'zero-opt-out input', 0)).not.toThrow();
    expect(() => ledger.charge(Number.MAX_SAFE_INTEGER, 'unsafe cumulative sum')).toThrow(
      /safe integer range/,
    );

    const planned: number[] = [];
    const decode = createBuiltinMeshoptDecode({
      maxCompressedInputBytes: 0,
      maxDecodedOutputBytes: 0,
      beforeDecodedAllocation: (plannedByteLength) => {
        planned.push(plannedByteLength);
        throw new Error('planned-without-fixed-cap');
      },
    });
    const count = Math.floor(BUILTIN_COMPRESSION_DECODE_BUDGET_BYTES / 4) + 1;
    await expect(decode(new Uint8Array(), count, 4, 'ATTRIBUTES', 'NONE')).rejects.toThrow(
      'planned-without-fixed-cap',
    );
    expect(planned).toEqual([count * 4]);
  });

  it('applies configured built-in ceilings before loading a codec', async () => {
    const inputLimited = createBuiltinMeshoptDecode({
      maxCompressedInputBytes: 3,
      maxDecodedOutputBytes: 0,
    });
    await expect(inputLimited(new Uint8Array(4), 1, 4, 'ATTRIBUTES', 'NONE')).rejects.toThrow(
      /compression input budget of 3 bytes/,
    );

    const outputLimited = createBuiltinMeshoptDecode({
      maxCompressedInputBytes: 0,
      maxDecodedOutputBytes: 3,
    });
    await expect(outputLimited(new Uint8Array(), 1, 4, 'ATTRIBUTES', 'NONE')).rejects.toThrow(
      /built-in decode memory budget of 3 bytes/,
    );
  });

  it('intrinsically sizes unsupported hook typed arrays for budget accounting', () => {
    const output = new Float64Array(5);
    Object.defineProperty(output, 'byteLength', { value: 1 });
    const info = compressionTypedArrayInfo(output);
    expect(info).toMatchObject({
      kind: 'Float64Array',
      byteLength: 40,
      length: 5,
      shared: false,
    });
    const ledger = new CompressionAllocationLedger(39);
    const state = { attemptsDisabled: false };
    expect(() =>
      chargeCompressionHookOutput(ledger, state, info!.byteLength, 'wrong-kind output'),
    ).toThrow(/cumulative allocation 40.*budget of 39/);
    expect(state.attemptsDisabled).toBe(true);
    expect(ledger.chargedBytes).toBe(0);
  });

  it('rejects invalid meshopt dimensions before loading or invoking the decoder', async () => {
    await expect(
      builtinMeshoptDecode(new Uint8Array(), 0, 4, 'ATTRIBUTES', 'NONE'),
    ).rejects.toThrow(/meshopt count must be a safe integer >= 1/);
    await expect(
      builtinMeshoptDecode(new Uint8Array(), Number.MAX_SAFE_INTEGER, 2, 'ATTRIBUTES', 'NONE'),
    ).rejects.toThrow(/count × byteStride is not a safe integer/);
  });

  it('rejects meshopt output beyond the explicit built-in memory budget before allocation', async () => {
    const count = Math.floor(BUILTIN_COMPRESSION_DECODE_BUDGET_BYTES / 4) + 1;
    await expect(
      builtinMeshoptDecode(new Uint8Array(), count, 4, 'ATTRIBUTES', 'NONE'),
    ).rejects.toThrow(/exceeds the built-in decode memory budget/);
  });
});

describe('strict Draco stream resolution', () => {
  it('rejects extension attribute IDs outside uint32 before invoking a custom hook', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const extension = gltf.meshes![0]!.primitives[0]!.extensions!.KHR_draco_mesh_compression as {
      attributes: Record<string, number>;
    };
    extension.attributes.POSITION = 2 ** 32;
    let calls = 0;

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => {
            calls += 1;
            return exactDraco(new Uint8Array(), {});
          },
        },
        [],
      ),
    ).rejects.toThrow(/attributes\.POSITION must be <= 4294967295/);
    expect(calls).toBe(0);
  });

  it('validates the compressed range against the declared buffer length before decoding', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    gltf.buffers![0]!.byteLength = 3;
    let calls = 0;

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => {
            calls += 1;
            return exactDraco(new Uint8Array(), {});
          },
        },
        [],
      ),
    ).rejects.toThrow(/bufferViews\[0\] range \[0, 4\).*declared buffers\[0\]\.byteLength 3/);
    expect(calls).toBe(0);
  });

  it('applies the Draco input budget only when compressed decoding is attempted', async () => {
    const oversized = BUILTIN_COMPRESSION_DECODE_BUDGET_BYTES + 1;

    {
      const { gltf, buffers } = makeDracoAsset(true);
      gltf.bufferViews![0]!.byteLength = oversized;
      gltf.buffers![0]!.byteLength = oversized;
      const out = await resolveCompression(gltf, buffers, {}, []);
      expect(out.meshes![0]!.primitives[0]!.extensions?.KHR_draco_mesh_compression).toBeUndefined();
    }

    {
      const { gltf, buffers } = makeDracoAsset(true);
      gltf.bufferViews![0]!.byteLength = oversized;
      gltf.buffers![0]!.byteLength = oversized;
      let calls = 0;
      const out = await resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => {
            calls += 1;
            return exactDraco(new Uint8Array(), {});
          },
        },
        [],
      );
      expect(calls).toBe(0);
      expect(out.meshes![0]!.primitives[0]!.extensions?.KHR_draco_mesh_compression).toBeUndefined();
    }

    {
      const { gltf, buffers } = makeDracoAsset(true);
      gltf.bufferViews![0]!.byteLength = oversized;
      gltf.buffers![0]!.byteLength = oversized;
      gltf.extensionsRequired = ['KHR_draco_mesh_compression'];
      let calls = 0;
      await expect(
        resolveCompression(
          gltf,
          buffers,
          {
            dracoDecode: () => {
              calls += 1;
              return exactDraco(new Uint8Array(), {});
            },
          },
          [],
        ),
      ).rejects.toThrow(/compression input budget|compression budget/);
      expect(calls).toBe(0);
    }
  });

  it('rejects inherited required Draco extension fields before decoding', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const primitive = gltf.meshes![0]!.primitives[0]!;
    const inherited = primitive.extensions!.KHR_draco_mesh_compression;
    primitive.extensions!.KHR_draco_mesh_compression = Object.create(inherited as object);
    let calls = 0;

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => {
            calls += 1;
            return exactDraco(new Uint8Array(), {});
          },
        },
        [],
      ),
    ).rejects.toThrow(/\.bufferView and .*\.attributes must be own properties/);
    expect(calls).toBe(0);
  });

  it('does not activate a Draco extension inherited through the primitive', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const primitive = gltf.meshes![0]!.primitives[0]!;
    const inheritedExtensions = primitive.extensions;
    delete primitive.extensions;
    Object.setPrototypeOf(primitive, { extensions: inheritedExtensions });
    let calls = 0;

    const out = await resolveCompression(
      gltf,
      buffers,
      {
        dracoDecode: () => {
          calls += 1;
          return exactDraco(new Uint8Array(), {});
        },
      },
      [],
    );

    expect(out).toBe(gltf);
    expect(calls).toBe(0);
  });

  it('rejects an inherited primitive accessor mapping before decoding', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const primitive = gltf.meshes![0]!.primitives[0]!;
    primitive.attributes = Object.create({ POSITION: 0 }) as typeof primitive.attributes;
    primitive.extensions!.KHR_draco_mesh_compression = {
      bufferView: 0,
      attributes: { POSITION: 10 },
    };
    let calls = 0;

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => {
            calls += 1;
            return exactDraco(new Uint8Array(), {});
          },
        },
        [],
      ),
    ).rejects.toThrow(/semantic "POSITION" has no exact primitive accessor mapping/);
    expect(calls).toBe(0);
  });

  it('does not treat an inherited POSITION mapping as authored point-count evidence', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const primitive = gltf.meshes![0]!.primitives[0]!;
    primitive.attributes = Object.assign(
      Object.create({ POSITION: 0 }) as typeof primitive.attributes,
      { NORMAL: 1 },
    );
    primitive.extensions!.KHR_draco_mesh_compression = {
      bufferView: 0,
      attributes: { NORMAL: 11 },
    };
    gltf.accessors![0]!.count = 4;
    let calls = 0;

    const out = await resolveCompression(
      gltf,
      buffers,
      {
        dracoDecode: () => {
          calls += 1;
          return {
            attributes: { NORMAL: new Float32Array(NORMALS) },
            indices: new Uint32Array([0, 1, 2]),
          };
        },
      },
      [],
    );

    expect(calls).toBe(1);
    expect(
      Object.prototype.hasOwnProperty.call(out.meshes![0]!.primitives[0]!.attributes, 'POSITION'),
    ).toBe(false);
    expect(out.meshes![0]!.primitives[0]!.attributes.NORMAL).not.toBe(1);
  });

  it('does not default explicit invalid Draco topology fields', async () => {
    for (const mutate of [
      (gltf: GltfJson) => {
        gltf.meshes![0]!.primitives[0]!.mode = null as unknown as number;
      },
      (gltf: GltfJson) => {
        Object.defineProperty(gltf.meshes![0]!.primitives[0]!, 'indices', {
          configurable: true,
          enumerable: true,
          writable: true,
          value: undefined,
        });
      },
    ]) {
      const { gltf, buffers } = makeDracoAsset(false);
      mutate(gltf);
      let calls = 0;
      await expect(
        resolveCompression(
          gltf,
          buffers,
          {
            dracoDecode: () => {
              calls += 1;
              return exactDraco(new Uint8Array(), {});
            },
          },
          [],
        ),
      ).rejects.toThrow(/mode must be TRIANGLES|indices must be a safe integer/);
      expect(calls).toBe(0);
    }
  });
  it('falls back the whole primitive atomically when one decoded semantic is omitted', async () => {
    const { gltf, buffers } = makeDracoAsset(true);
    const diagnostics: GltfCompressionDiagnostic[] = [];
    const out = await resolveCompression(
      gltf,
      buffers,
      {
        dracoDecode: () => ({
          attributes: { POSITION: new Float32Array(POSITIONS.map((value) => value + 9)) },
          indices: new Uint32Array([0, 1, 2]),
        }),
      },
      [],
      (diagnostic) => diagnostics.push(diagnostic),
    );

    const primitive = out.meshes![0]!.primitives[0]!;
    expect(primitive.attributes.POSITION).toBe(0);
    expect(primitive.attributes.NORMAL).toBe(1);
    expect(primitive.indices).toBe(2);
    expect(out.accessors).toHaveLength(3);
    expect(out.bufferViews).toHaveLength(4);
    expect([...buffers.keys()]).toEqual([0, 1, 2, 3]);
    expect(diagnostics).toEqual([
      expect.objectContaining({
        code: 'draco-fallback-accessors-used',
        path: 'meshes[0].primitives[0].extensions.KHR_draco_mesh_compression',
      }),
    ]);
  });

  it('uses an independently valid whole fallback when an optional Draco declaration is malformed', async () => {
    const { gltf, buffers } = makeDracoAsset(true);
    const diagnostics: GltfCompressionDiagnostic[] = [];
    gltf.meshes![0]!.primitives[0]!.extensions!.KHR_draco_mesh_compression = {
      bufferView: Number.NaN,
      attributes: { POSITION: 10, NORMAL: 11 },
    };

    const out = await resolveCompression(
      gltf,
      buffers,
      {},
      [],
      (diagnostic) => diagnostics.push(diagnostic),
    );

    const primitive = out.meshes![0]!.primitives[0]!;
    expect(primitive.attributes).toEqual({ POSITION: 0, NORMAL: 1 });
    expect(primitive.indices).toBe(2);
    expect(primitive.extensions?.KHR_draco_mesh_compression).toBeUndefined();
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: 'draco-fallback-accessors-used',
    }));
  });

  it('rejects the same malformed Draco declaration when the extension is required', async () => {
    const { gltf, buffers } = makeDracoAsset(true);
    gltf.extensionsRequired = ['KHR_draco_mesh_compression'];
    gltf.meshes![0]!.primitives[0]!.extensions!.KHR_draco_mesh_compression = {
      bufferView: Number.NaN,
      attributes: { POSITION: 10, NORMAL: 11 },
    };

    await expect(resolveCompression(gltf, buffers, {}, []))
      .rejects.toThrow(/bufferView must be a safe integer >= 0/);
  });

  it('falls back the whole primitive atomically when decoded indices are invalid', async () => {
    const { gltf, buffers } = makeDracoAsset(true);
    const diagnostics: GltfCompressionDiagnostic[] = [];
    const out = await resolveCompression(
      gltf,
      buffers,
      {
        dracoDecode: () => ({
          attributes: {
            POSITION: new Float32Array(POSITIONS.map((value) => value + 9)),
            NORMAL: new Float32Array(NORMALS),
          },
          indices: new Uint32Array([0, 1, 3]),
        }),
      },
      [],
      (diagnostic) => diagnostics.push(diagnostic),
    );

    const primitive = out.meshes![0]!.primitives[0]!;
    expect(primitive.attributes).toEqual({ POSITION: 0, NORMAL: 1 });
    expect(primitive.indices).toBe(2);
    expect(out.accessors).toHaveLength(3);
    expect([...buffers.keys()]).toEqual([0, 1, 2, 3]);
    expect(diagnostics).toEqual([
      expect.objectContaining({ code: 'draco-fallback-accessors-used' }),
    ]);
  });

  it('validates non-Draco primitive attributes before claiming a whole fallback', async () => {
    const { gltf, buffers } = makeDracoAsset(true);
    gltf.accessors!.push({ componentType: 5126, count: 3, type: 'VEC4' });
    gltf.meshes![0]!.primitives[0]!.attributes.COLOR_0 = 3;

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => ({
            attributes: { POSITION: new Float32Array(POSITIONS) },
            indices: new Uint32Array([0, 1, 2]),
          }),
        },
        [],
      ),
    ).rejects.toThrow(
      /COLOR_0.*neither a fallback bufferView nor sparse data|accessor 3 has neither a fallback bufferView nor sparse data/,
    );
  });

  it('accepts sparse-only attribute and index accessors as an atomic fallback', async () => {
    const { gltf, buffers } = makeDracoAsset(true);
    const sparseBytes = new ArrayBuffer(32);
    const sparseView = new DataView(sparseBytes);
    sparseView.setUint8(0, 0);
    sparseView.setUint8(1, 1);
    sparseView.setUint8(2, 2);
    sparseView.setUint32(4, 0, true);
    sparseView.setUint32(8, 1, true);
    sparseView.setUint32(12, 2, true);
    sparseView.setUint8(16, 1);
    sparseView.setFloat32(20, 0.25, true);
    sparseView.setFloat32(24, 0.5, true);
    sparseView.setFloat32(28, 0.75, true);
    buffers.set(4, sparseBytes);
    gltf.buffers!.push({ byteLength: sparseBytes.byteLength });
    gltf.bufferViews!.push(
      { buffer: 4, byteOffset: 0, byteLength: 3 },
      { buffer: 4, byteOffset: 4, byteLength: 12 },
      { buffer: 4, byteOffset: 16, byteLength: 1 },
      { buffer: 4, byteOffset: 20, byteLength: 12 },
    );
    const indexAccessor = gltf.accessors![2]!;
    delete indexAccessor.bufferView;
    indexAccessor.sparse = {
      count: 3,
      indices: { bufferView: 4, componentType: 5121 },
      values: { bufferView: 5 },
    };
    gltf.accessors!.push({
      componentType: 5126,
      count: 3,
      type: 'SCALAR',
      sparse: {
        count: 1,
        indices: { bufferView: 6, componentType: 5121 },
        values: { bufferView: 7 },
      },
    });
    gltf.meshes![0]!.primitives[0]!.attributes._WEIGHT = 3;

    const out = await resolveCompression(
      gltf,
      buffers,
      {
        dracoDecode: () => {
          throw new Error('force exact fallback');
        },
      },
      [],
    );

    const primitive = out.meshes![0]!.primitives[0]!;
    expect(primitive.indices).toBe(2);
    expect(primitive.attributes._WEIGHT).toBe(3);
    expect(out.accessors![2]!.bufferView).toBeUndefined();
    expect(out.accessors![2]!.sparse).toBeDefined();
    expect(out.accessors![3]!.bufferView).toBeUndefined();
    expect(out.accessors![3]!.sparse).toBeDefined();
  });

  it('accepts glTFProperty fields, aliased accessors/IDs, and __proto__ semantics', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const primitive = gltf.meshes![0]!.primitives[0]!;
    primitive.attributes = { POSITION: 0 };
    Object.defineProperty(primitive.attributes, '__proto__', {
      enumerable: true,
      configurable: true,
      writable: true,
      value: 0,
    });
    primitive.indices = 1;
    gltf.accessors = [
      { componentType: 5126, count: 3, type: 'VEC3' },
      { componentType: 5125, count: 3, type: 'SCALAR' },
    ];
    const extension = primitive.extensions!.KHR_draco_mesh_compression as Record<string, unknown>;
    const extensionAttributes: Record<string, number> = { POSITION: 10 };
    Object.defineProperty(extensionAttributes, '__proto__', {
      enumerable: true,
      configurable: true,
      writable: true,
      value: 10,
    });
    extension.attributes = extensionAttributes;
    extension.extras = { source: 'test' };
    extension.extensions = { VENDOR_metadata: { enabled: true } };
    let receivedIds: Readonly<Record<string, number>> | undefined;

    const out = await resolveCompression(
      gltf,
      buffers,
      {
        dracoDecode: (_compressed, ids) => {
          receivedIds = ids;
          const attributes = Object.create(null) as Record<string, Float32Array>;
          attributes.POSITION = new Float32Array(POSITIONS);
          attributes.__proto__ = new Float32Array(POSITIONS);
          return { attributes, indices: new Uint32Array([0, 1, 2]) };
        },
      },
      [],
    );

    expect(Object.prototype.hasOwnProperty.call(receivedIds, '__proto__')).toBe(true);
    expect(receivedIds?.__proto__).toBe(10);
    expect(
      Object.prototype.hasOwnProperty.call(out.meshes![0]!.primitives[0]!.attributes, '__proto__'),
    ).toBe(true);
    expect(extension.extras).toEqual({ source: 'test' });
  });

  it('rejects inherited Draco result fields and inherited semantic values', async () => {
    for (const decode of [
      () =>
        Object.create({
          attributes: exactAttributes(),
          indices: new Uint32Array([0, 1, 2]),
        }) as ReturnType<DracoDecodeFn>,
      () => {
        const attributes = Object.create(exactAttributes()) as Record<string, Float32Array>;
        return {
          attributes,
          indices: new Uint32Array([0, 1, 2]),
        };
      },
      () => {
        const result = Object.create({
          indices: new Uint32Array([0, 1, 2]),
        }) as Record<string, unknown>;
        result.attributes = exactAttributes();
        return result as unknown as ReturnType<DracoDecodeFn>;
      },
    ]) {
      const { gltf, buffers } = makeDracoAsset(false);
      await expect(resolveCompression(gltf, buffers, { dracoDecode: decode }, [])).rejects.toThrow(
        /must own its attributes field|must own its indices field|omitted semantic "POSITION"/,
      );
    }
  });

  it('accepts genuine cross-realm typed arrays from a Draco hook', async () => {
    const { runInNewContext } = await import('node:vm');
    const { gltf, buffers } = makeDracoAsset(false);
    const crossRealm = runInNewContext(`({
      attributes: {
        POSITION: new Float32Array(${JSON.stringify(POSITIONS)}),
        NORMAL: new Float32Array(${JSON.stringify(NORMALS)})
      },
      indices: new Uint32Array([0, 1, 2])
    })`) as ReturnType<DracoDecodeFn>;

    const out = await resolveCompression(gltf, buffers, { dracoDecode: () => crossRealm }, []);

    const primitive = out.meshes![0]!.primitives[0]!;
    expect(primitive.extensions?.KHR_draco_mesh_compression).toBeUndefined();
    expect(out.accessors).toHaveLength(6);
  });

  it('uses intrinsic Draco array lengths instead of shadowable instance fields', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const position = new Float32Array(POSITIONS);
    const normal = new Float32Array(NORMALS);
    const indices = new Uint32Array([0, 1, 2]);
    for (const value of [position, normal, indices]) {
      Object.defineProperty(value, 'byteLength', { value: 0 });
      Object.defineProperty(value, 'length', { value: 0 });
    }

    const out = await resolveCompression(
      gltf,
      buffers,
      {
        dracoDecode: () => ({
          attributes: { POSITION: position, NORMAL: normal },
          indices,
        }),
      },
      [],
    );

    expect(out.meshes![0]!.primitives[0]!.extensions?.KHR_draco_mesh_compression).toBeUndefined();
    expect(out.accessors).toHaveLength(6);
  });

  it('uses intrinsic Draco source size instead of a shadowable ArrayBuffer byteLength', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const source = buffers.get(0)!;
    Object.defineProperty(source, 'byteLength', { value: 0 });

    const out = await resolveCompression(
      gltf,
      buffers,
      {
        dracoDecode: () => ({
          attributes: exactAttributes(),
          indices: new Uint32Array([0, 1, 2]),
        }),
      },
      [],
    );

    expect(out.meshes![0]!.primitives[0]!.extensions?.KHR_draco_mesh_compression).toBeUndefined();
  });

  it('rejects shared Draco output and falls back atomically', async () => {
    const { gltf, buffers } = makeDracoAsset(true);
    const sharedPosition = new Float32Array(
      new SharedArrayBuffer(POSITIONS.length * Float32Array.BYTES_PER_ELEMENT),
    );
    sharedPosition.set(POSITIONS);
    const diagnostics: GltfCompressionDiagnostic[] = [];

    const out = await resolveCompression(
      gltf,
      buffers,
      {
        dracoDecode: () => ({
          attributes: {
            POSITION: sharedPosition,
            NORMAL: new Float32Array(NORMALS),
          },
          indices: new Uint32Array([0, 1, 2]),
        }),
      },
      [],
      (diagnostic) => diagnostics.push(diagnostic),
    );

    expect(out.meshes![0]!.primitives[0]!.attributes).toEqual({
      POSITION: 0,
      NORMAL: 1,
    });
    expect(diagnostics).toEqual([
      expect.objectContaining({ code: 'draco-fallback-accessors-used' }),
    ]);
  });

  it('does not publish partial decoded streams when required Draco validation fails', async () => {
    const { gltf, buffers } = makeDracoAsset(true);
    gltf.extensionsRequired = ['KHR_draco_mesh_compression'];
    const originalKeys = [...buffers.keys()];
    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => ({
            attributes: { POSITION: new Float32Array(POSITIONS) },
            indices: new Uint32Array([0, 1, 2]),
          }),
        },
        [],
      ),
    ).rejects.toThrow(/extensionsRequired.*omitted semantic "NORMAL"/);
    expect([...buffers.keys()]).toEqual(originalKeys);
    expect(gltf.accessors).toHaveLength(3);
    expect(gltf.meshes![0]!.primitives[0]!.attributes).toEqual({ POSITION: 0, NORMAL: 1 });
  });

  it('rejects an omitted declared semantic when its accessor has no exact fallback', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => ({
            attributes: { POSITION: new Float32Array(POSITIONS) },
            indices: new Uint32Array([0, 1, 2]),
          }),
        },
        [],
      ),
    ).rejects.toThrow(/omitted semantic "NORMAL".*no fully valid accessor fallback/);
  });

  it('rejects non-finite decoded attributes without an exact fallback', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const position = new Float32Array(POSITIONS);
    position[4] = Number.NaN;
    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => ({
            attributes: { POSITION: position, NORMAL: new Float32Array(NORMALS) },
            indices: new Uint32Array([0, 1, 2]),
          }),
        },
        [],
      ),
    ).rejects.toThrow(/non-finite Float32 value.*no fully valid accessor fallback/);
  });

  it('rejects invalid Draco accessor normalization and U32 vertex attributes before decoding', async () => {
    const cases: ReadonlyArray<{
      readonly mutate: (gltf: GltfJson) => void;
      readonly error: RegExp;
    }> = [
      {
        mutate: (gltf) => {
          gltf.accessors![0]!.normalized = true;
        },
        error: /normalized may be true only for BYTE/,
      },
      {
        mutate: (gltf) => {
          gltf.accessors![0]!.normalized = null as unknown as boolean;
        },
        error: /normalized must be a boolean/,
      },
      {
        mutate: (gltf) => {
          gltf.accessors![0]!.componentType = 5125;
        },
        error: /UNSIGNED_INT is reserved for primitive index accessors/,
      },
      {
        mutate: (gltf) => {
          gltf.accessors![2]!.componentType = 5123;
          gltf.accessors![2]!.normalized = true;
        },
        error: /index accessor 2 must not be normalized/i,
      },
    ];

    for (const testCase of cases) {
      const { gltf, buffers } = makeDracoAsset(false);
      testCase.mutate(gltf);
      let calls = 0;
      await expect(
        resolveCompression(
          gltf,
          buffers,
          {
            dracoDecode: () => {
              calls += 1;
              return exactDraco(new Uint8Array(), {});
            },
          },
          [],
        ),
      ).rejects.toThrow(testCase.error);
      expect(calls).toBe(0);
    }
  });

  it('rejects decoded indices outside the declared component type without wrapping', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    gltf.accessors![2]!.componentType = 5121;
    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => ({
            attributes: exactAttributes(),
            indices: new Uint16Array([0, 1, 256]),
          }),
        },
        [],
      ),
    ).rejects.toThrow(/index 256.*componentType maximum 255.*no fully valid accessor fallback/);
  });

  it('rejects every primitive-restart sentinel in decoded Draco indices', async () => {
    for (const testCase of [
      {
        componentType: 5121 as const,
        indices: new Uint8Array([0, 1, 0xff]),
        sentinel: 0xff,
      },
      {
        componentType: 5123 as const,
        indices: new Uint16Array([0, 1, 0xffff]),
        sentinel: 0xffff,
      },
      {
        componentType: 5125 as const,
        indices: new Uint32Array([0, 1, 0xffff_ffff]),
        sentinel: 0xffff_ffff,
      },
    ]) {
      const { gltf, buffers } = makeDracoAsset(false);
      gltf.accessors![2]!.componentType = testCase.componentType;

      await expect(
        resolveCompression(
          gltf,
          buffers,
          {
            dracoDecode: () => ({
              attributes: exactAttributes(),
              indices: testCase.indices,
            }),
          },
          [],
        ),
      ).rejects.toThrow(
        new RegExp(
          `index ${testCase.sentinel}.*reserved componentType maximum ${testCase.sentinel}`,
        ),
      );
    }
  });

  it('rejects a primitive-restart sentinel in an exact Draco fallback accessor', async () => {
    const { gltf, buffers } = makeDracoAsset(true);
    const sentinelFallback = new Uint8Array([0, 1, 0xff]).buffer;
    buffers.set(3, sentinelFallback);
    gltf.buffers![3]!.byteLength = sentinelFallback.byteLength;
    gltf.bufferViews![3]!.byteLength = sentinelFallback.byteLength;
    gltf.accessors![2]!.componentType = 5121;

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => {
            throw new Error('force fallback');
          },
        },
        [],
      ),
    ).rejects.toThrow(/reserved primitive-restart value 255/);
  });

  it('rejects a TRIANGLES index accessor count that is not divisible by three before decoding', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    gltf.accessors![2]!.count = 2;
    let calls = 0;

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => {
            calls += 1;
            return exactDraco(new Uint8Array(), {});
          },
        },
        [],
      ),
    ).rejects.toThrow(/non-empty triangle-list index count divisible by 3; received 2/);
    expect(calls).toBe(0);
  });

  it('ignores a huge unrelated host buffer key when publishing Draco streams', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    buffers.set(Number.MAX_SAFE_INTEGER, new ArrayBuffer(0));

    const out = await resolveCompression(gltf, buffers, { dracoDecode: exactDraco }, []);

    expect(out.buffers).toHaveLength(4);
    expect([...buffers.keys()].filter((key) => key < 10).sort((a, b) => a - b)).toEqual([
      0, 1, 2, 3,
    ]);
    expect(buffers.has(Number.MAX_SAFE_INTEGER)).toBe(true);
  });

  it('rejects Float32 JOINTS substitution atomically without publishing synthetic data', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const primitive = gltf.meshes![0]!.primitives[0]!;
    primitive.attributes.JOINTS_0 = 3;
    gltf.accessors!.push({
      componentType: 5121,
      count: 3,
      type: 'VEC4',
    });
    const extension = primitive.extensions!.KHR_draco_mesh_compression as {
      attributes: Record<string, number>;
    };
    extension.attributes.JOINTS_0 = 12;
    const originalBufferKeys = [...buffers.keys()];

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => ({
            attributes: {
              ...exactAttributes(),
              JOINTS_0: new Float32Array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
            },
            indices: new Uint32Array([0, 1, 2]),
          }),
        },
        [],
      ),
    ).rejects.toThrow(/semantic "JOINTS_0" must preserve.*no fully valid accessor fallback/);

    expect([...buffers.keys()]).toEqual(originalBufferKeys);
    expect(gltf.accessors).toHaveLength(4);
    expect(primitive.attributes.JOINTS_0).toBe(3);
  });

  it.each([
    'JOINTS_00',
    'WEIGHTS_9007199254740992',
    'JOINTS_-1',
    'WEIGHTS_1e2',
    'TEXCOORD_',
  ])('rejects malformed reserved Draco semantic %s before invoking the decoder', async (semantic) => {
    const { gltf, buffers } = makeDracoAsset(false);
    const primitive = gltf.meshes![0]!.primitives[0]!;
    primitive.attributes[semantic] = 0;
    const extension = primitive.extensions!.KHR_draco_mesh_compression as {
      attributes: Record<string, number>;
    };
    extension.attributes[semantic] = 12;
    let calls = 0;

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => {
            calls += 1;
            return exactDraco(new Uint8Array(), {});
          },
        },
        [],
      ),
    ).rejects.toThrow(
      /not canonical|safe-integer semantic range|non-negative canonical integer/,
    );
    expect(calls).toBe(0);
  });

  it('rejects an unknown non-application Draco semantic before invoking the decoder', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const primitive = gltf.meshes![0]!.primitives[0]!;
    primitive.attributes.CUSTOM_WEIGHT = 0;
    const extension = primitive.extensions!.KHR_draco_mesh_compression as {
      attributes: Record<string, number>;
    };
    extension.attributes.CUSTOM_WEIGHT = 12;
    let calls = 0;

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => {
            calls += 1;
            return exactDraco(new Uint8Array(), {});
          },
        },
        [],
      ),
    ).rejects.toThrow(/unknown non-application Draco attribute semantic "CUSTOM_WEIGHT"/);
    expect(calls).toBe(0);
  });

  it('accepts an application-specific Draco semantic through the strict boundary', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const primitive = gltf.meshes![0]!.primitives[0]!;
    const customAccessor = gltf.accessors!.push({
      componentType: 5126,
      count: 3,
      type: 'SCALAR',
    }) - 1;
    primitive.attributes._CUSTOM_WEIGHT = customAccessor;
    const extension = primitive.extensions!.KHR_draco_mesh_compression as {
      attributes: Record<string, number>;
    };
    extension.attributes._CUSTOM_WEIGHT = 12;

    const out = await resolveCompression(
      gltf,
      buffers,
      {
        dracoDecode: () => ({
          attributes: {
            ...exactAttributes(),
            _CUSTOM_WEIGHT: new Float32Array([0.25, 0.5, 0.75]),
          },
          indices: new Uint32Array([0, 1, 2]),
        }),
      },
      [],
    );

    const decodedCustomAccessor =
      out.meshes![0]!.primitives[0]!.attributes._CUSTOM_WEIGHT;
    expect(Number.isSafeInteger(decodedCustomAccessor)).toBe(true);
    expect(out.accessors![decodedCustomAccessor!]!.type).toBe('SCALAR');
    expect(
      out.meshes![0]!.primitives[0]!.extensions?.KHR_draco_mesh_compression,
    ).toBeUndefined();
  });

  it('rejects decoded indices outside the POSITION vertex count', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => ({
            attributes: exactAttributes(),
            indices: new Uint32Array([0, 1, 3]),
          }),
        },
        [],
      ),
    ).rejects.toThrow(/index 3.*vertex count 3.*no fully valid accessor fallback/);
  });

  it('bounds decoded indices by the common point count when POSITION is not Draco-owned', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const extension = gltf.meshes![0]!.primitives[0]!.extensions!.KHR_draco_mesh_compression as {
      attributes: Record<string, number>;
    };
    extension.attributes = { NORMAL: 11 };

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => ({
            attributes: { NORMAL: new Float32Array(NORMALS) },
            indices: new Uint32Array([0, 1, 3]),
          }),
        },
        [],
      ),
    ).rejects.toThrow(/index 3.*vertex count 3.*no fully valid accessor fallback/);
  });

  it('rejects inconsistent mapped Draco accessor counts before invoking the decoder', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    gltf.accessors![1]!.count = 2;
    let calls = 0;

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => {
            calls += 1;
            return exactDraco(new Uint8Array(), {});
          },
        },
        [],
      ),
    ).rejects.toThrow(/semantic "NORMAL" accessor 1 count 2.*common Draco point count 3/);
    expect(calls).toBe(0);
  });

  it('cross-checks an uncompressed POSITION accessor against the Draco point count', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const primitive = gltf.meshes![0]!.primitives[0]!;
    const extension = primitive.extensions!.KHR_draco_mesh_compression as {
      attributes: Record<string, number>;
    };
    extension.attributes = { NORMAL: 11 };
    gltf.accessors![0]!.count = 4;
    let calls = 0;

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => {
            calls += 1;
            return { attributes: {}, indices: new Uint32Array([0, 1, 2]) };
          },
        },
        [],
      ),
    ).rejects.toThrow(/POSITION accessor 0 count 4.*common Draco point count 3/);
    expect(calls).toBe(0);
  });

  it('rejects malformed Draco glTFProperty extensions before invoking the decoder', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const extension = gltf.meshes![0]!.primitives[0]!.extensions!
      .KHR_draco_mesh_compression as Record<string, unknown>;
    extension.extensions = [];
    let calls = 0;

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => {
            calls += 1;
            return exactDraco(new Uint8Array(), {});
          },
        },
        [],
      ),
    ).rejects.toThrow(/KHR_draco_mesh_compression.extensions must be an object/);
    expect(calls).toBe(0);
  });

  it('rejects a matrix Draco attribute before invoking the decoder', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    gltf.accessors![0]!.type = 'MAT2';
    let calls = 0;

    await expect(
      resolveCompression(
        gltf,
        buffers,
        {
          dracoDecode: () => {
            calls += 1;
            return {
              attributes: exactAttributes(),
              indices: new Uint32Array([0, 1, 2]),
            };
          },
        },
        [],
      ),
    ).rejects.toThrow(/type must be SCALAR, VEC2, VEC3, or VEC4/);

    expect(calls).toBe(0);
  });

  it('appends a dequantized FLOAT accessor without mutating the source component type', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const primitive = gltf.meshes![0]!.primitives[0]!;
    primitive.attributes = { POSITION: 0 };
    delete primitive.indices;
    primitive.extensions!.KHR_draco_mesh_compression = {
      bufferView: 0,
      attributes: { POSITION: 10 },
    };
    gltf.accessors = [
      {
        componentType: 5123,
        normalized: true,
        count: 3,
        type: 'VEC3',
        min: [0, 0, 0],
        max: [65535, 65535, 65535],
      },
    ];
    const decodedPositions = new Float32Array([-0.5, 4, 2, 1.5, -3, 8, 0.25, 2, -1]);

    const out = await resolveCompression(
      gltf,
      buffers,
      {
        dracoDecode: () => ({
          attributes: { POSITION: decodedPositions },
          indices: new Uint32Array([0, 1, 2]),
        }),
      },
      [],
    );
    const outPrimitive = out.meshes![0]!.primitives[0]!;
    const decodedIndex = outPrimitive.attributes.POSITION;
    if (decodedIndex === undefined) throw new Error('decoded POSITION accessor was not published');
    expect(out.accessors![0]!.componentType).toBe(5123);
    expect(out.accessors![0]!.normalized).toBe(true);
    expect(decodedIndex).not.toBe(0);
    expect(out.accessors![decodedIndex]!.componentType).toBe(5126);
    expect(out.accessors![decodedIndex]!.normalized).toBe(false);
    expect(out.accessors![decodedIndex]!.min).toEqual([-0.5, -3, -1]);
    expect(out.accessors![decodedIndex]!.max).toEqual([1.5, 4, 8]);
    expect(outPrimitive.indices).toBeDefined();
    expect(out.accessors![outPrimitive.indices!]!.count).toBe(3);
    expect(out.accessors![outPrimitive.indices!]!.type).toBe('SCALAR');
    expect(gltf.accessors[0]!.componentType).toBe(5123);
    expect(gltf.accessors[0]!.min).toEqual([0, 0, 0]);
    expect(gltf.accessors[0]!.max).toEqual([65535, 65535, 65535]);
    expect(primitive.indices).toBeUndefined();
  });

  it('keeps caller-owned compressed bytes immutable even when a Draco hook mutates its input', async () => {
    const { gltf, buffers } = makeDracoAsset(false);
    const source = buffers.get(0)!;
    await resolveCompression(
      gltf,
      buffers,
      {
        dracoDecode: (compressed, ids) => {
          compressed.fill(0);
          return exactDraco(compressed, ids);
        },
      },
      [],
    );
    expect(Array.from(new Uint8Array(source))).toEqual(COMPRESSED);
  });
});
