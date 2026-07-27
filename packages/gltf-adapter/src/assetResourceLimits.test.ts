import { describe, expect, it, vi } from 'vitest';
import { PNG } from 'pngjs';
import type { GltfJson } from './gltfTypes.js';
import {
  GltfResourceLimitError,
  gltfToScene,
  loadGltfAndDecodeTextures,
  loadGltfAsset,
  loadGltfForEngine,
  releaseGltfResources,
  type GltfAssetCache,
  type GltfAssetFetchResponse,
  type GltfAssetReadableStream,
  type GltfAssetReadableStreamReadResult,
  type GltfAssetReadableStreamReader,
} from './index.js';
import {
  buildTextureHandleMap,
  type DecodeImageFn,
  type GltfImageBytesMap,
} from './textures.js';
import {
  DecodedImageHandleOwner,
  ImportResourceLedger,
  createAsyncResourceLimiter,
  type GltfImportResourceContext,
  type GltfImportResourceLimits,
} from './importResourceBudget.js';

const EMPTY_GLTF: GltfJson = {
  asset: { version: '2.0' },
  scene: 0,
  scenes: [{ nodes: [] }],
};

function jsonBuffer(gltf: GltfJson): ArrayBuffer {
  const bytes = new TextEncoder().encode(JSON.stringify(gltf));
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
}

function glbWithBin(gltf: GltfJson, bin: Uint8Array): ArrayBuffer {
  const jsonBytes = new TextEncoder().encode(JSON.stringify(gltf));
  const jsonLength = Math.ceil(jsonBytes.byteLength / 4) * 4;
  const binLength = Math.ceil(bin.byteLength / 4) * 4;
  const totalLength = 12 + 8 + jsonLength + 8 + binLength;
  const buffer = new ArrayBuffer(totalLength);
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, totalLength, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, 0x4e4f534a, true);
  bytes.fill(0x20, 20, 20 + jsonLength);
  bytes.set(jsonBytes, 20);
  const binHeaderOffset = 20 + jsonLength;
  view.setUint32(binHeaderOffset, binLength, true);
  view.setUint32(binHeaderOffset + 4, 0x004e4942, true);
  bytes.set(bin, binHeaderOffset + 8);
  return buffer;
}

function response(
  data: ArrayBuffer,
  headers?: Readonly<Record<string, string>>,
): GltfAssetFetchResponse {
  return {
    ok: true,
    status: 200,
    headers: {
      get(name) {
        return headers?.[name.toLowerCase()] ?? null;
      },
    },
    arrayBuffer: async () => data,
  };
}

function resourceContext(
  limits: GltfImportResourceLimits = {},
): GltfImportResourceContext {
  const ledger = new ImportResourceLedger(limits);
  return {
    ledger,
    limiter: createAsyncResourceLimiter(
      ledger.limits.maxConcurrentResourceOperations,
    ),
    decodedImageHandles: new DecodedImageHandleOwner(),
  };
}

function pngHeader(width: number, height: number): Uint8Array {
  const bytes = new Uint8Array(24);
  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const view = new DataView(bytes.buffer);
  view.setUint32(8, 13, false);
  bytes.set([0x49, 0x48, 0x44, 0x52], 12);
  view.setUint32(16, width, false);
  view.setUint32(20, height, false);
  return bytes;
}

function imageOnlyGltf(imageCount: number): GltfJson {
  return {
    asset: { version: '2.0' },
    images: Array.from(
      { length: imageCount },
      (_, index) => ({ uri: `image-${index}.bin`, mimeType: 'application/octet-stream' }),
    ),
    textures: Array.from(
      { length: imageCount },
      (_, source) => ({ source }),
    ),
  };
}

function texturedTriangleGltf(): {
  readonly gltf: GltfJson;
  readonly geometryBuffer: ArrayBuffer;
} {
  const positions = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ]);
  const uvs = new Float32Array([
    0, 0,
    1, 0,
    0, 1,
  ]);
  const geometry = new Uint8Array(
    positions.byteLength + uvs.byteLength,
  );
  geometry.set(new Uint8Array(positions.buffer), 0);
  geometry.set(new Uint8Array(uvs.buffer), positions.byteLength);
  return {
    geometryBuffer: geometry.buffer,
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{
        primitives: [{
          attributes: { POSITION: 0, TEXCOORD_0: 1 },
          material: 0,
        }],
      }],
      materials: [{
        pbrMetallicRoughness: {
          baseColorTexture: { index: 0 },
        },
      }],
      textures: [{ source: 0 }],
      images: [{ uri: 'pixel.png', mimeType: 'image/png' }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
        {
          buffer: 0,
          byteOffset: positions.byteLength,
          byteLength: uvs.byteLength,
        },
      ],
      buffers: [{ byteLength: geometry.byteLength }],
    },
  };
}

function externalImages(
  count: number,
): GltfImageBytesMap {
  return new Map(
    Array.from({ length: count }, (_, index) => [
      index,
      {
        bytes: new Uint8Array([index + 1]),
        mimeType: 'application/octet-stream',
      },
    ]),
  );
}

describe('glTF encoded-resource acquisition limits', () => {
  it('rejects a GLB BIN copy before allocation through both direct public paths', async () => {
    const glb = glbWithBin(
      {
        ...EMPTY_GLTF,
        buffers: [{ byteLength: 4 }],
      },
      new Uint8Array([1, 2, 3, 4]),
    );
    const maxTotalEncodedBytes = glb.byteLength + 3;

    const lowLevelAttempt = gltfToScene(glb, {
      resourceLimits: { maxTotalEncodedBytes },
    });
    await expect(lowLevelAttempt).rejects.toBeInstanceOf(GltfResourceLimitError);
    await expect(lowLevelAttempt).rejects.toMatchObject({
      limitKind: 'total-encoded-bytes',
      limit: maxTotalEncodedBytes,
      actual: glb.byteLength + 4,
      path: 'GLB BIN chunk',
      resourceKey: 'buffer:0',
    });

    const assetAttempt = loadGltfAsset(glb, {
      resourceLimits: { maxTotalEncodedBytes },
    });
    await expect(assetAttempt).rejects.toBeInstanceOf(GltfResourceLimitError);
    await expect(assetAttempt).rejects.toMatchObject({
      limitKind: 'total-encoded-bytes',
      limit: maxTotalEncodedBytes,
      actual: glb.byteLength + 4,
      path: 'buffers[0]',
      resourceKey: 'buffer:0',
    });
  });

  it('preflights a fetched GLB BIN copy against the already-charged input', async () => {
    const glb = glbWithBin(
      {
        ...EMPTY_GLTF,
        buffers: [{ byteLength: 4 }],
      },
      new Uint8Array([1, 2, 3, 4]),
    );
    const maxTotalEncodedBytes = glb.byteLength + 3;
    const fetch = vi.fn(async () => response(glb));

    const importAttempt = loadGltfAsset('https://example.test/scene.glb', {
      fetch,
      resourceLimits: { maxTotalEncodedBytes },
    });
    await expect(importAttempt).rejects.toBeInstanceOf(GltfResourceLimitError);
    await expect(importAttempt).rejects.toMatchObject({
      limitKind: 'total-encoded-bytes',
      actual: glb.byteLength + 4,
      resourceKey: 'buffer:0',
    });
    expect(fetch).toHaveBeenCalledOnce();
  });

  it('counts an overridden GLB BIN copy under its distinct unused-copy key', async () => {
    const glb = glbWithBin(
      {
        ...EMPTY_GLTF,
        buffers: [{ byteLength: 4 }],
      },
      new Uint8Array([1, 2, 3, 4]),
    );
    const override = new ArrayBuffer(1);
    const maxTotalEncodedBytes = override.byteLength + glb.byteLength + 3;
    const importAttempt = loadGltfAsset(glb, {
      buffers: new Map([[0, override]]),
      resourceLimits: { maxTotalEncodedBytes },
    });

    await expect(importAttempt).rejects.toBeInstanceOf(GltfResourceLimitError);
    await expect(importAttempt).rejects.toMatchObject({
      limitKind: 'total-encoded-bytes',
      actual: override.byteLength + glb.byteLength + 4,
      resourceKey: 'buffer:0:unused-glb-copy',
    });
  });

  it.each([
    ['null', null],
    ['an array', []],
    ['a string', 'not-an-object'],
  ])('rejects resourceLimits supplied as %s through both public loaders', async (
    _label,
    resourceLimits,
  ) => {
    const options = {
      resourceLimits: resourceLimits as never,
    };

    await expect(
      loadGltfAsset(EMPTY_GLTF, options),
    ).rejects.toThrow(
      '[vitrum/gltf-adapter] resourceLimits must be an object when supplied.',
    );
    await expect(
      loadGltfAndDecodeTextures(EMPTY_GLTF, options),
    ).rejects.toThrow(
      '[vitrum/gltf-adapter] resourceLimits must be an object when supplied.',
    );
  });

  it.each([
    ['null', null],
    ['an array', []],
    ['a string', 'not-an-object'],
  ])('rejects configureTextureDecode returning %s', async (
    _label,
    configuredValue,
  ) => {
    await expect(
      loadGltfAndDecodeTextures(EMPTY_GLTF, {
        configureTextureDecode: (() => configuredValue) as never,
      }),
    ).rejects.toThrow(
      '[vitrum/gltf-adapter] configureTextureDecode must return an options object or undefined.',
    );
  });

  it('rejects an invalid nested cap before applying a valid flat alias', async () => {
    const direct = loadGltfAndDecodeTextures(EMPTY_GLTF, {
      resourceLimits: {
        maxDecodedTexturePixels: -1,
      },
      maxDecodedTexturePixels: 1,
    });
    await expect(direct).rejects.toThrow(
      'resourceLimits.maxDecodedTexturePixels must be a non-negative safe integer',
    );

    const configured = loadGltfAndDecodeTextures(EMPTY_GLTF, {
      configureTextureDecode: () => ({
        resourceLimits: {
          maxDecodedTexturePixels: -1,
        },
        maxDecodedTexturePixels: 1,
      }),
    });
    await expect(configured).rejects.toThrow(
      'resourceLimits.maxDecodedTexturePixels must be a non-negative safe integer',
    );
  });

  it('rejects an oversized direct ArrayBuffer before JSON parsing', async () => {
    const input = jsonBuffer(EMPTY_GLTF);
    await expect(
      loadGltfAsset(input, {
        resourceLimits: {
          maxEncodedResourceBytes: input.byteLength - 1,
        },
      }),
    ).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      limitKind: 'encoded-resource-bytes',
      path: 'input',
    });
  });

  it('uses intrinsic ArrayBuffer length instead of an own shadow', async () => {
    const input = jsonBuffer(EMPTY_GLTF);
    const actualLength = input.byteLength;
    Object.defineProperty(input, 'byteLength', {
      configurable: true,
      value: 0,
    });

    await expect(
      loadGltfAsset(input, {
        resourceLimits: {
          maxEncodedResourceBytes: actualLength - 1,
        },
      }),
    ).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      actual: actualLength,
    });
  });

  it('uses Content-Length as a non-allocating preflight and cancels the body', async () => {
    const cancel = vi.fn(async () => undefined);
    const arrayBuffer = vi.fn(async () => jsonBuffer(EMPTY_GLTF));
    const body: GltfAssetReadableStream = {
      cancel,
      getReader: () => {
        throw new Error('body reader must not be acquired');
      },
    };
    let bodyReads = 0;
    const fetchResponse: GltfAssetFetchResponse = {
      ok: true,
      headers: {
        get: (name) => name.toLowerCase() === 'content-length' ? '1000' : null,
      },
      arrayBuffer,
    };
    Object.defineProperty(fetchResponse, 'body', {
      get() {
        bodyReads += 1;
        return body;
      },
    });
    const fetch = vi.fn(async (): Promise<GltfAssetFetchResponse> => fetchResponse);

    await expect(
      loadGltfAsset('https://example.test/scene.gltf', {
        fetch,
        resourceLimits: { maxEncodedResourceBytes: 100 },
      }),
    ).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      actual: 1000,
    });
    expect(cancel).toHaveBeenCalledOnce();
    expect(arrayBuffer).not.toHaveBeenCalled();
    expect(bodyReads).toBe(1);
  });

  it('wraps throwing fetch metadata getters in GltfFetchFailed', async () => {
    const responseWithThrowingOk = {
      get ok(): boolean {
        throw new Error('hostile ok getter');
      },
      arrayBuffer: async () => jsonBuffer(EMPTY_GLTF),
    } as GltfAssetFetchResponse;
    await expect(
      loadGltfAsset('https://example.test/scene.gltf', {
        fetch: async () => responseWithThrowingOk,
      }),
    ).rejects.toMatchObject({ code: 'GLTF_FETCH_FAILED' });

    const responseWithThrowingStatus = {
      ok: false,
      get status(): number {
        throw new Error('hostile status getter');
      },
      arrayBuffer: async () => jsonBuffer(EMPTY_GLTF),
    } as GltfAssetFetchResponse;
    await expect(
      loadGltfAsset('https://example.test/scene.gltf', {
        fetch: async () => responseWithThrowingStatus,
      }),
    ).rejects.toMatchObject({ code: 'GLTF_FETCH_FAILED' });
  });

  it('wraps a throwing body getter without attempting arrayBuffer fallback', async () => {
    const arrayBuffer = vi.fn(async () => jsonBuffer(EMPTY_GLTF));
    const fetchResponse = {
      ok: true,
      get body(): GltfAssetReadableStream {
        throw new Error('hostile body getter');
      },
      arrayBuffer,
    } as GltfAssetFetchResponse;

    await expect(
      loadGltfAsset('https://example.test/scene.gltf', {
        fetch: async () => fetchResponse,
      }),
    ).rejects.toMatchObject({ code: 'GLTF_FETCH_FAILED' });
    expect(arrayBuffer).not.toHaveBeenCalled();
  });

  it('bounds streaming bytes, cancels on overflow, and never caches the rejection', async () => {
    const chunks = [new Uint8Array(6), new Uint8Array(6)];
    const cancel = vi.fn(async () => undefined);
    let index = 0;
    const reader: GltfAssetReadableStreamReader = {
      read: async (): Promise<GltfAssetReadableStreamReadResult> =>
        index < chunks.length
          ? { done: false, value: chunks[index++]! }
          : { done: true },
      cancel,
    };
    const cache: GltfAssetCache = {
      get: vi.fn(async () => undefined),
      set: vi.fn(async () => undefined),
    };
    const fetch = vi.fn(async (): Promise<GltfAssetFetchResponse> => ({
      ok: true,
      body: { getReader: () => reader },
      arrayBuffer: async () => {
        throw new Error('stream path expected');
      },
    }));

    await expect(
      loadGltfAsset('https://example.test/scene.gltf', {
        fetch,
        cache,
        resourceLimits: { maxEncodedResourceBytes: 10 },
      }),
    ).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      actual: 12,
    });
    expect(cancel).toHaveBeenCalledOnce();
    expect(cache.set).not.toHaveBeenCalled();
  });

  it('caps stream operation count even when a body yields zero-byte chunks forever', async () => {
    const cancel = vi.fn(async () => undefined);
    const reader: GltfAssetReadableStreamReader = {
      read: async () => ({ done: false, value: new Uint8Array() }),
      cancel,
    };
    const fetch = vi.fn(async (): Promise<GltfAssetFetchResponse> => ({
      ok: true,
      body: { getReader: () => reader },
      arrayBuffer: async () => {
        throw new Error('stream path expected');
      },
    }));

    await expect(
      loadGltfAsset('https://example.test/scene.gltf', {
        fetch,
        resourceLimits: {
          maxEncodedResourceBytes: 0,
          maxTotalEncodedBytes: 0,
        },
      }),
    ).rejects.toMatchObject({ code: 'GLTF_FETCH_FAILED' });
    expect(cancel).toHaveBeenCalledOnce();
  });

  it('rejects shared response chunks and cancels the reader', async () => {
    if (typeof SharedArrayBuffer === 'undefined') return;
    const cancel = vi.fn(async () => undefined);
    let read = false;
    const sharedChunk = new Uint8Array(new SharedArrayBuffer(1));
    const reader: GltfAssetReadableStreamReader = {
      read: async () => {
        if (read) return { done: true };
        read = true;
        return {
          done: false,
          value: sharedChunk as unknown as Uint8Array,
        };
      },
      cancel,
    };
    const fetch = vi.fn(async (): Promise<GltfAssetFetchResponse> => ({
      ok: true,
      body: { getReader: () => reader },
      arrayBuffer: async () => {
        throw new Error('stream path expected');
      },
    }));

    await expect(
      loadGltfAsset('https://example.test/scene.gltf', { fetch }),
    ).rejects.toMatchObject({ code: 'GLTF_FETCH_FAILED' });
    expect(cancel).toHaveBeenCalledOnce();
  });

  it('postchecks arrayBuffer fallback bytes and leaves rejected bytes out of cache', async () => {
    const cache: GltfAssetCache = {
      get: vi.fn(async () => undefined),
      set: vi.fn(async () => undefined),
    };
    const fetch = vi.fn(async () => response(new ArrayBuffer(11)));

    await expect(
      loadGltfAsset('https://example.test/scene.gltf', {
        fetch,
        cache,
        resourceLimits: { maxEncodedResourceBytes: 10 },
      }),
    ).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      actual: 11,
    });
    expect(cache.set).not.toHaveBeenCalled();
  });

  it('checks cache hits before use and does not fall through to fetch', async () => {
    const fetch = vi.fn();
    const cache: GltfAssetCache = {
      get: vi.fn(async () => new ArrayBuffer(11)),
      set: vi.fn(async () => undefined),
    };

    await expect(
      loadGltfAsset('https://example.test/scene.gltf', {
        fetch,
        cache,
        resourceLimits: { maxEncodedResourceBytes: 10 },
      }),
    ).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      actual: 11,
    });
    expect(fetch).not.toHaveBeenCalled();
    expect(cache.set).not.toHaveBeenCalled();
  });

  it('waits for all started external fetches before reporting the first failure', async () => {
    let releaseSecond!: () => void;
    const secondGate = new Promise<void>((resolve) => {
      releaseSecond = resolve;
    });
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{
        primitives: [{
          attributes: { POSITION: 0, NORMAL: 1 },
        }],
      }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
      ],
      bufferViews: [
        { buffer: 0, byteLength: 36 },
        { buffer: 1, byteLength: 36 },
      ],
      buffers: [
        { uri: 'first.bin', byteLength: 36 },
        { uri: 'second.bin', byteLength: 36 },
      ],
    };
    const fetch = vi.fn(async (url: string): Promise<GltfAssetFetchResponse> => {
      if (url.endsWith('/first.bin')) throw new Error('first failed');
      await secondGate;
      return response(new ArrayBuffer(36));
    });
    let settled = false;
    const load = loadGltfAsset(gltf, {
      baseUri: 'https://example.test/assets/scene.gltf',
      fetch,
    });
    void load.then(
      () => { settled = true; },
      () => { settled = true; },
    );

    await vi.waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    await Promise.resolve();
    expect(settled).toBe(false);
    releaseSecond();
    await expect(load).rejects.toMatchObject({ code: 'GLTF_FETCH_FAILED' });
    expect(settled).toBe(true);
  });

  it('publishes stream response types usable by custom fetch implementations', () => {
    const reader: GltfAssetReadableStreamReader = {
      read: async () => ({ done: true }),
    };
    const body: GltfAssetReadableStream = {
      getReader: () => reader,
    };
    const value: GltfAssetFetchResponse = {
      body,
      arrayBuffer: async () => new ArrayBuffer(0),
    };
    expect(value.body).toBe(body);
  });
});

describe('glTF image acquisition budgets', () => {
  it('does not double-charge an embedded image view against its parent buffer', async () => {
    const { gltf, geometryBuffer } = texturedTriangleGltf();
    const geometryBytes = new Uint8Array(geometryBuffer);
    const combinedBytes = new Uint8Array(geometryBytes.byteLength + 4);
    combinedBytes.set(geometryBytes);
    combinedBytes.set([1, 2, 3, 4], geometryBytes.byteLength);
    gltf.buffers = [{ byteLength: combinedBytes.byteLength }];
    gltf.bufferViews?.push({
      buffer: 0,
      byteOffset: geometryBytes.byteLength,
      byteLength: 4,
    });
    gltf.images = [{
      bufferView: 2,
      mimeType: 'application/octet-stream',
    }];
    const decodeImage = vi.fn(
      async (_bytes: Uint8Array, _mimeType: string) => ({ opaque: true }),
    );

    await expect(
      loadGltfAsset(gltf, {
        buffers: new Map([[0, combinedBytes.buffer]]),
        decodeImage,
        resourceLimits: {
          maxEncodedResourceBytes: combinedBytes.byteLength,
          maxTotalEncodedBytes: combinedBytes.byteLength,
        },
      }),
    ).resolves.toMatchObject({
      scene: {
        primitives: [{
          material: {
            baseColorMap: { handle: { opaque: true } },
          },
        }],
      },
    });
    expect(decodeImage).toHaveBeenCalledOnce();
    expect(decodeImage.mock.calls[0]?.[0]).toEqual(
      combinedBytes.subarray(geometryBytes.byteLength),
    );
  });

  it.each([
    {
      name: 'fractional offset',
      declaredBufferBytes: 16,
      loadedBufferBytes: 16,
      byteOffset: 0.5,
      byteLength: 4,
      message: 'must be a safe integer',
    },
    {
      name: 'range beyond the declared buffer',
      declaredBufferBytes: 4,
      loadedBufferBytes: 16,
      byteOffset: 0,
      byteLength: 5,
      message: 'exceeds declared',
    },
    {
      name: 'range beyond the loaded buffer',
      declaredBufferBytes: 16,
      loadedBufferBytes: 4,
      byteOffset: 2,
      byteLength: 4,
      message: 'exceeds loaded',
    },
  ])('rejects malformed embedded-image bufferViews: $name', async ({
    declaredBufferBytes,
    loadedBufferBytes,
    byteOffset,
    byteLength,
    message,
  }) => {
    const decode = vi.fn(async () => ({ opaque: true }));
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      buffers: [{ byteLength: declaredBufferBytes }],
      bufferViews: [{
        buffer: 0,
        byteOffset,
        byteLength,
      }],
      images: [{
        bufferView: 0,
        mimeType: 'application/octet-stream',
      }],
      textures: [{ source: 0 }],
    };

    await expect(
      buildTextureHandleMap(
        gltf,
        new Map([[0, new ArrayBuffer(loadedBufferBytes)]]),
        decode,
        [],
      ),
    ).rejects.toThrow(message);
    expect(decode).not.toHaveBeenCalled();
  });

  it('copies browser decode bytes without trusting the backing buffer slice method', async () => {
    const previousDescriptor = Object.getOwnPropertyDescriptor(
      globalThis,
      'createImageBitmap',
    );
    const browserHandle = { width: 1, height: 1 };
    const createImageBitmapStub = vi.fn(
      async (_blob: Blob): Promise<typeof browserHandle> => browserHandle,
    );
    Object.defineProperty(globalThis, 'createImageBitmap', {
      configurable: true,
      writable: true,
      value: createImageBitmapStub,
    });

    try {
      const backing = new ArrayBuffer(4);
      const bytes = new Uint8Array(backing);
      bytes.set([1, 2, 3, 4]);
      Object.defineProperty(backing, 'slice', {
        configurable: true,
        value: () => {
          throw new Error('host backing-buffer slice must not run');
        },
      });
      const images = new Map([
        [0, { bytes, mimeType: 'application/octet-stream' }],
      ]);

      const handles = await buildTextureHandleMap(
        imageOnlyGltf(1),
        new Map(),
        undefined,
        [],
        images,
      );

      expect(handles.get(0)).toBe(browserHandle);
      expect(createImageBitmapStub).toHaveBeenCalledOnce();
      const blob = createImageBitmapStub.mock.calls[0]?.[0] as Blob;
      expect(Array.from(new Uint8Array(await blob.arrayBuffer())))
        .toEqual([1, 2, 3, 4]);
    } finally {
      if (previousDescriptor === undefined) {
        Reflect.deleteProperty(globalThis, 'createImageBitmap');
      } else {
        Object.defineProperty(
          globalThis,
          'createImageBitmap',
          previousDescriptor,
        );
      }
    }
  });

  it('preflights base64 data URIs before atob without rejecting legal whitespace', async () => {
    const decode = vi.fn(async () => ({ opaque: true }));
    const atobSpy = vi.spyOn(globalThis, 'atob');
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      images: [{
        uri: 'data:application/octet-stream;base64,Q U J D',
        mimeType: 'application/octet-stream',
      }],
      textures: [{ source: 0 }],
    };

    const accepted = await buildTextureHandleMap(
      gltf,
      new Map(),
      decode,
      [],
      undefined,
      [],
      undefined,
      undefined,
      resourceContext({ maxEncodedResourceBytes: 3 }),
    );
    expect(accepted.has(0)).toBe(true);

    atobSpy.mockClear();
    decode.mockClear();
    await expect(
      buildTextureHandleMap(
        gltf,
        new Map(),
        decode,
        [],
        undefined,
        [],
        undefined,
        undefined,
        resourceContext({ maxEncodedResourceBytes: 2 }),
      ),
    ).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      actual: 3,
    });
    expect(atobSpy).not.toHaveBeenCalled();
    expect(decode).not.toHaveBeenCalled();
    atobSpy.mockRestore();
  });

  it('bypasses the decoder when encoded dimensions exceed the pixel cap', async () => {
    const decode = vi.fn(async () => ({ width: 11, height: 1 }));
    const images = new Map([
      [0, { bytes: pngHeader(11, 1), mimeType: 'image/png' }],
    ]);

    await expect(
      buildTextureHandleMap(
        imageOnlyGltf(1),
        new Map(),
        decode,
        [],
        images,
        [],
        undefined,
        undefined,
        resourceContext({ maxDecodedTexturePixels: 10 }),
      ),
    ).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      limitKind: 'decoded-texture-pixels',
      actual: 11,
    });
    expect(decode).not.toHaveBeenCalled();
  });

  it('closes a decoded handle whose actual dimensions exceed the cap', async () => {
    const close = vi.fn();
    const decode = vi.fn(async () => ({ width: 11, height: 1, close }));
    const images = new Map([
      [0, { bytes: pngHeader(1, 1), mimeType: 'image/png' }],
    ]);

    await expect(
      buildTextureHandleMap(
        imageOnlyGltf(1),
        new Map(),
        decode,
        [],
        images,
        [],
        undefined,
        undefined,
        resourceContext({ maxDecodedTexturePixels: 10 }),
      ),
    ).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      actual: 11,
    });
    expect(close).toHaveBeenCalledOnce();
  });

  it('bounds unique image decode concurrency while retaining image-index dedup', async () => {
    let active = 0;
    let maximumActive = 0;
    const decode: DecodeImageFn = vi.fn(async (bytes) => {
      active += 1;
      maximumActive = Math.max(maximumActive, active);
      await new Promise((resolve) => setTimeout(resolve, 5));
      active -= 1;
      return { image: bytes[0] };
    });
    const gltf = imageOnlyGltf(5);
    gltf.textures = [{ source: 0 }, ...(gltf.textures ?? [])];

    const handles = await buildTextureHandleMap(
      gltf,
      new Map(),
      decode,
      [],
      externalImages(5),
      [],
      undefined,
      undefined,
      resourceContext({ maxConcurrentResourceOperations: 2 }),
    );
    expect(decode).toHaveBeenCalledTimes(5);
    expect(maximumActive).toBe(2);
    expect(handles.get(0)).toBe(handles.get(1));
  });

  it('closes every successful peer handle when any unique image decode fails', async () => {
    const close = vi.fn();
    const failure = new Error('decoder failed');
    const decode: DecodeImageFn = vi.fn(async (bytes) => {
      if (bytes[0] === 2) throw failure;
      return { close };
    });

    await expect(
      buildTextureHandleMap(
        imageOnlyGltf(2),
        new Map(),
        decode,
        [],
        externalImages(2),
        [],
        undefined,
        undefined,
        resourceContext({ maxConcurrentResourceOperations: 2 }),
      ),
    ).rejects.toBe(failure);
    expect(close).toHaveBeenCalledOnce();
  });

  it('does not double-charge raw acquisition before CPU pixel decoding', async () => {
    const png = PNG.sync.write({
      width: 1,
      height: 1,
      data: Buffer.from([255, 255, 255, 255]),
    });
    const { gltf, geometryBuffer } = texturedTriangleGltf();

    const result = await loadGltfAndDecodeTextures(gltf, {
      buffers: new Map([[0, geometryBuffer]]),
      imageBytes: new Map([
        [0, {
          bytes: new Uint8Array(png.buffer, png.byteOffset, png.byteLength),
          mimeType: 'image/png',
        }],
      ]),
      maxDecodedTexturePixels: 1,
      maxTotalDecodedTexturePixels: 1,
    });
    expect(result.decodedTextureCount).toBe(1);
  });

  it('keeps acquisition pixel charges when a decode policy overrides the total cap', async () => {
    const { gltf, geometryBuffer } = texturedTriangleGltf();
    const pixelHandle = {
      width: 2,
      height: 1,
      data: new Uint8Array([
        255, 255, 255, 255,
        255, 255, 255, 255,
      ]),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: 'srgb' as const,
    };

    const load = loadGltfAndDecodeTextures(gltf, {
      buffers: new Map([[0, geometryBuffer]]),
      imageBytes: new Map([
        [0, {
          bytes: new Uint8Array([1]),
          mimeType: 'application/octet-stream',
        }],
      ]),
      decodeImage: async () => pixelHandle,
      resourceLimits: {
        maxDecodedTexturePixels: 4,
        maxTotalDecodedTexturePixels: 4,
      },
      configureTextureDecode: () => ({
        maxTotalDecodedTexturePixels: 1,
      }),
    });

    await expect(load).rejects.toBeInstanceOf(GltfResourceLimitError);
    await expect(load).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      limitKind: 'total-decoded-texture-pixels',
      limit: 1,
      actual: 2,
      path: 'images[0]',
    });
  });

  it('applies configured encoded-byte limits to the shared raw-image snapshot ledger', async () => {
    const png = PNG.sync.write({
      width: 1,
      height: 1,
      data: Buffer.from([255, 255, 255, 255]),
    });
    const { gltf, geometryBuffer } = texturedTriangleGltf();

    const load = loadGltfAndDecodeTextures(gltf, {
      buffers: new Map([[0, geometryBuffer]]),
      imageBytes: new Map([
        [0, {
          bytes: new Uint8Array(png.buffer, png.byteOffset, png.byteLength),
          mimeType: 'image/png',
        }],
      ]),
      configureTextureDecode: () => ({
        resourceLimits: {
          maxEncodedResourceBytes: png.byteLength - 1,
        },
      }),
    });

    await expect(load).rejects.toBeInstanceOf(GltfResourceLimitError);
    await expect(load).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      limitKind: 'encoded-resource-bytes',
      limit: png.byteLength - 1,
      actual: png.byteLength,
      path: 'options.imageBytes[0].bytes',
      resourceKey: 'image:0',
    });
  });

  it('does not let an undefined nested patch relax an inherited encoded-byte cap', async () => {
    const { gltf, geometryBuffer } = texturedTriangleGltf();
    const decodePixels = vi.fn(async () => ({
      width: 1,
      height: 1,
      data: new Uint8Array([255, 255, 255, 255]),
      channels: 4 as const,
      dataType: 'uint8' as const,
    }));

    const result = await loadGltfAndDecodeTextures(gltf, {
      buffers: new Map([[0, geometryBuffer]]),
      imageBytes: new Map([
        [0, {
          bytes: new Uint8Array([1]),
          mimeType: 'application/octet-stream',
        }],
      ]),
      decodeImage: async () => ({
        kind: 'raw-image' as const,
        data: new Uint8Array(128),
        mimeType: 'application/octet-stream',
      }),
      decodePixels,
      resourceLimits: {
        maxEncodedResourceBytes: 64,
      },
      configureTextureDecode: (() => ({
        resourceLimits: {
          maxEncodedResourceBytes: undefined,
        },
      })) as never,
    });

    expect(result.decodedTextureCount).toBe(0);
    expect(decodePixels).not.toHaveBeenCalled();
    expect(result.textureDecodeDiagnostics).toContainEqual(
      expect.objectContaining({
        code: 'encoded-texture-exceeds-byte-budget',
        maxEncodedTextureBytes: 64,
        encodedTextureBytes: 128,
      }),
    );
  });

  it('rolls back a decoded image handle when later scene geometry allocation fails', async () => {
    const { gltf, geometryBuffer } = texturedTriangleGltf();
    const close = vi.fn();

    await expect(
      gltfToScene(gltf, {
        buffers: new Map([[0, geometryBuffer]]),
        imageBytes: externalImages(1),
        decodeImage: async () => ({ width: 1, height: 1, close }),
        resourceLimits: {
          maxDecodedGeometryBytes: 1,
        },
      }),
    ).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      limitKind: 'decoded-geometry-bytes',
    });
    expect(close).toHaveBeenCalledOnce();
  });

  it('rolls back an acquired image handle when the decode policy hook rejects', async () => {
    const { gltf, geometryBuffer } = texturedTriangleGltf();
    const close = vi.fn();
    const failure = new Error('decode policy rejected');

    await expect(
      loadGltfAndDecodeTextures(gltf, {
        buffers: new Map([[0, geometryBuffer]]),
        imageBytes: externalImages(1),
        decodeImage: async () => ({ width: 1, height: 1, close }),
        configureTextureDecode: () => {
          throw failure;
        },
      }),
    ).rejects.toBe(failure);
    expect(close).toHaveBeenCalledOnce();
  });

  it('rolls back an acquired image handle when a tighter decode policy rejects prior charges', async () => {
    const { gltf, geometryBuffer } = texturedTriangleGltf();
    const close = vi.fn();

    await expect(
      loadGltfAndDecodeTextures(gltf, {
        buffers: new Map([[0, geometryBuffer]]),
        imageBytes: externalImages(1),
        decodeImage: async () => ({ width: 1, height: 1, close }),
        configureTextureDecode: () => ({
          resourceLimits: {
            maxTotalEncodedBytes: 1,
          },
        }),
      }),
    ).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      limitKind: 'total-encoded-bytes',
      limit: 1,
    });
    expect(close).toHaveBeenCalledOnce();
  });

  it('closes a shared decoded identity once across local and transaction rollback', async () => {
    const close = vi.fn();
    const sharedHandle = { width: 1, height: 1, close };

    await expect(
      buildTextureHandleMap(
        imageOnlyGltf(2),
        new Map(),
        async () => sharedHandle,
        [],
        externalImages(2),
        [],
        undefined,
        undefined,
        resourceContext({
          maxDecodedTexturePixels: 1,
          maxTotalDecodedTexturePixels: 1,
          maxConcurrentResourceOperations: 2,
        }),
      ),
    ).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      limitKind: 'total-decoded-texture-pixels',
      actual: 2,
    });
    expect(close).toHaveBeenCalledOnce();
  });

  it('closes a shared oversized decoded identity once across sibling local failures', async () => {
    const close = vi.fn();
    const sharedHandle = { width: 2, height: 1, close };

    await expect(
      buildTextureHandleMap(
        imageOnlyGltf(2),
        new Map(),
        async () => sharedHandle,
        [],
        externalImages(2),
        [],
        undefined,
        undefined,
        resourceContext({
          maxDecodedTexturePixels: 1,
          maxConcurrentResourceOperations: 2,
        }),
      ),
    ).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
      limitKind: 'decoded-texture-pixels',
      actual: 2,
    });
    expect(close).toHaveBeenCalledOnce();
  });

  it('keeps successful result handles open until idempotent explicit release', async () => {
    const { gltf, geometryBuffer } = texturedTriangleGltf();
    const close = vi.fn();
    const handle = { width: 1, height: 1, close };
    const result = await gltfToScene(gltf, {
      buffers: new Map([[0, geometryBuffer]]),
      imageBytes: externalImages(1),
      decodeImage: async () => handle,
    });

    expect(result.scene.primitives[0]?.material.baseColorMap?.handle).toBe(handle);
    expect(close).not.toHaveBeenCalled();
    releaseGltfResources(result);
    releaseGltfResources(result);
    expect(close).toHaveBeenCalledOnce();
  });

  it('closes a successfully normalized acquisition handle as soon as it is superseded', async () => {
    const { gltf, geometryBuffer } = texturedTriangleGltf();
    const close = vi.fn();
    const rawHandle = {
      kind: 'raw-image' as const,
      mimeType: 'application/octet-stream',
      data: new Uint8Array([1]),
      close,
    };
    const result = await loadGltfAndDecodeTextures(gltf, {
      buffers: new Map([[0, geometryBuffer]]),
      imageBytes: externalImages(1),
      decodeImage: async () => rawHandle,
      decodePixels: async () => ({
        width: 1,
        height: 1,
        data: new Uint8Array([255, 255, 255, 255]),
        channels: 4,
        dataType: 'uint8',
      }),
    });

    expect(result.decodedTextureCount).toBeGreaterThan(0);
    expect(result.scene.primitives[0]?.material.baseColorMap?.handle).not.toBe(rawHandle);
    expect(close).toHaveBeenCalledOnce();
    releaseGltfResources(result);
    expect(close).toHaveBeenCalledOnce();
  });

  it('rolls back an acquired image handle when engine construction rejects', async () => {
    const { gltf, geometryBuffer } = texturedTriangleGltf();
    const close = vi.fn();
    const failure = new Error('engine factory failed');

    await expect(
      loadGltfForEngine(gltf, {
        buffers: new Map([[0, geometryBuffer]]),
        imageBytes: externalImages(1),
        decodeImage: async () => ({ width: 1, height: 1, close }),
        decodeTextures: false,
        backend: 'pt-webgl2',
        createEngine: async () => {
          throw failure;
        },
      }),
    ).rejects.toBe(failure);
    expect(close).toHaveBeenCalledOnce();
  });

  it('keeps successful engine-result handles alive until the nested asset owner is released', async () => {
    const { gltf, geometryBuffer } = texturedTriangleGltf();
    const close = vi.fn();
    const engine = { backendId: 'pt-webgl2' as const, setScene: vi.fn() };
    const result = await loadGltfForEngine(gltf, {
      buffers: new Map([[0, geometryBuffer]]),
      imageBytes: externalImages(1),
      decodeImage: async () => ({ width: 1, height: 1, close }),
      decodeTextures: false,
      backend: 'pt-webgl2',
      engine,
    });

    expect(close).not.toHaveBeenCalled();
    releaseGltfResources(result);
    releaseGltfResources(result.asset);
    expect(close).toHaveBeenCalledOnce();
  });
});
