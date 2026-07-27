import { describe, expect, it, vi } from 'vitest';
import type { GltfJson } from '@vitrum/gltf-adapter';
import type { EngineWithBackendId } from '../createEngine.js';
import {
  DEFAULT_GLTF_IMPORT_RESOURCE_LIMITS,
  GltfResourceLimitError,
  loadGltfForEngine,
  loadGltfWithEngine,
  loadGltfWithProgressiveEngine,
  normalizeGltfImportResourceLimits,
  releaseGltfResources,
  type DecodeSceneTextureDiagnostic,
  type DecodeSceneTextureDiagnosticCode,
  type GltfBackendTextureStatus,
  type GltfImportResourceLimits,
  type GltfMaterialTextureField,
  type GltfResourceLimitErrorInit,
  type GltfResourceLimitKind,
  type GltfTextureColorSpace,
  type GltfTextureDecodeReport,
  type GltfTextureDecodeReportEntry,
  type GltfTextureHandleKind,
  type LoadGltfWithProgressiveEngineOptions,
  type LoadGltfWithEngineOptions,
  type NormalizedGltfImportResourceLimits,
} from '../gltf.js';

describe('@vitrum/engine/gltf subpath', () => {
  it('exports the adapter bridge and createEngine-backed convenience wrapper', () => {
    const opts: LoadGltfWithEngineOptions = {
      compatibilityMode: 'best-effort',
      decodeTextures: true,
      decodePixels: () => ({
        width: 1,
        height: 1,
        data: new Uint8Array([255, 255, 255, 255]),
        channels: 4,
        dataType: 'uint8',
        colorSpace: 'srgb',
      }),
    };

    expect(opts.compatibilityMode).toBe('best-effort');
    expect(opts.decodeTextures).toBe(true);
    expect(typeof loadGltfForEngine).toBe('function');
    expect(typeof loadGltfWithEngine).toBe('function');
    expect(typeof loadGltfWithProgressiveEngine).toBe('function');
    expect(typeof releaseGltfResources).toBe('function');
  });

  it('exports the createProgressiveEngine-backed glTF convenience wrapper types', () => {
    const canvas = {} as HTMLCanvasElement;
    const opts: LoadGltfWithProgressiveEngineOptions = {
      engineOptions: { canvas },
    };

    expect(opts.engineOptions.canvas).toBe(canvas);
  });

  it('exports resource-limit values and types from the one-import subpath', () => {
    const limits: GltfImportResourceLimits = {
      maxDecodedGeometryBytes: 1024,
      maxConcurrentResourceOperations: 2,
    };
    const normalized: NormalizedGltfImportResourceLimits =
      normalizeGltfImportResourceLimits(limits);
    const limitKind: GltfResourceLimitKind = 'decoded-geometry-bytes';
    const init: GltfResourceLimitErrorInit = {
      limitKind,
      limit: 1024,
      actual: 1025,
      path: 'accessors[0]',
    };
    const error = new GltfResourceLimitError(init);
    const opts: LoadGltfWithEngineOptions = {
      resourceLimits: limits,
      maxDecodedTexturePixels: 256,
      maxTotalDecodedTexturePixels: 512,
      maxImageDecodeConcurrency: 2,
    };

    expect(error.code).toBe('GLTF_RESOURCE_LIMIT_EXCEEDED');
    expect(normalized.maxDecodedGeometryBytes).toBe(1024);
    expect(DEFAULT_GLTF_IMPORT_RESOURCE_LIMITS.maxConcurrentResourceOperations)
      .toBeGreaterThan(0);
    expect(opts.resourceLimits).toBe(limits);
  });

  it.each([
    ['null', null],
    ['an array', []],
    ['a string', 'not-an-object'],
  ])('rejects a host texture policy patch returned as %s before wrapper spreading', async (
    _label,
    configuredValue,
  ) => {
    const input = {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [] }],
    } as Parameters<typeof loadGltfWithEngine>[0];

    await expect(
      loadGltfWithEngine(input, {
        configureTextureDecode: (() => configuredValue) as never,
      }),
    ).rejects.toThrow(
      '[vitrum/engine/gltf] configureTextureDecode must return an options object or undefined.',
    );
  });

  it('exports texture decode report and diagnostic types from the one-import subpath', () => {
    const status: GltfBackendTextureStatus = 'ready';
    const field: GltfMaterialTextureField = 'baseColorMap';
    const colorSpace: GltfTextureColorSpace = 'srgb';
    const kind: GltfTextureHandleKind = 'pixel-data';
    const code: DecodeSceneTextureDiagnosticCode = 'unsupported-handle-kind';
    const diagnostic: DecodeSceneTextureDiagnostic = {
      code,
      severity: 'warning',
      path: 'materials[0].baseColorTexture',
      materialField: field,
      primitiveId: 'p0',
      primitiveIndex: 0,
      message: 'decoded',
    };
    const entry: GltfTextureDecodeReportEntry = {
      primitiveId: 'p0',
      primitiveKind: 'mesh',
      primitiveIndex: 0,
      materialField: field,
      path: 'materials[0].baseColorTexture',
      texCoord: 0,
      hasTransform: false,
      wrapS: 'repeat',
      wrapT: 'clamp-to-edge',
      handleKind: kind,
      colorSpace,
      backendReadiness: {
        ptWebgl2: status,
        ptWebgpu: status,
        walkaroundHybrid: status,
      },
    };
    const report: GltfTextureDecodeReport = {
      mapCount: 1,
      uniqueHandleCount: 1,
      rawImageCount: 0,
      imageBitmapCount: 0,
      opaqueHandleCount: 0,
      cpuReadableCount: 1,
      rawImageRefs: [],
      imageBitmapRefs: [],
      entries: [entry],
    };

    expect(report.entries[0]?.backendReadiness.ptWebgpu).toBe('ready');
    expect(diagnostic.code).toBe('unsupported-handle-kind');
  });

  it('releases acquired image handles when existing-engine attachment rejects', async () => {
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
    const gltf: GltfJson = {
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
      images: [{ uri: 'pixel.bin', mimeType: 'application/octet-stream' }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' },
      ],
      bufferViews: [
        { buffer: 0, byteLength: positions.byteLength },
        { buffer: 1, byteLength: uvs.byteLength },
      ],
      buffers: [
        { byteLength: positions.byteLength },
        { byteLength: uvs.byteLength },
      ],
    };
    const close = vi.fn();
    const failure = new Error('setScene failed');
    const engine = {
      backendId: 'pt-webgl2',
      setScene: vi.fn(() => {
        throw failure;
      }),
    } as unknown as EngineWithBackendId;

    await expect(
      loadGltfWithEngine(gltf, {
        engine,
        buffers: new Map([
          [0, positions.buffer],
          [1, uvs.buffer],
        ]),
        imageBytes: new Map([
          [0, {
            bytes: new Uint8Array([1]),
            mimeType: 'application/octet-stream',
          }],
        ]),
        decodeImage: async () => ({ width: 1, height: 1, close }),
        decodeTextures: false,
      }),
    ).rejects.toBe(failure);
    expect(close).toHaveBeenCalledOnce();
  });
});
