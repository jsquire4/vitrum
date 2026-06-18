import { describe, expect, it } from 'vitest';
import {
  loadGltfForEngine,
  loadGltfWithEngine,
  loadGltfWithProgressiveEngine,
  type DecodeSceneTextureDiagnostic,
  type DecodeSceneTextureDiagnosticCode,
  type GltfBackendTextureStatus,
  type GltfMaterialTextureField,
  type GltfTextureColorSpace,
  type GltfTextureDecodeReport,
  type GltfTextureDecodeReportEntry,
  type GltfTextureHandleKind,
  type LoadGltfWithProgressiveEngineOptions,
  type LoadGltfWithEngineOptions,
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
  });

  it('exports the createProgressiveEngine-backed glTF convenience wrapper types', () => {
    const canvas = {} as HTMLCanvasElement;
    const opts: LoadGltfWithProgressiveEngineOptions = {
      engineOptions: { canvas },
    };

    expect(opts.engineOptions.canvas).toBe(canvas);
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
});
