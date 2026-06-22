/**
 * glTF viewer example — one-call glTF loading through @vitrum/engine/gltf.
 *
 * The example builds a tiny glTF asset in-memory so it does not depend on a
 * network model. The code path is still the real public path:
 * loadGltfWithEngine() -> adapter feature report -> backend recommendation ->
 * createEngine({ gltfAsset }) -> GltfSceneController attachment.
 *
 * Capture protocol: sets globalThis.VITRUM_CAPTURE_READY = true after the
 * engine reports samplesAccumulated >= targetSpp.
 */

import { asMat4 } from '@vitrum/core';
import type { Mat4, Scene, ScenePrimitive } from '@vitrum/core';
import { loadGltfWithEngine } from '@vitrum/engine/gltf';
import DracoDecoderModule from 'draco3d/draco_decoder_nodejs.js';
import dracoDecoderWasmUrl from 'draco3d/draco_decoder.wasm?url';
import { MeshoptDecoder } from 'meshoptimizer';

const params = new URLSearchParams(location.search);
const targetSpp = Number(params.get('vitrumSpp')) || 128;
const requestedAssetId = params.get('vitrumGltfAsset') ?? '';
const requestedBackend = params.get('vitrumBackend') ?? '';

type BrowserDecodeKind = 'draco' | 'meshopt';
type MeshoptMode = 'ATTRIBUTES' | 'TRIANGLES' | 'INDICES';
type MeshoptFilter = 'NONE' | 'OCTAHEDRAL' | 'QUATERNION' | 'EXPONENTIAL';
type DracoDecodeHook = (
  compressed: Uint8Array,
  attributeIds: Record<string, number>,
) => { attributes: Record<string, Float32Array>; indices?: Uint32Array };
type MeshoptDecodeHook = (
  compressed: Uint8Array,
  count: number,
  byteStride: number,
  mode: MeshoptMode,
  filter: MeshoptFilter,
) => Promise<Uint8Array> | Uint8Array;
interface BrowserDecodeHooks {
  readonly dracoDecode?: DracoDecodeHook;
  readonly meshoptDecode?: MeshoptDecodeHook;
  readonly report: Record<string, boolean | readonly string[]>;
}

interface RealGltfAsset {
  readonly url: string;
  readonly kind: string;
  readonly minPrimitives: number;
  readonly minTextures: number;
  readonly requiredExtensions?: readonly string[];
  readonly browserDecode?: readonly BrowserDecodeKind[];
}

const REAL_GLTF_ASSETS: Record<string, RealGltfAsset> = {
  'box-textured-glb': {
    url: 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/BoxTextured/glTF-Binary/BoxTextured.glb',
    kind: 'textured-glb',
    minPrimitives: 1,
    minTextures: 1,
  },
  'cesium-milk-truck-draco': {
    url: 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CesiumMilkTruck/glTF-Draco/CesiumMilkTruck.gltf',
    kind: 'draco',
    minPrimitives: 1,
    minTextures: 0,
    requiredExtensions: ['KHR_draco_mesh_compression'],
    browserDecode: ['draco'],
  },
  'meshopt-cube-real': {
    url: 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/MeshoptCubeTest/glTF-Meshopt/MeshoptCubeTest.gltf',
    kind: 'meshopt',
    minPrimitives: 1,
    minTextures: 0,
    requiredExtensions: ['KHR_meshopt_compression'],
    browserDecode: ['meshopt'],
  },
};

const canvas = document.getElementById('vitrum-canvas') as HTMLCanvasElement;

const viewMatrix = asMat4(new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, -3, 1,
]));

function projectionForCanvas(): Float32Array {
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  const aspect = width / height;
  const fovY = Math.PI / 3;
  const near = 0.1;
  const far = 100;
  const f = 1 / Math.tan(fovY / 2);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), -1,
    0, 0, (2 * far * near) / (near - far), 0,
  ]);
}

async function main(): Promise<void> {
  const realAsset = requestedAssetId ? REAL_GLTF_ASSETS[requestedAssetId] : undefined;
  if (requestedAssetId && realAsset == null) {
    throw new Error(`[gltf-viewer example] unsupported vitrumGltfAsset "${requestedAssetId}"`);
  }
  const browserDecodeHooks = realAsset == null ? emptyBrowserDecodeHooks() : await createBrowserDecodeHooks(realAsset);
  const result = await loadGltfWithEngine(realAsset?.url ?? createEmbeddedGltf(), {
    compatibilityMode: 'best-effort',
    ...(realAsset == null ? { decodeImage: decodeEmbeddedDemoImage } : {}),
    ...(realAsset != null ? {
      decodeTextures: true,
      decodePixels: decodeBrowserImagePixels,
      maxTextureSize: 4096,
      warnOnNpotRepeatWrap: true,
    } : {}),
    ...browserDecodeHooks,
    ...(requestedBackend === 'pt-webgl2' ? { backend: 'pt-webgl2' as const } : {}),
    engineOptions: {
      canvas,
      prefer: requestedBackend === 'pt-webgl2' ? 'quality' : 'auto',
      onWarning: (warning) => {
        console.warn('[gltf-viewer example]', warning.message, warning.details ?? '');
      },
    },
  });
  const engine = requireEngine(result.engine);
  const renderScene = realAsset == null ? result.controller.scene : addRealAssetLighting(normalizeSceneForViewer(result.controller.scene));
  if (renderScene !== result.controller.scene) {
    engine.setScene(renderScene);
  }

  (globalThis as Record<string, unknown>).VITRUM_CAPTURE_FRAME = async (colorSpace: 'linear' | 'output' = 'output') => {
    if (typeof engine.captureFrame !== 'function') return null;
    const frame = await engine.captureFrame({ colorSpace });
    if (frame == null) return null;
    return {
      width: frame.width,
      height: frame.height,
      rgba: Array.from(frame.rgba),
    };
  };
  (globalThis as Record<string, unknown>).VITRUM_GLTF_BACKEND = result.backend;
  (globalThis as Record<string, unknown>).VITRUM_GLTF_TEXTURE_REPORT = result.textureDecodeReport;
  (globalThis as Record<string, unknown>).VITRUM_GLTF_WARNINGS = result.warnings;
  (globalThis as Record<string, unknown>).VITRUM_CAPTURE_TELEMETRY = {
    assetId: requestedAssetId || 'embedded-demo',
    backend: result.backend,
    profileId: result.profileId,
    primitiveCount: result.controller.scene.primitives.length,
    extensionsUsed: result.asset.featureReport.extensions.used,
    extensionsRequired: result.asset.featureReport.extensions.required,
    browserDecodeHooks: browserDecodeHooks.report,
    textureDecodeReport: result.textureDecodeReport,
    warnings: result.warnings,
    diagnostics: result.diagnostics,
    realAssetReady:
      realAsset == null ||
      (
        result.controller.scene.primitives.length >= realAsset.minPrimitives &&
        result.textureDecodeReport.mapCount >= realAsset.minTextures &&
        (realAsset.requiredExtensions ?? []).every((ext) => result.asset.featureReport.extensions.used.includes(ext))
      ),
  };

  let frameIndex = 0;
  let captureSignalled = false;
  let lastSpp = 0;
  let lastAnimationNowMs: number | null = null;

  engine.onFrame?.((stats) => {
    lastSpp = stats.spp ?? lastSpp;
    (globalThis as Record<string, unknown>).VITRUM_MS_PER_SAMPLE =
      stats.frameTimeMs > 0 && lastSpp > 0 ? stats.frameTimeMs / lastSpp : 0;
  });

  function tick(now: number): void {
    if ((globalThis as Record<string, unknown>).VITRUM_CAPTURE_PAUSED === true) {
      requestAnimationFrame(tick);
      return;
    }

    if (result.asset.animations.length > 0) {
      if (lastAnimationNowMs != null) {
        const deltaSeconds = Math.max(0, (now - lastAnimationNowMs) / 1000);
        if (deltaSeconds > 0) result.controller.advance(deltaSeconds, { engine });
      }
      lastAnimationNowMs = now;
    }

    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    canvas.width = width;
    canvas.height = height;

    const output = engine.renderFrame({
      viewMatrix,
      projMatrix: asMat4(projectionForCanvas()),
      cameraPosition: [0, 0, 3],
      viewport: { width, height, devicePixelRatio: 1 },
      frameIndex,
      frameSeed: (frameIndex * 1664525 + 1013904223) >>> 0,
    });

    if (output.kind === 'rendered') {
      frameIndex += 1;
      if (!captureSignalled && output.samplesAccumulated >= targetSpp) {
        (globalThis as Record<string, unknown>).VITRUM_CAPTURE_READY = true;
        captureSignalled = true;
      }
    }

    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}

function emptyBrowserDecodeHooks(): BrowserDecodeHooks {
  return {
    report: {
      requested: [],
      draco: false,
      meshopt: false,
    },
  };
}

async function createBrowserDecodeHooks(asset: RealGltfAsset): Promise<BrowserDecodeHooks> {
  const needed = new Set(asset.browserDecode ?? []);
  const report: Record<string, boolean | readonly string[]> = {
    requested: [...needed],
    draco: false,
    meshopt: false,
  };
  const hooks: { dracoDecode?: DracoDecodeHook; meshoptDecode?: MeshoptDecodeHook } = {};

  if (needed.has('draco')) {
    hooks.dracoDecode = await createBrowserDracoDecode();
    report.draco = true;
  }
  if (needed.has('meshopt')) {
    hooks.meshoptDecode = await createBrowserMeshoptDecode();
    report.meshopt = true;
  }
  return { ...hooks, report };
}

async function createBrowserDracoDecode(): Promise<DracoDecodeHook> {
  const wasmBinary = await fetch(dracoDecoderWasmUrl).then((response) => {
    if (!response.ok) throw new Error(`[gltf-viewer example] failed to fetch Draco WASM: ${response.status}`);
    return response.arrayBuffer();
  });
  const module = await DracoDecoderModule({ wasmBinary });
  return (compressed, attributeIds) => {
    const decoder = new module.Decoder();
    const buffer = new module.DecoderBuffer();
    let mesh: ReturnType<typeof createDracoMesh> | null = null;
    try {
      buffer.Init(new Int8Array(compressed.buffer, compressed.byteOffset, compressed.byteLength), compressed.byteLength);
      const geometryType = decoder.GetEncodedGeometryType(buffer);
      if (geometryType !== module.TRIANGULAR_MESH) {
        throw new Error(`[gltf-viewer example] unsupported Draco geometry type ${geometryType}`);
      }
      mesh = new module.Mesh();
      const status = decoder.DecodeBufferToMesh(buffer, mesh);
      if (!status.ok()) throw new Error(status.error_msg());

      const attributes: Record<string, Float32Array> = {};
      for (const [semantic, uniqueId] of Object.entries(attributeIds)) {
        const attr = decoder.GetAttributeByUniqueId(mesh, uniqueId);
        if (attr == null || attr.ptr === 0) {
          throw new Error(`[gltf-viewer example] missing Draco attribute ${semantic} (${uniqueId})`);
        }
        const values = new module.DracoFloat32Array();
        try {
          decoder.GetAttributeFloatForAllPoints(mesh, attr, values);
          const out = new Float32Array(values.size());
          for (let i = 0; i < out.length; i += 1) out[i] = values.GetValue(i);
          attributes[semantic] = out;
        } finally {
          module.destroy(values);
        }
      }

      const face = new module.DracoInt32Array();
      const indices = new Uint32Array(mesh.num_faces() * 3);
      try {
        for (let f = 0; f < mesh.num_faces(); f += 1) {
          decoder.GetFaceFromMesh(mesh, f, face);
          const offset = f * 3;
          indices[offset] = face.GetValue(0);
          indices[offset + 1] = face.GetValue(1);
          indices[offset + 2] = face.GetValue(2);
        }
      } finally {
        module.destroy(face);
      }
      return { attributes, indices };
    } finally {
      module.destroy(buffer);
      if (mesh != null) module.destroy(mesh);
      module.destroy(decoder);
    }
  };
}

function createDracoMesh(module: Awaited<ReturnType<typeof DracoDecoderModule>>) {
  return new module.Mesh();
}

async function createBrowserMeshoptDecode(): Promise<MeshoptDecodeHook> {
  await MeshoptDecoder.ready;
  return (compressed, count, byteStride, mode, filter) => {
    if (typeof MeshoptDecoder.decodeGltfBufferAsync === 'function') {
      return MeshoptDecoder.decodeGltfBufferAsync(
        count,
        byteStride,
        compressed,
        mode,
        filter === 'NONE' ? undefined : filter,
      );
    }
    const target = new Uint8Array(count * byteStride);
    MeshoptDecoder.decodeGltfBuffer(
      target,
      count,
      byteStride,
      compressed,
      mode,
      filter === 'NONE' ? undefined : filter,
    );
    return target;
  };
}

function createEmbeddedGltf(): Parameters<typeof loadGltfWithEngine>[0] {
  const binary = createTriangleBuffer();
  return {
    asset: { version: '2.0', generator: 'vitrum examples/gltf-viewer' },
    scene: 0,
    extensionsUsed: ['KHR_materials_unlit', 'KHR_texture_transform'],
    buffers: [{
      byteLength: binary.byteLength,
      uri: dataUri(binary, 'application/octet-stream'),
    }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36, target: 34962 },
      { buffer: 0, byteOffset: 36, byteLength: 36, target: 34962 },
      { buffer: 0, byteOffset: 72, byteLength: 24, target: 34962 },
      { buffer: 0, byteOffset: 96, byteLength: 6, target: 34963 },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [-0.9, -0.65, 0], max: [0.9, 0.8, 0] },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3', min: [0, 0, 1], max: [0, 0, 1] },
      { bufferView: 2, componentType: 5126, count: 3, type: 'VEC2', min: [0, 0], max: [1, 1] },
      { bufferView: 3, componentType: 5123, count: 3, type: 'SCALAR', min: [0], max: [2] },
    ],
    samplers: [{ wrapS: 33071, wrapT: 33648 }],
    images: [{ uri: 'data:image/png;base64,AAAA', mimeType: 'image/png' }],
    textures: [{ source: 0, sampler: 0 }],
    materials: [{
      name: 'unlit-textured-demo',
      pbrMetallicRoughness: {
        baseColorFactor: [1, 1, 1, 1],
        baseColorTexture: {
          index: 0,
          texCoord: 0,
          extensions: {
            KHR_texture_transform: {
              offset: [0, 0],
              scale: [1, 1],
              rotation: 0,
            },
          },
        },
        metallicFactor: 0,
        roughnessFactor: 1,
      },
      extensions: { KHR_materials_unlit: {} },
    }],
    meshes: [{
      primitives: [{
        attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 },
        indices: 3,
        material: 0,
        mode: 4,
      }],
    }],
    nodes: [{ mesh: 0, name: 'triangle' }],
    scenes: [{ nodes: [0] }],
  };
}

async function decodeEmbeddedDemoImage(): Promise<{ width: number; height: number; data: Uint8Array }> {
  return {
    width: 2,
    height: 2,
    data: new Uint8Array([
      255, 80, 40, 255,
      40, 180, 255, 255,
      255, 235, 80, 255,
      180, 80, 255, 255,
    ]),
  };
}

async function decodeBrowserImagePixels(
  handle: { readonly mimeType?: string; readonly data?: Uint8Array },
  context: { readonly colorSpace: 'srgb' | 'linear' },
): Promise<{
  width: number;
  height: number;
  channels: 4;
  dataType: 'uint8';
  colorSpace: 'srgb' | 'linear';
  data: Uint8ClampedArray;
}> {
  if (!(handle.data instanceof Uint8Array)) {
    throw new Error('[gltf-viewer example] decodePixels expected a raw-image Uint8Array handle.');
  }
  const bytes = new Uint8Array(handle.data);
  const blob = new Blob([bytes.buffer], { type: handle.mimeType ?? 'application/octet-stream' });
  const bitmap = await createImageBitmap(blob);
  const width = bitmap.width;
  const height = bitmap.height;
  const canvas2d = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(width, height)
    : document.createElement('canvas');
  canvas2d.width = width;
  canvas2d.height = height;
  const ctx = canvas2d.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  if (ctx == null) throw new Error('[gltf-viewer example] 2D canvas unavailable for texture decode.');
  ctx.drawImage(bitmap, 0, 0);
  const pixels = ctx.getImageData(0, 0, width, height).data;
  bitmap.close();
  return {
    width,
    height,
    channels: 4,
    dataType: 'uint8',
    colorSpace: context.colorSpace,
    data: pixels,
  };
}

function normalizeSceneForViewer(scene: Scene): Scene {
  const aabb = sceneBounds(scene.primitives);
  if (aabb == null) return scene;
  const center: [number, number, number] = [
    (aabb.min[0] + aabb.max[0]) * 0.5,
    (aabb.min[1] + aabb.max[1]) * 0.5,
    (aabb.min[2] + aabb.max[2]) * 0.5,
  ];
  const extent = Math.max(
    aabb.max[0] - aabb.min[0],
    aabb.max[1] - aabb.min[1],
    aabb.max[2] - aabb.min[2],
  );
  if (!(extent > 0)) return scene;
  const s = 1.35 / extent;
  const normalization = asMat4(new Float32Array([
    s, 0, 0, 0,
    0, s, 0, 0,
    0, 0, s, 0,
    -center[0] * s, -center[1] * s, -center[2] * s, 1,
  ]));
  return {
    ...scene,
    primitives: scene.primitives.map((primitive) => {
      if (primitive.kind === 'instanced-mesh') {
        return {
          ...primitive,
          instances: primitive.instances.map((instance) => asMat4(multiplyMat4(normalization, instance))),
        };
      }
      return {
        ...primitive,
        transform: asMat4(multiplyMat4(normalization, primitive.transform ?? IDENTITY_MAT4)),
      };
    }),
  };
}

function addRealAssetLighting(scene: Scene): Scene {
  return {
    ...scene,
    emitters: [
      ...(scene.emitters ?? []),
      {
        kind: 'rect-area',
        id: 'gltf-viewer-real-asset-key-light',
        position: [0, 0.95, 0.65],
        uAxis: [0.45, 0, 0],
        vAxis: [0, 0.45, 0],
        color: [1, 1, 1],
        intensity: 18.0,
      },
    ],
    environment: {
      kind: 'procedural-sky',
      sunDirection: [0.4, 1.0, 0.25],
      turbidity: 2.0,
      rayleigh: 1.0,
      mieCoefficient: 0.005,
      mieDirectionalG: 0.8,
      intensity: 1.0,
    },
  };
}

const IDENTITY_MAT4 = asMat4(new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]));

function multiplyMat4(a: Float32Array, b: Float32Array): Mat4 {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[col * 4 + row] =
        a[0 * 4 + row]! * b[col * 4 + 0]! +
        a[1 * 4 + row]! * b[col * 4 + 1]! +
        a[2 * 4 + row]! * b[col * 4 + 2]! +
        a[3 * 4 + row]! * b[col * 4 + 3]!;
    }
  }
  return asMat4(out);
}

function sceneBounds(primitives: readonly ScenePrimitive[]): { min: [number, number, number]; max: [number, number, number] } | null {
  let found = false;
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const primitive of primitives) {
    if (primitive.kind !== 'mesh' && primitive.kind !== 'skinned-mesh') continue;
    const transform = primitive.transform ?? IDENTITY_MAT4;
    const positions = primitive.positions;
    for (let i = 0; i + 2 < positions.length; i += 3) {
      const p = transformPoint(transform, positions[i]!, positions[i + 1]!, positions[i + 2]!);
      for (let axis = 0; axis < 3; axis += 1) {
        const currentMin = min[axis]!;
        const currentMax = max[axis]!;
        const value = p[axis]!;
        min[axis] = Math.min(currentMin, value);
        max[axis] = Math.max(currentMax, value);
      }
      found = true;
    }
  }
  return found ? { min, max } : null;
}

function transformPoint(m: Float32Array, x: number, y: number, z: number): [number, number, number] {
  return [
    m[0]! * x + m[4]! * y + m[8]! * z + m[12]!,
    m[1]! * x + m[5]! * y + m[9]! * z + m[13]!,
    m[2]! * x + m[6]! * y + m[10]! * z + m[14]!,
  ];
}

function createTriangleBuffer(): Uint8Array {
  const bytes = new Uint8Array(104);
  writeF32(bytes, 0, [
    -0.9, -0.65, 0,
    0.9, -0.65, 0,
    0.0, 0.8, 0,
  ]);
  writeF32(bytes, 36, [
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
  ]);
  writeF32(bytes, 72, [
    0, 0,
    1, 0,
    0.5, 1,
  ]);
  writeU16(bytes, 96, [0, 1, 2]);
  return bytes;
}

function writeF32(bytes: Uint8Array, byteOffset: number, values: readonly number[]): void {
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < values.length; i += 1) {
    view.setFloat32(byteOffset + i * 4, values[i]!, true);
  }
}

function writeU16(bytes: Uint8Array, byteOffset: number, values: readonly number[]): void {
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < values.length; i += 1) {
    view.setUint16(byteOffset + i * 2, values[i]!, true);
  }
}

function dataUri(bytes: Uint8Array, mimeType: string): string {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return `data:${mimeType};base64,${btoa(binary)}`;
}

function requireEngine<T>(engine: T | undefined): T {
  if (!engine) throw new Error('loadGltfWithEngine did not return an engine.');
  return engine;
}

main().catch((err: unknown) => {
  console.error('[gltf-viewer example] fatal:', err);
  (globalThis as Record<string, unknown>).VITRUM_CAPTURE_ERROR = String(err);
});
