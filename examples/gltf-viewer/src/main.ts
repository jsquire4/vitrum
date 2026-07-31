/**
 * glTF viewer example — one-call glTF loading through @vitrum/engine/gltf.
 *
 * The example builds a tiny glTF asset in-memory so it does not depend on a
 * network model. The code path is still the real public path:
 * loadGltfWithEngine() -> adapter feature report -> backend recommendation ->
 * createEngine({ gltfAsset }) -> GltfSceneController attachment.
 *
 * Capture protocol: waits for targetSpp accumulated PT samples, or the same
 * number of rendered frames when the selected backend is real-time.
 */

import { asMat4 } from '@vitrum/core';
import type { FrameStats, ProgressStats, Scene } from '@vitrum/core';
import { attachVitrum, computeSceneAABB } from '@vitrum/engine';
import type { AttachVitrumHandle, CameraLike, SceneAABB } from '@vitrum/engine';
import type { AttachVitrumSceneController } from '@vitrum/engine/lifecycle';
import { loadGltfWithEngine, releaseGltfResources } from '@vitrum/engine/gltf';
import {
  createAxisAlignedView,
  createPerspectiveProjection,
  syncCanvasToDisplaySize,
  writePerspectiveProjection,
  type ExampleVec3,
  type PerspectiveOptions,
} from '../../shared/exampleHost.js';

const params = new URLSearchParams(location.search);
const targetSpp = Number(params.get('vitrumSpp')) || 128;
const requestedAssetId = params.get('vitrumGltfAsset') ?? '';
const requestedBackend = params.get('vitrumBackend') ?? '';

type BrowserCompressionKind = 'draco' | 'meshopt';

interface RealGltfAsset {
  readonly url: string;
  readonly kind: string;
  readonly minPrimitives: number;
  readonly minTextures: number;
  readonly requiredExtensions?: readonly string[];
  readonly compression?: readonly BrowserCompressionKind[];
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
    compression: ['draco'],
  },
  'meshopt-cube-real': {
    url: 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/MeshoptCubeTest/glTF-Meshopt/MeshoptCubeTest.gltf',
    kind: 'meshopt',
    minPrimitives: 1,
    minTextures: 0,
    requiredExtensions: ['KHR_meshopt_compression'],
    compression: ['meshopt'],
  },
};

const canvas = document.getElementById('vitrum-canvas') as HTMLCanvasElement;
const initialViewport = syncCanvasToDisplaySize(canvas);
const viewMatrix = asMat4(createAxisAlignedView([0, 0, 3]));
const projMatrix = asMat4(
  createPerspectiveProjection(initialViewport.width, initialViewport.height),
);
let disposeAfterFatalError: (() => void) | undefined;

async function main(): Promise<void> {
  const realAsset = requestedAssetId ? REAL_GLTF_ASSETS[requestedAssetId] : undefined;
  if (requestedAssetId && realAsset == null) {
    throw new Error(`[gltf-viewer example] unsupported vitrumGltfAsset "${requestedAssetId}"`);
  }
  const browserCompressionDecoders = compressionDecoderReport(realAsset);
  const result = await loadGltfWithEngine(realAsset?.url ?? createEmbeddedGltf(), {
    compatibilityMode: 'best-effort',
    ...(realAsset == null ? { decodeImage: decodeEmbeddedDemoImage } : {}),
    ...(realAsset != null
      ? {
          decodeTextures: true,
          decodePixels: decodeBrowserImagePixels,
          maxTextureSize: 4096,
          warnOnNpotRepeatWrap: true,
        }
      : {}),
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
  let resourcesReleased = false;
  const releaseResources = (): void => {
    if (resourcesReleased) return;
    resourcesReleased = true;
    releaseGltfResources(result);
  };
  disposeAfterFatalError = () => {
    try {
      engine.dispose();
    } finally {
      releaseResources();
    }
  };
  const sceneBounds = computeSceneAABB(result.controller.scene);
  const renderScene =
    realAsset == null
      ? result.controller.scene
      : addRealAssetLighting(result.controller.scene, sceneBounds);
  if (renderScene !== result.controller.scene) {
    engine.setScene(renderScene);
  }
  const cameraPosition = { x: 0, y: 0, z: 0 };
  const refreshCameraFrame = (width: number, height: number): void => {
    const frame = deriveViewerCamera(sceneBounds, width / Math.max(height, 1));
    viewMatrix.set(createAxisAlignedView(frame.position));
    writePerspectiveProjection(projMatrix, width, height, frame.projection);
    cameraPosition.x = frame.position[0];
    cameraPosition.y = frame.position[1];
    cameraPosition.z = frame.position[2];
  };
  refreshCameraFrame(initialViewport.width, initialViewport.height);
  const camera: CameraLike = {
    updateMatrixWorld() {
      // Re-fit the asset when aspect changes, rather than updating projection
      // alone and allowing narrow resizes to crop a previously fitted model.
      refreshCameraFrame(canvas.width, canvas.height);
    },
    matrixWorldInverse: { elements: viewMatrix },
    projectionMatrix: { elements: projMatrix },
    position: cameraPosition,
  };

  (globalThis as Record<string, unknown>).VITRUM_CAPTURE_FRAME = async (
    colorSpace: 'linear' | 'output' = 'output',
  ) => {
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
    browserCompressionDecoders,
    textureDecodeReport: result.textureDecodeReport,
    warnings: result.warnings,
    diagnostics: result.diagnostics,
    realAssetReady:
      realAsset == null ||
      (result.controller.scene.primitives.length >= realAsset.minPrimitives &&
        result.textureDecodeReport.mapCount >= realAsset.minTextures &&
        (realAsset.requiredExtensions ?? []).every((ext) =>
          result.asset.featureReport.extensions.used.includes(ext),
        )),
  };

  let captureSignalled = false;
  let renderedRealtimeFrames = 0;
  let capturePaused = false;

  const playbackController: AttachVitrumSceneController = {
    animations: result.controller.animations,
    attachEngine(target, options) {
      result.controller.attachEngine(target, options);
    },
    advance(deltaSeconds, options) {
      if (!capturePaused) result.controller.advance(deltaSeconds, options);
    },
  };

  // loadGltfWithEngine owns import/backend selection; attachVitrum owns the
  // selected engine's presentation mode, resize propagation, and RAF cadence.
  // This is required for both swapchain-required and offscreen WebGPU engines.
  const handle = await attachVitrum({
    canvas,
    scene: renderScene,
    camera,
    engine,
    sceneController: playbackController,
    sceneControllerPlayback: result.asset.animations.length > 0,
    quality: { samplesTarget: targetSpp },
    onFrame(stats: FrameStats) {
      const spp = stats.spp ?? 0;
      (globalThis as Record<string, unknown>).VITRUM_MS_PER_SAMPLE =
        stats.frameTimeMs > 0 && spp > 0 ? stats.frameTimeMs / spp : 0;
      if (
        !engine.capabilities.accumulates &&
        spp > 0 &&
        !captureSignalled &&
        ++renderedRealtimeFrames >= targetSpp
      ) {
        (globalThis as Record<string, unknown>).VITRUM_CAPTURE_READY = true;
        captureSignalled = true;
      }
    },
    onProgress(progress: ProgressStats) {
      if (
        progress.kind === 'pt-spp' &&
        !captureSignalled &&
        progress.current >= targetSpp
      ) {
        (globalThis as Record<string, unknown>).VITRUM_CAPTURE_READY = true;
        captureSignalled = true;
      }
    },
  });
  const dispose = (): void => {
    try {
      handle.dispose();
    } finally {
      releaseResources();
    }
  };
  disposeAfterFatalError = dispose;

  Object.defineProperty(globalThis, 'VITRUM_CAPTURE_PAUSED', {
    configurable: true,
    enumerable: true,
    get: () => capturePaused,
    set: (value: unknown) => {
      capturePaused = value === true;
      if (capturePaused) handle.engine.pause();
      else handle.engine.resume();
    },
  });
  const publicHandle: AttachVitrumHandle = {
    get engine() { return handle.engine; },
    get backendId() { return handle.backendId; },
    get backendProfileId() { return handle.backendProfileId; },
    get profileId() { return handle.profileId; },
    dispose,
    captureFrame: (options) => handle.captureFrame(options),
  };
  (globalThis as Record<string, unknown>).VITRUM_HANDLE = publicHandle;
  (globalThis as Record<string, unknown>).VITRUM_DISPOSE = dispose;
}

function compressionDecoderReport(asset: RealGltfAsset | undefined): Record<string, unknown> {
  const requested = asset?.compression ?? [];
  return {
    policy: 'builtin',
    requested,
    hostOverrides: [],
    draco: requested.includes('draco') ? 'builtin' : 'unused',
    meshopt: requested.includes('meshopt') ? 'builtin' : 'unused',
  };
}

function createEmbeddedGltf(): Parameters<typeof loadGltfWithEngine>[0] {
  const binary = createTriangleBuffer();
  return {
    asset: { version: '2.0', generator: 'vitrum examples/gltf-viewer' },
    scene: 0,
    extensionsUsed: ['KHR_materials_unlit', 'KHR_texture_transform'],
    buffers: [
      {
        byteLength: binary.byteLength,
        uri: dataUri(binary, 'application/octet-stream'),
      },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36, target: 34962 },
      { buffer: 0, byteOffset: 36, byteLength: 36, target: 34962 },
      { buffer: 0, byteOffset: 72, byteLength: 24, target: 34962 },
      { buffer: 0, byteOffset: 96, byteLength: 6, target: 34963 },
    ],
    accessors: [
      {
        bufferView: 0,
        componentType: 5126,
        count: 3,
        type: 'VEC3',
        min: [-0.9, -0.65, 0],
        max: [0.9, 0.8, 0],
      },
      {
        bufferView: 1,
        componentType: 5126,
        count: 3,
        type: 'VEC3',
        min: [0, 0, 1],
        max: [0, 0, 1],
      },
      { bufferView: 2, componentType: 5126, count: 3, type: 'VEC2', min: [0, 0], max: [1, 1] },
      { bufferView: 3, componentType: 5123, count: 3, type: 'SCALAR', min: [0], max: [2] },
    ],
    samplers: [{ wrapS: 33071, wrapT: 33648 }],
    images: [{ uri: 'data:image/png;base64,AAAA', mimeType: 'image/png' }],
    textures: [{ source: 0, sampler: 0 }],
    materials: [
      {
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
      },
    ],
    meshes: [
      {
        primitives: [
          {
            attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 },
            indices: 3,
            material: 0,
            mode: 4,
          },
        ],
      },
    ],
    nodes: [{ mesh: 0, name: 'triangle' }],
    scenes: [{ nodes: [0] }],
  };
}

async function decodeEmbeddedDemoImage(): Promise<{
  width: number;
  height: number;
  data: Uint8Array;
}> {
  return {
    width: 2,
    height: 2,
    data: new Uint8Array([
      255, 80, 40, 255, 40, 180, 255, 255, 255, 235, 80, 255, 180, 80, 255, 255,
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
  try {
    const width = bitmap.width;
    const height = bitmap.height;
    const canvas2d =
      typeof OffscreenCanvas !== 'undefined'
        ? new OffscreenCanvas(width, height)
        : document.createElement('canvas');
    canvas2d.width = width;
    canvas2d.height = height;
    const ctx = canvas2d.getContext('2d') as
      | CanvasRenderingContext2D
      | OffscreenCanvasRenderingContext2D
      | null;
    if (ctx == null) {
      throw new Error('[gltf-viewer example] 2D canvas unavailable for texture decode.');
    }
    ctx.drawImage(bitmap, 0, 0);
    const pixels = ctx.getImageData(0, 0, width, height).data;
    return {
      width,
      height,
      channels: 4,
      dataType: 'uint8',
      colorSpace: context.colorSpace,
      data: pixels,
    };
  } finally {
    bitmap.close();
  }
}

interface ViewerCameraFrame {
  readonly position: ExampleVec3;
  readonly projection: PerspectiveOptions;
}

function deriveViewerCamera(bounds: SceneAABB, aspect: number): ViewerCameraFrame {
  const safeAspect = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  const fovY = Math.PI / 3;
  const tanHalfFovY = Math.tan(fovY / 2);
  const halfWidth = Math.max(bounds.extent[0] * 0.5, bounds.diagonal * 0.005);
  const halfHeight = Math.max(bounds.extent[1] * 0.5, bounds.diagonal * 0.005);
  const halfDepth = Math.max(bounds.extent[2] * 0.5, 0);
  const distanceToFront =
    1.15 * Math.max(halfHeight / tanHalfFovY, halfWidth / (tanHalfFovY * safeAspect));
  const distance = Math.max(
    halfDepth + distanceToFront,
    bounds.diagonal,
    0.01,
  );
  const near = Math.max(bounds.diagonal * 0.0001, 0.0001);
  const far = Math.max(near * 2, distance + halfDepth + bounds.diagonal * 2);
  return {
    position: [
      bounds.center[0],
      bounds.center[1],
      bounds.center[2] + distance,
    ],
    projection: { fovY, near, far },
  };
}

function addRealAssetLighting(scene: Scene, bounds: SceneAABB): Scene {
  const scale = Math.max(bounds.extent[0], bounds.extent[1], bounds.extent[2], 0.01);
  return {
    ...scene,
    emitters: [
      ...(scene.emitters ?? []),
      {
        kind: 'rect-area',
        id: 'gltf-viewer-real-asset-key-light',
        position: [
          bounds.center[0],
          bounds.max[1] + scale * 0.2,
          bounds.max[2] + scale * 0.45,
        ],
        uAxis: [scale * 0.3, 0, 0],
        // uAxis × vAxis points toward -Z, back into the framed asset.
        vAxis: [0, -scale * 0.3, 0],
        color: [1, 1, 1],
        intensity: 18.0,
      },
    ],
    environment: {
      kind: 'procedural-sky',
      // Core's procedural-sky contract requires a unit vector. This is the
      // normalized form of the intended [0.4, 1.0, 0.25] key-light direction.
      sunDirection: [0.36177250531690763, 0.9044312632922691, 0.22610781582306727],
      turbidity: 2.0,
      rayleigh: 1.0,
      mieCoefficient: 0.005,
      mieDirectionalG: 0.8,
      intensity: 1.0,
    },
  };
}

function createTriangleBuffer(): Uint8Array {
  const bytes = new Uint8Array(104);
  writeF32(bytes, 0, [-0.9, -0.65, 0, 0.9, -0.65, 0, 0.0, 0.8, 0]);
  writeF32(bytes, 36, [0, 0, 1, 0, 0, 1, 0, 0, 1]);
  writeF32(bytes, 72, [0, 0, 1, 0, 0.5, 1]);
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
  try {
    disposeAfterFatalError?.();
  } catch (cleanupError) {
    console.error('[gltf-viewer example] cleanup after fatal error failed:', cleanupError);
  }
  console.error('[gltf-viewer example] fatal:', err);
  (globalThis as Record<string, unknown>).VITRUM_CAPTURE_ERROR = String(err);
});
