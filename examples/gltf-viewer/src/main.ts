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
import { loadGltfWithEngine } from '@vitrum/engine/gltf';

const params = new URLSearchParams(location.search);
const targetSpp = Number(params.get('vitrumSpp')) || 128;

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
  const result = await loadGltfWithEngine(createEmbeddedGltf(), {
    compatibilityMode: 'best-effort',
    decodeImage: async () => ({
      width: 2,
      height: 2,
      data: new Uint8Array([
        255, 80, 40, 255,
        40, 180, 255, 255,
        255, 235, 80, 255,
        180, 80, 255, 255,
      ]),
    }),
    engineOptions: {
      canvas,
      prefer: 'auto',
      onWarning: (warning) => {
        console.warn('[gltf-viewer example]', warning.message, warning.details ?? '');
      },
    },
  });
  const engine = requireEngine(result.engine);

  (globalThis as Record<string, unknown>).VITRUM_GLTF_BACKEND = result.backend;
  (globalThis as Record<string, unknown>).VITRUM_GLTF_TEXTURE_REPORT = result.textureDecodeReport;
  (globalThis as Record<string, unknown>).VITRUM_GLTF_WARNINGS = result.warnings;

  let frameIndex = 0;
  let captureSignalled = false;
  let lastSpp = 0;

  engine.onFrame?.((stats) => {
    lastSpp = stats.spp ?? lastSpp;
    (globalThis as Record<string, unknown>).VITRUM_MS_PER_SAMPLE =
      stats.frameTimeMs > 0 && lastSpp > 0 ? stats.frameTimeMs / lastSpp : 0;
  });

  function tick(now: number): void {
    if (result.asset.animations.length > 0) {
      result.controller.advance(now * 0.001, { engine });
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
});
