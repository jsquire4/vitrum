import { gltfToScene } from '@vitrum/gltf-adapter';
import type { GltfJson } from '@vitrum/gltf-adapter';

interface ProofResult {
  readonly status: 'PASS' | 'FAIL';
  readonly hostHooksPresent: boolean;
  readonly draco?: {
    readonly positions: readonly number[];
    readonly normals: readonly number[];
    readonly indices: readonly number[];
  };
  readonly meshopt?: {
    readonly positions: readonly number[];
    readonly colors: readonly number[];
    readonly indices: readonly number[];
  };
  readonly error?: string;
}

declare global {
  // Browser proof result consumed by check-builtin-compression.mjs.
  var __VITRUM_BUILTIN_COMPRESSION_PROOF__: ProofResult | undefined;
}

const DRACO_TRIANGLE = new Uint8Array([
  68, 82, 65, 67, 79, 2, 2, 1, 0, 0, 0, 1, 3, 1, 0, 1, 2, 1, 2, 0, 9, 3, 0, 0, 1, 9, 3, 0, 1, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128,
  63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63,
]);

// meshoptimizer 1.1.1 streams generated from the triangle below. The proof
// covers ATTRIBUTES, TRIANGLES, and the KHR codec-v1 COLOR filter.
const MESHOPT_POSITIONS = new Uint8Array([
  160, 0, 0, 1, 60, 0, 0, 0, 255, 255, 1, 60, 0, 0, 0, 126, 125, 0, 0, 1, 12, 0, 0, 0, 255, 1, 12,
  0, 0, 0, 126, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
]);

const MESHOPT_INDICES = new Uint8Array([
  225, 240, 0, 118, 135, 86, 103, 120, 169, 134, 101, 137, 104, 152, 1, 105, 0, 0,
]);

const MESHOPT_COLOR = new Uint8Array([
  161, 191, 0, 94, 61, 0, 189, 125, 0, 99, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 96, 63, 161, 255, 0,
]);

function exactArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

function createDracoAsset(): {
  readonly gltf: GltfJson;
  readonly buffers: ReadonlyMap<number, ArrayBuffer>;
} {
  const gltf: GltfJson = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [
      {
        primitives: [
          {
            attributes: { POSITION: 0, NORMAL: 1 },
            extensions: {
              KHR_draco_mesh_compression: {
                bufferView: 0,
                attributes: { POSITION: 0, NORMAL: 1 },
              },
            },
          },
        ],
      },
    ],
    accessors: [
      { componentType: 5126, count: 3, type: 'VEC3' },
      { componentType: 5126, count: 3, type: 'VEC3' },
    ],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: DRACO_TRIANGLE.byteLength }],
    buffers: [{ byteLength: DRACO_TRIANGLE.byteLength }],
    extensionsUsed: ['KHR_draco_mesh_compression'],
    extensionsRequired: ['KHR_draco_mesh_compression'],
  };
  return { gltf, buffers: new Map([[0, exactArrayBuffer(DRACO_TRIANGLE)]]) };
}

function createMeshoptAsset(): {
  readonly gltf: GltfJson;
  readonly buffers: ReadonlyMap<number, ArrayBuffer>;
} {
  // Deliberately poison all parent/fallback ranges with zeroes. Exact nonzero
  // positions, indices, and colors can therefore only come from decoding all
  // three required KHR meshopt streams.
  const logical = new Uint8Array(54);

  const gltf: GltfJson = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [
      {
        primitives: [
          {
            attributes: { POSITION: 0, COLOR_0: 2 },
            indices: 1,
          },
        ],
      },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
      { bufferView: 2, componentType: 5121, normalized: true, count: 3, type: 'VEC4' },
    ],
    bufferViews: [
      {
        buffer: 0,
        byteOffset: 0,
        byteLength: 36,
        byteStride: 12,
        extensions: {
          KHR_meshopt_compression: {
            buffer: 1,
            byteOffset: 0,
            byteLength: MESHOPT_POSITIONS.byteLength,
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
          KHR_meshopt_compression: {
            buffer: 2,
            byteOffset: 0,
            byteLength: MESHOPT_INDICES.byteLength,
            byteStride: 2,
            count: 3,
            mode: 'TRIANGLES',
          },
        },
      },
      {
        buffer: 0,
        byteOffset: 42,
        byteLength: 12,
        byteStride: 4,
        extensions: {
          KHR_meshopt_compression: {
            buffer: 3,
            byteOffset: 0,
            byteLength: MESHOPT_COLOR.byteLength,
            byteStride: 4,
            count: 3,
            mode: 'ATTRIBUTES',
            filter: 'COLOR',
          },
        },
      },
    ],
    buffers: [
      { byteLength: logical.byteLength },
      { byteLength: MESHOPT_POSITIONS.byteLength },
      { byteLength: MESHOPT_INDICES.byteLength },
      { byteLength: MESHOPT_COLOR.byteLength },
    ],
    extensionsUsed: ['KHR_meshopt_compression'],
    extensionsRequired: ['KHR_meshopt_compression'],
  };
  return {
    gltf,
    buffers: new Map([
      [0, logical.buffer],
      [1, exactArrayBuffer(MESHOPT_POSITIONS)],
      [2, exactArrayBuffer(MESHOPT_INDICES)],
      [3, exactArrayBuffer(MESHOPT_COLOR)],
    ]),
  };
}

function requireMeshPrimitive(value: unknown): {
  readonly kind: 'mesh';
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly colors?: Float32Array;
  readonly indices?: Uint32Array;
} {
  if (value == null || typeof value !== 'object' || (value as { kind?: unknown }).kind !== 'mesh') {
    throw new Error('Expected one decoded mesh primitive.');
  }
  return value as ReturnType<typeof requireMeshPrimitive>;
}

async function run(): Promise<void> {
  globalThis.__VITRUM_BUILTIN_COMPRESSION_PROOF__ = undefined;
  const status = document.querySelector<HTMLOutputElement>('#status');
  try {
    const dracoAsset = createDracoAsset();
    const meshoptAsset = createMeshoptAsset();
    const dracoOptions = { buffers: dracoAsset.buffers };
    const meshoptOptions = { buffers: meshoptAsset.buffers };
    const hostHooksPresent = [dracoOptions, meshoptOptions].some(
      (options) => 'dracoDecode' in options || 'meshoptDecode' in options,
    );
    const [dracoResult, meshoptResult] = await Promise.all([
      gltfToScene(dracoAsset.gltf, dracoOptions),
      gltfToScene(meshoptAsset.gltf, meshoptOptions),
    ]);
    const draco = requireMeshPrimitive(dracoResult.scene.primitives[0]);
    const meshopt = requireMeshPrimitive(meshoptResult.scene.primitives[0]);
    if (meshopt.colors === undefined) throw new Error('meshopt COLOR_0 was not published.');

    globalThis.__VITRUM_BUILTIN_COMPRESSION_PROOF__ = {
      status: 'PASS',
      hostHooksPresent,
      draco: {
        positions: Array.from(draco.positions),
        normals: Array.from(draco.normals),
        indices: Array.from(draco.indices ?? []),
      },
      meshopt: {
        positions: Array.from(meshopt.positions),
        colors: Array.from(meshopt.colors),
        indices: Array.from(meshopt.indices ?? []),
      },
    };
    if (status) status.value = 'PASS';
  } catch (error) {
    const message =
      error instanceof Error ? `${error.message}\n${error.stack ?? ''}` : String(error);
    globalThis.__VITRUM_BUILTIN_COMPRESSION_PROOF__ = {
      status: 'FAIL',
      hostHooksPresent: false,
      error: message,
    };
    if (status) status.value = 'FAIL';
  }
}

void run();
