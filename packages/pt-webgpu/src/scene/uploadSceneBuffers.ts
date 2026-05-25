import { asMat4, type Scene, type SceneEmitter } from '@vitrum/core';
import { transformPoint } from '../math/mat4.js';
import { invertMat4 } from '../math/mat4.js';
import { buildCpuBvh } from './buildCpuBvh.js';
import { buildSceneTlas } from './tlasBridge.js';
import { MATERIAL_FLOAT_STRIDE, materialToPackedVec4s } from './materialPacking.js';
import { environmentParams } from './environmentPacking.js';
import {
  defaultDirectionalIrradiance,
  defaultDirectionalLight,
  packEmitterArrays,
} from './emitterPacking.js';

// 8 dead re-exports of MAX_*_LIGHTS / *_FLOAT_STRIDE constants (originally
// surfaced here for hosts that might assemble emitter arrays directly) were
// removed 2026-05-18 — no in-tree consumer reached them. The canonical
// definitions stay in `./emitterPacking.ts` and are file-locally used.

interface PackedSceneData {
  readonly positions: Float32Array; // vec4f packed
  readonly normals: Float32Array; // vec4f packed
  /**
   * Triangle indices — stride 4 (vec4u): 3 u32 vertex indices + `.w = 0`
   * (zero-fill contract). The pt-webgpu WGSL reads `.x,.y,.z` from
   * `array<vec4u>` — stride 4 is required for correct WGSL alignment.
   *
   * This differs from `shared-bvh/buildSceneBVH` which returns stride 3.
   * Upload-time assertion: `indices.byteLength % (4 * 4) === 0`.
   */
  readonly indices: Uint32Array; // vec4u packed (xyz = vertex indices, w = 0)
  readonly triMaterialIds: Uint32Array;
  readonly materials: Float32Array; // MATERIAL_VEC4_STRIDE * vec4f per material
  readonly bvhNodes: Float32Array; // 8 floats (32 bytes) per node
  readonly tlasNodes: Uint32Array; // 8 u32 words (32 bytes) per node
  readonly tlasInstanceIndices: Uint32Array;
  readonly tlasBlasRoots: Uint32Array;
  /** World-to-local matrices, 16 floats per instance. */
  readonly tlasInstanceWorldToLocal: Float32Array;
  /** Local-to-world matrices, 16 floats per instance. */
  readonly tlasInstanceLocalToWorld: Float32Array;
  readonly analyticHeaders: Float32Array; // vec4f per analytic primitive: [shapeId, materialId, paramsOffset, 0]
  readonly analyticParams: Float32Array; // vec4f array, two vec4f per analytic primitive (8 floats)
  readonly analyticLocalToWorld: Float32Array; // 4 vec4f (mat4) per analytic primitive
  readonly analyticWorldToLocal: Float32Array; // 4 vec4f (mat4) per analytic primitive
  readonly triangleCount: number;
  readonly analyticCount: number;
  readonly warnings: readonly string[];
  readonly directionalLight: readonly [number, number, number];
  readonly directionalIrradiance: readonly [number, number, number];
  readonly pointLightCount: number;
  readonly spotLightCount: number;
  readonly rectAreaLightCount: number;
  readonly meshAreaLightCount: number;
  readonly pointLightsData: Float32Array;
  readonly spotLightsData: Float32Array;
  readonly rectAreaLightsData: Float32Array;
  readonly meshAreaLightsData: Float32Array;
  readonly environmentTint: readonly [number, number, number];
  readonly environmentSunDirection: readonly [number, number, number];
  readonly environmentSunStrength: number;
  readonly environmentMapWidth: number;
  readonly environmentMapHeight: number;
  readonly hasEnvironmentMap: boolean;
  readonly environmentMapTexels: Float32Array; // rgba = radiance.rgb + pdfOmega
  readonly environmentMapCdf: Float32Array; // length N + 1
}

/**
 * `UploadedSceneBuffers` is `PackedSceneData` plus the matching GPU buffer
 * handles and derived counts. Embedding `PackedSceneData` directly removes
 * the ~30-field re-declaration that used to live here and the
 * field-by-field copy in `uploadPackedScene` (now a single spread).
 */
export interface UploadedSceneBuffers extends PackedSceneData {
  readonly bvhNodeCount: number;
  readonly tlasNodeCount: number;
  readonly materialCount: number;
  readonly positionsBuffer: GPUBuffer;
  readonly normalsBuffer: GPUBuffer;
  readonly indicesBuffer: GPUBuffer;
  readonly triMaterialIdsBuffer: GPUBuffer;
  readonly materialsBuffer: GPUBuffer;
  readonly bvhNodesBuffer: GPUBuffer;
  readonly analyticHeadersBuffer: GPUBuffer;
  readonly analyticParamsBuffer: GPUBuffer;
  readonly analyticLocalToWorldBuffer: GPUBuffer;
  readonly analyticWorldToLocalBuffer: GPUBuffer;
  readonly environmentMapTexelsBuffer: GPUBuffer;
  readonly environmentMapCdfBuffer: GPUBuffer;
  readonly pointLightsBuffer: GPUBuffer;
  readonly spotLightsBuffer: GPUBuffer;
  readonly rectAreaLightsBuffer: GPUBuffer;
  readonly meshAreaLightsBuffer: GPUBuffer;
  readonly tlasNodesBuffer: GPUBuffer;
  readonly tlasInstanceIndicesBuffer: GPUBuffer;
  readonly tlasBlasRootsBuffer: GPUBuffer;
  readonly tlasInstanceWorldToLocalBuffer: GPUBuffer;
  readonly tlasInstanceLocalToWorldBuffer: GPUBuffer;
  readonly destroy: () => void;
}

function createStorageBuffer(device: GPUDevice, label: string, data: ArrayBufferView): GPUBuffer {
  const minSize = data.byteLength === 0 ? 16 : data.byteLength;
  const buffer = device.createBuffer({
    label,
    size: Math.ceil(minSize / 4) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  if (data.byteLength > 0) {
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  } else {
    device.queue.writeBuffer(buffer, 0, new Uint32Array([0, 0, 0, 0]));
  }
  return buffer;
}

const IDENTITY_MAT4 = asMat4([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

/** Supported analytic-shape discriminator strings, in numeric-id order.
 *  The pt-webgpu WGSL shader reads `analyticHeader.x` and switches on these
 *  integer ids; the array index is the id. Slot 0 is reserved for "unknown". */
export const PT_WEBGPU_ANALYTIC_SHAPES = [
  /* 0 */ 'unknown',
  /* 1 */ 'sphere',
  /* 2 */ 'box',
  /* 3 */ 'capsule',
  /* 4 */ 'cylinder',
  /* 5 */ 'h-channel-came',
] as const;

function analyticShapeId(shape: string): number {
  const idx = (PT_WEBGPU_ANALYTIC_SHAPES as readonly string[]).indexOf(shape);
  return idx > 0 ? idx : 0;
}

interface PendingTlasInstance {
  readonly aabbMin: readonly [number, number, number];
  readonly aabbMax: readonly [number, number, number];
  readonly worldToLocal: Float32Array;
  readonly localToWorld: Float32Array;
  readonly blasRoot: number;
}

function buildTlasFromInstances(instances: readonly PendingTlasInstance[]): {
  tlasNodes: Uint32Array;
  tlasInstanceIndices: Uint32Array;
  tlasBlasRoots: Uint32Array;
  tlasInstanceWorldToLocal: Float32Array;
  tlasInstanceLocalToWorld: Float32Array;
} {
  if (instances.length === 0) {
    return {
      tlasNodes: new Uint32Array(0),
      tlasInstanceIndices: new Uint32Array(0),
      tlasBlasRoots: new Uint32Array(0),
      tlasInstanceWorldToLocal: new Float32Array(0),
      tlasInstanceLocalToWorld: new Float32Array(0),
    };
  }
  const tlas = buildSceneTlas(
    instances.map((instance) => ({
      blasId: instance.blasRoot,
      aabbMin: instance.aabbMin,
      aabbMax: instance.aabbMax,
      worldToLocal: instance.worldToLocal,
    })),
  );
  const l2w = new Float32Array(instances.length * 16);
  for (let i = 0; i < instances.length; i += 1) {
    l2w.set(instances[i]!.localToWorld, i * 16);
  }
  return {
    tlasNodes: tlas.nodes,
    tlasInstanceIndices: tlas.instanceIndices,
    tlasBlasRoots: tlas.blasRoots,
    tlasInstanceWorldToLocal: tlas.instanceTransforms,
    tlasInstanceLocalToWorld: l2w,
  };
}

function computeLocalAabb(positions: Float32Array): {
  min: readonly [number, number, number];
  max: readonly [number, number, number];
} | null {
  if (positions.length < 3) return null;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let i = 0; i + 2 < positions.length; i += 3) {
    const x = positions[i] ?? 0;
    const y = positions[i + 1] ?? 0;
    const z = positions[i + 2] ?? 0;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

function transformAabb(
  min: readonly [number, number, number],
  max: readonly [number, number, number],
  localToWorld: Float32Array,
): {
  min: readonly [number, number, number];
  max: readonly [number, number, number];
} {
  let outMinX = Number.POSITIVE_INFINITY;
  let outMinY = Number.POSITIVE_INFINITY;
  let outMinZ = Number.POSITIVE_INFINITY;
  let outMaxX = Number.NEGATIVE_INFINITY;
  let outMaxY = Number.NEGATIVE_INFINITY;
  let outMaxZ = Number.NEGATIVE_INFINITY;

  for (let c = 0; c < 8; c += 1) {
    const corner: [number, number, number] = [
      (c & 1) === 0 ? min[0] : max[0],
      (c & 2) === 0 ? min[1] : max[1],
      (c & 4) === 0 ? min[2] : max[2],
    ];
    const p = transformPoint(asMat4(localToWorld), corner);
    outMinX = Math.min(outMinX, p[0]);
    outMinY = Math.min(outMinY, p[1]);
    outMinZ = Math.min(outMinZ, p[2]);
    outMaxX = Math.max(outMaxX, p[0]);
    outMaxY = Math.max(outMaxY, p[1]);
    outMaxZ = Math.max(outMaxZ, p[2]);
  }

  return { min: [outMinX, outMinY, outMinZ], max: [outMaxX, outMaxY, outMaxZ] };
}

export function buildPackedScene(scene: Scene): PackedSceneData {
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  const triMaterialIds: number[] = [];
  const bvhNodeWords: number[] = [];
  const pendingTlasInstances: PendingTlasInstance[] = [];
  const materials: number[] = [];
  const analyticHeaders: number[] = [];
  const analyticParams: number[] = [];
  const analyticLocalToWorld: number[] = [];
  const analyticWorldToLocal: number[] = [];
  const warnings: string[] = [];
  for (const emitter of scene.emitters) {
    const k = emitter.kind;
    const supported =
      k === 'directional' ||
      k === 'point' ||
      k === 'spot' ||
      k === 'rect-area' ||
      k === 'disc-area' ||
      k === 'mesh-area';
    if (!supported) {
      warnings.push(
        `Emitter "${(emitter as SceneEmitter).id}" (${String(k)}) ignored; prototype supports directional, point, spot, rect-area, disc-area (packed as rect), and mesh-area emitters only.`,
      );
    }
  }

  let nextMaterialId = 0;

  for (const primitive of scene.primitives) {
    if (primitive.kind === 'analytic') {
      const shapeId = analyticShapeId(primitive.shape);
      if (shapeId === 0) {
        warnings.push(`Analytic primitive "${primitive.id}" has unsupported shape "${primitive.shape}".`);
        continue;
      }
      const matId = nextMaterialId++;
      materials.push(...materialToPackedVec4s(primitive.material));
      const transform = primitive.transform ?? IDENTITY_MAT4;
      const maybeInvTransform = invertMat4(transform);
      if (maybeInvTransform == null) {
        warnings.push(
          `Analytic primitive "${primitive.id}" has non-invertible transform; using identity worldToLocal fallback.`,
        );
      }
      const invTransform = asMat4(maybeInvTransform ?? IDENTITY_MAT4);
      const paramsOffset = Math.floor(analyticParams.length / 4);
      const p = primitive.params;
      analyticParams.push(
        p[0] ?? 0,
        p[1] ?? 0,
        p[2] ?? 0,
        p[3] ?? 0,
        p[4] ?? 0,
        p[5] ?? 0,
        p[6] ?? 0,
        p[7] ?? 0,
      );
      analyticHeaders.push(shapeId, matId, paramsOffset, 0);
      analyticLocalToWorld.push(...transform);
      analyticWorldToLocal.push(...invTransform);
      continue;
    }

    const matId = nextMaterialId++;
    materials.push(...materialToPackedVec4s(primitive.material));

    const basePositions = primitive.positions;
    const vertexCount = Math.floor(basePositions.length / 3);
    if (vertexCount < 3) {
      warnings.push(`Primitive "${primitive.id}" has fewer than 3 vertices; skipping.`);
      continue;
    }
    const baseIndices =
      primitive.indices ??
      (() => {
        const generated = new Uint32Array(vertexCount);
        for (let i = 0; i < generated.length; i += 1) generated[i] = i;
        return generated;
      })();
    const triCount = Math.floor(baseIndices.length / 3);
    if (triCount === 0) {
      warnings.push(`Primitive "${primitive.id}" has no triangles; skipping.`);
      continue;
    }

    const localPositions = new Float32Array(vertexCount * 4);
    const localNormals = new Float32Array(vertexCount * 4);
    for (let i = 0; i < vertexCount; i += 1) {
      localPositions[i * 4] = basePositions[i * 3] ?? 0;
      localPositions[i * 4 + 1] = basePositions[i * 3 + 1] ?? 0;
      localPositions[i * 4 + 2] = basePositions[i * 3 + 2] ?? 0;
      localPositions[i * 4 + 3] = 0;

      localNormals[i * 4] = primitive.normals[i * 3] ?? 0;
      localNormals[i * 4 + 1] = primitive.normals[i * 3 + 1] ?? 1;
      localNormals[i * 4 + 2] = primitive.normals[i * 3 + 2] ?? 0;
      localNormals[i * 4 + 3] = 0;
    }

    const localIndices = new Uint32Array(triCount * 4);
    for (let t = 0; t < triCount; t += 1) {
      localIndices[t * 4] = baseIndices[t * 3] ?? 0;
      localIndices[t * 4 + 1] = baseIndices[t * 3 + 1] ?? 0;
      localIndices[t * 4 + 2] = baseIndices[t * 3 + 2] ?? 0;
      localIndices[t * 4 + 3] = 0;
    }
    const localTriMaterialIds = new Uint32Array(triCount);
    localTriMaterialIds.fill(matId);
    const localBvh = buildCpuBvh(localPositions, localIndices, localTriMaterialIds);

    const vertexBase = Math.floor(positions.length / 4);
    const triBase = triMaterialIds.length;
    const nodeBase = Math.floor(bvhNodeWords.length / 8);
    const localNodeWords = new Uint32Array(localBvh.bvhNodes.buffer, localBvh.bvhNodes.byteOffset, localBvh.bvhNodes.length);

    for (let i = 0; i < localPositions.length; i += 1) positions.push(localPositions[i] ?? 0);
    for (let i = 0; i < localNormals.length; i += 1) normals.push(localNormals[i] ?? 0);
    for (let i = 0; i + 3 < localBvh.reorderedIndices.length; i += 4) {
      indices.push(
        (localBvh.reorderedIndices[i] ?? 0) + vertexBase,
        (localBvh.reorderedIndices[i + 1] ?? 0) + vertexBase,
        (localBvh.reorderedIndices[i + 2] ?? 0) + vertexBase,
        localBvh.reorderedIndices[i + 3] ?? 0,
      );
    }
    for (let i = 0; i < localBvh.reorderedTriMaterialIds.length; i += 1) {
      triMaterialIds.push(localBvh.reorderedTriMaterialIds[i] ?? matId);
    }

    for (let n = 0; n + 7 < localNodeWords.length; n += 8) {
      const splitOrCount = localNodeWords[n + 7] ?? 0;
      const isLeaf = (splitOrCount & 0xffff0000) === 0xffff0000;
      bvhNodeWords.push(
        localNodeWords[n] ?? 0,
        localNodeWords[n + 1] ?? 0,
        localNodeWords[n + 2] ?? 0,
        localNodeWords[n + 3] ?? 0,
        localNodeWords[n + 4] ?? 0,
        localNodeWords[n + 5] ?? 0,
        isLeaf ? (localNodeWords[n + 6] ?? 0) + triBase : (localNodeWords[n + 6] ?? 0),
        splitOrCount,
      );
    }

    const localAabb = computeLocalAabb(basePositions);
    if (localAabb == null) {
      continue;
    }
    const transforms =
      primitive.kind === 'instanced-mesh' ? primitive.instances : [primitive.transform ?? undefined];
    if (transforms.length === 0) {
      warnings.push(`Instanced primitive "${primitive.id}" has no instances; skipping TLAS instance upload.`);
      continue;
    }
    for (const transform of transforms) {
      const candidateLocalToWorld = asMat4(transform ?? IDENTITY_MAT4);
      const maybeWorldToLocal = invertMat4(candidateLocalToWorld);
      if (maybeWorldToLocal == null) {
        warnings.push(
          `Primitive "${primitive.id}" has non-invertible instance transform; using identity fallback for TLAS transform.`,
        );
      }
      const localToWorld = maybeWorldToLocal == null ? IDENTITY_MAT4 : candidateLocalToWorld;
      const worldToLocal = asMat4(maybeWorldToLocal ?? IDENTITY_MAT4);
      const worldAabb = transformAabb(localAabb.min, localAabb.max, localToWorld);
      pendingTlasInstances.push({
        blasRoot: nodeBase,
        worldToLocal,
        localToWorld,
        aabbMin: worldAabb.min,
        aabbMax: worldAabb.max,
      });
    }
  }

  const packedPositions = new Float32Array(positions);
  const packedNormals = new Float32Array(normals);
  const packedIndices = new Uint32Array(indices);
  const packedTriMaterialIds = new Uint32Array(triMaterialIds);
  const packedBvhNodes = new Float32Array(new Uint32Array(bvhNodeWords).buffer);
  const tlasBuild = buildTlasFromInstances(pendingTlasInstances);
  const emitArrays = packEmitterArrays(scene);
  const environment = environmentParams(scene);
  warnings.push(...environment.warnings);
  warnings.push(...emitArrays.warnings);

  return {
    positions: packedPositions,
    normals: packedNormals,
    indices: packedIndices,
    triMaterialIds: packedTriMaterialIds,
    materials: new Float32Array(materials),
    bvhNodes: packedBvhNodes,
    tlasNodes: tlasBuild.tlasNodes,
    tlasInstanceIndices: tlasBuild.tlasInstanceIndices,
    tlasBlasRoots: tlasBuild.tlasBlasRoots,
    tlasInstanceWorldToLocal: tlasBuild.tlasInstanceWorldToLocal,
    tlasInstanceLocalToWorld: tlasBuild.tlasInstanceLocalToWorld,
    analyticHeaders: new Float32Array(analyticHeaders),
    analyticParams: new Float32Array(analyticParams),
    analyticLocalToWorld: new Float32Array(analyticLocalToWorld),
    analyticWorldToLocal: new Float32Array(analyticWorldToLocal),
    triangleCount: packedTriMaterialIds.length,
    analyticCount: Math.floor(analyticHeaders.length / 4),
    warnings,
    directionalLight: defaultDirectionalLight(scene),
    directionalIrradiance: defaultDirectionalIrradiance(scene),
    pointLightCount: emitArrays.pointLightCount,
    spotLightCount: emitArrays.spotLightCount,
    rectAreaLightCount: emitArrays.rectAreaLightCount,
    meshAreaLightCount: emitArrays.meshAreaLightCount,
    pointLightsData: emitArrays.pointLightsData,
    spotLightsData: emitArrays.spotLightsData,
    rectAreaLightsData: emitArrays.rectAreaLightsData,
    meshAreaLightsData: emitArrays.meshAreaLightsData,
    environmentTint: environment.tint,
    environmentSunDirection: environment.sunDirection,
    environmentSunStrength: environment.sunStrength,
    environmentMapWidth: environment.hdriWidth,
    environmentMapHeight: environment.hdriHeight,
    hasEnvironmentMap: environment.hasHdri,
    environmentMapTexels: environment.hdriTexels,
    environmentMapCdf: environment.hdriCdf,
  };
}

export function uploadPackedScene(device: GPUDevice, packed: PackedSceneData): UploadedSceneBuffers {
  // Upload-time assertion: pt-webgpu uses stride-4 indices (vec4u, .w = 0).
  // byteLength must be a multiple of 4 u32 = 16 bytes.
  if (packed.indices.byteLength > 0 && packed.indices.byteLength % 16 !== 0) {
    throw new Error(
      `[pt-webgpu/uploadPackedScene] Index buffer byteLength (${packed.indices.byteLength}) ` +
        `is not aligned to BvhIndexStride 4 (16 bytes per triangle). ` +
        `pt-webgpu requires stride-4 indices — 3 vertex u32 + 1 zero-fill u32.`,
    );
  }
  const positionsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.positions', packed.positions);
  const normalsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.normals', packed.normals);
  const indicesBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.indices', packed.indices);
  const triMaterialIdsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.triMaterialIds', packed.triMaterialIds);
  const materialsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.materials', packed.materials);
  const bvhNodesBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.bvhNodes', packed.bvhNodes);
  const analyticHeadersBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.analyticHeaders', packed.analyticHeaders);
  const analyticParamsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.analyticParams', packed.analyticParams);
  const analyticLocalToWorldBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.analyticLocalToWorld', packed.analyticLocalToWorld);
  const analyticWorldToLocalBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.analyticWorldToLocal', packed.analyticWorldToLocal);
  const environmentMapTexelsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.environmentMapTexels', packed.environmentMapTexels);
  const environmentMapCdfBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.environmentMapCdf', packed.environmentMapCdf);
  const pointLightsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.pointLights', packed.pointLightsData);
  const spotLightsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.spotLights', packed.spotLightsData);
  const rectAreaLightsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.rectAreaLights', packed.rectAreaLightsData);
  const meshAreaLightsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.meshAreaLights', packed.meshAreaLightsData);
  const tlasNodesBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.tlasNodes', packed.tlasNodes);
  const tlasInstanceIndicesBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.tlasInstanceIndices', packed.tlasInstanceIndices);
  const tlasBlasRootsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.tlasBlasRoots', packed.tlasBlasRoots);
  const tlasInstanceWorldToLocalBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.tlasInstanceWorldToLocal',
    packed.tlasInstanceWorldToLocal,
  );
  const tlasInstanceLocalToWorldBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.tlasInstanceLocalToWorld',
    packed.tlasInstanceLocalToWorld,
  );

  return {
    ...packed,
    bvhNodeCount: Math.floor(packed.bvhNodes.length / 8),
    tlasNodeCount: Math.floor(packed.tlasNodes.length / 8),
    materialCount: Math.floor(packed.materials.length / MATERIAL_FLOAT_STRIDE),
    positionsBuffer,
    normalsBuffer,
    indicesBuffer,
    triMaterialIdsBuffer,
    materialsBuffer,
    bvhNodesBuffer,
    analyticHeadersBuffer,
    analyticParamsBuffer,
    analyticLocalToWorldBuffer,
    analyticWorldToLocalBuffer,
    environmentMapTexelsBuffer,
    environmentMapCdfBuffer,
    pointLightsBuffer,
    spotLightsBuffer,
    rectAreaLightsBuffer,
    meshAreaLightsBuffer,
    tlasNodesBuffer,
    tlasInstanceIndicesBuffer,
    tlasBlasRootsBuffer,
    tlasInstanceWorldToLocalBuffer,
    tlasInstanceLocalToWorldBuffer,
    destroy: () => {
      positionsBuffer.destroy();
      normalsBuffer.destroy();
      indicesBuffer.destroy();
      triMaterialIdsBuffer.destroy();
      materialsBuffer.destroy();
      bvhNodesBuffer.destroy();
      analyticHeadersBuffer.destroy();
      analyticParamsBuffer.destroy();
      analyticLocalToWorldBuffer.destroy();
      analyticWorldToLocalBuffer.destroy();
      environmentMapTexelsBuffer.destroy();
      environmentMapCdfBuffer.destroy();
      pointLightsBuffer.destroy();
      spotLightsBuffer.destroy();
      rectAreaLightsBuffer.destroy();
      meshAreaLightsBuffer.destroy();
      tlasNodesBuffer.destroy();
      tlasInstanceIndicesBuffer.destroy();
      tlasBlasRootsBuffer.destroy();
      tlasInstanceWorldToLocalBuffer.destroy();
      tlasInstanceLocalToWorldBuffer.destroy();
    },
  };
}
