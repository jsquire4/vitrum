import type { Scene, SceneEmitter } from '@vitrum/core';
import { transformNormal, transformPoint } from '../math/mat4.js';
import { invertMat4 } from '../math/mat4.js';
import { buildCpuBvh } from './buildCpuBvh.js';
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

const IDENTITY_MAT4 = new Float32Array([
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

export function buildPackedScene(scene: Scene): PackedSceneData {
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  const triMaterialIds: number[] = [];
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

  let nextVertexOffset = 0;
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
      const invTransform = maybeInvTransform ?? IDENTITY_MAT4;
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
    const baseIndices =
      primitive.indices ??
      (() => {
        const generated = new Uint32Array(basePositions.length / 3);
        for (let i = 0; i < generated.length; i += 1) generated[i] = i;
        return generated;
      })();

    const transforms =
      primitive.kind === 'instanced-mesh' ? primitive.instances : [primitive.transform ?? undefined];

    for (const transform of transforms) {
      const vertexCount = Math.floor(basePositions.length / 3);
      for (let i = 0; i < vertexCount; i += 1) {
        const p: [number, number, number] = [
          basePositions[i * 3] ?? 0,
          basePositions[i * 3 + 1] ?? 0,
          basePositions[i * 3 + 2] ?? 0,
        ];
        const n: [number, number, number] = [
          primitive.normals[i * 3] ?? 0,
          primitive.normals[i * 3 + 1] ?? 1,
          primitive.normals[i * 3 + 2] ?? 0,
        ];
        const tp = transform == null ? p : transformPoint(transform, p);
        const tn = transform == null ? n : transformNormal(transform, n);
        positions.push(tp[0], tp[1], tp[2], 0);
        normals.push(tn[0], tn[1], tn[2], 0);
      }

      const triCount = Math.floor(baseIndices.length / 3);
      for (let t = 0; t < triCount; t += 1) {
        const i0 = (baseIndices[t * 3] ?? 0) + nextVertexOffset;
        const i1 = (baseIndices[t * 3 + 1] ?? 0) + nextVertexOffset;
        const i2 = (baseIndices[t * 3 + 2] ?? 0) + nextVertexOffset;
        indices.push(i0, i1, i2, 0);
        triMaterialIds.push(matId);
      }

      nextVertexOffset += vertexCount;
    }
  }

  const packedPositions = new Float32Array(positions);
  const packedNormals = new Float32Array(normals);
  const packedIndices = new Uint32Array(indices);
  const packedTriMaterialIds = new Uint32Array(triMaterialIds);
  const bvhBuild = buildCpuBvh(packedPositions, packedIndices, packedTriMaterialIds);
  const emitArrays = packEmitterArrays(scene);
  const environment = environmentParams(scene);
  warnings.push(...environment.warnings);
  warnings.push(...emitArrays.warnings);

  return {
    positions: packedPositions,
    normals: packedNormals,
    indices: bvhBuild.reorderedIndices,
    triMaterialIds: bvhBuild.reorderedTriMaterialIds,
    materials: new Float32Array(materials),
    bvhNodes: bvhBuild.bvhNodes,
    analyticHeaders: new Float32Array(analyticHeaders),
    analyticParams: new Float32Array(analyticParams),
    analyticLocalToWorld: new Float32Array(analyticLocalToWorld),
    analyticWorldToLocal: new Float32Array(analyticWorldToLocal),
    triangleCount: bvhBuild.reorderedTriMaterialIds.length,
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

  return {
    ...packed,
    bvhNodeCount: Math.floor(packed.bvhNodes.length / 8),
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
    },
  };
}
