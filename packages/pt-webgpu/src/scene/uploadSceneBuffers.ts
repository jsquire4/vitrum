import {
  asMat4,
  partitionSceneBySupport,
  type AnalyticShape,
  type MaterialSpec,
  type Scene,
  type SceneEmitter,
  type ScenePrimitive,
  type SupportSets,
} from '@vitrum/core';
import {
  packSceneFromCore,
  refitTlasTransforms,
  type PrimitiveTlasBinding,
  type ScenePackResult,
} from '@vitrum/shared-bvh';
import { buildLightTree, packLightTreeForGPU } from '@vitrum/shared-samplers';
import { invertMat4 } from '../math/mat4.js';
import { MATERIAL_FLOAT_STRIDE, materialToPackedVec4s } from './materialPacking.js';
import { collectMaterialTextures } from './materialTextures.js';
import { createMaterialTextureArray } from './materialTextureArray.js';
import { environmentParams } from './environmentPacking.js';
import {
  buildLightTreeInputForScene,
  defaultDirectionalIrradiance,
  defaultDirectionalLight,
  packEmitterArrays,
  type PackedEmitterArrays,
} from './emitterPacking.js';

// 8 dead re-exports of MAX_*_LIGHTS / *_FLOAT_STRIDE constants (originally
// surfaced here for hosts that might assemble emitter arrays directly) were
// removed 2026-05-18 — no in-tree consumer reached them. The canonical
// definitions stay in `./emitterPacking.ts` and are file-locally used.

interface PackedSceneData {
  readonly positions: Float32Array; // vec4f packed
  readonly normals: Float32Array; // vec4f packed
  readonly uvs: Float32Array; // vec4f packed (.xy = uv0, .zw = uv1) — P2 texture sampling
  /** Per-material texture descriptor floats (MATERIAL_TEX_FLOAT_STRIDE each):
   *  texture indices + alpha-mode + KHR UV transform. Indexed by matId. */
  readonly materialTexDescriptors: Float32Array;
  /** Dedup'd, upload-ordered sRGB texture handles (baseColor + emissive; layer
   *  i = sources[i]); the GPU upload turns these into a texture_2d_array. */
  readonly materialTextureSources: readonly unknown[];
  /** Dedup'd, upload-ordered LINEAR texture handles (normal + ORM) → a second
   *  texture_2d_array sampled without sRGB decode. */
  readonly materialTextureLinearSources: readonly unknown[];
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
  /** Stable primitive→TLAS binding metadata for transform-only fast paths. */
  readonly primitiveTlasBindings: readonly PrimitiveTlasBinding[];
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
  /**
   * WS2 — packed power-weighted light-tree nodes (12 f32 / node, see
   * `packLightTreeForGPU`). Empty when < 2 selectable lights (the NEE falls back
   * to the uniform pick). `lightTreeNodeCount` is the descent loop bound.
   */
  readonly lightTreeNodes: Float32Array;
  readonly lightTreeNodeCount: number;
  readonly lightTreeEnabled: boolean;
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
  /** WS2 — light-tree node storage buffer (group 3, full tier only). */
  readonly lightTreeBuffer: GPUBuffer;
  /** P2 — per-vertex UV storage buffer (group 3, full tier only). */
  readonly uvsBuffer: GPUBuffer;
  /** P2 — per-material texture descriptor storage buffer (group 3). */
  readonly materialTexDescriptorsBuffer: GPUBuffer;
  /** P2 — sampled sRGB texture_2d_array handle (baseColor/emissive; for dispose). */
  readonly materialTexture: GPUTexture;
  /** P2 — sRGB 2d-array view bound in group 3 (binding 3). */
  readonly materialTextureView: GPUTextureView;
  /** P2 — filtering sampler bound in group 3 (shared by both arrays). */
  readonly materialTextureSampler: GPUSampler;
  /** P2 — LINEAR texture_2d_array handle (normal/ORM; for dispose). */
  readonly materialLinearTexture: GPUTexture;
  /** P2 — linear 2d-array view bound in group 3 (binding 5). */
  readonly materialLinearTextureView: GPUTextureView;
  /** Live GPU footprint of the scene resources, split so the debug surface can
   *  honour `GpuMemoryBreakdown`'s invariants: `bufferBytes` (all 24 scene
   *  STORAGE buffers) + `textureBytesByFormat` (the two material arrays, keyed by
   *  their actual `GPUTextureFormat`). Read off the CURRENT handles (not a
   *  creation-time constant) so it stays correct after a realloc fast-path swaps
   *  BLAS/TLAS/uvs/light-tree buffers onto this struct. */
  readonly gpuMemoryBytes: () => {
    readonly bufferBytes: number;
    readonly textureBytesByFormat: Readonly<Record<string, number>>;
  };
  readonly destroy: () => void;
}

export type { PrimitiveTlasBinding };

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

/** The capability sets `buildPackedScene` partitions a scene against. These are
 *  the single source of truth for what pt-webgpu ingests — `index.ts` derives
 *  the engine's advertised `EngineCapabilities.supported*Kinds` from the same
 *  values, so the declared sets and the ingestion behavior can no longer drift.
 *
 *  Slot 0 of `PT_WEBGPU_ANALYTIC_SHAPES` is the "unknown" sentinel; the real
 *  supported shapes start at index 1. */
export const PT_WEBGPU_SUPPORT: Required<SupportSets> = {
  supportedPrimitiveKinds: new Set<ScenePrimitive['kind']>([
    'mesh',
    'instanced-mesh',
    'analytic',
    'skinned-mesh',
  ]),
  supportedEmitterKinds: new Set<SceneEmitter['kind']>([
    'directional',
    'point',
    'spot',
    'rect-area',
    'disc-area',
    'mesh-area',
  ]),
  supportedAnalyticShapes: new Set<AnalyticShape>(
    PT_WEBGPU_ANALYTIC_SHAPES.slice(1) as readonly AnalyticShape[],
  ),
  supportedEnvironmentKinds: new Set<Scene['environment']['kind']>([
    'none',
    'hdri',
    'procedural-sky',
  ]),
};

function isMeshLikePrimitive(
  primitive: ScenePrimitive,
): primitive is Extract<ScenePrimitive, { kind: 'mesh' | 'skinned-mesh' | 'instanced-mesh' }> {
  return primitive.kind === 'mesh'
    || primitive.kind === 'skinned-mesh'
    || primitive.kind === 'instanced-mesh';
}

/** Options for {@link buildPackedScene}. */
export interface BuildPackedSceneOptions {
  /**
   * Re-attach emissive radiance onto emissive-mesh primitives so they glow when
   * seen directly by the camera / through refraction (not only via NEE). See
   * `PTEngineWebGPUOptions.cameraVisibleEmitters`. Default `false` (caller passes
   * the engine's resolved value); `false` keeps the pre-2026-05-30 NEE-only
   * behaviour (emissive stays at whatever the scene primitive carries — zero for
   * `sceneFromThreeJS`-converted emissive meshes).
   */
  readonly cameraVisibleEmitters?: boolean;
}

export function buildPackedScene(
  inputScene: Scene,
  options: BuildPackedSceneOptions = {},
): PackedSceneData {
  // Capability filter (warn + skip). Consume pt-webgpu's OWN declared support
  // sets to drop unsupported primitive kinds / analytic shapes / emitter kinds,
  // replacing the hand-rolled boolean-chain skip+warn that previously lived
  // here. `partitionSceneBySupport` is pure; `scene` below is the supported
  // subset and `warnings` carries one message per dropped node.
  const { supported: scene, warnings } = partitionSceneBySupport(inputScene, PT_WEBGPU_SUPPORT);

  // Camera-visible emitters: `sceneFromThreeJS` converts an emissive mesh into a
  // `mesh-area` emitter and ZEROES the primitive's emissive (so the surface is
  // sampled via NEE, not double-counted). To make the emitter glow on the camera
  // ray + through refraction (the paths the analytic BSDF↔light connection cannot
  // reach), re-attach the emitter's radiance (`color · intensity`, EXACTLY the
  // value mesh-area NEE samples — emitterPacking.ts:173-176) onto the referenced
  // primitive's material `emissive`. The kernel's emissive-on-hit term is gated
  // to the non-MIS paths, so this does not double-count against NEE. Keyed by
  // `emitter.meshId` → primitive `id` (`emissiveMeshAreaEmitter` sets
  // `meshId = mesh.uuid` and `convertMesh` sets `id = mesh.uuid`).
  const emissiveByMeshId = new Map<string, { emissive: [number, number, number]; intensity: number }>();
  if (options.cameraVisibleEmitters === true) {
    for (const emitter of scene.emitters) {
      if (emitter.kind !== 'mesh-area') continue;
      emissiveByMeshId.set(emitter.meshId, {
        emissive: [emitter.color[0], emitter.color[1], emitter.color[2]],
        intensity: emitter.intensity,
      });
    }
  }
  const packMaterial = (primitive: { id: string; material: MaterialSpec }): number[] => {
    const reattach = emissiveByMeshId.get(primitive.id);
    // Pre-multiply intensity into emissive and set emissiveIntensity=1 so the
    // kernel decode (`emissive · emissiveIntensity`) reproduces `color · intensity`
    // exactly, matching the NEE radiance regardless of the primitive's own
    // (zeroed) emissiveIntensity.
    const mat = reattach == null
      ? primitive.material
      : { ...primitive.material, emissive: reattach.emissive.map((c) => c * reattach.intensity) as [number, number, number], emissiveIntensity: 1 };
    return materialToPackedVec4s(mat);
  };

  const materials: number[] = [];
  // Ordered MaterialSpec list, matId-aligned with `materials` (P2): drives the
  // texture-descriptor collection below. Reattachment only rewrites `emissive`,
  // which textures don't read, so the primitive's own material is sufficient.
  const materialSpecs: MaterialSpec[] = [];
  const meshMaterialIds = new Map<string, number>();
  const analyticHeaders: number[] = [];
  const analyticParams: number[] = [];
  const analyticLocalToWorld: number[] = [];
  const analyticWorldToLocal: number[] = [];

  let nextMaterialId = 0;

  for (const primitive of scene.primitives) {
    if (primitive.kind === 'analytic') {
      // Shape support was already vetted by the capability filter above; any
      // analytic primitive that reaches here has a known shape id (> 0).
      const shapeId = analyticShapeId(primitive.shape);
      const matId = nextMaterialId++;
      materials.push(...packMaterial(primitive));
      materialSpecs.push(primitive.material);
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

    if (!isMeshLikePrimitive(primitive)) {
      continue;
    }
    const matId = nextMaterialId++;
    meshMaterialIds.set(primitive.id, matId);
    materials.push(...packMaterial(primitive));
    materialSpecs.push(primitive.material);
  }

  const texCollection = collectMaterialTextures(materialSpecs);

  const geo = packSceneFromCore(scene, {
    tlas: true,
    resolveMaterialId: (id) => meshMaterialIds.get(id) ?? 0,
  });
  warnings.push(...geo.warnings);
  const emitArrays = packEmitterArrays(scene);
  const environment = environmentParams(scene);
  warnings.push(...environment.warnings);
  warnings.push(...emitArrays.warnings);

  // WS2 — build the power-weighted light tree over the selectable lights, in the
  // SAME order the kernel NEE walk visits them (directional · point · spot ·
  // rect · mesh · env). Only worthwhile with ≥ 2 lights: a 0/1-light tree adds
  // no variance reduction over the uniform pick (and `buildLightTree` requires a
  // non-empty input), so below 2 lights we ship an empty buffer + the uniform
  // fallback gate (`lightTreeEnabled = false`).
  const lightTreeInput = buildLightTreeInputForScene(scene);
  let lightTreeNodes = new Float32Array(0);
  let lightTreeNodeCount = 0;
  let lightTreeEnabled = false;
  if (lightTreeInput.powers.length >= 2) {
    const { nodes } = buildLightTree(lightTreeInput);
    lightTreeNodes = new Float32Array(packLightTreeForGPU(nodes));
    lightTreeNodeCount = nodes.length;
    lightTreeEnabled = true;
  }

  return {
    positions: geo.positions,
    normals: geo.normals,
    uvs: geo.uvs,
    indices: geo.indices,
    triMaterialIds: geo.triMaterialIds,
    materials: new Float32Array(materials),
    materialTexDescriptors: texCollection.descriptors,
    materialTextureSources: texCollection.sources,
    materialTextureLinearSources: texCollection.linearSources,
    bvhNodes: geo.bvhNodes,
    tlasNodes: geo.tlasNodes,
    tlasInstanceIndices: geo.tlasInstanceIndices,
    tlasBlasRoots: geo.tlasBlasRoots,
    tlasInstanceWorldToLocal: geo.tlasInstanceWorldToLocal,
    tlasInstanceLocalToWorld: geo.tlasInstanceLocalToWorld,
    primitiveTlasBindings: geo.primitiveTlasBindings,
    analyticHeaders: new Float32Array(analyticHeaders),
    analyticParams: new Float32Array(analyticParams),
    analyticLocalToWorld: new Float32Array(analyticLocalToWorld),
    analyticWorldToLocal: new Float32Array(analyticWorldToLocal),
    triangleCount: geo.triangleCount,
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
    lightTreeNodes,
    lightTreeNodeCount,
    lightTreeEnabled,
  };
}

/** Snapshot geometry + TLAS from a full pack for {@link rebuildPrimitiveBlas} fast paths. */
export function scenePackResultFromPacked(packed: PackedSceneData): ScenePackResult {
  return {
    positions: packed.positions,
    normals: packed.normals,
    uvs: packed.uvs,
    indices: packed.indices,
    triMaterialIds: packed.triMaterialIds,
    bvhNodes: packed.bvhNodes,
    triangleCount: packed.triangleCount,
    tlasNodes: packed.tlasNodes,
    tlasInstanceIndices: packed.tlasInstanceIndices,
    tlasBlasRoots: packed.tlasBlasRoots,
    tlasInstanceWorldToLocal: packed.tlasInstanceWorldToLocal,
    tlasInstanceLocalToWorld: packed.tlasInstanceLocalToWorld,
    tlasNodeCount: Math.floor(packed.tlasNodes.length / 8),
    primitiveTlasBindings: packed.primitiveTlasBindings,
    warnings: packed.warnings,
  };
}

/** In-place GPU + CPU mirror update after BLAS splice / refit (WG-6). */
export function uploadScenePackGeometry(
  device: GPUDevice,
  sb: UploadedSceneBuffers,
  pack: ScenePackResult,
): void {
  const write = (buffer: GPUBuffer, data: ArrayBufferView): void => {
    if (data.byteLength > 0) {
      device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    }
  };
  write(sb.positionsBuffer, pack.positions);
  write(sb.normalsBuffer, pack.normals);
  write(sb.uvsBuffer, pack.uvs);
  write(sb.indicesBuffer, pack.indices);
  write(sb.triMaterialIdsBuffer, pack.triMaterialIds);
  write(sb.bvhNodesBuffer, pack.bvhNodes);
  write(sb.tlasNodesBuffer, pack.tlasNodes);
  write(sb.tlasInstanceIndicesBuffer, pack.tlasInstanceIndices);
  write(sb.tlasBlasRootsBuffer, pack.tlasBlasRoots);
  write(sb.tlasInstanceWorldToLocalBuffer, pack.tlasInstanceWorldToLocal);
  write(sb.tlasInstanceLocalToWorldBuffer, pack.tlasInstanceLocalToWorld);
  sb.positions.set(pack.positions);
  sb.normals.set(pack.normals);
  sb.uvs.set(pack.uvs);
  sb.indices.set(pack.indices);
  sb.triMaterialIds.set(pack.triMaterialIds);
  sb.bvhNodes.set(pack.bvhNodes);
  sb.tlasNodes.set(pack.tlasNodes);
  sb.tlasInstanceIndices.set(pack.tlasInstanceIndices);
  sb.tlasBlasRoots.set(pack.tlasBlasRoots);
  sb.tlasInstanceWorldToLocal.set(pack.tlasInstanceWorldToLocal);
  sb.tlasInstanceLocalToWorld.set(pack.tlasInstanceLocalToWorld);
  const mutable = asMutableSceneBuffers(sb);
  mutable.tlasNodeCount = pack.tlasNodeCount;
  mutable.bvhNodeCount = Math.floor(pack.bvhNodes.length / 8);
  mutable.triangleCount = pack.triangleCount;
  mutable.primitiveTlasBindings = pack.primitiveTlasBindings;
}

/** C2 — upload BLAS concat buffers only (TLAS instance data unchanged). */
export function uploadScenePackBlasOnly(
  device: GPUDevice,
  sb: UploadedSceneBuffers,
  pack: Pick<
    ScenePackResult,
    | 'positions'
    | 'normals'
    | 'uvs'
    | 'indices'
    | 'triMaterialIds'
    | 'bvhNodes'
    | 'triangleCount'
    | 'primitiveTlasBindings'
  >,
): void {
  const write = (buffer: GPUBuffer, data: ArrayBufferView): void => {
    if (data.byteLength > 0) {
      device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    }
  };
  write(sb.positionsBuffer, pack.positions);
  write(sb.normalsBuffer, pack.normals);
  write(sb.uvsBuffer, pack.uvs);
  write(sb.indicesBuffer, pack.indices);
  write(sb.triMaterialIdsBuffer, pack.triMaterialIds);
  write(sb.bvhNodesBuffer, pack.bvhNodes);
  sb.positions.set(pack.positions);
  sb.normals.set(pack.normals);
  sb.uvs.set(pack.uvs);
  sb.indices.set(pack.indices);
  sb.triMaterialIds.set(pack.triMaterialIds);
  sb.bvhNodes.set(pack.bvhNodes);
  const mutable = asMutableSceneBuffers(sb);
  mutable.bvhNodeCount = Math.floor(pack.bvhNodes.length / 8);
  mutable.triangleCount = pack.triangleCount;
  mutable.primitiveTlasBindings = pack.primitiveTlasBindings;
}

/**
 * Slice-1 — reallocate ONLY the (5) TLAS GPU buffers after an instanced-mesh
 * instance-COUNT change, leaving every BLAS buffer untouched.
 *
 * The transform-only fast path ({@link uploadScenePackTlasOnly}) requires the
 * new TLAS arrays to be byte-length-identical to the live buffers (in-place
 * `writeBuffer`). An instance-count change grows/shrinks those arrays, so the
 * five TLAS buffers must be destroyed and recreated at the new size. The BLAS
 * buffers (positions/normals/indices/triMaterialIds/bvhNodes) are byte-identical
 * across an instance-count change (shared geometry) and are NOT touched here.
 *
 * After swapping the new buffer handles onto {@link UploadedSceneBuffers}, the
 * caller MUST invalidate any cached bind groups so the next frame rebinds the
 * fresh TLAS buffers (`gpuResources.ts` reads `sb.tlas*Buffer` at bind-group
 * build time).
 */
export function uploadScenePackTlasRealloc(
  device: GPUDevice,
  sb: UploadedSceneBuffers,
  pack: Pick<
    ScenePackResult,
    | 'tlasNodes'
    | 'tlasInstanceIndices'
    | 'tlasBlasRoots'
    | 'tlasInstanceWorldToLocal'
    | 'tlasInstanceLocalToWorld'
    | 'tlasNodeCount'
    | 'primitiveTlasBindings'
  >,
): void {
  // Destroy the stale TLAS buffers (BLAS buffers untouched).
  sb.tlasNodesBuffer.destroy();
  sb.tlasInstanceIndicesBuffer.destroy();
  sb.tlasBlasRootsBuffer.destroy();
  sb.tlasInstanceWorldToLocalBuffer.destroy();
  sb.tlasInstanceLocalToWorldBuffer.destroy();

  const tlasNodesBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.tlasNodes', pack.tlasNodes);
  const tlasInstanceIndicesBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.tlasInstanceIndices',
    pack.tlasInstanceIndices,
  );
  const tlasBlasRootsBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.tlasBlasRoots',
    pack.tlasBlasRoots,
  );
  const tlasInstanceWorldToLocalBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.tlasInstanceWorldToLocal',
    pack.tlasInstanceWorldToLocal,
  );
  const tlasInstanceLocalToWorldBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.tlasInstanceLocalToWorld',
    pack.tlasInstanceLocalToWorld,
  );

  // Swap the new handles + CPU mirrors onto the (otherwise readonly) struct. The
  // `destroy` closure built in `uploadPackedScene` resolves these TLAS handles
  // off the struct at teardown-time, so no closure rewire is needed here.
  const buffers = asMutableSceneBufferBuffers(sb);
  buffers.tlasNodesBuffer = tlasNodesBuffer;
  buffers.tlasInstanceIndicesBuffer = tlasInstanceIndicesBuffer;
  buffers.tlasBlasRootsBuffer = tlasBlasRootsBuffer;
  buffers.tlasInstanceWorldToLocalBuffer = tlasInstanceWorldToLocalBuffer;
  buffers.tlasInstanceLocalToWorldBuffer = tlasInstanceLocalToWorldBuffer;
  buffers.tlasNodes = new Uint32Array(pack.tlasNodes);
  buffers.tlasInstanceIndices = new Uint32Array(pack.tlasInstanceIndices);
  buffers.tlasBlasRoots = new Uint32Array(pack.tlasBlasRoots);
  buffers.tlasInstanceWorldToLocal = new Float32Array(pack.tlasInstanceWorldToLocal);
  buffers.tlasInstanceLocalToWorld = new Float32Array(pack.tlasInstanceLocalToWorld);

  const mutable = asMutableSceneBuffers(sb);
  mutable.tlasNodeCount = pack.tlasNodeCount;
  mutable.primitiveTlasBindings = pack.primitiveTlasBindings;
}

/**
 * Slice-2 — reallocate the BLAS concat buffers AND the TLAS buffers after a mesh
 * vertex/index-COUNT change spliced one primitive's BLAS to a new size (growing
 * or shrinking the concat arrays + rebasing downstream offsets, see
 * `rebuildPrimitiveBlas` → `spliceResizedPrimitiveBlasIntoPack`). Unlike
 * {@link uploadScenePackGeometry} (in-place `writeBuffer`, same byte lengths),
 * the resized concat arrays no longer fit the live buffers, so the five BLAS
 * buffers + the five TLAS buffers are destroyed and recreated at the new size.
 *
 * As with {@link uploadScenePackTlasRealloc}, the swapped handles are resolved
 * off the struct at teardown-time by the `destroy` closure built in
 * {@link uploadPackedScene}, so no closure rewire is needed. The caller MUST
 * invalidate any cached bind groups so the next frame rebinds the fresh buffers.
 */
export function uploadScenePackGeometryRealloc(
  device: GPUDevice,
  sb: UploadedSceneBuffers,
  pack: ScenePackResult,
): void {
  // Destroy the stale BLAS + TLAS buffers (everything geometry-sized).
  sb.positionsBuffer.destroy();
  sb.normalsBuffer.destroy();
  sb.uvsBuffer.destroy();
  sb.indicesBuffer.destroy();
  sb.triMaterialIdsBuffer.destroy();
  sb.bvhNodesBuffer.destroy();
  sb.tlasNodesBuffer.destroy();
  sb.tlasInstanceIndicesBuffer.destroy();
  sb.tlasBlasRootsBuffer.destroy();
  sb.tlasInstanceWorldToLocalBuffer.destroy();
  sb.tlasInstanceLocalToWorldBuffer.destroy();

  const blas = asMutableSceneBufferBlasHandles(sb);
  blas.positionsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.positions', pack.positions);
  blas.normalsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.normals', pack.normals);
  blas.uvsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.uvs', pack.uvs);
  blas.indicesBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.indices', pack.indices);
  blas.triMaterialIdsBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.triMaterialIds',
    pack.triMaterialIds,
  );
  blas.bvhNodesBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.bvhNodes', pack.bvhNodes);
  blas.positions = new Float32Array(pack.positions);
  blas.normals = new Float32Array(pack.normals);
  blas.uvs = new Float32Array(pack.uvs);
  blas.indices = new Uint32Array(pack.indices);
  blas.triMaterialIds = new Uint32Array(pack.triMaterialIds);
  blas.bvhNodes = new Float32Array(pack.bvhNodes);

  const tlas = asMutableSceneBufferBuffers(sb);
  tlas.tlasNodesBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.tlasNodes', pack.tlasNodes);
  tlas.tlasInstanceIndicesBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.tlasInstanceIndices',
    pack.tlasInstanceIndices,
  );
  tlas.tlasBlasRootsBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.tlasBlasRoots',
    pack.tlasBlasRoots,
  );
  tlas.tlasInstanceWorldToLocalBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.tlasInstanceWorldToLocal',
    pack.tlasInstanceWorldToLocal,
  );
  tlas.tlasInstanceLocalToWorldBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.tlasInstanceLocalToWorld',
    pack.tlasInstanceLocalToWorld,
  );
  tlas.tlasNodes = new Uint32Array(pack.tlasNodes);
  tlas.tlasInstanceIndices = new Uint32Array(pack.tlasInstanceIndices);
  tlas.tlasBlasRoots = new Uint32Array(pack.tlasBlasRoots);
  tlas.tlasInstanceWorldToLocal = new Float32Array(pack.tlasInstanceWorldToLocal);
  tlas.tlasInstanceLocalToWorld = new Float32Array(pack.tlasInstanceLocalToWorld);

  const mutable = asMutableSceneBuffers(sb);
  mutable.bvhNodeCount = Math.floor(pack.bvhNodes.length / 8);
  mutable.triangleCount = pack.triangleCount;
  mutable.tlasNodeCount = pack.tlasNodeCount;
  mutable.primitiveTlasBindings = pack.primitiveTlasBindings;
}

/** C2 — upload TLAS SSBOs only (transform-only refit; BLAS buffers unchanged). */
export function uploadScenePackTlasOnly(
  device: GPUDevice,
  sb: UploadedSceneBuffers,
  pack: Pick<
    ScenePackResult,
    | 'tlasNodes'
    | 'tlasInstanceIndices'
    | 'tlasBlasRoots'
    | 'tlasInstanceWorldToLocal'
    | 'tlasInstanceLocalToWorld'
    | 'tlasNodeCount'
    | 'primitiveTlasBindings'
  >,
): void {
  const write = (buffer: GPUBuffer, data: ArrayBufferView): void => {
    if (data.byteLength > 0) {
      device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    }
  };
  write(sb.tlasNodesBuffer, pack.tlasNodes);
  write(sb.tlasInstanceIndicesBuffer, pack.tlasInstanceIndices);
  write(sb.tlasBlasRootsBuffer, pack.tlasBlasRoots);
  write(sb.tlasInstanceWorldToLocalBuffer, pack.tlasInstanceWorldToLocal);
  write(sb.tlasInstanceLocalToWorldBuffer, pack.tlasInstanceLocalToWorld);
  sb.tlasNodes.set(pack.tlasNodes);
  sb.tlasInstanceIndices.set(pack.tlasInstanceIndices);
  sb.tlasBlasRoots.set(pack.tlasBlasRoots);
  sb.tlasInstanceWorldToLocal.set(pack.tlasInstanceWorldToLocal);
  sb.tlasInstanceLocalToWorld.set(pack.tlasInstanceLocalToWorld);
  const mutable = asMutableSceneBuffers(sb);
  mutable.tlasNodeCount = pack.tlasNodeCount;
  mutable.primitiveTlasBindings = pack.primitiveTlasBindings;
}

/**
 * Typed mutators for the (otherwise `readonly`) derived count / lighting /
 * environment fields on {@link UploadedSceneBuffers}. The CPU-side mirror of an
 * `UploadedSceneBuffers` is logically immutable except for these fields, which
 * the incremental `updateEmitter` / `updateEnvironment` fast paths rewrite in
 * place after a partial GPU upload. Centralizing the structural
 * `as unknown as { … }` casts here (instead of scattering them across `index.ts`
 * and the geometry/BLAS/TLAS uploaders) keeps the unsafe surface in one typed
 * place. Behavior-preserving.
 */
interface MutableSceneBufferFields {
  // Geometry-pack derived counts (BLAS / TLAS uploads).
  bvhNodeCount: number;
  tlasNodeCount: number;
  triangleCount: number;
  primitiveTlasBindings: readonly PrimitiveTlasBinding[];
  // Emitter counts + directional aggregate (updateEmitter fast path).
  pointLightCount: number;
  spotLightCount: number;
  rectAreaLightCount: number;
  meshAreaLightCount: number;
  directionalLight: readonly [number, number, number];
  directionalIrradiance: readonly [number, number, number];
  // Environment fields (updateEnvironment fast path).
  environmentTint: readonly [number, number, number];
  environmentSunDirection: readonly [number, number, number];
  environmentSunStrength: number;
  environmentMapWidth: number;
  environmentMapHeight: number;
  hasEnvironmentMap: boolean;
  // WS2 — light-tree state + buffer handle / CPU mirror (rebuilt on emitter /
  // environment incremental patches; the buffer reallocates if the node count
  // — hence its byte length — changes).
  lightTreeNodeCount: number;
  lightTreeEnabled: boolean;
  lightTreeNodes: Float32Array;
  lightTreeBuffer: GPUBuffer;
}

/** Single typed view onto the mutable subset of an UploadedSceneBuffers. */
function asMutableSceneBuffers(sb: UploadedSceneBuffers): MutableSceneBufferFields {
  return sb as unknown as MutableSceneBufferFields;
}

/**
 * The five TLAS GPU buffer handles + their CPU-mirror typed arrays, normally
 * `readonly`, are reassigned in place by {@link uploadScenePackTlasRealloc} when
 * an instanced-mesh instance count changes (the buffers must be reallocated at
 * the new size). This single typed view localizes that one unsafe write site.
 */
interface MutableTlasBufferHandles {
  tlasNodesBuffer: GPUBuffer;
  tlasInstanceIndicesBuffer: GPUBuffer;
  tlasBlasRootsBuffer: GPUBuffer;
  tlasInstanceWorldToLocalBuffer: GPUBuffer;
  tlasInstanceLocalToWorldBuffer: GPUBuffer;
  tlasNodes: Uint32Array;
  tlasInstanceIndices: Uint32Array;
  tlasBlasRoots: Uint32Array;
  tlasInstanceWorldToLocal: Float32Array;
  tlasInstanceLocalToWorld: Float32Array;
}

function asMutableSceneBufferBuffers(sb: UploadedSceneBuffers): MutableTlasBufferHandles {
  return sb as unknown as MutableTlasBufferHandles;
}

/**
 * Light-buffer handles + CPU mirrors are reassigned by the incremental
 * updateEmitter path when dynamic emitter expansion changes byte lengths.
 */
interface MutableEmitterBufferHandles {
  pointLightsBuffer: GPUBuffer;
  spotLightsBuffer: GPUBuffer;
  rectAreaLightsBuffer: GPUBuffer;
  meshAreaLightsBuffer: GPUBuffer;
  pointLightsData: Float32Array;
  spotLightsData: Float32Array;
  rectAreaLightsData: Float32Array;
  meshAreaLightsData: Float32Array;
}

function asMutableSceneBufferEmitterHandles(sb: UploadedSceneBuffers): MutableEmitterBufferHandles {
  return sb as unknown as MutableEmitterBufferHandles;
}

/**
 * The five BLAS GPU buffer handles + their CPU-mirror typed arrays, normally
 * `readonly`, are reassigned in place by {@link uploadScenePackGeometryRealloc}
 * when a mesh vertex/index count changes (the concat buffers grow/shrink, so the
 * buffers must be reallocated at the new size). This single typed view localizes
 * that one unsafe write site, mirroring {@link MutableTlasBufferHandles}.
 */
interface MutableBlasBufferHandles {
  positionsBuffer: GPUBuffer;
  normalsBuffer: GPUBuffer;
  uvsBuffer: GPUBuffer;
  indicesBuffer: GPUBuffer;
  triMaterialIdsBuffer: GPUBuffer;
  bvhNodesBuffer: GPUBuffer;
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  indices: Uint32Array;
  triMaterialIds: Uint32Array;
  bvhNodes: Float32Array;
}

function asMutableSceneBufferBlasHandles(sb: UploadedSceneBuffers): MutableBlasBufferHandles {
  return sb as unknown as MutableBlasBufferHandles;
}

/** Rewrite emitter counts + directional aggregate after an in-place light upload. */
export function applyEmitterCountMutation(
  sb: UploadedSceneBuffers,
  next: {
    readonly pointLightCount: number;
    readonly spotLightCount: number;
    readonly rectAreaLightCount: number;
    readonly meshAreaLightCount: number;
    readonly directionalLight: readonly [number, number, number];
    readonly directionalIrradiance: readonly [number, number, number];
  },
): void {
  const mutable = asMutableSceneBuffers(sb);
  mutable.pointLightCount = next.pointLightCount;
  mutable.spotLightCount = next.spotLightCount;
  mutable.rectAreaLightCount = next.rectAreaLightCount;
  mutable.meshAreaLightCount = next.meshAreaLightCount;
  mutable.directionalLight = next.directionalLight;
  mutable.directionalIrradiance = next.directionalIrradiance;
}

function storageBufferMinByteLength(data: ArrayBufferView): number {
  return data.byteLength === 0 ? 16 : Math.ceil(data.byteLength / 4) * 4;
}

function uploadOrReallocateEmitterBuffer(
  device: GPUDevice,
  currentBuffer: GPUBuffer,
  currentData: Float32Array,
  nextData: Float32Array,
  label: string,
  assign: (buffer: GPUBuffer, data: Float32Array) => void,
): boolean {
  if (storageBufferMinByteLength(nextData) !== storageBufferMinByteLength(currentData)) {
    const nextBuffer = createStorageBuffer(device, label, nextData);
    currentBuffer.destroy();
    assign(nextBuffer, new Float32Array(nextData));
    return true;
  }
  if (nextData.byteLength > 0) {
    device.queue.writeBuffer(
      currentBuffer,
      0,
      nextData.buffer,
      nextData.byteOffset,
      nextData.byteLength,
    );
  }
  currentData.set(nextData);
  return false;
}

export function uploadEmitterArrays(
  device: GPUDevice,
  sb: UploadedSceneBuffers,
  packed: PackedEmitterArrays,
  nextDirectional: {
    readonly directionalLight: readonly [number, number, number];
    readonly directionalIrradiance: readonly [number, number, number];
  },
): boolean {
  const handles = asMutableSceneBufferEmitterHandles(sb);
  let reallocated = false;
  reallocated = uploadOrReallocateEmitterBuffer(
    device,
    sb.pointLightsBuffer,
    sb.pointLightsData,
    packed.pointLightsData,
    'vitrum.pt-webgpu.scene.pointLights',
    (buffer, data) => {
      handles.pointLightsBuffer = buffer;
      handles.pointLightsData = data;
    },
  ) || reallocated;
  reallocated = uploadOrReallocateEmitterBuffer(
    device,
    sb.spotLightsBuffer,
    sb.spotLightsData,
    packed.spotLightsData,
    'vitrum.pt-webgpu.scene.spotLights',
    (buffer, data) => {
      handles.spotLightsBuffer = buffer;
      handles.spotLightsData = data;
    },
  ) || reallocated;
  reallocated = uploadOrReallocateEmitterBuffer(
    device,
    sb.rectAreaLightsBuffer,
    sb.rectAreaLightsData,
    packed.rectAreaLightsData,
    'vitrum.pt-webgpu.scene.rectAreaLights',
    (buffer, data) => {
      handles.rectAreaLightsBuffer = buffer;
      handles.rectAreaLightsData = data;
    },
  ) || reallocated;
  reallocated = uploadOrReallocateEmitterBuffer(
    device,
    sb.meshAreaLightsBuffer,
    sb.meshAreaLightsData,
    packed.meshAreaLightsData,
    'vitrum.pt-webgpu.scene.meshAreaLights',
    (buffer, data) => {
      handles.meshAreaLightsBuffer = buffer;
      handles.meshAreaLightsData = data;
    },
  ) || reallocated;
  applyEmitterCountMutation(sb, {
    pointLightCount: packed.pointLightCount,
    spotLightCount: packed.spotLightCount,
    rectAreaLightCount: packed.rectAreaLightCount,
    meshAreaLightCount: packed.meshAreaLightCount,
    directionalLight: nextDirectional.directionalLight,
    directionalIrradiance: nextDirectional.directionalIrradiance,
  });
  return reallocated;
}

/**
 * WS2 — rebuild the power-weighted light tree from `scene` and re-upload it after
 * an incremental emitter / environment patch (which change light powers /
 * positions / the env gate but do NOT trigger a full repack).
 *
 * Returns `true` if the light-tree GPU BUFFER was REALLOCATED (node count — hence
 * byte length — changed), which means the caller MUST invalidate cached bind
 * groups so the fresh buffer is rebound. An in-place `writeBuffer` (same size)
 * returns `false`. The `lightTreeEnabled` / `lightTreeNodeCount` mirror fields are
 * always updated so the next `#buildParamsBuffer` reflects the new tree.
 *
 * Mirrors the `uploadScenePack*Realloc` pattern: the `lightTreeBuffer` handle is
 * read off the struct at destroy-time, so swapping a fresh handle here needs no
 * `destroy`-closure rewire.
 */
export function rebuildLightTreeForScene(
  device: GPUDevice,
  sb: UploadedSceneBuffers,
  scene: Scene,
): boolean {
  const input = buildLightTreeInputForScene(scene);
  let nodes = new Float32Array(0);
  let nodeCount = 0;
  let enabled = false;
  if (input.powers.length >= 2) {
    const built = buildLightTree(input);
    nodes = new Float32Array(packLightTreeForGPU(built.nodes));
    nodeCount = built.nodes.length;
    enabled = true;
  }
  const mutable = asMutableSceneBuffers(sb);
  mutable.lightTreeNodeCount = nodeCount;
  mutable.lightTreeEnabled = enabled;

  const prevByteLen = sb.lightTreeNodes.byteLength;
  // `createStorageBuffer` rounds empty arrays up to a 16-byte placeholder, so the
  // live buffer's minimum size is 16; compare against the same flooring.
  const nextMinBytes = storageBufferMinByteLength(nodes);
  const liveMinBytes = prevByteLen === 0 ? 16 : Math.ceil(prevByteLen / 4) * 4;
  mutable.lightTreeNodes = nodes;
  if (nextMinBytes !== liveMinBytes) {
    // Size changed — reallocate.
    sb.lightTreeBuffer.destroy();
    mutable.lightTreeBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.lightTree', nodes);
    return true;
  }
  // Same size — in-place write when there is node data. When the tree is (and
  // was already) disabled the live buffer is an untouched zeroed placeholder, so
  // skip a redundant write; the gate (`lightTreeEnabled = false`) keeps the
  // shader off it regardless.
  if (nodes.byteLength > 0) {
    device.queue.writeBuffer(sb.lightTreeBuffer, 0, nodes.buffer, nodes.byteOffset, nodes.byteLength);
  } else if (prevByteLen > 0) {
    // enabled → disabled at the same (16-byte placeholder) size is impossible
    // here (prevByteLen>0 ⇒ liveMinBytes>16 ⇒ size changed ⇒ realloc above), so
    // this branch is unreachable; left as a defensive zero.
    device.queue.writeBuffer(sb.lightTreeBuffer, 0, new Uint32Array([0, 0, 0, 0]));
  }
  return false;
}

/** Rewrite environment fields after an in-place HDRI/sky upload. */
export function applyEnvironmentMutation(
  sb: UploadedSceneBuffers,
  next: {
    readonly environmentTint: readonly [number, number, number];
    readonly environmentSunDirection: readonly [number, number, number];
    readonly environmentSunStrength: number;
    readonly environmentMapWidth: number;
    readonly environmentMapHeight: number;
    readonly hasEnvironmentMap: boolean;
  },
): void {
  const mutable = asMutableSceneBuffers(sb);
  mutable.environmentTint = next.environmentTint;
  mutable.environmentSunDirection = next.environmentSunDirection;
  mutable.environmentSunStrength = next.environmentSunStrength;
  mutable.environmentMapWidth = next.environmentMapWidth;
  mutable.environmentMapHeight = next.environmentMapHeight;
  mutable.hasEnvironmentMap = next.hasEnvironmentMap;
}

export function rebuildTlasForSceneTransforms(
  scene: Scene,
  primitiveTlasBindings: readonly PrimitiveTlasBinding[],
  prevTlas?: {
    readonly tlasNodes: Uint32Array;
    readonly tlasInstanceIndices: Uint32Array;
    readonly tlasBlasRoots: Uint32Array;
    readonly tlasInstanceWorldToLocal: Float32Array;
  },
) {
  return refitTlasTransforms(scene, primitiveTlasBindings, prevTlas);
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
  const lightTreeBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.lightTree', packed.lightTreeNodes);
  // P2 — per-vertex UVs + per-material texture descriptors + the baseColor
  // texture_2d_array (all group 3, full tier). A textureless scene gets a 1×1
  // white dummy so the binding is always satisfied; descriptors all hold -1 so
  // the kernel never samples it (textureless render stays byte-identical).
  const uvsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.uvs', packed.uvs);
  const materialTexDescriptorsBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.materialTexDescriptors',
    packed.materialTexDescriptors,
  );
  const materialTextureArray = createMaterialTextureArray(device, packed.materialTextureSources);
  // Linear array (normal + ORM) — rgba8unorm so the sampler does NOT sRGB-decode.
  const materialLinearArray = createMaterialTextureArray(
    device,
    packed.materialTextureLinearSources,
    'rgba8unorm',
  );
  // Surface texture-array warnings (heterogeneous source sizes → wrong UVs, or an
  // unusable image) instead of dropping them — these were silently discarded.
  for (const w of materialTextureArray.warnings) {
    console.warn(`[vitrum/pt-webgpu] material texture array (sRGB): ${w}`);
  }
  for (const w of materialLinearArray.warnings) {
    console.warn(`[vitrum/pt-webgpu] material texture array (linear): ${w}`);
  }
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

  const uploaded: UploadedSceneBuffers = {
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
    lightTreeBuffer,
    tlasNodesBuffer,
    tlasInstanceIndicesBuffer,
    tlasBlasRootsBuffer,
    tlasInstanceWorldToLocalBuffer,
    tlasInstanceLocalToWorldBuffer,
    uvsBuffer,
    materialTexDescriptorsBuffer,
    materialTexture: materialTextureArray.texture,
    materialTextureView: materialTextureArray.view,
    materialTextureSampler: materialTextureArray.sampler,
    materialLinearTexture: materialLinearArray.texture,
    materialLinearTextureView: materialLinearArray.view,
    // BLAS + TLAS buffers are resolved off `uploaded` at destroy-time (not the
    // captured locals), because the realloc fast paths swap fresh handles onto
    // the struct: {@link uploadScenePackTlasRealloc} (instance-count change) and
    // {@link uploadScenePackGeometryRealloc} (mesh vertex/index-count change).
    // Reading them late keeps `destroy` free of stale handles (no double-free /
    // leak) without a closure rewire on every realloc. The non-resized buffers
    // (materials / analytic / environment) stay captured.
    destroy: () => {
      uploaded.positionsBuffer.destroy();
      uploaded.normalsBuffer.destroy();
      uploaded.indicesBuffer.destroy();
      uploaded.triMaterialIdsBuffer.destroy();
      materialsBuffer.destroy();
      uploaded.bvhNodesBuffer.destroy();
      analyticHeadersBuffer.destroy();
      analyticParamsBuffer.destroy();
      analyticLocalToWorldBuffer.destroy();
      analyticWorldToLocalBuffer.destroy();
      environmentMapTexelsBuffer.destroy();
      environmentMapCdfBuffer.destroy();
      uploaded.pointLightsBuffer.destroy();
      uploaded.spotLightsBuffer.destroy();
      uploaded.rectAreaLightsBuffer.destroy();
      uploaded.meshAreaLightsBuffer.destroy();
      // Light-tree buffer is realloc-swapped by rebuildLightTreeForScene when
      // the node count changes — resolve it late off `uploaded` like the
      // BLAS/TLAS handles, or the swapped-in buffer leaks (and the original
      // gets a benign double-destroy).
      uploaded.lightTreeBuffer.destroy();
      uploaded.tlasNodesBuffer.destroy();
      uploaded.tlasInstanceIndicesBuffer.destroy();
      uploaded.tlasBlasRootsBuffer.destroy();
      uploaded.tlasInstanceWorldToLocalBuffer.destroy();
      uploaded.tlasInstanceLocalToWorldBuffer.destroy();
      // P2 — uvsBuffer is realloc-swapped on a vertex-count change, so resolve it
      // late off `uploaded`. The descriptor buffer + texture array are
      // material-indexed (never resized by a geometry realloc) → captured locals.
      uploaded.uvsBuffer.destroy();
      materialTexDescriptorsBuffer.destroy();
      materialTextureArray.texture.destroy();
      materialLinearArray.texture.destroy();
    },
    // Sum the CURRENT GPUBuffer sizes off `uploaded` (realloc-swapped handles
    // included) + the two material texture arrays (GPUTexture has no `.size`, so
    // derive w·h·layers·4 at rgba8 = 4 B/texel, keyed by the actual format).
    // Keeps `debug.estimatedGpuMemoryBytes` honest instead of under-reporting by
    // the whole scene.
    gpuMemoryBytes: () => {
      const buffers: readonly GPUBuffer[] = [
        uploaded.positionsBuffer, uploaded.normalsBuffer, uploaded.indicesBuffer,
        uploaded.triMaterialIdsBuffer, uploaded.materialsBuffer, uploaded.bvhNodesBuffer,
        uploaded.analyticHeadersBuffer, uploaded.analyticParamsBuffer,
        uploaded.analyticLocalToWorldBuffer, uploaded.analyticWorldToLocalBuffer,
        uploaded.environmentMapTexelsBuffer, uploaded.environmentMapCdfBuffer,
        uploaded.pointLightsBuffer, uploaded.spotLightsBuffer, uploaded.rectAreaLightsBuffer,
        uploaded.meshAreaLightsBuffer, uploaded.lightTreeBuffer,
        uploaded.tlasNodesBuffer, uploaded.tlasInstanceIndicesBuffer, uploaded.tlasBlasRootsBuffer,
        uploaded.tlasInstanceWorldToLocalBuffer, uploaded.tlasInstanceLocalToWorldBuffer,
        uploaded.uvsBuffer, uploaded.materialTexDescriptorsBuffer,
      ];
      let bufferBytes = 0;
      for (const b of buffers) bufferBytes += b.size;
      const textureBytesByFormat: Record<string, number> = {};
      const addTex = (t: GPUTexture): void => {
        const bytes = t.width * t.height * t.depthOrArrayLayers * 4;
        textureBytesByFormat[t.format] = (textureBytesByFormat[t.format] ?? 0) + bytes;
      };
      addTex(uploaded.materialTexture);
      addTex(uploaded.materialLinearTexture);
      return { bufferBytes, textureBytesByFormat };
    },
  };
  return uploaded;
}
