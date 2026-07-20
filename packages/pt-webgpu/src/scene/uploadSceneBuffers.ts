import {
  asMat4,
  partitionSceneBySupport,
  solveSkin,
  type AnalyticShape,
  type EngineWarning,
  type MaterialSpec,
  type Scene,
  type SceneEmitter,
  type ScenePrimitive,
  type SupportSets,
} from '@vitrum/core';
import {
  BVH_NODE_FLOATS,
  CWBVH_CHILD_META_WORDS,
  CWBVH_CHILD_NODE,
  buildCompressedWideBvhFromArrayBvh,
  mergeUv1FromCore,
  mergeWorldSpaceFromCore,
  packCwbvhBuildBoundsForWgsl,
  packSceneFromCore,
  refitTlasTransforms,
  type PrimitiveTlasBinding,
  type ScenePackResult,
} from '@vitrum/shared-bvh';
import { buildLightTree, packLightTreeForGPU } from '@vitrum/shared-samplers';
import { invertMat4 } from '../math/mat4.js';
import { MATERIAL_FLOAT_STRIDE, materialToPackedVec4s } from './materialPacking.js';
import {
  applyMaterialTextureUvFitScales,
  collectMaterialTextures,
  type MaterialTextureLayerInfo,
} from './materialTextures.js';
import {
  createMaterialTextureArray,
  type MaterialTextureArray,
  type MaterialTextureArrayWarning,
} from './materialTextureArray.js';
import { environmentParams } from './environmentPacking.js';
import {
  buildLightTreeInputForScene,
  defaultDirectionalAngularDiameter,
  defaultDirectionalIrradiance,
  defaultDirectionalLight,
  packEmitterArrays,
  type EnvSummaryForTree,
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
  /** vec4f packed authored/generated tangents (.xyz = tangent, .w = handedness);
   *  zero .w means absent and the shader falls back to UV-gradient derivation. */
  readonly tangents: Float32Array;
  /** vec4f packed vertex colors (.rgba); absent authored colors are 1,1,1,1. */
  readonly colors: Float32Array;
  /** Per-material texture descriptor floats (MATERIAL_TEX_FLOAT_STRIDE each):
   *  texture indices + alpha-mode + KHR UV transform. Indexed by matId. */
  readonly materialTexDescriptors: Float32Array;
  /** Dedup'd, upload-ordered sRGB texture handles (baseColor, emissive, and
   *  extension color-tint maps; layer i = sources[i]); the GPU upload turns
   *  these into a texture_2d_array. */
  readonly materialTextureSources: readonly unknown[];
  /** Provenance for each sRGB material texture layer, for structured upload warnings. */
  readonly materialTextureSourceInfos: readonly MaterialTextureLayerInfo[];
  /** Dedup'd, upload-ordered LINEAR texture handles (normal, ORM, scalar maps,
   *  height/coverage/radiance data) → a second texture_2d_array sampled without
   *  sRGB decode. */
  readonly materialTextureLinearSources: readonly unknown[];
  /** Provenance for each LINEAR material texture layer, for structured upload warnings. */
  readonly materialTextureLinearSourceInfos: readonly MaterialTextureLayerInfo[];
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
  /** CWBVH prototype forest: parent wide-node bounds, 6 f32 per node. */
  readonly cwbvhNodeBounds: Float32Array;
  /** CWBVH prototype forest: child bounds as packed u16 pairs (3 u32 / child). */
  readonly cwbvhChildBoundsPacked: Uint32Array;
  /** CWBVH prototype forest: child kind / wide-node-or-triangle offset / count. */
  readonly cwbvhChildMeta: Uint32Array;
  /** CWBVH prototype forest: number of live child slots per wide node. */
  readonly cwbvhChildCount: Uint32Array;
  /** TLAS instance BLAS roots remapped from binary BVH node roots to CWBVH roots. */
  readonly cwbvhTlasBlasRoots: Uint32Array;
  /** Sparse CPU mirror: binary-BLAS root index -> CWBVH wide-node root index. */
  readonly cwbvhBinaryRootToWideRoot: Uint32Array;
  readonly cwbvhNodeCount: number;
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
  readonly structuredWarnings: readonly EngineWarning[];
  readonly directionalLight: readonly [number, number, number];
  readonly directionalIrradiance: readonly [number, number, number];
  /** D3 — soft-sun angular diameter (radians); 0 = exact delta directional. */
  readonly directionalAngularDiameter: number;
  /** N-directional: total count of directional emitters with non-zero irradiance. */
  readonly directionalLightCount: number;
  /** N-directional: packed flat array, DIRECTIONAL_LIGHT_FLOAT_STRIDE (8) floats per entry. */
  readonly directionalLightsData: Float32Array;
  /** Current scene bounds center, used to place pseudo-distant BDPT emitters. */
  readonly sceneCenter: readonly [number, number, number];
  /** Current scene half-diagonal radius, used to scale pseudo-distant BDPT emitters. */
  readonly sceneRadius: number;
  readonly pointLightCount: number;
  readonly spotLightCount: number;
  readonly rectAreaLightCount: number;
  readonly meshAreaLightCount: number;
  readonly pointLightsData: Float32Array;
  readonly spotLightsData: Float32Array;
  readonly rectAreaLightsData: Float32Array;
  readonly meshAreaLightsData: Float32Array;
  readonly meshAreaLightSourceFactorsData: Float32Array;
  readonly environmentTint: readonly [number, number, number];
  readonly environmentSunDirection: readonly [number, number, number];
  readonly environmentSunStrength: number;
  /**
   * H14-E: HDRI radiance intensity multiplier — separate from `environmentSunStrength`
   * (which drives the procedural-sky sun gate, `environmentSun.w`).
   * Value = `scene.environment.intensity ?? 1` when a valid HDRI is present; 0 otherwise.
   * Uploaded to `params.environmentHdriIntensity` so the equirect lookup is NOT gated
   * by the procedural-sky sun-strength lane.
   */
  readonly environmentHdriIntensity: number;
  /**
   * H6: HDRI dome rotation around the world +Y axis, radians (default 0).
   * Packed into `params.environmentTint.w` (the previously-zero .w lane — no layout
   * change).  Zero means no rotation (identity). Non-HDRI environments always supply 0.
   */
  readonly environmentHdriRotationY: number;
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
  readonly cwbvhNodeBoundsBuffer: GPUBuffer;
  readonly cwbvhChildBoundsPackedBuffer: GPUBuffer;
  readonly cwbvhChildMetaBuffer: GPUBuffer;
  readonly cwbvhChildCountBuffer: GPUBuffer;
  readonly cwbvhTlasBlasRootsBuffer: GPUBuffer;
  readonly analyticHeadersBuffer: GPUBuffer;
  readonly analyticParamsBuffer: GPUBuffer;
  readonly analyticLocalToWorldBuffer: GPUBuffer;
  readonly analyticWorldToLocalBuffer: GPUBuffer;
  readonly environmentMapTexelsBuffer: GPUBuffer;
  readonly environmentMapCdfBuffer: GPUBuffer;
  /** N-directional: GPU storage buffer, group(1) binding(10). */
  readonly directionalLightsBuffer: GPUBuffer;
  readonly pointLightsBuffer: GPUBuffer;
  readonly spotLightsBuffer: GPUBuffer;
  readonly rectAreaLightsBuffer: GPUBuffer;
  readonly meshAreaLightsBuffer: GPUBuffer;
  readonly meshAreaLightSourceFactorsBuffer: GPUBuffer;
  readonly tlasNodesBuffer: GPUBuffer;
  readonly tlasInstanceIndicesBuffer: GPUBuffer;
  readonly tlasBlasRootsBuffer: GPUBuffer;
  readonly tlasInstanceWorldToLocalBuffer: GPUBuffer;
  readonly tlasInstanceLocalToWorldBuffer: GPUBuffer;
  /** WS2 — light-tree node storage buffer (group 3, full tier only). */
  readonly lightTreeBuffer: GPUBuffer;
  /** P2 — per-vertex UV storage buffer (group 3, full tier only). */
  readonly uvsBuffer: GPUBuffer;
  /** Per-vertex tangent.xyzw storage buffer (group 3, full tier only). */
  readonly tangentsBuffer: GPUBuffer;
  /** Per-vertex color.rgba storage buffer (group 3, full tier only). */
  readonly colorsBuffer: GPUBuffer;
  /** P2 — per-material texture descriptor storage buffer (group 3). */
  readonly materialTexDescriptorsBuffer: GPUBuffer;
  /** P2 — sampled sRGB texture_2d_array handle (baseColor/emissive; for dispose). */
  readonly materialTexture: GPUTexture;
  /** P2 — sRGB 2d-array view bound in group 3 (binding 3). */
  readonly materialTextureView: GPUTextureView;
  /** P2 — filtering sampler bound in group 3 (shared by both arrays). */
  readonly materialTextureSampler: GPUSampler;
  /** P2 — LINEAR texture_2d_array handle (normal/scalar data maps; for dispose). */
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

/**
 * D8.7 — One entry per GPU storage buffer created by {@link uploadPackedScene}.
 *
 * `key`         — the field name on `PackedSceneData` whose data populates the buffer.
 * `bufferField` — the corresponding GPUBuffer handle field on `UploadedSceneBuffers`.
 * `label`       — the label string passed to `createStorageBuffer` (and visible in GPU
 *                 debuggers as `vitrum.pt-webgpu.scene.<name>`).
 * `excludeFromMemorySum` — (optional) when `true` the buffer is NOT counted by
 *                 `gpuMemoryBytes`. This flag exists ONLY to preserve the exact
 *                 pre-registry-driven behavior: `meshAreaLightSourceFactorsBuffer`
 *                 was historically omitted from the `gpuMemoryBytes` buffer sum
 *                 (present in destroy + create, absent from the debug memory
 *                 estimate). The omission looks unintentional; it is kept here as
 *                 data (not silently changed) so the debug estimate is byte-stable.
 *
 * **T2-A single-source invariant** — `uploadPackedScene`'s create loop, the
 * `destroy` closure, and `gpuMemoryBytes` are ALL driven off this registry, so a
 * buffer added here appears in creation, teardown, and the memory estimate with
 * no further hand-listing.
 *
 * **Binding sync invariant** — render-consumed buffers in this registry must be
 * reflected in the bind-group layout declarations in `gpuResources.ts`:
 *   - Group 0 (bindings 0–13): `#makeGroup0LayoutEntries()` / `#buildSharedPipelineLayout()`
 *   - Group 1 (bindings 0–10): analytic + env + area-light buffers in
 *     `#buildSharedPipelineLayout()` → `bindGroupLayout1`
 *   - Group 2 (bindings 0–4): TLAS table in `#buildSharedPipelineLayout()` →
 *     `bindGroupLayout2`
 *   - Group 3 (bindings 0–10): light-tree + P2 textures/descriptors + SPPM in
 *     `#buildSharedPipelineLayout()` → `bindGroupLayout3`
 *   - `buildBindGroups` and `buildReservoirBindGroups` in `gpuResources.ts` bind
 *     the render-consumed subset. Prototype CWBVH buffers are uploaded here as
 *     opt-in traversal plumbing and are bound by the full-tier
 *     `bvhTraversal:'cwbvh-closest-experimental'` path.
 *
 * The TLAS entries must remain contiguous at the END so realloc/registry tests
 * can keep the TLAS table as one tail block.
 */
export const SCENE_BUFFER_REGISTRY = [
  // ── BLAS geometry ─────────────────────────────────────────────────────────
  { key: 'positions',          bufferField: 'positionsBuffer',          label: 'vitrum.pt-webgpu.scene.positions' },
  { key: 'normals',            bufferField: 'normalsBuffer',            label: 'vitrum.pt-webgpu.scene.normals' },
  { key: 'indices',            bufferField: 'indicesBuffer',            label: 'vitrum.pt-webgpu.scene.indices' },
  { key: 'triMaterialIds',     bufferField: 'triMaterialIdsBuffer',     label: 'vitrum.pt-webgpu.scene.triMaterialIds' },
  { key: 'materials',          bufferField: 'materialsBuffer',          label: 'vitrum.pt-webgpu.scene.materials' },
  { key: 'bvhNodes',           bufferField: 'bvhNodesBuffer',           label: 'vitrum.pt-webgpu.scene.bvhNodes' },
  { key: 'cwbvhNodeBounds',         bufferField: 'cwbvhNodeBoundsBuffer',         label: 'vitrum.pt-webgpu.scene.cwbvhNodeBounds' },
  { key: 'cwbvhChildBoundsPacked',  bufferField: 'cwbvhChildBoundsPackedBuffer',  label: 'vitrum.pt-webgpu.scene.cwbvhChildBoundsPacked' },
  { key: 'cwbvhChildMeta',          bufferField: 'cwbvhChildMetaBuffer',          label: 'vitrum.pt-webgpu.scene.cwbvhChildMeta' },
  { key: 'cwbvhChildCount',         bufferField: 'cwbvhChildCountBuffer',         label: 'vitrum.pt-webgpu.scene.cwbvhChildCount' },
  { key: 'cwbvhTlasBlasRoots',      bufferField: 'cwbvhTlasBlasRootsBuffer',      label: 'vitrum.pt-webgpu.scene.cwbvhTlasBlasRoots' },
  // ── Analytic primitives ───────────────────────────────────────────────────
  { key: 'analyticHeaders',         bufferField: 'analyticHeadersBuffer',         label: 'vitrum.pt-webgpu.scene.analyticHeaders' },
  { key: 'analyticParams',          bufferField: 'analyticParamsBuffer',          label: 'vitrum.pt-webgpu.scene.analyticParams' },
  { key: 'analyticLocalToWorld',    bufferField: 'analyticLocalToWorldBuffer',    label: 'vitrum.pt-webgpu.scene.analyticLocalToWorld' },
  { key: 'analyticWorldToLocal',    bufferField: 'analyticWorldToLocalBuffer',    label: 'vitrum.pt-webgpu.scene.analyticWorldToLocal' },
  // ── Environment map ───────────────────────────────────────────────────────
  { key: 'environmentMapTexels', bufferField: 'environmentMapTexelsBuffer', label: 'vitrum.pt-webgpu.scene.environmentMapTexels' },
  { key: 'environmentMapCdf',    bufferField: 'environmentMapCdfBuffer',    label: 'vitrum.pt-webgpu.scene.environmentMapCdf' },
  // ── Emitter arrays ────────────────────────────────────────────────────────
  { key: 'directionalLightsData', bufferField: 'directionalLightsBuffer', label: 'vitrum.pt-webgpu.scene.directionalLights' },
  { key: 'pointLightsData',       bufferField: 'pointLightsBuffer',       label: 'vitrum.pt-webgpu.scene.pointLights' },
  { key: 'spotLightsData',        bufferField: 'spotLightsBuffer',        label: 'vitrum.pt-webgpu.scene.spotLights' },
  { key: 'rectAreaLightsData',    bufferField: 'rectAreaLightsBuffer',    label: 'vitrum.pt-webgpu.scene.rectAreaLights' },
  { key: 'meshAreaLightsData',    bufferField: 'meshAreaLightsBuffer',    label: 'vitrum.pt-webgpu.scene.meshAreaLights' },
  { key: 'meshAreaLightSourceFactorsData', bufferField: 'meshAreaLightSourceFactorsBuffer', label: 'vitrum.pt-webgpu.scene.meshAreaLightSourceFactors', excludeFromMemorySum: true },
  // ── WS2 light tree ────────────────────────────────────────────────────────
  { key: 'lightTreeNodes', bufferField: 'lightTreeBuffer', label: 'vitrum.pt-webgpu.scene.lightTree' },
  // ── P2 per-vertex UVs/tangents/colors + material texture descriptors ─────
  { key: 'uvs',                    bufferField: 'uvsBuffer',                    label: 'vitrum.pt-webgpu.scene.uvs' },
  { key: 'tangents',               bufferField: 'tangentsBuffer',               label: 'vitrum.pt-webgpu.scene.tangents' },
  { key: 'colors',                 bufferField: 'colorsBuffer',                 label: 'vitrum.pt-webgpu.scene.colors' },
  { key: 'materialTexDescriptors', bufferField: 'materialTexDescriptorsBuffer', label: 'vitrum.pt-webgpu.scene.materialTexDescriptors' },
  // ── TLAS (must be contiguous at the END; index 22 = TLAS_START_INDEX) ─────
  { key: 'tlasNodes',                  bufferField: 'tlasNodesBuffer',                  label: 'vitrum.pt-webgpu.scene.tlasNodes' },
  { key: 'tlasInstanceIndices',        bufferField: 'tlasInstanceIndicesBuffer',        label: 'vitrum.pt-webgpu.scene.tlasInstanceIndices' },
  { key: 'tlasBlasRoots',             bufferField: 'tlasBlasRootsBuffer',             label: 'vitrum.pt-webgpu.scene.tlasBlasRoots' },
  { key: 'tlasInstanceWorldToLocal',   bufferField: 'tlasInstanceWorldToLocalBuffer',   label: 'vitrum.pt-webgpu.scene.tlasInstanceWorldToLocal' },
  { key: 'tlasInstanceLocalToWorld',   bufferField: 'tlasInstanceLocalToWorldBuffer',   label: 'vitrum.pt-webgpu.scene.tlasInstanceLocalToWorld' },
] as const;


/**
 * Write `data` into `buffer` when non-empty (shared by all four upload-variant
 * functions — hoisted from the identical local closures that previously lived in
 * each one separately).
 */
function writeBufferIfNonEmpty(buffer: GPUBuffer, data: ArrayBufferView, device: GPUDevice): void {
  if (data.byteLength > 0) {
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  }
}

const BVH_NODE_BUFFER_BYTES = BVH_NODE_FLOATS * Float32Array.BYTES_PER_ELEMENT;

const MIN_STORAGE_BUFFER_BYTES_BY_LABEL: Readonly<Record<string, number>> = {
  'vitrum.pt-webgpu.scene.bvhNodes': BVH_NODE_BUFFER_BYTES,
  'vitrum.pt-webgpu.scene.tlasNodes': BVH_NODE_BUFFER_BYTES,
};

function createStorageBuffer(device: GPUDevice, label: string, data: ArrayBufferView): GPUBuffer {
  const minSize = Math.max(data.byteLength, MIN_STORAGE_BUFFER_BYTES_BY_LABEL[label] ?? 16);
  const buffer = device.createBuffer({
    label,
    size: Math.ceil(minSize / 4) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  if (data.byteLength > 0) {
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  } else {
    device.queue.writeBuffer(buffer, 0, new Uint32Array(Math.ceil(minSize / Uint32Array.BYTES_PER_ELEMENT)));
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
   * behaviour (emissive stays at whatever the scene primitive carries).
   */
  readonly cameraVisibleEmitters?: boolean;

  /**
   * Geometry pack shape.
   *
   * `tlas` is the full-tier path: local-space BLAS streams plus TLAS instance
   * tables. `merged` is the lite-tier path: one baked world-space BLAS rooted at
   * node 0 so the single-group lite shader, which cannot bind TLAS buffers, still
   * sees every static mesh/skinned/instanced primitive.
   */
  readonly geometryMode?: 'tlas' | 'merged';

  /** Structured warning sink used by engine-owned scene ingestion. */
  readonly onWarning?: (warning: EngineWarning) => void;
  readonly warningPhase?: EngineWarning['phase'];
  readonly warningMethod?: string;
}

/**
 * H10 — Pack the GPU material floats for `primitive`, applying the emissive
 * fold when `cameraVisibleEmitters` is true AND the primitive is the mesh
 * referenced by a mesh-area emitter in `scene`.
 *
 * The fold re-attaches `emitter.color * emitter.intensity` onto the
 * primitive's emissive channel (emissiveIntensity=1) so the kernel's
 * emissive-on-hit term produces radiance equal to the NEE sample. Without
 * the fold, camera-visible emitters look dark when hit directly.
 *
 * Used by: `buildPackedScene` (full pack) + `updateEmitter`/material fast
 * paths in `SceneMutationRouter` (incremental patch). Sharing the same code
 * prevents the desync described in H10.
 */
export function packFoldedMaterialEntry(
  primitive: { id: string; material: MaterialSpec; castShadow?: boolean },
  scene: Scene,
  cameraVisibleEmitters: boolean,
): number[] {
  // SHADOW-01 — material slots are per-primitive, so the primitive's castShadow
  // flag rides the material payload (vec4 #25 .w). Analytic primitives carry no
  // castShadow field in the contract → undefined → packs the 0.0 default.
  const packContext = { castShadow: primitive.castShadow };
  if (cameraVisibleEmitters) {
    for (const emitter of scene.emitters) {
      if (emitter.kind !== 'mesh-area') continue;
      if (emitter.meshId !== primitive.id) continue;
      // Pre-multiply intensity so `emissive · emissiveIntensity` == NEE radiance.
      const foldedMat = {
        ...primitive.material,
        emissive: [
          emitter.color[0] * emitter.intensity,
          emitter.color[1] * emitter.intensity,
          emitter.color[2] * emitter.intensity,
        ] as [number, number, number],
        emissiveIntensity: 1,
      };
      return materialToPackedVec4s(foldedMat, packContext);
    }
  }
  return materialToPackedVec4s(primitive.material, packContext);
}

/**
 * Item 1 — run solveSkin (CPU LBS) on every skinned-mesh primitive that carries
 * bone data, replacing rest-pose positions/normals with the solved deformed pose.
 * Primitives without bones (or with no skin data) are returned unchanged.
 *
 * morphTargets ARE handled by solveSkin (morph-blend is applied before LBS when
 * morphTargets + morphWeights are present). solveSkin handles position, normal,
 * tangent, and UV morph deltas; solved tangents/UVs are preserved for
 * tangent-space and texture-space material maps.
 *
 * Returns a new scene whose skinned-mesh primitives carry solved
 * positions/normals/tangents/uvs so that packSceneFromCore uses the correct
 * deformed geometry, tangent frame, and texture coordinates.
 */
function applySolveSkinToScene(scene: Scene, warningOptions: BuildPackedSceneOptions): Scene {
  let anyChanged = false;
  const nextPrimitives = scene.primitives.map((p) => {
    if (p.kind !== 'skinned-mesh') return p;
    const boneCount = p.bones.length / 16;
    if (boneCount === 0) return p; // no bones — rest pose is correct
    try {
      const solved = solveSkin(p);
      anyChanged = true;
      // Return a structural override: only solved vertex attributes change; all
      // other fields (uvs, indices, skinIndices, skinWeights, bones, …) are
      // preserved so downstream packs (emitters, BVH) see the full primitive data.
      return {
        ...p,
        positions: solved.positions,
        normals: solved.normals,
        ...(solved.tangents ? { tangents: solved.tangents } : {}),
        ...(solved.uvs ? { uvs: solved.uvs } : {}),
        ...(solved.uv1 ? { uv1: solved.uv1 } : {}),
      };
    } catch (err) {
      emitSolveSkinFallbackWarning(p.id, err, warningOptions);
      return p;
    }
  });
  if (!anyChanged) return scene;
  return { ...scene, primitives: nextPrimitives };
}

function emitSolveSkinFallbackWarning(
  primitiveId: ScenePrimitive['id'],
  err: unknown,
  warningOptions: BuildPackedSceneOptions,
): void {
  const message = `[vitrum/pt-webgpu] solveSkin failed for primitive "${primitiveId}"; using rest pose. ${String(err)}`;
  const warning: EngineWarning = {
    code: 'pt-webgpu.set-scene-skin-fallback',
    backend: 'pt-webgpu',
    phase: warningOptions.warningPhase ?? 'setScene',
    method: warningOptions.warningMethod ?? 'setScene',
    message,
    details: {
      primitiveId: String(primitiveId),
      fallback: 'rest-pose',
    },
    raw: err,
  };
  if (warningOptions.onWarning) {
    warningOptions.onWarning(warning);
    return;
  }
  console.warn(message);
}

function hdriHandleDiagnostics(scene: Scene): Record<string, unknown> {
  if (scene.environment.kind !== 'hdri') return {};
  const handle = scene.environment.hdri;
  const record =
    handle != null && typeof handle === 'object'
      ? handle as {
          width?: unknown;
          height?: unknown;
          data?: unknown;
          image?: { width?: unknown; height?: unknown; data?: unknown };
          source?: unknown;
        }
      : undefined;
  const data = record?.data ?? record?.image?.data;
  const dataLength =
    data != null && typeof data === 'object' && 'length' in data
      ? Number((data as { length?: unknown }).length)
      : undefined;
  const handleType =
    handle == null
      ? String(handle)
      : ArrayBuffer.isView(handle)
        ? handle.constructor.name
        : typeof handle;
  const dataType =
    data == null
      ? undefined
      : ArrayBuffer.isView(data)
        ? data.constructor.name
        : typeof data;
  return {
    width: record?.width ?? record?.image?.width,
    height: record?.height ?? record?.image?.height,
    dataLength: Number.isFinite(dataLength) ? dataLength : undefined,
    handleType,
    dataType,
    sourceType: record?.source == null ? undefined : typeof record.source,
  };
}

function structuredEnvironmentWarnings(
  scene: Scene,
  environment: ReturnType<typeof environmentParams>,
  warningOptions: BuildPackedSceneOptions,
): EngineWarning[] {
  if (scene.environment.kind !== 'hdri' || environment.hasHdri) return [];
  const structured: EngineWarning[] = [];
  for (const warning of environment.warnings) {
    const isUnreadable = warning.includes('lacks CPU pixel data');
    const isZeroLuminance = warning.includes('zero total luminance');
    if (!isUnreadable && !isZeroLuminance) continue;
    const code = isUnreadable
      ? 'pt-webgpu.hdri-unreadable'
      : 'pt-webgpu.hdri-zero-luminance';
    const message = isUnreadable
      ? '[vitrum/pt-webgpu] HDRI environment is present but has no usable CPU pixel data; ' +
        'pt-webgpu requires a raw {width, height, data} or DataTexture-shaped {image:{width,height,data}} RGB/RGBA payload ' +
        'and will use a black no-environment fallback.'
      : '[vitrum/pt-webgpu] HDRI environment has zero total luminance; using black no-environment fallback.';
    structured.push({
      code,
      backend: 'pt-webgpu',
      phase: warningOptions.warningPhase ?? 'setScene',
      method: warningOptions.warningMethod ?? 'setScene',
      message,
      details: {
        warning,
        fallback: 'no-environment',
        ...hdriHandleDiagnostics(scene),
      },
    });
  }
  return structured;
}

function padTriangleIndicesToVec4(indices: Uint32Array): Uint32Array {
  const triCount = Math.floor(indices.length / 3);
  const out = new Uint32Array(triCount * 4);
  for (let t = 0; t < triCount; t += 1) {
    out[t * 4] = indices[t * 3] ?? 0;
    out[t * 4 + 1] = indices[t * 3 + 1] ?? 0;
    out[t * 4 + 2] = indices[t * 3 + 2] ?? 0;
    out[t * 4 + 3] = 0;
  }
  return out;
}

function packMergedUvs(scene: Scene, merged: ReturnType<typeof mergeWorldSpaceFromCore>): Float32Array {
  const vertexCount = merged.vertexCount;
  const uv1 = mergeUv1FromCore(scene, merged.meshVertexRanges, vertexCount);
  const out = new Float32Array(vertexCount * 4);
  for (let i = 0; i < vertexCount; i += 1) {
    out[i * 4] = merged.uvs[i * 2] ?? 0;
    out[i * 4 + 1] = merged.uvs[i * 2 + 1] ?? 0;
    out[i * 4 + 2] = uv1?.[i * 2] ?? out[i * 4]!;
    out[i * 4 + 3] = uv1?.[i * 2 + 1] ?? out[i * 4 + 1]!;
  }
  return out;
}

function packMergedMaterial(
  material: MaterialSpec & { readonly castShadow?: boolean },
): number[] {
  return materialToPackedVec4s(material, { castShadow: material.castShadow });
}

function finiteRootBoundsFromFloatWords(words: ArrayLike<number>): {
  readonly center: readonly [number, number, number];
  readonly radius: number;
} | null {
  if (words.length < BVH_NODE_FLOATS) return null;
  const minX = words[0] ?? 0;
  const minY = words[1] ?? 0;
  const minZ = words[2] ?? 0;
  const maxX = words[3] ?? 0;
  const maxY = words[4] ?? 0;
  const maxZ = words[5] ?? 0;
  if (![minX, minY, minZ, maxX, maxY, maxZ].every(Number.isFinite)) return null;
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  const dx = maxX - minX;
  const dy = maxY - minY;
  const dz = maxZ - minZ;
  return {
    center: [cx, cy, cz],
    radius: Math.max(1e-3, 0.5 * Math.hypot(dx, dy, dz)),
  };
}

function sceneCenterRadiusFromPack(
  pack: Pick<ScenePackResult, 'bvhNodes'> & Partial<Pick<ScenePackResult, 'tlasNodes'>>,
): {
  readonly center: readonly [number, number, number];
  readonly radius: number;
} {
  if (pack.tlasNodes && pack.tlasNodes.length >= BVH_NODE_FLOATS) {
    const tlasWords = new Float32Array(
      pack.tlasNodes.buffer,
      pack.tlasNodes.byteOffset,
      pack.tlasNodes.length,
    );
    const bounds = finiteRootBoundsFromFloatWords(tlasWords);
    if (bounds) return bounds;
  }
  const bounds = finiteRootBoundsFromFloatWords(pack.bvhNodes);
  return bounds ?? { center: [0, 0, 0], radius: 1 };
}

interface PackedCwbvhSceneData {
  readonly cwbvhNodeBounds: Float32Array;
  readonly cwbvhChildBoundsPacked: Uint32Array;
  readonly cwbvhChildMeta: Uint32Array;
  readonly cwbvhChildCount: Uint32Array;
  readonly cwbvhTlasBlasRoots: Uint32Array;
  readonly cwbvhBinaryRootToWideRoot: Uint32Array;
  readonly cwbvhNodeCount: number;
}

function identityTriangleMap(triangleCount: number): Uint32Array {
  const out = new Uint32Array(Math.max(0, triangleCount));
  for (let i = 0; i < out.length; i += 1) out[i] = i;
  return out;
}

function sortedBlasRootSpans(pack: Pick<ScenePackResult, 'bvhNodes' | 'primitiveTlasBindings'>): readonly {
  readonly binaryRoot: number;
  readonly binaryEnd: number;
}[] {
  const totalNodes = Math.floor(pack.bvhNodes.length / BVH_NODE_FLOATS);
  if (totalNodes === 0) return [];
  const roots = Array.from(new Set(pack.primitiveTlasBindings.map((binding) => binding.blasRoot)))
    .filter((root) => Number.isFinite(root) && root >= 0 && root < totalNodes)
    .sort((a, b) => a - b);
  if (roots.length === 0) return [{ binaryRoot: 0, binaryEnd: totalNodes }];
  return roots.map((binaryRoot, i) => ({
    binaryRoot,
    binaryEnd: roots[i + 1] ?? totalNodes,
  }));
}

function remapCwbvhTlasBlasRoots(
  tlasBlasRoots: Uint32Array,
  binaryRootToWideRoot: Uint32Array,
): Uint32Array {
  const out = new Uint32Array(tlasBlasRoots.length);
  for (let i = 0; i < tlasBlasRoots.length; i += 1) {
    const binaryRoot = tlasBlasRoots[i] ?? 0;
    out[i] = binaryRootToWideRoot[binaryRoot] ?? 0;
  }
  return out;
}

/**
 * Build the pt-webgpu CWBVH forest beside the canonical binary BVH.
 *
 * CWBVH traversal is opt-in rather than default-promoted. This pack is still
 * renderer-shaped: full-tier TLAS scenes concatenate one CWBVH tree per BLAS
 * subtree and remap `tlasBlasRoots` to wide-node roots, while merged/lite
 * scenes produce a single root-0 wide tree. Leaf metadata keeps the existing
 * global triangle offsets, so the opt-in traversal can read the same `indices`,
 * `positions`, material ids, and cast/visibility payloads as the binary path.
 */
function buildPackedCwbvhSceneData(pack: Pick<
  ScenePackResult,
  | 'positions'
  | 'indices'
  | 'triMaterialIds'
  | 'bvhNodes'
  | 'triangleCount'
  | 'tlasBlasRoots'
  | 'primitiveTlasBindings'
>): PackedCwbvhSceneData {
  const nodeBounds: number[] = [];
  const childBoundsPacked: number[] = [];
  const childMeta: number[] = [];
  const childCount: number[] = [];
  const totalBinaryNodes = Math.floor(pack.bvhNodes.length / BVH_NODE_FLOATS);
  const binaryRootToWideRoot = new Uint32Array(totalBinaryNodes);
  const sourceTriangles = identityTriangleMap(pack.triangleCount);

  for (const span of sortedBlasRootSpans(pack)) {
    if (span.binaryEnd <= span.binaryRoot) continue;
    const wideBase = Math.floor(nodeBounds.length / 6);
    binaryRootToWideRoot[span.binaryRoot] = wideBase;
    const binaryNodeWords = pack.bvhNodes.subarray(
      span.binaryRoot * BVH_NODE_FLOATS,
      span.binaryEnd * BVH_NODE_FLOATS,
    );
    const cwbvh = buildCompressedWideBvhFromArrayBvh({
      bvhNodes: new Float32Array(binaryNodeWords),
      reorderedIndices: pack.indices,
      reorderedTriMaterialIds: pack.triMaterialIds,
      reorderedToSourceTriangle: sourceTriangles,
    });
    for (const word of cwbvh.cwbvhNodeBounds) nodeBounds.push(word);
    for (const word of packCwbvhBuildBoundsForWgsl(cwbvh)) childBoundsPacked.push(word);
    for (let i = 0; i < cwbvh.cwbvhChildMeta.length; i += CWBVH_CHILD_META_WORDS) {
      const kind = cwbvh.cwbvhChildMeta[i] ?? 0;
      childMeta.push(
        kind,
        kind === CWBVH_CHILD_NODE ? (cwbvh.cwbvhChildMeta[i + 1] ?? 0) + wideBase : cwbvh.cwbvhChildMeta[i + 1] ?? 0,
        cwbvh.cwbvhChildMeta[i + 2] ?? 0,
      );
    }
    for (const count of cwbvh.cwbvhChildCount) childCount.push(count);
  }

  const rootMap = binaryRootToWideRoot;
  return {
    cwbvhNodeBounds: new Float32Array(nodeBounds),
    cwbvhChildBoundsPacked: new Uint32Array(childBoundsPacked),
    cwbvhChildMeta: new Uint32Array(childMeta),
    cwbvhChildCount: new Uint32Array(childCount),
    cwbvhTlasBlasRoots: remapCwbvhTlasBlasRoots(pack.tlasBlasRoots, rootMap),
    cwbvhBinaryRootToWideRoot: rootMap,
    cwbvhNodeCount: childCount.length,
  };
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
  const { supported: filteredScene, warnings } = partitionSceneBySupport(inputScene, PT_WEBGPU_SUPPORT);
  // Item 1 — apply CPU LBS to skinned-mesh primitives so packSceneFromCore uses
  // solved (deformed) positions instead of rest-pose. morphTargets are also
  // handled by solveSkin (blend applied before LBS).
  const scene = applySolveSkinToScene(filteredScene, options);
  const geometryMode = options.geometryMode ?? 'tlas';

  // Camera-visible emitters: delegate to the shared packFoldedMaterialEntry helper
  // (H10). This ensures the fold logic is identical across the full pack and the
  // incremental fast-path patches in SceneMutationRouter.
  const cameraVisible = options.cameraVisibleEmitters === true;
  const packMaterial = (primitive: { id: string; material: MaterialSpec; castShadow?: boolean }): number[] =>
    packFoldedMaterialEntry(primitive, scene, cameraVisible);

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
    if (geometryMode === 'merged') {
      continue;
    }
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

  const geo = geometryMode === 'merged'
    ? (() => {
        const merged = mergeWorldSpaceFromCore(scene, {
          positionStride: 4,
          splitMaterialsByCastShadow: true,
          bakeConstantVertexColorIntoMaterial: true,
        });
        for (const material of merged.materials) {
          const withShadow = material as MaterialSpec & { readonly castShadow?: boolean };
          materials.push(...packMergedMaterial(withShadow));
          materialSpecs.push(withShadow);
        }
        return {
          positions: merged.positions,
          normals: merged.normals,
          uvs: packMergedUvs(scene, merged),
          tangents: new Float32Array(merged.positions.length),
          colors: merged.colors,
          indices: padTriangleIndicesToVec4(merged.indices),
          triMaterialIds: merged.triMaterialId,
          bvhNodes: merged.bvhNodes,
          triangleCount: merged.triangleCount,
          tlasNodes: new Uint32Array(0),
          tlasInstanceIndices: new Uint32Array(0),
          tlasBlasRoots: new Uint32Array(0),
          tlasInstanceWorldToLocal: new Float32Array(0),
          tlasInstanceLocalToWorld: new Float32Array(0),
          tlasNodeCount: 0,
          primitiveTlasBindings: [] as readonly PrimitiveTlasBinding[],
          warnings: merged.warnings,
        } satisfies ScenePackResult;
      })()
    : packSceneFromCore(scene, {
        tlas: true,
        resolveMaterialId: (id) => meshMaterialIds.get(id) ?? 0,
      });
  const analyticIds = new Set(
    scene.primitives
      .filter((primitive) => primitive.kind === 'analytic')
      .map((primitive) => primitive.id),
  );
  warnings.push(
    ...geo.warnings.filter((warning) => {
      if (analyticIds.size === 0) return true;
      for (const id of analyticIds) {
        if (
          warning ===
          `Primitive "${id}" (analytic) skipped; scenePack supports mesh, skinned-mesh, and instanced-mesh only.`
        ) {
          return false;
        }
      }
      return true;
    }),
  );
  const texCollection = collectMaterialTextures(materialSpecs);
  const emitArrays = packEmitterArrays(scene);
  const environment = environmentParams(scene);
  const structuredWarnings = [
    ...structuredEnvironmentWarnings(scene, environment, options),
    ...texCollection.unsupportedTexCoordWarnings,
  ];
  const sceneBounds = sceneCenterRadiusFromPack(geo);
  warnings.push(...environment.warnings);
  warnings.push(...emitArrays.warnings);

  // WS2 — build the power-weighted light tree over the selectable lights, in the
  // SAME order the kernel NEE walk visits them (directional · point · spot ·
  // rect · mesh · env). Only worthwhile with ≥ 2 lights: a 0/1-light tree adds
  // no variance reduction over the uniform pick (and `buildLightTree` requires a
  // non-empty input), so below 2 lights we ship an empty buffer + the uniform
  // fallback gate (`lightTreeEnabled = false`).
  //
  // Pass the already-computed `emitArrays` (from packEmitterArrays above) and the
  // env summary derived from `environment` (from environmentParams above) so
  // buildLightTreeInputForScene does NOT re-run either expensive call a second time.
  const envSummaryForTree: EnvSummaryForTree = {
    hasHdri: environment.hasHdri,
    sunStrength: environment.sunStrength,
    tint: environment.tint,
  };
  const lightTreeInput = buildLightTreeInputForScene(scene, {
    packed: emitArrays,
    envSummary: envSummaryForTree,
  });
  let lightTreeNodes = new Float32Array(0);
  let lightTreeNodeCount = 0;
  let lightTreeEnabled = false;
  if (lightTreeInput.powers.length >= 2) {
    const { nodes } = buildLightTree(lightTreeInput);
    lightTreeNodes = new Float32Array(packLightTreeForGPU(nodes));
    lightTreeNodeCount = nodes.length;
    lightTreeEnabled = true;
  }
  const cwbvh = buildPackedCwbvhSceneData(geo);

  return {
    positions: geo.positions,
    normals: geo.normals,
    uvs: geo.uvs,
    tangents: geo.tangents,
    colors: geo.colors,
    indices: geo.indices,
    triMaterialIds: geo.triMaterialIds,
    materials: new Float32Array(materials),
    cwbvhNodeBounds: cwbvh.cwbvhNodeBounds,
    cwbvhChildBoundsPacked: cwbvh.cwbvhChildBoundsPacked,
    cwbvhChildMeta: cwbvh.cwbvhChildMeta,
    cwbvhChildCount: cwbvh.cwbvhChildCount,
    cwbvhTlasBlasRoots: cwbvh.cwbvhTlasBlasRoots,
    cwbvhBinaryRootToWideRoot: cwbvh.cwbvhBinaryRootToWideRoot,
    cwbvhNodeCount: cwbvh.cwbvhNodeCount,
    materialTexDescriptors: texCollection.descriptors,
    materialTextureSources: texCollection.sources,
    materialTextureSourceInfos: texCollection.sourceInfos,
    materialTextureLinearSources: texCollection.linearSources,
    materialTextureLinearSourceInfos: texCollection.linearSourceInfos,
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
    structuredWarnings,
    directionalLight: defaultDirectionalLight(scene),
    directionalIrradiance: defaultDirectionalIrradiance(scene),
    directionalAngularDiameter: defaultDirectionalAngularDiameter(scene),
    directionalLightCount: emitArrays.directionalLightCount,
    directionalLightsData: emitArrays.directionalLightsData,
    sceneCenter: sceneBounds.center,
    sceneRadius: sceneBounds.radius,
    pointLightCount: emitArrays.pointLightCount,
    spotLightCount: emitArrays.spotLightCount,
    rectAreaLightCount: emitArrays.rectAreaLightCount,
    meshAreaLightCount: emitArrays.meshAreaLightCount,
    pointLightsData: emitArrays.pointLightsData,
    spotLightsData: emitArrays.spotLightsData,
    rectAreaLightsData: emitArrays.rectAreaLightsData,
    meshAreaLightsData: emitArrays.meshAreaLightsData,
    meshAreaLightSourceFactorsData: emitArrays.meshAreaLightSourceFactorsData,
    environmentTint: environment.tint,
    environmentSunDirection: environment.sunDirection,
    environmentSunStrength: environment.sunStrength,
    environmentHdriIntensity: environment.hdriIntensity,
    environmentHdriRotationY: environment.hdriRotationY,
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
    tangents: packed.tangents,
    colors: packed.colors,
    indices: packed.indices,
    triMaterialIds: packed.triMaterialIds,
    bvhNodes: packed.bvhNodes,
    triangleCount: packed.triangleCount,
    tlasNodes: packed.tlasNodes,
    tlasInstanceIndices: packed.tlasInstanceIndices,
    tlasBlasRoots: packed.tlasBlasRoots,
    tlasInstanceWorldToLocal: packed.tlasInstanceWorldToLocal,
    tlasInstanceLocalToWorld: packed.tlasInstanceLocalToWorld,
    tlasNodeCount: Math.floor(packed.tlasNodes.length / BVH_NODE_FLOATS),
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
  writeBufferIfNonEmpty(sb.positionsBuffer, pack.positions, device);
  writeBufferIfNonEmpty(sb.normalsBuffer, pack.normals, device);
  writeBufferIfNonEmpty(sb.uvsBuffer, pack.uvs, device);
  writeBufferIfNonEmpty(sb.tangentsBuffer, pack.tangents, device);
  writeBufferIfNonEmpty(sb.colorsBuffer, pack.colors, device);
  writeBufferIfNonEmpty(sb.indicesBuffer, pack.indices, device);
  writeBufferIfNonEmpty(sb.triMaterialIdsBuffer, pack.triMaterialIds, device);
  writeBufferIfNonEmpty(sb.bvhNodesBuffer, pack.bvhNodes, device);
  writeBufferIfNonEmpty(sb.tlasNodesBuffer, pack.tlasNodes, device);
  writeBufferIfNonEmpty(sb.tlasInstanceIndicesBuffer, pack.tlasInstanceIndices, device);
  writeBufferIfNonEmpty(sb.tlasBlasRootsBuffer, pack.tlasBlasRoots, device);
  writeBufferIfNonEmpty(sb.tlasInstanceWorldToLocalBuffer, pack.tlasInstanceWorldToLocal, device);
  writeBufferIfNonEmpty(sb.tlasInstanceLocalToWorldBuffer, pack.tlasInstanceLocalToWorld, device);
  sb.positions.set(pack.positions);
  sb.normals.set(pack.normals);
  sb.uvs.set(pack.uvs);
  sb.tangents.set(pack.tangents);
  sb.colors.set(pack.colors);
  sb.indices.set(pack.indices);
  sb.triMaterialIds.set(pack.triMaterialIds);
  sb.bvhNodes.set(pack.bvhNodes);
  sb.tlasNodes.set(pack.tlasNodes);
  sb.tlasInstanceIndices.set(pack.tlasInstanceIndices);
  sb.tlasBlasRoots.set(pack.tlasBlasRoots);
  sb.tlasInstanceWorldToLocal.set(pack.tlasInstanceWorldToLocal);
  sb.tlasInstanceLocalToWorld.set(pack.tlasInstanceLocalToWorld);
  rebuildCwbvhFromPack(device, sb, pack);
  applyScenePackCounts(sb, pack);
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
    | 'tangents'
    | 'colors'
    | 'indices'
    | 'triMaterialIds'
    | 'bvhNodes'
    | 'triangleCount'
    | 'primitiveTlasBindings'
  >,
): void {
  writeBufferIfNonEmpty(sb.positionsBuffer, pack.positions, device);
  writeBufferIfNonEmpty(sb.normalsBuffer, pack.normals, device);
  writeBufferIfNonEmpty(sb.uvsBuffer, pack.uvs, device);
  writeBufferIfNonEmpty(sb.tangentsBuffer, pack.tangents, device);
  writeBufferIfNonEmpty(sb.colorsBuffer, pack.colors, device);
  writeBufferIfNonEmpty(sb.indicesBuffer, pack.indices, device);
  writeBufferIfNonEmpty(sb.triMaterialIdsBuffer, pack.triMaterialIds, device);
  writeBufferIfNonEmpty(sb.bvhNodesBuffer, pack.bvhNodes, device);
  sb.positions.set(pack.positions);
  sb.normals.set(pack.normals);
  sb.uvs.set(pack.uvs);
  sb.tangents.set(pack.tangents);
  sb.colors.set(pack.colors);
  sb.indices.set(pack.indices);
  sb.triMaterialIds.set(pack.triMaterialIds);
  sb.bvhNodes.set(pack.bvhNodes);
  rebuildCwbvhFromPack(device, sb, {
    positions: pack.positions,
    indices: pack.indices,
    triMaterialIds: pack.triMaterialIds,
    bvhNodes: pack.bvhNodes,
    triangleCount: pack.triangleCount,
    tlasBlasRoots: sb.tlasBlasRoots,
    primitiveTlasBindings: pack.primitiveTlasBindings,
  });
  applyScenePackCounts(sb, pack);
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
  const buffers = asMutableSceneBuffers(sb);
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
  updateCwbvhTlasRootMirror(device, sb, buffers.tlasBlasRoots);

  applyScenePackCounts(sb, pack);
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
  sb.tangentsBuffer.destroy();
  sb.colorsBuffer.destroy();
  sb.indicesBuffer.destroy();
  sb.triMaterialIdsBuffer.destroy();
  sb.bvhNodesBuffer.destroy();
  sb.tlasNodesBuffer.destroy();
  sb.tlasInstanceIndicesBuffer.destroy();
  sb.tlasBlasRootsBuffer.destroy();
  sb.tlasInstanceWorldToLocalBuffer.destroy();
  sb.tlasInstanceLocalToWorldBuffer.destroy();

  const handles = asMutableSceneBuffers(sb);
  handles.positionsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.positions', pack.positions);
  handles.normalsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.normals', pack.normals);
  handles.uvsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.uvs', pack.uvs);
  handles.tangentsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.tangents', pack.tangents);
  handles.colorsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.colors', pack.colors);
  handles.indicesBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.indices', pack.indices);
  handles.triMaterialIdsBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.triMaterialIds',
    pack.triMaterialIds,
  );
  handles.bvhNodesBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.bvhNodes', pack.bvhNodes);
  handles.positions = new Float32Array(pack.positions);
  handles.normals = new Float32Array(pack.normals);
  handles.uvs = new Float32Array(pack.uvs);
  handles.tangents = new Float32Array(pack.tangents);
  handles.colors = new Float32Array(pack.colors);
  handles.indices = new Uint32Array(pack.indices);
  handles.triMaterialIds = new Uint32Array(pack.triMaterialIds);
  handles.bvhNodes = new Float32Array(pack.bvhNodes);

  handles.tlasNodesBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.tlasNodes', pack.tlasNodes);
  handles.tlasInstanceIndicesBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.tlasInstanceIndices',
    pack.tlasInstanceIndices,
  );
  handles.tlasBlasRootsBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.tlasBlasRoots',
    pack.tlasBlasRoots,
  );
  handles.tlasInstanceWorldToLocalBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.tlasInstanceWorldToLocal',
    pack.tlasInstanceWorldToLocal,
  );
  handles.tlasInstanceLocalToWorldBuffer = createStorageBuffer(
    device,
    'vitrum.pt-webgpu.scene.tlasInstanceLocalToWorld',
    pack.tlasInstanceLocalToWorld,
  );
  handles.tlasNodes = new Uint32Array(pack.tlasNodes);
  handles.tlasInstanceIndices = new Uint32Array(pack.tlasInstanceIndices);
  handles.tlasBlasRoots = new Uint32Array(pack.tlasBlasRoots);
  handles.tlasInstanceWorldToLocal = new Float32Array(pack.tlasInstanceWorldToLocal);
  handles.tlasInstanceLocalToWorld = new Float32Array(pack.tlasInstanceLocalToWorld);
  rebuildCwbvhFromPack(device, sb, pack);

  applyScenePackCounts(sb, pack);
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
  writeBufferIfNonEmpty(sb.tlasNodesBuffer, pack.tlasNodes, device);
  writeBufferIfNonEmpty(sb.tlasInstanceIndicesBuffer, pack.tlasInstanceIndices, device);
  writeBufferIfNonEmpty(sb.tlasBlasRootsBuffer, pack.tlasBlasRoots, device);
  writeBufferIfNonEmpty(sb.tlasInstanceWorldToLocalBuffer, pack.tlasInstanceWorldToLocal, device);
  writeBufferIfNonEmpty(sb.tlasInstanceLocalToWorldBuffer, pack.tlasInstanceLocalToWorld, device);
  sb.tlasNodes.set(pack.tlasNodes);
  sb.tlasInstanceIndices.set(pack.tlasInstanceIndices);
  sb.tlasBlasRoots.set(pack.tlasBlasRoots);
  sb.tlasInstanceWorldToLocal.set(pack.tlasInstanceWorldToLocal);
  sb.tlasInstanceLocalToWorld.set(pack.tlasInstanceLocalToWorld);
  updateCwbvhTlasRootMirror(device, sb, sb.tlasBlasRoots);
  applyScenePackCounts(sb, pack);
}

/**
 * `MutableSceneBuffers` — the SINGLE mutable view onto an otherwise-readonly
 * {@link UploadedSceneBuffers}. It unions all four field sets that the incremental
 * fast paths need to rewrite:
 *
 *   - Geometry-pack derived counts (BLAS / TLAS uploads).
 *   - TLAS GPU buffer handles + CPU mirrors (realloc'd on instance-count change).
 *   - BLAS GPU buffer handles + CPU mirrors (realloc'd on vertex/index-count change).
 *   - Emitter buffer handles + CPU mirrors (realloc'd when emitter arrays grow/shrink).
 *   - Emitter counts + directional aggregate (incremental emitter patches).
 *   - Environment fields (incremental environment patches).
 *   - Light-tree state + buffer handle (rebuilt after emitter/environment patches).
 *
 * Having ONE interface + ONE cast function (`asMutableSceneBuffers`) is the single
 * unsafe-cast surface. The former four separate interfaces
 * (`MutableSceneBufferFields`, `MutableTlasBufferHandles`, `MutableEmitterBufferHandles`,
 * `MutableBlasBufferHandles`) and their four cast functions have been merged here
 * to eliminate the scattered `as unknown as { … }` sites. Behavior-preserving.
 */
interface MutableSceneBuffers {
  // ── Geometry-pack derived counts ─────────────────────────────────────────
  bvhNodeCount: number;
  cwbvhNodeCount: number;
  tlasNodeCount: number;
  triangleCount: number;
  primitiveTlasBindings: readonly PrimitiveTlasBinding[];
  sceneCenter: readonly [number, number, number];
  sceneRadius: number;

  // ── TLAS GPU handles + CPU mirrors (realloc'd on instance-count change) ──
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

  // ── BLAS GPU handles + CPU mirrors (realloc'd on vertex/index-count change)
  positionsBuffer: GPUBuffer;
  normalsBuffer: GPUBuffer;
  uvsBuffer: GPUBuffer;
  tangentsBuffer: GPUBuffer;
  colorsBuffer: GPUBuffer;
  indicesBuffer: GPUBuffer;
  triMaterialIdsBuffer: GPUBuffer;
  bvhNodesBuffer: GPUBuffer;
  cwbvhNodeBoundsBuffer: GPUBuffer;
  cwbvhChildBoundsPackedBuffer: GPUBuffer;
  cwbvhChildMetaBuffer: GPUBuffer;
  cwbvhChildCountBuffer: GPUBuffer;
  cwbvhTlasBlasRootsBuffer: GPUBuffer;
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  tangents: Float32Array;
  colors: Float32Array;
  indices: Uint32Array;
  triMaterialIds: Uint32Array;
  bvhNodes: Float32Array;
  cwbvhNodeBounds: Float32Array;
  cwbvhChildBoundsPacked: Uint32Array;
  cwbvhChildMeta: Uint32Array;
  cwbvhChildCount: Uint32Array;
  cwbvhTlasBlasRoots: Uint32Array;
  cwbvhBinaryRootToWideRoot: Uint32Array;

  // ── Emitter buffer handles + CPU mirrors (realloc'd on array-size change) ─
  directionalLightsBuffer: GPUBuffer;
  pointLightsBuffer: GPUBuffer;
  spotLightsBuffer: GPUBuffer;
  rectAreaLightsBuffer: GPUBuffer;
  meshAreaLightsBuffer: GPUBuffer;
  meshAreaLightSourceFactorsBuffer: GPUBuffer;
  directionalLightsData: Float32Array;
  pointLightsData: Float32Array;
  spotLightsData: Float32Array;
  rectAreaLightsData: Float32Array;
  meshAreaLightsData: Float32Array;
  meshAreaLightSourceFactorsData: Float32Array;

  // ── Emitter counts + directional aggregate (incremental emitter patches) ──
  directionalLightCount: number;
  pointLightCount: number;
  spotLightCount: number;
  rectAreaLightCount: number;
  meshAreaLightCount: number;
  directionalLight: readonly [number, number, number];
  directionalIrradiance: readonly [number, number, number];
  /** D3/SHADOW-01 — signed angular-diameter mirror must be kept in sync on
   *  incremental emitter patches; negative encodes first directional
   *  castShadow:false for the lite-tier UBO path. */
  directionalAngularDiameter: number;

  // ── Environment fields (incremental environment patches) ─────────────────
  environmentTint: readonly [number, number, number];
  environmentSunDirection: readonly [number, number, number];
  environmentSunStrength: number;
  environmentHdriIntensity: number;
  environmentHdriRotationY: number;
  environmentMapWidth: number;
  environmentMapHeight: number;
  hasEnvironmentMap: boolean;

  // ── Light-tree state + buffer handle (rebuilt on emitter/env patches) ─────
  lightTreeNodeCount: number;
  lightTreeEnabled: boolean;
  lightTreeNodes: Float32Array;
  lightTreeBuffer: GPUBuffer;
}

/**
 * Single unsafe-cast entry point for all in-place mutations of an
 * {@link UploadedSceneBuffers}. All incremental fast paths (geometry realloc,
 * TLAS realloc, emitter patch, environment patch, light-tree rebuild) go through
 * this one function instead of four separate typed-cast helpers.
 */
function asMutableSceneBuffers(sb: UploadedSceneBuffers): MutableSceneBuffers {
  return sb;
}

function replaceCwbvhBuffers(
  device: GPUDevice,
  sb: UploadedSceneBuffers,
  cwbvh: PackedCwbvhSceneData,
): void {
  if (
    sb.cwbvhNodeBoundsBuffer == null ||
    sb.cwbvhChildBoundsPackedBuffer == null ||
    sb.cwbvhChildMetaBuffer == null ||
    sb.cwbvhChildCountBuffer == null ||
    sb.cwbvhTlasBlasRootsBuffer == null
  ) {
    return;
  }
  const mutable = asMutableSceneBuffers(sb);
  const updateBuffer = (
    current: GPUBuffer | undefined,
    label: string,
    data: ArrayBufferView,
    previousData: ArrayBufferView | undefined,
    assign: (buffer: GPUBuffer) => void,
  ): void => {
    const targetSize = Math.max(data.byteLength, MIN_STORAGE_BUFFER_BYTES_BY_LABEL[label] ?? 16);
    const currentSize = current?.size ?? Math.max(previousData?.byteLength ?? -1, MIN_STORAGE_BUFFER_BYTES_BY_LABEL[label] ?? 16);
    if (current != null && currentSize === targetSize) {
      writeBufferIfNonEmpty(current, data, device);
      return;
    }
    current?.destroy();
    assign(createStorageBuffer(device, label, data));
  };
  updateBuffer(
    sb.cwbvhNodeBoundsBuffer,
    'vitrum.pt-webgpu.scene.cwbvhNodeBounds',
    cwbvh.cwbvhNodeBounds,
    sb.cwbvhNodeBounds,
    (buffer) => {
      mutable.cwbvhNodeBoundsBuffer = buffer;
    },
  );
  updateBuffer(
    sb.cwbvhChildBoundsPackedBuffer,
    'vitrum.pt-webgpu.scene.cwbvhChildBoundsPacked',
    cwbvh.cwbvhChildBoundsPacked,
    sb.cwbvhChildBoundsPacked,
    (buffer) => {
      mutable.cwbvhChildBoundsPackedBuffer = buffer;
    },
  );
  updateBuffer(
    sb.cwbvhChildMetaBuffer,
    'vitrum.pt-webgpu.scene.cwbvhChildMeta',
    cwbvh.cwbvhChildMeta,
    sb.cwbvhChildMeta,
    (buffer) => {
      mutable.cwbvhChildMetaBuffer = buffer;
    },
  );
  updateBuffer(
    sb.cwbvhChildCountBuffer,
    'vitrum.pt-webgpu.scene.cwbvhChildCount',
    cwbvh.cwbvhChildCount,
    sb.cwbvhChildCount,
    (buffer) => {
      mutable.cwbvhChildCountBuffer = buffer;
    },
  );
  updateBuffer(
    sb.cwbvhTlasBlasRootsBuffer,
    'vitrum.pt-webgpu.scene.cwbvhTlasBlasRoots',
    cwbvh.cwbvhTlasBlasRoots,
    sb.cwbvhTlasBlasRoots,
    (buffer) => {
      mutable.cwbvhTlasBlasRootsBuffer = buffer;
    },
  );
  mutable.cwbvhNodeBounds = cwbvh.cwbvhNodeBounds;
  mutable.cwbvhChildBoundsPacked = cwbvh.cwbvhChildBoundsPacked;
  mutable.cwbvhChildMeta = cwbvh.cwbvhChildMeta;
  mutable.cwbvhChildCount = cwbvh.cwbvhChildCount;
  mutable.cwbvhTlasBlasRoots = cwbvh.cwbvhTlasBlasRoots;
  mutable.cwbvhBinaryRootToWideRoot = cwbvh.cwbvhBinaryRootToWideRoot;
  mutable.cwbvhNodeCount = cwbvh.cwbvhNodeCount;
}

function updateCwbvhTlasRootMirror(
  device: GPUDevice,
  sb: UploadedSceneBuffers,
  tlasBlasRoots: Uint32Array,
): void {
  if (sb.cwbvhBinaryRootToWideRoot == null || sb.cwbvhTlasBlasRootsBuffer == null) {
    return;
  }
  const roots = remapCwbvhTlasBlasRoots(tlasBlasRoots, sb.cwbvhBinaryRootToWideRoot);
  const mutable = asMutableSceneBuffers(sb);
  const label = 'vitrum.pt-webgpu.scene.cwbvhTlasBlasRoots';
  const expectedSize = Math.max(roots.byteLength, MIN_STORAGE_BUFFER_BYTES_BY_LABEL[label] ?? 16);
  const currentSize = sb.cwbvhTlasBlasRootsBuffer.size
    ?? Math.max(sb.cwbvhTlasBlasRoots?.byteLength ?? -1, MIN_STORAGE_BUFFER_BYTES_BY_LABEL[label] ?? 16);
  if (currentSize !== expectedSize) {
    sb.cwbvhTlasBlasRootsBuffer.destroy();
    mutable.cwbvhTlasBlasRootsBuffer = createStorageBuffer(
      device,
      label,
      roots,
    );
  } else {
    writeBufferIfNonEmpty(sb.cwbvhTlasBlasRootsBuffer, roots, device);
  }
  mutable.cwbvhTlasBlasRoots = roots;
}

function rebuildCwbvhFromPack(
  device: GPUDevice,
  sb: UploadedSceneBuffers,
  pack: Pick<
    ScenePackResult,
    | 'positions'
    | 'indices'
    | 'triMaterialIds'
    | 'bvhNodes'
    | 'triangleCount'
    | 'tlasBlasRoots'
    | 'primitiveTlasBindings'
  >,
): void {
  replaceCwbvhBuffers(device, sb, buildPackedCwbvhSceneData(pack));
}

/**
 * Apply geometry-pack derived count fields onto `sb` from `pack`. Extracted from
 * the repeated tail blocks in all four upload-variant functions; each variant
 * only supplies the fields it recomputes.
 *
 * - `bvhNodes` present → updates `bvhNodeCount` (BLAS uploads).
 * - `triangleCount` present → updates `triangleCount` (BLAS uploads).
 * - `tlasNodeCount` present → updates `tlasNodeCount` (TLAS uploads).
 * - `primitiveTlasBindings` always required (every upload variant refreshes it).
 * - root TLAS/BLAS bounds refresh `sceneCenter` + `sceneRadius` when present.
 */
function applyScenePackCounts(
  sb: UploadedSceneBuffers,
  pack: {
    readonly primitiveTlasBindings: readonly PrimitiveTlasBinding[];
    readonly bvhNodes?: Float32Array;
    readonly tlasNodes?: Uint32Array;
    readonly triangleCount?: number;
    readonly tlasNodeCount?: number;
  },
): void {
  const mutable = asMutableSceneBuffers(sb);
  if (pack.bvhNodes !== undefined) {
    mutable.bvhNodeCount = Math.floor(pack.bvhNodes.length / BVH_NODE_FLOATS);
  }
  if (pack.triangleCount !== undefined) {
    mutable.triangleCount = pack.triangleCount;
  }
  if (pack.tlasNodeCount !== undefined) {
    mutable.tlasNodeCount = pack.tlasNodeCount;
  }
  if (pack.bvhNodes !== undefined) {
    const bounds = sceneCenterRadiusFromPack({
      bvhNodes: pack.bvhNodes,
      ...(pack.tlasNodes !== undefined ? { tlasNodes: pack.tlasNodes } : {}),
    });
    mutable.sceneCenter = bounds.center;
    mutable.sceneRadius = bounds.radius;
  } else if (pack.tlasNodes !== undefined) {
    const bounds = sceneCenterRadiusFromPack({
      bvhNodes: sb.bvhNodes,
      tlasNodes: pack.tlasNodes,
    });
    mutable.sceneCenter = bounds.center;
    mutable.sceneRadius = bounds.radius;
  }
  mutable.primitiveTlasBindings = pack.primitiveTlasBindings;
}

/** Rewrite emitter counts + directional aggregate after an in-place light upload. */
export function applyEmitterCountMutation(
  sb: UploadedSceneBuffers,
  next: {
    readonly directionalLightCount: number;
    readonly pointLightCount: number;
    readonly spotLightCount: number;
    readonly rectAreaLightCount: number;
    readonly meshAreaLightCount: number;
    readonly directionalLight: readonly [number, number, number];
    readonly directionalIrradiance: readonly [number, number, number];
    /** Item 2d + SHADOW-01 — signed D3 soft-sun angular diameter mirror
     *  (0 = delta directional; negative encodes castShadow:false). */
    readonly directionalAngularDiameter: number;
  },
): void {
  const mutable = asMutableSceneBuffers(sb);
  mutable.directionalLightCount = next.directionalLightCount;
  mutable.pointLightCount = next.pointLightCount;
  mutable.spotLightCount = next.spotLightCount;
  mutable.rectAreaLightCount = next.rectAreaLightCount;
  mutable.meshAreaLightCount = next.meshAreaLightCount;
  mutable.directionalLight = next.directionalLight;
  mutable.directionalIrradiance = next.directionalIrradiance;
  // Item 2d / SHADOW-01 — angular diameter + castShadow mirror must be kept in
  // sync so frameParamsPacker packs the correct value after an incremental
  // directional-emitter patch.
  mutable.directionalAngularDiameter = next.directionalAngularDiameter;
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
    /** Item 2d — D3 soft-sun angular diameter; default 0 (delta directional). */
    readonly directionalAngularDiameter?: number;
  },
): boolean {
  const handles = asMutableSceneBuffers(sb);
  let reallocated = false;
  reallocated = uploadOrReallocateEmitterBuffer(
    device,
    sb.directionalLightsBuffer,
    sb.directionalLightsData,
    packed.directionalLightsData,
    'vitrum.pt-webgpu.scene.directionalLights',
    (buffer, data) => {
      handles.directionalLightsBuffer = buffer;
      handles.directionalLightsData = data;
    },
  ) || reallocated;
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
  reallocated = uploadOrReallocateEmitterBuffer(
    device,
    sb.meshAreaLightSourceFactorsBuffer,
    sb.meshAreaLightSourceFactorsData,
    packed.meshAreaLightSourceFactorsData,
    'vitrum.pt-webgpu.scene.meshAreaLightSourceFactors',
    (buffer, data) => {
      handles.meshAreaLightSourceFactorsBuffer = buffer;
      handles.meshAreaLightSourceFactorsData = data;
    },
  ) || reallocated;
  applyEmitterCountMutation(sb, {
    directionalLightCount: packed.directionalLightCount,
    pointLightCount: packed.pointLightCount,
    spotLightCount: packed.spotLightCount,
    rectAreaLightCount: packed.rectAreaLightCount,
    meshAreaLightCount: packed.meshAreaLightCount,
    directionalLight: nextDirectional.directionalLight,
    directionalIrradiance: nextDirectional.directionalIrradiance,
    directionalAngularDiameter: nextDirectional.directionalAngularDiameter ?? 0,
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
 *
 * `precomputed` — optional already-computed sub-results (same-scene). When provided,
 * the internal `packEmitterArrays` / `environmentParams` calls inside
 * `buildLightTreeInputForScene` are skipped:
 *   - `packed`: result of `packEmitterArrays(scene)` already held by the caller.
 *   - `envSummary`: narrow env metadata from `environmentParams(scene)` held by caller.
 * Incremental callers (updateEmitter / updateEnvironment / updatePrimitive) always
 * hold one or both of these — threading them here avoids a redundant recomputation.
 */
export function rebuildLightTreeForScene(
  device: GPUDevice,
  sb: UploadedSceneBuffers,
  scene: Scene,
  precomputed?: { packed?: PackedEmitterArrays; envSummary?: EnvSummaryForTree },
): boolean {
  const input = buildLightTreeInputForScene(scene, precomputed);
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
    readonly environmentHdriIntensity: number;
    readonly environmentHdriRotationY: number;
    readonly environmentMapWidth: number;
    readonly environmentMapHeight: number;
    readonly hasEnvironmentMap: boolean;
  },
): void {
  const mutable = asMutableSceneBuffers(sb);
  mutable.environmentTint = next.environmentTint;
  mutable.environmentSunDirection = next.environmentSunDirection;
  mutable.environmentSunStrength = next.environmentSunStrength;
  mutable.environmentHdriIntensity = next.environmentHdriIntensity;
  mutable.environmentHdriRotationY = next.environmentHdriRotationY;
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

function materialTextureWarningCode(warning: MaterialTextureArrayWarning): string {
  switch (warning.code) {
    case 'texture-unreadable':
      return 'pt-webgpu.material-texture-unreadable';
    case 'texture-size-mismatch':
      return 'pt-webgpu.material-texture-size-mismatch';
    case 'texture-unsupported-layout':
      return 'pt-webgpu.material-texture-unsupported-layout';
    case 'texture-sampler-policy-approximation':
      return 'pt-webgpu.material-texture-sampler-policy-approximation';
  }
}

function materialTextureEngineWarnings(
  warnings: readonly MaterialTextureArrayWarning[],
  colorSpace: 'sRGB' | 'linear',
): readonly EngineWarning[] {
  return warnings.map((warning) => {
    const messageWarning = `material texture array (${colorSpace}): ${warning.warning}`;
    const materialIndices = Array.from(new Set(warning.uses.map((use) => use.materialIndex)));
    const fields = Array.from(new Set(warning.uses.map((use) => use.field)));
    return {
      code: materialTextureWarningCode(warning),
      backend: 'pt-webgpu',
      phase: 'setScene',
      method: 'setScene',
      message: `[vitrum/pt-webgpu] ${messageWarning}`,
      details: {
        warning: messageWarning,
        colorSpace,
        layer: warning.layer,
        uses: warning.uses,
        materialIndices,
        fields,
        fallback: warning.fallback,
        width: warning.width,
        height: warning.height,
        arrayWidth: warning.arrayWidth,
        arrayHeight: warning.arrayHeight,
        byteLength: warning.byteLength,
        requestedSamplerPolicies: warning.requestedSamplerPolicies,
      },
    };
  });
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
  // V2-2: failure cleanup. uploadPackedScene creates ~30 GPU resources
  // sequentially; a mid-sequence throw (e.g. createMaterialTextureArray on a
  // malformed texture, or a buffer-size validation error) previously leaked every
  // resource already created. Track every created buffer/texture and, on any throw
  // in the body, destroy them all and rethrow so the caller (setScene) sees the
  // error with no leaked GPU memory. `createStorageBuffer` is shadowed by a local
  // tracking wrapper so all 27 storage-buffer creates below register automatically;
  // the two material texture arrays register their textures explicitly.
  const createdResources: Array<{ destroy: () => void }> = [];
  const trackedCreateStorageBuffer = (
    dev: GPUDevice,
    label: string,
    data: ArrayBufferView,
  ): GPUBuffer => {
    const buf = createStorageBuffer(dev, label, data);
    createdResources.push(buf);
    return buf;
  };
  const trackMaterialArray = (arr: MaterialTextureArray): MaterialTextureArray => {
    // GPUSampler has no destroy(); only the texture holds device memory.
    createdResources.push(arr.texture);
    return arr;
  };
  try {
    return uploadPackedSceneInner(device, packed, trackedCreateStorageBuffer, trackMaterialArray);
  } catch (err) {
    for (const r of createdResources) {
      try {
        r.destroy();
      } catch {
        // best-effort teardown — swallow secondary destroy errors so the original
        // upload failure propagates.
      }
    }
    throw err;
  }
}

function uploadPackedSceneInner(
  device: GPUDevice,
  packed: PackedSceneData,
  createStorageBuffer: (dev: GPUDevice, label: string, data: ArrayBufferView) => GPUBuffer,
  trackMaterialArray: (arr: MaterialTextureArray) => MaterialTextureArray,
): UploadedSceneBuffers {
  // P2 — the two material texture arrays + the UV-fit descriptor patch MUST run
  // before the `materialTexDescriptors` storage buffer is created, because
  // `applyMaterialTextureUvFitScales` mutates `packed.materialTexDescriptors`
  // in place (with the arrays' per-layer source rects) and the buffer is written
  // from that array at creation time. Everything else is order-independent, so
  // the storage buffers are created by a single registry-driven loop below.
  const materialTextureArray = trackMaterialArray(createMaterialTextureArray(
    device,
    packed.materialTextureSources,
    'rgba8unorm-srgb',
    packed.materialTextureSourceInfos,
  ));
  // Linear array (normal + ORM) — rgba8unorm so the sampler does NOT sRGB-decode.
  const materialLinearArray = trackMaterialArray(createMaterialTextureArray(
    device,
    packed.materialTextureLinearSources,
    'rgba8unorm',
    packed.materialTextureLinearSourceInfos,
  ));
  applyMaterialTextureUvFitScales(
    packed.materialTexDescriptors,
    materialTextureArray.layerUvScales,
    materialLinearArray.layerUvScales,
  );

  // D8.7 / T2-A — create every scene storage buffer from the single-source
  // SCENE_BUFFER_REGISTRY. Each entry maps a `PackedSceneData` field (`key`)
  // to the `UploadedSceneBuffers` GPUBuffer handle (`bufferField`) + label.
  // Driving the create loop off the registry (instead of ~30 hand-written
  // `createStorageBuffer` lines) keeps the buffer set single-sourced with the
  // destroy closure + gpuMemoryBytes below. `createStorageBuffer` is the tracked
  // wrapper, so each create still registers for the atomic-rollback path.
  // Registry order == the previous hand-written create order (BLAS geometry →
  // CWBVH → analytic → env → emitters → light-tree → P2 UVs/tangents/colors/
  // descriptors → TLAS), so the sequence GPU debuggers see is unchanged.
  const buffers = {} as Record<(typeof SCENE_BUFFER_REGISTRY)[number]['bufferField'], GPUBuffer>;
  for (const entry of SCENE_BUFFER_REGISTRY) {
    const data = packed[entry.key] as ArrayBufferView;
    buffers[entry.bufferField] = createStorageBuffer(device, entry.label, data);
  }

  // Surface texture-array warnings (heterogeneous source sizes → wrong UVs, or an
  // unusable image). Accumulate them onto UploadedSceneBuffers.warnings so the
  // engine drains them through its structured warning/onWarning path.
  const materialTextureWarnings = [
    ...materialTextureArray.warnings.map((w) => `material texture array (sRGB): ${w}`),
    ...materialLinearArray.warnings.map((w) => `material texture array (linear): ${w}`),
  ];
  const materialTextureStructuredWarnings = [
    ...materialTextureEngineWarnings(materialTextureArray.structuredWarnings, 'sRGB'),
    ...materialTextureEngineWarnings(materialLinearArray.structuredWarnings, 'linear'),
  ];

  const uploaded: UploadedSceneBuffers = {
    ...packed,
    warnings: [...packed.warnings, ...materialTextureWarnings],
    structuredWarnings: [...packed.structuredWarnings, ...materialTextureStructuredWarnings],
    bvhNodeCount: Math.floor(packed.bvhNodes.length / BVH_NODE_FLOATS),
    tlasNodeCount: Math.floor(packed.tlasNodes.length / BVH_NODE_FLOATS),
    materialCount: Math.floor(packed.materials.length / MATERIAL_FLOAT_STRIDE),
    ...buffers,
    materialTexture: materialTextureArray.texture,
    materialTextureView: materialTextureArray.view,
    materialTextureSampler: materialTextureArray.sampler,
    materialLinearTexture: materialLinearArray.texture,
    materialLinearTextureView: materialLinearArray.view,
    // T2-A — every scene storage buffer is destroyed via a single registry-driven
    // loop reading `uploaded[bufferField]`. Resolving each handle LATE off
    // `uploaded` (not a captured local) is required for the realloc fast paths
    // that swap fresh handles onto the struct — {@link uploadScenePackTlasRealloc}
    // (instance-count change), {@link uploadScenePackGeometryRealloc} (mesh
    // vertex/index-count change), {@link rebuildLightTreeForScene} (node-count
    // change) — so `destroy` never touches a stale (leaked/double-freed) handle.
    // Non-resized buffers (materials / analytic / environment / descriptors) are
    // never reassigned, so reading them off `uploaded` is identical to the
    // previously-captured locals. The two material texture arrays are destroyed
    // explicitly (textures are not in the buffer registry).
    destroy: () => {
      for (const entry of SCENE_BUFFER_REGISTRY) {
        uploaded[entry.bufferField].destroy();
      }
      materialTextureArray.texture.destroy();
      materialLinearArray.texture.destroy();
    },
    // T2-A — sum the CURRENT GPUBuffer sizes via the same registry (realloc-swapped
    // handles included) + the two material texture arrays (GPUTexture has no
    // `.size`, so derive w·h·layers·4 at rgba8 = 4 B/texel, keyed by the actual
    // format). Entries flagged `excludeFromMemorySum` are skipped to preserve the
    // exact pre-registry behavior (meshAreaLightSourceFactors was historically not
    // counted here). Keeps `debug.estimatedGpuMemoryBytes` byte-stable.
    gpuMemoryBytes: () => {
      let bufferBytes = 0;
      for (const entry of SCENE_BUFFER_REGISTRY) {
        if ('excludeFromMemorySum' in entry && entry.excludeFromMemorySum) continue;
        bufferBytes += uploaded[entry.bufferField].size;
      }
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
