import {
  analyticPrimitiveToMesh,
  asMat4,
  partitionSceneBySupport,
  type EngineWarning,
  type MaterialSpec,
  type Scene,
  type ScenePrimitive,
} from '@vitrum/core';
import {
  BVH_NODE_FLOATS,
  CWBVH_CHILD_META_WORDS,
  CWBVH_CHILD_COUNT_INVALID,
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
import {
  MATERIAL_FLOAT_STRIDE,
  THIN_FILM_LAYER_LIMIT,
  materialToPackedVec4s,
  thinFilmRgbLutForMaterial,
} from './materialPacking.js';
import {
  applyMaterialTextureUvFitScales,
  collectMaterialTextures,
  type MaterialTextureLayerInfo,
  MATERIAL_TEX_FLOAT_STRIDE,
} from './materialTextures.js';
import {
  hasActiveMorphTargets,
  solveSkinnedPrimitive,
} from './solveSkinnedPrimitive.js';
import { packGpuUvSets, type GpuUvRange } from './gpuUvPacking.js';
import {
  assertMneeInterfaceDomainSupported,
  buildMneeFacetCandidateTable,
} from './mneeFacetCandidates.js';
import {
  createMaterialTextureArray,
  estimateMaterialTextureArrayPeakBytes,
  MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES,
  type MaterialTextureArray,
  type MaterialTextureArrayWarning,
} from './materialTextureArray.js';
import { environmentParams } from './environmentPacking.js';
import {
  buildLightTreeInputForScene,
  packEmitterArrays,
  type EnvSummaryForTree,
  type PackedEmitterArrays,
} from './emitterPacking.js';
import { assertLiteSceneSupported } from './liteSceneWarnings.js';
import {
  PT_WEBGPU_FULL_SUPPORT,
  ptWebgpuSupportManifest,
  ptWebgpuSupportSets,
} from '../supportManifest.js';
import {
  collectUnsupportedMaterialFieldsForTraceTier,
} from '../supportDetails.js';

const ANALYTIC_MATERIAL_TEXTURE_FIELDS = [
  'baseColorMap',
  'normalMap',
  'roughnessMap',
  'metallicMap',
  'transmissionMap',
  'thicknessMap',
  'emissiveMap',
  'alphaMap',
  'aoMap',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'sheenColorMap',
  'sheenRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'anisotropyMap',
  'specularColorMap',
  'specularIntensityMap',
  'bumpMap',
  'displacementMap',
  'lightMap',
] as const satisfies readonly (keyof MaterialSpec)[];

/**
 * Native analytic hits do not carry triangle barycentrics or mesh UV streams.
 * Keep that limitation explicit: mapped analytics must be rendered through
 * their authored fallback mesh rather than reaching the texture helpers, where
 * their synthetic triangle index would otherwise make every map sample white.
 */
export function analyticMaterialTextureFields(
  primitive: ScenePrimitive,
): readonly string[] {
  if (primitive.kind !== 'analytic') return [];
  const fields = ANALYTIC_MATERIAL_TEXTURE_FIELDS.filter(
    (field) => primitive.material[field] != null,
  ) as string[];
  if (primitive.material.frontLayer?.normalMap != null) {
    fields.push('frontLayer.normalMap');
  }
  if (primitive.material.backLayer?.normalMap != null) {
    fields.push('backLayer.normalMap');
  }
  return fields;
}

/** True when any authored analytic must use its mesh fallback for texturing. */
export function sceneHasMappedAnalytic(scene: Scene): boolean {
  return scene.primitives.some(
    (primitive) => analyticMaterialTextureFields(primitive).length > 0,
  );
}

function applyAnalyticTextureFallbacks(
  scene: Scene,
  warnings: string[],
): Scene {
  let changed = false;
  const primitives = scene.primitives.map((primitive) => {
    const textureFields = analyticMaterialTextureFields(primitive);
    if (textureFields.length === 0 || primitive.kind !== 'analytic') {
      return primitive;
    }
    if (primitive.fallbackMesh == null) {
      throw new TypeError(
        `@vitrum/pt-webgpu: analytic primitive "${primitive.id}" uses material ` +
          `texture maps [${textureFields.join(', ')}], but native analytic hits ` +
          'do not expose mesh UVs and no fallbackMesh was supplied.',
      );
    }
    changed = true;
    warnings.push(
      `Analytic primitive "${primitive.id}" uses material texture maps ` +
        `[${textureFields.join(', ')}]; rendering its fallbackMesh so authored ` +
        'UV streams are sampled.',
    );
    return analyticPrimitiveToMesh(primitive);
  });
  return changed ? { ...scene, primitives } : scene;
}

// 8 dead re-exports of MAX_*_LIGHTS / *_FLOAT_STRIDE constants (originally
// surfaced here for hosts that might assemble emitter arrays directly) were
// removed 2026-05-18 — no in-tree consumer reached them. The canonical
// definitions stay in `./emitterPacking.ts` and are file-locally used.

interface PackedSceneData {
  readonly positions: Float32Array; // vec4f packed
  readonly normals: Float32Array; // vec4f packed
  readonly uvs: Float32Array; // primary uv0/uv1 plane + compact arbitrary-UV tail planes
  /** GPU UV slot -> authored TextureRef.texCoord; slots 0/1 are ABI-stable. */
  readonly uvSetTexCoords: readonly number[];
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
  /** Dedup'd, upload-ordered EMISSIVE texture handles → a dedicated rgba16float
   *  texture_2d_array so HDR emissive texture values survive packing (the sRGB
   *  8-bit array clamped them to [0,1]). LDR sources are sRGB→linear decoded on
   *  upload so they stay visually identical to the previous sRGB-array path. */
  readonly materialTextureEmissiveSources: readonly unknown[];
  /** Provenance for each EMISSIVE material texture layer, for structured upload warnings. */
  readonly materialTextureEmissiveSourceInfos: readonly MaterialTextureLayerInfo[];
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
  /** CWBVH forest: parent wide-node bounds, 6 f32 per node. */
  readonly cwbvhNodeBounds: Float32Array;
  /** CWBVH forest: child bounds as packed u16 pairs (3 u32 / child). */
  readonly cwbvhChildBoundsPacked: Uint32Array;
  /** CWBVH forest: child kind / wide-node-or-triangle offset / count. */
  readonly cwbvhChildMeta: Uint32Array;
  /** CWBVH forest: number of live child slots per wide node. */
  readonly cwbvhChildCount: Uint32Array;
  /** TLAS root records: magic, canonical binary root, CWBVH root, integrity word. */
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
  /** CPU-only power proxy retained so emitter-only mutations rebuild the light
   * tree from the same environment integral without rebaking the map. */
  readonly environmentLightTreePower: number;
  /**
   * H14-E: map-backed environment-radiance intensity multiplier. The legacy
   * `environmentSunStrength` field is retained for stable packed-scene state,
   * but shader environment evaluation no longer consumes it.
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
  /** T1-6 — EMISSIVE rgba16float texture_2d_array handle (HDR emissive; for dispose). */
  readonly materialEmissiveTexture: GPUTexture;
  /** T1-6 — emissive rgba16float 2d-array view bound in group 3 (binding 17). */
  readonly materialEmissiveTextureView: GPUTextureView;
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
 *     the render-consumed subset. CWBVH buffers are uploaded beside the binary
 *     BVH and bound by the full-tier `bvhTraversal:'cwbvh-closest'` path.
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


interface BufferRewrite {
  readonly buffer: GPUBuffer;
  readonly next: ArrayBufferView;
  readonly previous: ArrayBufferView;
}

/**
 * Apply a same-size buffer set and return a rollback closure.  Synchronous
 * injected queue failures are undone immediately; the returned closure lets a
 * later dependent allocation/upload failure restore this set as well.
 */
function writeBufferSetWithRollback(
  device: GPUDevice,
  rewrites: readonly BufferRewrite[],
): () => void {
  const uniqueBuffers = new Set<GPUBuffer>();
  for (const rewrite of rewrites) {
    if (uniqueBuffers.has(rewrite.buffer)) {
      throw new Error(
        '[pt-webgpu] transactional write set aliases one GPUBuffer across multiple fields',
      );
    }
    uniqueBuffers.add(rewrite.buffer);
  }
  const written: BufferRewrite[] = [];
  const restore = (entries: readonly BufferRewrite[]): void => {
    for (let i = entries.length - 1; i >= 0; i -= 1) {
      const entry = entries[i]!;
      try { writeBufferIfNonEmpty(entry.buffer, entry.previous, device); } catch { /* best effort */ }
    }
  };
  try {
    for (const rewrite of rewrites) {
      writeBufferIfNonEmpty(rewrite.buffer, rewrite.next, device);
      written.push(rewrite);
    }
  } catch (error) {
    restore(written);
    throw error;
  }
  let rolledBack = false;
  return () => {
    if (rolledBack) return;
    rolledBack = true;
    restore(rewrites);
  };
}
const BVH_NODE_BUFFER_BYTES = BVH_NODE_FLOATS * Float32Array.BYTES_PER_ELEMENT;

const MIN_STORAGE_BUFFER_BYTES_BY_LABEL: Readonly<Record<string, number>> = {
  'vitrum.pt-webgpu.scene.bvhNodes': BVH_NODE_BUFFER_BYTES,
  'vitrum.pt-webgpu.scene.tlasNodes': BVH_NODE_BUFFER_BYTES,
};

function createStorageBuffer(
  device: GPUDevice,
  label: string,
  data: ArrayBufferView,
  forbiddenResources: ReadonlySet<object> = new Set(),
): GPUBuffer {
  const minSize = Math.max(data.byteLength, MIN_STORAGE_BUFFER_BYTES_BY_LABEL[label] ?? 16);
  const buffer = device.createBuffer({
    label,
    size: Math.ceil(minSize / 4) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  try {
    if (forbiddenResources.has(buffer)) {
      throw new Error(
        `[pt-webgpu] candidate allocation for ${label} aliased an existing GPU resource`,
      );
    }
    if (data.byteLength > 0) {
      device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    } else {
      device.queue.writeBuffer(buffer, 0, new Uint32Array(Math.ceil(minSize / Uint32Array.BYTES_PER_ELEMENT)));
    }
    return buffer;
  } catch (error) {
    // A buffer is already live once createBuffer returns.  Keep upload failure
    // atomic even when queue.writeBuffer is the failing stage: callers cannot
    // register the candidate for rollback until this helper returns.
    try {
      // An aliased return value belongs to the existing/live set. It must not
      // be destroyed merely because a non-conformant device returned it again.
      if (!forbiddenResources.has(buffer)) buffer.destroy();
    } catch {
      // Preserve the original upload error.
    }
    throw error;
  }
}

function destroyResourcesBestEffort(
  resources: readonly { destroy(): void }[],
  preservedResources: readonly object[] = [],
): void {
  const destroyed = new Set<object>(preservedResources);
  for (const resource of resources) {
    if (destroyed.has(resource)) continue;
    destroyed.add(resource);
    try { resource.destroy(); } catch { /* preserve the transaction outcome */ }
  }
}

interface StorageBufferCandidateSpec {
  readonly key: string;
  readonly label: string;
  readonly data: ArrayBufferView;
}

function createStorageBufferCandidates(
  device: GPUDevice,
  specs: readonly StorageBufferCandidateSpec[],
  preservedResources: readonly object[] = [],
): Readonly<Record<string, GPUBuffer>> {
  const created: GPUBuffer[] = [];
  const result: Record<string, GPUBuffer> = {};
  const forbiddenResources = new Set<object>(preservedResources);
  try {
    for (const spec of specs) {
      const buffer = createStorageBuffer(device, spec.label, spec.data, forbiddenResources);
      created.push(buffer);
      forbiddenResources.add(buffer);
      result[spec.key] = buffer;
    }
    return result;
  } catch (error) {
    destroyResourcesBestEffort(created, preservedResources);
    throw error;
  }
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

/**
 * Backward-compatible name for the full-tier support sets. The sets are
 * derived from the exhaustive backend-local manifest; this file no longer
 * owns a parallel kind list.
 */
export const PT_WEBGPU_SUPPORT = PT_WEBGPU_FULL_SUPPORT;

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
   * behaviour. For a primitive explicitly owned by a mesh-area emitter, `false`
   * zeros surface-hit emission while retaining the emitter's NEE proposal; an
   * unrelated or implicit emissive material keeps its authored emission.
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

  /**
   * Include the compact valid-facet table consumed by manifold NEE. The table
   * is appended after the real analytic parameter records in the same storage
   * allocation, so no extra shader binding is required.
   */
  readonly includeMneeFacetCandidates?: boolean;
  /** Device-derived byte ceiling for the shared analytic-params allocation. */
  readonly mneeFacetCandidateStorageLimitBytes?: number;

  /** Structured warning sink used by engine-owned scene ingestion. */
  readonly onWarning?: (warning: EngineWarning) => void;
  readonly warningPhase?: EngineWarning['phase'];
  readonly warningMethod?: string;
}

/**
 * H10 — Pack the GPU material floats for `primitive`, applying the emissive
 * fold when the primitive is the mesh referenced by a mesh-area emitter in
 * `scene`. With `cameraVisibleEmitters=true`, the explicit emitter's authored
 * radiance is folded into the surface material. With it disabled, that explicit
 * emitter-owned surface emission is zeroed so camera/refraction forward hits
 * cannot evaluate a different `primitive.material` integrand than NEE/MIS.
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
  const matchingEmitters = scene.emitters.filter(
    (emitter) => emitter.kind === 'mesh-area' && emitter.meshId === primitive.id,
  );
  if (matchingEmitters.length > 1) {
    throw new TypeError(
      `@vitrum/pt-webgpu: mesh primitive "${primitive.id}" is referenced by multiple ` +
        `mesh-area emitters (${matchingEmitters.map((emitter) => `"${emitter.id}"`).join(', ')}). ` +
        'A surface may have only one explicit mesh-area emitter so forward-hit radiance ' +
        'and its MIS proposal have one unambiguous owner.',
    );
  }
  const emitter = matchingEmitters[0];
  if (emitter != null) {
    // Pre-multiply intensity so `emissive · emissiveIntensity` == NEE radiance.
    // Keep the primitive's emissiveMap: both the forward hit and mapped proposal
    // apply that same texture sample to this scalar/base radiance.
    const foldedMat = {
      ...primitive.material,
      emissive: cameraVisibleEmitters
        ? [
            emitter.color[0] * emitter.intensity,
            emitter.color[1] * emitter.intensity,
            emitter.color[2] * emitter.intensity,
          ] as [number, number, number]
        : [0, 0, 0] as [number, number, number],
      emissiveIntensity: 1,
    };
    return materialToPackedVec4s(foldedMat, packContext);
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
 * positions/normals/tangents/uvs/uvSets so that packSceneFromCore uses the correct
 * deformed geometry, tangent frame, and texture coordinates.
 */
export function applySolveSkinToScene(scene: Scene): Scene {
  let anyChanged = false;
  const nextPrimitives = scene.primitives.map((p) => {
    if (p.kind !== 'skinned-mesh') return p;
    if (p.bones.length === 0 && !hasActiveMorphTargets(p)) return p;
    try {
      const solved = solveSkinnedPrimitive(p);
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
        ...(solved.uvSets ? { uvSets: solved.uvSets } : {}),
      };
    } catch (err) {
      throw new Error(
        `[vitrum/pt-webgpu] solveSkin failed for primitive "${p.id}"; ` +
          `scene upload was rejected before GPU mutation. ${String(err)}`,
      );
    }
  });
  if (!anyChanged) return scene;
  return { ...scene, primitives: nextPrimitives };
}

/**
 * The coherent TMM stack has a finite shader ABI. Reject excess layers before
 * material packing so authored optical interfaces are never silently truncated.
 */
function assertThinFilmLayerCapacity(scene: Scene): void {
  for (const primitive of scene.primitives) {
    const layers = primitive.material.thinFilmStack?.layers;
    const requested = layers?.length ?? 0;
    if (requested <= THIN_FILM_LAYER_LIMIT) continue;
    throw new Error(
      `[vitrum/pt-webgpu] thin-film scene validation: primitive "${primitive.id}" ` +
        `declares ${requested} coherent layers, but this backend's exact limit is ` +
        `${THIN_FILM_LAYER_LIMIT}; scene upload was rejected before GPU mutation.`,
    );
  }
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

/** @internal Binary/wide-root record layout shared with the CWBVH WGSL binding. */
export const CWBVH_ROOT_PAIR_MAGIC = 0x43574256;
/** @internal */
export const CWBVH_ROOT_PAIR_WORDS = 4;
const CWBVH_BINARY_ROOT_FACTOR = 0x9e3779b1;
const CWBVH_WIDE_ROOT_FACTOR = 0x85ebca6b;

/** @internal */
export function packCwbvhRootPair(binaryRoot: number, wideRoot: number): Uint32Array {
  if (
    !Number.isInteger(binaryRoot) || binaryRoot < 0 || binaryRoot >= CWBVH_CHILD_COUNT_INVALID ||
    !Number.isInteger(wideRoot) || wideRoot < 0 || wideRoot >= CWBVH_CHILD_COUNT_INVALID
  ) {
    throw new RangeError('[pt-webgpu] CWBVH root pair indices must be valid u32 values below the invalid sentinel');
  }
  const checksum = (
    CWBVH_ROOT_PAIR_MAGIC ^
    Math.imul(binaryRoot, CWBVH_BINARY_ROOT_FACTOR) ^
    Math.imul(wideRoot, CWBVH_WIDE_ROOT_FACTOR)
  ) >>> 0;
  return new Uint32Array([CWBVH_ROOT_PAIR_MAGIC, binaryRoot, wideRoot, checksum]);
}

/** @internal */
export function isValidCwbvhRootPair(record: Uint32Array, offset = 0): boolean {
  if (!Number.isInteger(offset) || offset < 0 || offset + CWBVH_ROOT_PAIR_WORDS > record.length) {
    return false;
  }
  const magic = record[offset]!;
  const binaryRoot = record[offset + 1]!;
  const wideRoot = record[offset + 2]!;
  const checksum = record[offset + 3]!;
  return magic === CWBVH_ROOT_PAIR_MAGIC &&
    binaryRoot !== CWBVH_CHILD_COUNT_INVALID &&
    wideRoot !== CWBVH_CHILD_COUNT_INVALID &&
    checksum === (magic ^ Math.imul(binaryRoot, CWBVH_BINARY_ROOT_FACTOR) ^
      Math.imul(wideRoot, CWBVH_WIDE_ROOT_FACTOR)) >>> 0;
}

function remapCwbvhTlasBlasRoots(
  tlasBlasRoots: Uint32Array,
  binaryRootToWideRoot: Uint32Array,
): Uint32Array {
  const out = new Uint32Array(tlasBlasRoots.length * CWBVH_ROOT_PAIR_WORDS);
  for (let i = 0; i < tlasBlasRoots.length; i += 1) {
    const binaryRoot = tlasBlasRoots[i] ?? 0;
    const wideRoot = binaryRootToWideRoot[binaryRoot];
    if (wideRoot === undefined || wideRoot === CWBVH_CHILD_COUNT_INVALID) {
      throw new Error(
        `[pt-webgpu] binary BLAS root ${binaryRoot} has no valid CWBVH root mapping`,
      );
    }
    const pair = packCwbvhRootPair(binaryRoot, wideRoot);
    out.set(pair, i * CWBVH_ROOT_PAIR_WORDS);
  }
  return out;
}

/**
 * Build the pt-webgpu CWBVH forest beside the canonical binary BVH.
 *
 * Full-tier TLAS scenes concatenate one CWBVH tree per BLAS
 * subtree and remap `tlasBlasRoots` to wide-node roots, while merged/lite
 * scenes produce a single root-0 wide tree. Leaf metadata keeps the existing
 * global triangle offsets, so the traversal reads the same `indices`,
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

  binaryRootToWideRoot.fill(CWBVH_CHILD_COUNT_INVALID);
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

function assertKnownManifestDiscriminators(
  scene: Scene,
  manifest: ReturnType<typeof ptWebgpuSupportManifest>,
): void {
  for (const primitive of scene.primitives) {
    if (
      primitive.kind === 'analytic' &&
      !Object.prototype.hasOwnProperty.call(
        manifest.analyticShapes,
        primitive.shape,
      )
    ) {
      throw new TypeError(
        `@vitrum/pt-webgpu: scene contains unsupported content: ` +
        `Scene primitive "${primitive.id}" has unknown analytic shape ` +
        `"${String(primitive.shape)}".`,
      );
    }
  }
}

function assertManifestMaterialDomain(
  scene: Scene,
  traceTier: 'full' | 'lite',
): void {
  const violations: string[] = [];
  for (const primitive of scene.primitives) {
    const fields = collectUnsupportedMaterialFieldsForTraceTier(
      primitive.material,
      traceTier,
    );
    if (fields.length > 0) {
      violations.push(`"${primitive.id}" [${fields.join(', ')}]`);
    }
  }
  if (violations.length > 0) {
    throw new TypeError(
      `@vitrum/pt-webgpu: scene contains unsupported content for the ` +
      `${traceTier} tier: material fields ${violations.join('; ')}.`,
    );
  }
}

export function buildPackedScene(
  inputScene: Scene,
  options: BuildPackedSceneOptions = {},
): PackedSceneData {
  const geometryMode = options.geometryMode ?? 'tlas';
  const traceTier = geometryMode === 'merged' ? 'lite' : 'full';
  const supportManifest = ptWebgpuSupportManifest(traceTier);
  const supportSets = ptWebgpuSupportSets(traceTier);
  // Preserve the detailed lite-domain diagnostic for known unsupported
  // geometry/emitter families before generic partitioning.
  if (traceTier === 'lite') assertLiteSceneSupported(inputScene);
  assertKnownManifestDiscriminators(inputScene, supportManifest);
  assertManifestMaterialDomain(inputScene, traceTier);
  // Capability preflight. Partitioning is used only to identify unsupported
  // content; silently packing the supported subset would change the scene.
  const { supported: filteredScene, warnings } =
    partitionSceneBySupport(inputScene, supportSets);
  if (warnings.length > 0) {
    throw new TypeError(
      `@vitrum/pt-webgpu: scene contains unsupported content: ${warnings.join(' | ')}`,
    );
  }
  const textureSafeScene = applyAnalyticTextureFallbacks(filteredScene, warnings);
  assertThinFilmLayerCapacity(textureSafeScene);
  // Item 1 — apply CPU LBS to skinned-mesh primitives so packSceneFromCore uses
  // solved (deformed) positions instead of rest-pose. morphTargets are also
  // handled by solveSkin (blend applied before LBS).
  const scene = applySolveSkinToScene(textureSafeScene);
  if (options.includeMneeFacetCandidates) {
    assertMneeInterfaceDomainSupported(scene);
  }

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

  let gpuUvRanges: readonly GpuUvRange[] = [];
  const emitterMaterialIds = new Map(meshMaterialIds);
  const geo = geometryMode === 'merged'
    ? (() => {
        const merged = mergeWorldSpaceFromCore(scene, {
          positionStride: 4,
          splitMaterialsByCastShadow: true,
          bakeConstantVertexColorIntoMaterial: true,
        });
        for (const range of merged.meshVertexRanges) {
          const primitiveId = range.sourcePrimitiveId ?? range.name;
          const materialId = merged.mergedTriMaterialId[range.triStart];
          if (materialId != null && !emitterMaterialIds.has(primitiveId)) {
            emitterMaterialIds.set(primitiveId, materialId);
          }
        }
        for (const material of merged.materials) {
          const withShadow = material as MaterialSpec & { readonly castShadow?: boolean };
          materials.push(...packMergedMaterial(withShadow));
          materialSpecs.push(withShadow);
        }
        gpuUvRanges = merged.meshVertexRanges;
        return {
          positions: merged.positions,
          normals: merged.normals,
          uvs: packMergedUvs(scene, merged),
          tangents: merged.tangents,
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
    : (() => {
        const packed = packSceneFromCore(scene, {
          tlas: true,
          resolveMaterialId: (id) => meshMaterialIds.get(id) ?? 0,
        });
        gpuUvRanges = packed.primitiveTlasBindings;
        return packed;
      })();
  let mneeFacetCandidateRecords: Float32Array | undefined;
  if (options.includeMneeFacetCandidates) {
    if (geometryMode !== 'tlas') {
      throw new Error(
        '@vitrum/pt-webgpu: manifold-nee facet candidates require the full TLAS tier.',
      );
    }
    const storageLimit = options.mneeFacetCandidateStorageLimitBytes;
    if (storageLimit == null || !Number.isFinite(storageLimit) || storageLimit < 16) {
      throw new Error(
        '@vitrum/pt-webgpu: manifold-nee requires a finite device storage-buffer byte limit.',
      );
    }
    const table = buildMneeFacetCandidateTable(
      scene,
      geo.primitiveTlasBindings,
      storageLimit,
      analyticParams.length * Float32Array.BYTES_PER_ELEMENT,
    );
    mneeFacetCandidateRecords = table.records;
  }
  const packedAnalyticParams = new Float32Array(
    analyticParams.length + (mneeFacetCandidateRecords?.length ?? 0),
  );
  packedAnalyticParams.set(analyticParams);
  if (mneeFacetCandidateRecords != null) {
    packedAnalyticParams.set(mneeFacetCandidateRecords, analyticParams.length);
  }
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
  const gpuUvs = packGpuUvSets(
    scene,
    geo.uvs,
    gpuUvRanges,
    texCollection.uvSetTexCoords,
  );
  const emitArrays = packEmitterArrays(scene, {
    materialIdByPrimitive: emitterMaterialIds,
  });
  const environment = environmentParams(scene);
  // RGB thin-film LUTs are a sparse tail after all fixed material records.
  // `matId * MATERIAL_FLOAT_STRIDE` therefore remains valid, and non-film
  // scenes retain the exact pre-thin-film buffer size. Vec4 #28.z stores the
  // absolute vec4 offset for film materials (zero is the no-LUT sentinel).
  const fixedMaterialFloatCount = materials.length;
  if (fixedMaterialFloatCount !== materialSpecs.length * MATERIAL_FLOAT_STRIDE) {
    throw new Error('pt-webgpu fixed material-table packing invariant failed.');
  }
  const thinFilmRgbLutTail: number[] = [];
  for (let matId = 0; matId < materialSpecs.length; matId += 1) {
    const lut = thinFilmRgbLutForMaterial(materialSpecs[matId]!);
    if (lut.length === 0) continue;
    const absoluteBaseVec4 =
      fixedMaterialFloatCount / 4 + thinFilmRgbLutTail.length / 4;
    // Numeric f32 represents every integer through 2^24 exactly. Validate the
    // descriptor before conversion so WGSL's round()+u32 recovers it bit-exactly.
    if (!Number.isSafeInteger(absoluteBaseVec4) || absoluteBaseVec4 >= 2 ** 24) {
      throw new Error(
        'pt-webgpu thin-film RGB LUT offset exceeds exact f32 integer range.',
      );
    }
    materials[matId * MATERIAL_FLOAT_STRIDE + 28 * 4 + 2] = absoluteBaseVec4;
    thinFilmRgbLutTail.push(...lut);
  }
  materials.push(...thinFilmRgbLutTail);

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
    lightTreePower: environment.lightTreePower,
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
    uvs: gpuUvs,
    uvSetTexCoords: texCollection.uvSetTexCoords,
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
    materialTextureEmissiveSources: texCollection.emissiveSources,
    materialTextureEmissiveSourceInfos: texCollection.emissiveSourceInfos,
    bvhNodes: geo.bvhNodes,
    tlasNodes: geo.tlasNodes,
    tlasInstanceIndices: geo.tlasInstanceIndices,
    tlasBlasRoots: geo.tlasBlasRoots,
    tlasInstanceWorldToLocal: geo.tlasInstanceWorldToLocal,
    tlasInstanceLocalToWorld: geo.tlasInstanceLocalToWorld,
    primitiveTlasBindings: geo.primitiveTlasBindings,
    analyticHeaders: new Float32Array(analyticHeaders),
    analyticParams: packedAnalyticParams,
    analyticLocalToWorld: new Float32Array(analyticLocalToWorld),
    analyticWorldToLocal: new Float32Array(analyticWorldToLocal),
    triangleCount: geo.triangleCount,
    analyticCount: Math.floor(analyticHeaders.length / 4),
    warnings,
    structuredWarnings,
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
    environmentLightTreePower: environment.lightTreePower,
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
  const primaryUvFloatCount = packed.positions.length;
  return {
    positions: packed.positions,
    normals: packed.normals,
    uvs: packed.uvs.slice(0, primaryUvFloatCount),
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
  const rewrites: BufferRewrite[] = [
    { buffer: sb.positionsBuffer, next: pack.positions, previous: sb.positions },
    { buffer: sb.normalsBuffer, next: pack.normals, previous: sb.normals },
    { buffer: sb.uvsBuffer, next: pack.uvs, previous: sb.uvs },
    { buffer: sb.tangentsBuffer, next: pack.tangents, previous: sb.tangents },
    { buffer: sb.colorsBuffer, next: pack.colors, previous: sb.colors },
    { buffer: sb.indicesBuffer, next: pack.indices, previous: sb.indices },
    { buffer: sb.triMaterialIdsBuffer, next: pack.triMaterialIds, previous: sb.triMaterialIds },
    { buffer: sb.bvhNodesBuffer, next: pack.bvhNodes, previous: sb.bvhNodes },
    { buffer: sb.tlasNodesBuffer, next: pack.tlasNodes, previous: sb.tlasNodes },
    { buffer: sb.tlasInstanceIndicesBuffer, next: pack.tlasInstanceIndices, previous: sb.tlasInstanceIndices },
    { buffer: sb.tlasBlasRootsBuffer, next: pack.tlasBlasRoots, previous: sb.tlasBlasRoots },
    { buffer: sb.tlasInstanceWorldToLocalBuffer, next: pack.tlasInstanceWorldToLocal, previous: sb.tlasInstanceWorldToLocal },
    { buffer: sb.tlasInstanceLocalToWorldBuffer, next: pack.tlasInstanceLocalToWorld, previous: sb.tlasInstanceLocalToWorld },
  ];
  const cwbvh = buildPackedCwbvhSceneData(pack);
  const rollbackBase: { current?: () => void } = {};
  try {
    replaceCwbvhBuffers(device, sb, cwbvh, () => {
      rollbackBase.current = writeBufferSetWithRollback(device, rewrites);
    }, rewrites.map((rewrite) => rewrite.buffer));
  } catch (error) {
    rollbackBase.current?.();
    throw error;
  }
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
  const rewrites: BufferRewrite[] = [
    { buffer: sb.positionsBuffer, next: pack.positions, previous: sb.positions },
    { buffer: sb.normalsBuffer, next: pack.normals, previous: sb.normals },
    { buffer: sb.uvsBuffer, next: pack.uvs, previous: sb.uvs },
    { buffer: sb.tangentsBuffer, next: pack.tangents, previous: sb.tangents },
    { buffer: sb.colorsBuffer, next: pack.colors, previous: sb.colors },
    { buffer: sb.indicesBuffer, next: pack.indices, previous: sb.indices },
    { buffer: sb.triMaterialIdsBuffer, next: pack.triMaterialIds, previous: sb.triMaterialIds },
    { buffer: sb.bvhNodesBuffer, next: pack.bvhNodes, previous: sb.bvhNodes },
  ];
  const cwbvh = buildPackedCwbvhSceneData({
    positions: pack.positions,
    indices: pack.indices,
    triMaterialIds: pack.triMaterialIds,
    bvhNodes: pack.bvhNodes,
    triangleCount: pack.triangleCount,
    tlasBlasRoots: sb.tlasBlasRoots,
    primitiveTlasBindings: pack.primitiveTlasBindings,
  });
  const rollbackBase: { current?: () => void } = {};
  try {
    replaceCwbvhBuffers(device, sb, cwbvh, () => {
      rollbackBase.current = writeBufferSetWithRollback(device, rewrites);
    }, rewrites.map((rewrite) => rewrite.buffer));
  } catch (error) {
    rollbackBase.current?.();
    throw error;
  }
  sb.positions.set(pack.positions);
  sb.normals.set(pack.normals);
  sb.uvs.set(pack.uvs);
  sb.tangents.set(pack.tangents);
  sb.colors.set(pack.colors);
  sb.indices.set(pack.indices);
  sb.triMaterialIds.set(pack.triMaterialIds);
  sb.bvhNodes.set(pack.bvhNodes);
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
  // Build every CPU mirror and every replacement GPU handle before touching the
  // live set.  The CWBVH TLAS-root mirror is part of this transaction: swapping
  // the binary TLAS while leaving a stale/destroyed wide-root table is not a
  // renderable state.
  const nextTlasNodes = new Uint32Array(pack.tlasNodes);
  const nextTlasInstanceIndices = new Uint32Array(pack.tlasInstanceIndices);
  const nextTlasBlasRoots = new Uint32Array(pack.tlasBlasRoots);
  const nextTlasInstanceWorldToLocal = new Float32Array(pack.tlasInstanceWorldToLocal);
  const nextTlasInstanceLocalToWorld = new Float32Array(pack.tlasInstanceLocalToWorld);
  const nextCwbvhTlasBlasRoots = remapCwbvhTlasBlasRoots(
    nextTlasBlasRoots,
    sb.cwbvhBinaryRootToWideRoot,
  );
  const cwbvhRootBuffer = sb.cwbvhTlasBlasRootsBuffer as GPUBuffer | undefined;
  const hasCwbvhRoot = cwbvhRootBuffer != null;
  const candidateSpecs: StorageBufferCandidateSpec[] = [
    { key: 'tlasNodes', label: 'vitrum.pt-webgpu.scene.tlasNodes', data: nextTlasNodes },
    { key: 'tlasInstanceIndices', label: 'vitrum.pt-webgpu.scene.tlasInstanceIndices', data: nextTlasInstanceIndices },
    { key: 'tlasBlasRoots', label: 'vitrum.pt-webgpu.scene.tlasBlasRoots', data: nextTlasBlasRoots },
    { key: 'tlasInstanceWorldToLocal', label: 'vitrum.pt-webgpu.scene.tlasInstanceWorldToLocal', data: nextTlasInstanceWorldToLocal },
    { key: 'tlasInstanceLocalToWorld', label: 'vitrum.pt-webgpu.scene.tlasInstanceLocalToWorld', data: nextTlasInstanceLocalToWorld },
  ];
  const rootLabel = 'vitrum.pt-webgpu.scene.cwbvhTlasBlasRoots';
  const expectedRootSize = Math.ceil(Math.max(
    nextCwbvhTlasBlasRoots.byteLength,
    MIN_STORAGE_BUFFER_BYTES_BY_LABEL[rootLabel] ?? 16,
  ) / 4) * 4;
  const currentRootSize = cwbvhRootBuffer?.size ?? Math.ceil(Math.max(
    sb.cwbvhTlasBlasRoots.byteLength,
    MIN_STORAGE_BUFFER_BYTES_BY_LABEL[rootLabel] ?? 16,
  ) / 4) * 4;
  const replaceRoot = hasCwbvhRoot && currentRootSize !== expectedRootSize;
  if (replaceRoot) {
    candidateSpecs.push({
      key: 'cwbvhTlasBlasRoots',
      label: rootLabel,
      data: nextCwbvhTlasBlasRoots,
    });
  }
  const previous = [
    sb.tlasNodesBuffer,
    sb.tlasInstanceIndicesBuffer,
    sb.tlasBlasRootsBuffer,
    sb.tlasInstanceWorldToLocalBuffer,
    sb.tlasInstanceLocalToWorldBuffer,
  ];
  assertCwbvhTlasRootIsDistinct(sb, previous);
  const liveResources = sceneBufferResourceHandles(sb);
  const candidates = createStorageBufferCandidates(device, candidateSpecs, liveResources);

  if (hasCwbvhRoot && !replaceRoot) {
    try {
      writeBufferIfNonEmpty(
        cwbvhRootBuffer,
        nextCwbvhTlasBlasRoots,
        device,
      );
    } catch (error) {
      try {
        writeBufferIfNonEmpty(
          cwbvhRootBuffer,
          sb.cwbvhTlasBlasRoots,
          device,
        );
      } catch { /* best effort */ }
      destroyResourcesBestEffort(Object.values(candidates), liveResources);
      throw error;
    }
  }

  if (replaceRoot) previous.push(cwbvhRootBuffer);
  const buffers = asMutableSceneBuffers(sb);
  buffers.tlasNodesBuffer = candidates.tlasNodes!;
  buffers.tlasInstanceIndicesBuffer = candidates.tlasInstanceIndices!;
  buffers.tlasBlasRootsBuffer = candidates.tlasBlasRoots!;
  buffers.tlasInstanceWorldToLocalBuffer = candidates.tlasInstanceWorldToLocal!;
  buffers.tlasInstanceLocalToWorldBuffer = candidates.tlasInstanceLocalToWorld!;
  if (replaceRoot) {
    buffers.cwbvhTlasBlasRootsBuffer = candidates.cwbvhTlasBlasRoots!;
  }
  buffers.tlasNodes = nextTlasNodes;
  buffers.tlasInstanceIndices = nextTlasInstanceIndices;
  buffers.tlasBlasRoots = nextTlasBlasRoots;
  buffers.tlasInstanceWorldToLocal = nextTlasInstanceWorldToLocal;
  buffers.tlasInstanceLocalToWorld = nextTlasInstanceLocalToWorld;
  buffers.cwbvhTlasBlasRoots = nextCwbvhTlasBlasRoots;
  applyScenePackCounts(sb, pack);
  destroyResourcesBestEffort(previous, sceneBufferResourceHandles(sb));
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
  const next = {
    positions: new Float32Array(pack.positions),
    normals: new Float32Array(pack.normals),
    uvs: new Float32Array(pack.uvs),
    tangents: new Float32Array(pack.tangents),
    colors: new Float32Array(pack.colors),
    indices: new Uint32Array(pack.indices),
    triMaterialIds: new Uint32Array(pack.triMaterialIds),
    bvhNodes: new Float32Array(pack.bvhNodes),
    tlasNodes: new Uint32Array(pack.tlasNodes),
    tlasInstanceIndices: new Uint32Array(pack.tlasInstanceIndices),
    tlasBlasRoots: new Uint32Array(pack.tlasBlasRoots),
    tlasInstanceWorldToLocal: new Float32Array(pack.tlasInstanceWorldToLocal),
    tlasInstanceLocalToWorld: new Float32Array(pack.tlasInstanceLocalToWorld),
  };
  const cwbvh = buildPackedCwbvhSceneData(pack);
  const liveResources = sceneBufferResourceHandles(sb);
  const candidates = createStorageBufferCandidates(device, [
    { key: 'positions', label: 'vitrum.pt-webgpu.scene.positions', data: next.positions },
    { key: 'normals', label: 'vitrum.pt-webgpu.scene.normals', data: next.normals },
    { key: 'uvs', label: 'vitrum.pt-webgpu.scene.uvs', data: next.uvs },
    { key: 'tangents', label: 'vitrum.pt-webgpu.scene.tangents', data: next.tangents },
    { key: 'colors', label: 'vitrum.pt-webgpu.scene.colors', data: next.colors },
    { key: 'indices', label: 'vitrum.pt-webgpu.scene.indices', data: next.indices },
    { key: 'triMaterialIds', label: 'vitrum.pt-webgpu.scene.triMaterialIds', data: next.triMaterialIds },
    { key: 'bvhNodes', label: 'vitrum.pt-webgpu.scene.bvhNodes', data: next.bvhNodes },
    { key: 'tlasNodes', label: 'vitrum.pt-webgpu.scene.tlasNodes', data: next.tlasNodes },
    { key: 'tlasInstanceIndices', label: 'vitrum.pt-webgpu.scene.tlasInstanceIndices', data: next.tlasInstanceIndices },
    { key: 'tlasBlasRoots', label: 'vitrum.pt-webgpu.scene.tlasBlasRoots', data: next.tlasBlasRoots },
    { key: 'tlasInstanceWorldToLocal', label: 'vitrum.pt-webgpu.scene.tlasInstanceWorldToLocal', data: next.tlasInstanceWorldToLocal },
    { key: 'tlasInstanceLocalToWorld', label: 'vitrum.pt-webgpu.scene.tlasInstanceLocalToWorld', data: next.tlasInstanceLocalToWorld },
  ], liveResources);
  const previous = [
    sb.positionsBuffer, sb.normalsBuffer, sb.uvsBuffer, sb.tangentsBuffer,
    sb.colorsBuffer, sb.indicesBuffer, sb.triMaterialIdsBuffer, sb.bvhNodesBuffer,
    sb.tlasNodesBuffer, sb.tlasInstanceIndicesBuffer, sb.tlasBlasRootsBuffer,
    sb.tlasInstanceWorldToLocalBuffer, sb.tlasInstanceLocalToWorldBuffer,
  ];
  const handles = asMutableSceneBuffers(sb);
  let binaryPublished = false;
  try {
    // replaceCwbvhBuffers first allocates every size-changing wide-BVH
    // candidate. Only then does this callback publish the complete binary
    // replacement set, immediately before the rollbackable in-place writes.
    replaceCwbvhBuffers(device, sb, cwbvh, () => {
      handles.positionsBuffer = candidates.positions!;
      handles.normalsBuffer = candidates.normals!;
      handles.uvsBuffer = candidates.uvs!;
      handles.tangentsBuffer = candidates.tangents!;
      handles.colorsBuffer = candidates.colors!;
      handles.indicesBuffer = candidates.indices!;
      handles.triMaterialIdsBuffer = candidates.triMaterialIds!;
      handles.bvhNodesBuffer = candidates.bvhNodes!;
      handles.tlasNodesBuffer = candidates.tlasNodes!;
      handles.tlasInstanceIndicesBuffer = candidates.tlasInstanceIndices!;
      handles.tlasBlasRootsBuffer = candidates.tlasBlasRoots!;
      handles.tlasInstanceWorldToLocalBuffer = candidates.tlasInstanceWorldToLocal!;
      handles.tlasInstanceLocalToWorldBuffer = candidates.tlasInstanceLocalToWorld!;
      binaryPublished = true;
    });
  } catch (error) {
    if (binaryPublished) {
      handles.positionsBuffer = previous[0]!;
      handles.normalsBuffer = previous[1]!;
      handles.uvsBuffer = previous[2]!;
      handles.tangentsBuffer = previous[3]!;
      handles.colorsBuffer = previous[4]!;
      handles.indicesBuffer = previous[5]!;
      handles.triMaterialIdsBuffer = previous[6]!;
      handles.bvhNodesBuffer = previous[7]!;
      handles.tlasNodesBuffer = previous[8]!;
      handles.tlasInstanceIndicesBuffer = previous[9]!;
      handles.tlasBlasRootsBuffer = previous[10]!;
      handles.tlasInstanceWorldToLocalBuffer = previous[11]!;
      handles.tlasInstanceLocalToWorldBuffer = previous[12]!;
    }
    destroyResourcesBestEffort(Object.values(candidates), liveResources);
    throw error;
  }
  Object.assign(handles, next);
  applyScenePackCounts(sb, pack);
  destroyResourcesBestEffort(previous, sceneBufferResourceHandles(sb));
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
  const rewrites: BufferRewrite[] = [
    { buffer: sb.tlasNodesBuffer, next: pack.tlasNodes, previous: sb.tlasNodes },
    { buffer: sb.tlasInstanceIndicesBuffer, next: pack.tlasInstanceIndices, previous: sb.tlasInstanceIndices },
    { buffer: sb.tlasBlasRootsBuffer, next: pack.tlasBlasRoots, previous: sb.tlasBlasRoots },
    { buffer: sb.tlasInstanceWorldToLocalBuffer, next: pack.tlasInstanceWorldToLocal, previous: sb.tlasInstanceWorldToLocal },
    { buffer: sb.tlasInstanceLocalToWorldBuffer, next: pack.tlasInstanceLocalToWorld, previous: sb.tlasInstanceLocalToWorld },
  ];
  assertCwbvhTlasRootIsDistinct(sb, rewrites.map((rewrite) => rewrite.buffer));
  const rollback = writeBufferSetWithRollback(device, rewrites);
  try {
    updateCwbvhTlasRootMirror(device, sb, pack.tlasBlasRoots);
  } catch (error) {
    rollback();
    try { writeBufferIfNonEmpty(sb.cwbvhTlasBlasRootsBuffer, sb.cwbvhTlasBlasRoots, device); } catch { /* best effort */ }
    throw error;
  }
  sb.tlasNodes.set(pack.tlasNodes);
  sb.tlasInstanceIndices.set(pack.tlasInstanceIndices);
  sb.tlasBlasRoots.set(pack.tlasBlasRoots);
  sb.tlasInstanceWorldToLocal.set(pack.tlasInstanceWorldToLocal);
  sb.tlasInstanceLocalToWorld.set(pack.tlasInstanceLocalToWorld);
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

  // ── Material + analytic GPU handles / CPU mirrors ───────────────────────
  materialsBuffer: GPUBuffer;
  materialTexDescriptorsBuffer: GPUBuffer;
  analyticHeadersBuffer: GPUBuffer;
  analyticParamsBuffer: GPUBuffer;
  analyticLocalToWorldBuffer: GPUBuffer;
  analyticWorldToLocalBuffer: GPUBuffer;
  materials: Float32Array;
  materialTexDescriptors: Float32Array;
  analyticHeaders: Float32Array;
  analyticParams: Float32Array;
  analyticLocalToWorld: Float32Array;
  analyticWorldToLocal: Float32Array;

  // ── Environment-map GPU handles / CPU mirrors ──────────────────────────
  environmentMapTexelsBuffer: GPUBuffer;
  environmentMapCdfBuffer: GPUBuffer;
  environmentMapTexels: Float32Array;
  environmentMapCdf: Float32Array;
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

  // ── Emitter counts (incremental emitter patches) ─────────────────────────
  directionalLightCount: number;
  pointLightCount: number;
  spotLightCount: number;
  rectAreaLightCount: number;
  meshAreaLightCount: number;

  // ── Environment fields (incremental environment patches) ─────────────────
  environmentTint: readonly [number, number, number];
  environmentSunDirection: readonly [number, number, number];
  environmentSunStrength: number;
  environmentLightTreePower: number;
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

/**
 * A prepared incremental scene-buffer replacement. Candidate buffers are fully
 * allocated and uploaded before this token is returned; the live scene remains
 * untouched until {@link commit}. Old handles stay alive until {@link finalize}
 * and can therefore be restored by {@link rollback} if publication or reset
 * fails.
 */
export interface PreparedSceneBufferMutation {
  /** Whether commit publishes fresh GPUBuffer handles and invalidates bind groups. */
  readonly replacesBufferHandles: boolean;
  /** Read-only preview used to stage dependent resources (for example lite textures). */
  readonly preview: UploadedSceneBuffers;
  /** Publish candidate handles and CPU mirrors without retiring the live set. */
  commit(): void;
  /** Restore the live set, if published, and destroy all candidates. */
  rollback(): void;
  /** Retire the previous handles after the complete mutation succeeds. */
  finalize(): void;
}

export type SceneBufferMutationPatch = Partial<MutableSceneBuffers>;

function cloneSceneBufferData(data: ArrayBufferView): Float32Array | Uint32Array {
  if (data instanceof Float32Array) return new Float32Array(data);
  if (data instanceof Uint32Array) return new Uint32Array(data);
  throw new Error('[pt-webgpu] incremental scene buffer data must be Float32Array or Uint32Array');
}

interface SceneMutationDataEntry {
  readonly key: (typeof SCENE_BUFFER_REGISTRY)[number]['key'];
  readonly bufferField: (typeof SCENE_BUFFER_REGISTRY)[number]['bufferField'];
  readonly label: string;
  readonly next: Float32Array | Uint32Array;
  readonly previousData: Float32Array | Uint32Array;
  readonly previousBuffer: GPUBuffer;
}

interface SceneMutationScalarEntry {
  readonly key: keyof MutableSceneBuffers;
  readonly next: unknown;
  readonly previous: unknown;
}

interface DirtyByteRange {
  readonly byteOffset: number;
  readonly byteLength: number;
}

/**
 * Return the minimal coalesced 4-byte ranges whose contents differ. WebGPU's
 * copyBufferToBuffer contract requires source offset, destination offset, and
 * size to be multiples of four, so comparisons are performed one word at a
 * time. Adjacent dirty words become one copy.
 */
function coalescedDirtyByteRanges(
  previous: ArrayBufferView,
  next: ArrayBufferView,
): readonly DirtyByteRange[] {
  if (previous.byteLength !== next.byteLength) {
    throw new Error('[pt-webgpu] in-place mutation requires equal byte lengths');
  }
  if (previous.byteLength % 4 !== 0) {
    throw new Error('[pt-webgpu] in-place mutation payloads must be 4-byte aligned');
  }
  const previousBytes = new Uint8Array(
    previous.buffer,
    previous.byteOffset,
    previous.byteLength,
  );
  const nextBytes = new Uint8Array(next.buffer, next.byteOffset, next.byteLength);
  const ranges: DirtyByteRange[] = [];
  let start = -1;
  for (let byteOffset = 0; byteOffset < nextBytes.byteLength; byteOffset += 4) {
    let dirty = false;
    for (let lane = 0; lane < 4; lane += 1) {
      if (previousBytes[byteOffset + lane] !== nextBytes[byteOffset + lane]) {
        dirty = true;
        break;
      }
    }
    if (dirty && start < 0) start = byteOffset;
    if (!dirty && start >= 0) {
      ranges.push({ byteOffset: start, byteLength: byteOffset - start });
      start = -1;
    }
  }
  if (start >= 0) {
    ranges.push({ byteOffset: start, byteLength: nextBytes.byteLength - start });
  }
  return ranges;
}

interface InPlaceCopySpec {
  readonly label: string;
  readonly destination: GPUBuffer;
  readonly data: ArrayBufferView;
  readonly ranges: readonly DirtyByteRange[];
}

interface PreparedInPlaceCopyBatch {
  submit(): void;
  destroy(): void;
}

/**
 * Stage every dirty range in one allocation, encode every copy, and submit the
 * whole mutation as one batch. No GPU work is created for a byte-identical set.
 */
function prepareInPlaceCopyBatch(
  device: GPUDevice,
  specs: readonly InPlaceCopySpec[],
  protectedResources: readonly object[],
): PreparedInPlaceCopyBatch {
  const forbiddenResources = new Set<object>(protectedResources);
  const copies: Array<{
    readonly destination: GPUBuffer;
    readonly destinationByteOffset: number;
    readonly stagingByteOffset: number;
    readonly byteLength: number;
    readonly data: ArrayBufferView;
  }> = [];
  let stagingByteLength = 0;
  for (const spec of specs) {
    for (const range of spec.ranges) {
      if (
        range.byteOffset % 4 !== 0 ||
        range.byteLength <= 0 ||
        range.byteLength % 4 !== 0
      ) {
        throw new Error('[pt-webgpu] incremental copy range is not WebGPU-aligned');
      }
      copies.push({
        destination: spec.destination,
        destinationByteOffset: range.byteOffset,
        stagingByteOffset: stagingByteLength,
        byteLength: range.byteLength,
        data: new Uint8Array(
          spec.data.buffer,
          spec.data.byteOffset + range.byteOffset,
          range.byteLength,
        ),
      });
      stagingByteLength += range.byteLength;
    }
  }

  if (copies.length === 0) {
    return {
      submit: () => {},
      destroy: () => {},
    };
  }

  let staging: GPUBuffer | null = null;
  try {
    staging = device.createBuffer({
      label: 'vitrum.pt-webgpu.scene.incremental-staging',
      size: stagingByteLength,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    if (forbiddenResources.has(staging)) {
      throw new Error(
        '[pt-webgpu] incremental staging allocation aliased a live GPU resource',
      );
    }
    const stagingBytes = new Uint8Array(stagingByteLength);
    for (const copy of copies) {
      stagingBytes.set(
        copy.data as Uint8Array,
        copy.stagingByteOffset,
      );
    }
    device.queue.writeBuffer(
      staging,
      0,
      stagingBytes.buffer,
      stagingBytes.byteOffset,
      stagingBytes.byteLength,
    );
    const encoder = device.createCommandEncoder({
      label: 'vitrum.pt-webgpu.scene.incremental-copy',
    });
    for (const copy of copies) {
      encoder.copyBufferToBuffer(
        staging,
        copy.stagingByteOffset,
        copy.destination,
        copy.destinationByteOffset,
        copy.byteLength,
      );
    }
    const commandBuffer = encoder.finish();
    let destroyed = false;
    return {
      submit: () => { device.queue.submit([commandBuffer]); },
      destroy: () => {
        if (destroyed) return;
        destroyed = true;
        destroyResourcesBestEffort([staging!], protectedResources);
      },
    };
  } catch (error) {
    if (staging != null) {
      destroyResourcesBestEffort([staging], protectedResources);
    }
    throw error;
  }
}

function prepareSameSizeSceneBufferMutation(
  device: GPUDevice,
  sb: UploadedSceneBuffers,
  dataEntries: readonly SceneMutationDataEntry[],
  scalarEntries: readonly SceneMutationScalarEntry[],
): PreparedSceneBufferMutation {
  const mutable = asMutableSceneBuffers(sb) as unknown as Record<string, unknown>;
  const preview = { ...sb } as unknown as Record<string, unknown>;
  const dirtyEntries = dataEntries.map((entry) => ({
    entry,
    ranges: coalescedDirtyByteRanges(entry.previousData, entry.next),
  }));
  const protectedResources = sceneBufferResourceHandles(sb);
  const initialBatch = prepareInPlaceCopyBatch(
    device,
    dirtyEntries.map(({ entry, ranges }) => ({
      label: entry.label,
      destination: entry.previousBuffer,
      data: entry.next,
      ranges,
    })),
    protectedResources,
  );
  for (const entry of dataEntries) preview[entry.key] = entry.next;
  for (const entry of scalarEntries) preview[entry.key as string] = entry.next;

  const publish = (): void => {
    for (const entry of dataEntries) mutable[entry.key] = entry.next;
    for (const entry of scalarEntries) mutable[entry.key as string] = entry.next;
  };
  const restoreMirrors = (): void => {
    for (const entry of dataEntries) mutable[entry.key] = entry.previousData;
    for (const entry of scalarEntries) mutable[entry.key as string] = entry.previous;
  };
  const restoreGpu = (): void => {
    const restoreBatch = prepareInPlaceCopyBatch(
      device,
      dirtyEntries.map(({ entry, ranges }) => ({
        label: `${entry.label}.rollback`,
        destination: entry.previousBuffer,
        data: entry.previousData,
        ranges,
      })),
      protectedResources,
    );
    try {
      restoreBatch.submit();
    } finally {
      restoreBatch.destroy();
    }
  };

  let state: 'prepared' | 'committed' | 'closed' = 'prepared';
  return {
    replacesBufferHandles: false,
    preview: preview as unknown as UploadedSceneBuffers,
    commit: () => {
      if (state === 'closed') {
        throw new Error('[pt-webgpu] scene mutation transaction is closed');
      }
      if (state === 'committed') return;
      try {
        initialBatch.submit();
      } catch (error) {
        initialBatch.destroy();
        state = 'closed';
        throw error;
      }
      initialBatch.destroy();
      try {
        publish();
        state = 'committed';
      } catch (error) {
        try { restoreGpu(); } catch { /* preserve publication failure */ }
        restoreMirrors();
        state = 'closed';
        throw error;
      }
    },
    rollback: () => {
      if (state === 'closed') return;
      if (state === 'prepared') {
        initialBatch.destroy();
        state = 'closed';
        return;
      }
      let restoreError: unknown;
      try { restoreGpu(); } catch (error) { restoreError = error; }
      restoreMirrors();
      state = 'closed';
      if (restoreError instanceof Error) throw restoreError;
      if (restoreError != null) {
        throw new Error('Failed to restore pt-webgpu scene buffers.', {
          cause: restoreError,
        });
      }
    },
    finalize: () => {
      if (state === 'closed') return;
      if (state === 'prepared') initialBatch.destroy();
      state = 'closed';
    },
  };
}

/**
 * Prepare a bounded transaction over selected scene buffers and CPU mirrors.
 * Equal-size arrays stage only coalesced dirty words, then copy them into the
 * existing buffers in one submission before publishing the mirrors. Size
 * changes use isolated candidate buffers and publish their handles only after
 * every allocation and upload succeeds.
 */
export function prepareSceneBufferMutation(
  device: GPUDevice,
  sb: UploadedSceneBuffers,
  requestedPatch: SceneBufferMutationPatch,
): PreparedSceneBufferMutation {
  const dataKeys = new Set<string>();
  const dataEntries: SceneMutationDataEntry[] = [];
  for (const entry of SCENE_BUFFER_REGISTRY) {
    const requested = requestedPatch[entry.key as keyof SceneBufferMutationPatch];
    if (requested === undefined) continue;
    if (!ArrayBuffer.isView(requested)) {
      throw new Error(`[pt-webgpu] incremental scene field ${entry.key} must be an ArrayBufferView`);
    }
    dataKeys.add(entry.key);
    dataEntries.push({
      key: entry.key,
      bufferField: entry.bufferField,
      label: entry.label,
      next: cloneSceneBufferData(requested),
      previousData: sb[entry.key],
      previousBuffer: sb[entry.bufferField],
    });
  }
  if (dataEntries.length === 0) {
    throw new Error('[pt-webgpu] incremental scene mutation must replace at least one GPU buffer');
  }

  const scalarEntries = Object.entries(requestedPatch)
    .filter(([key, value]) => value !== undefined && !dataKeys.has(key))
    .map(([key, next]) => ({
      key: key as keyof MutableSceneBuffers,
      next,
      previous: (sb as unknown as Record<string, unknown>)[key],
    }));
  if (
    dataEntries.every(
      (entry) => entry.next.byteLength === entry.previousData.byteLength,
    )
  ) {
    return prepareSameSizeSceneBufferMutation(device, sb, dataEntries, scalarEntries);
  }
  const liveResources = sceneBufferResourceHandles(sb);
  const candidates = createStorageBufferCandidates(
    device,
    dataEntries.map((entry) => ({ key: entry.key, label: entry.label, data: entry.next })),
    liveResources,
  );
  const mutable = asMutableSceneBuffers(sb) as unknown as Record<string, unknown>;
  const preview = { ...sb } as unknown as Record<string, unknown>;
  for (const entry of dataEntries) {
    preview[entry.key] = entry.next;
    preview[entry.bufferField] = candidates[entry.key]!;
  }
  for (const entry of scalarEntries) preview[entry.key as string] = entry.next;

  let state: 'prepared' | 'committed' | 'closed' = 'prepared';
  const restorePrevious = (): void => {
    for (const entry of dataEntries) {
      mutable[entry.key] = entry.previousData;
      mutable[entry.bufferField] = entry.previousBuffer;
    }
    for (const entry of scalarEntries) mutable[entry.key as string] = entry.previous;
  };
  const destroyCandidates = (): void => {
    destroyResourcesBestEffort(Object.values(candidates), sceneBufferResourceHandles(sb));
  };

  return {
    preview: preview as unknown as UploadedSceneBuffers,
    replacesBufferHandles: true,
    commit: () => {
      if (state === 'closed') throw new Error('[pt-webgpu] scene mutation transaction is closed');
      if (state === 'committed') return;
      try {
        for (const entry of dataEntries) {
          mutable[entry.key] = entry.next;
          mutable[entry.bufferField] = candidates[entry.key]!;
        }
        for (const entry of scalarEntries) mutable[entry.key as string] = entry.next;
        state = 'committed';
      } catch (error) {
        restorePrevious();
        destroyCandidates();
        state = 'closed';
        throw error;
      }
    },
    rollback: () => {
      if (state === 'closed') return;
      if (state === 'committed') restorePrevious();
      destroyCandidates();
      state = 'closed';
    },
    finalize: () => {
      if (state === 'closed') return;
      if (state !== 'committed') {
        destroyCandidates();
        state = 'closed';
        return;
      }
      destroyResourcesBestEffort(
        dataEntries.map((entry) => entry.previousBuffer),
        sceneBufferResourceHandles(sb),

      );
      state = 'closed';
    },
  };
}

/** Build the BLAS/CWBVH mirror patch, optionally including TLAS. */
export function scenePackGeometryMutationPatch(
  sb: UploadedSceneBuffers,
  pack: ScenePackResult,
  includeTlas: boolean,
  uvsOverride?: Float32Array,
): SceneBufferMutationPatch {
  const cwbvh = buildPackedCwbvhSceneData(pack);
  const bounds = sceneCenterRadiusFromPack(pack);
  const next: SceneBufferMutationPatch = {
    positions: pack.positions,
    normals: pack.normals,
    uvs: uvsOverride ?? pack.uvs,
    tangents: pack.tangents,
    colors: pack.colors,
    indices: pack.indices,
    triMaterialIds: pack.triMaterialIds,
    bvhNodes: pack.bvhNodes,
    cwbvhNodeBounds: cwbvh.cwbvhNodeBounds,
    cwbvhChildBoundsPacked: cwbvh.cwbvhChildBoundsPacked,
    cwbvhChildMeta: cwbvh.cwbvhChildMeta,
    cwbvhChildCount: cwbvh.cwbvhChildCount,
    cwbvhTlasBlasRoots: cwbvh.cwbvhTlasBlasRoots,
    cwbvhBinaryRootToWideRoot: cwbvh.cwbvhBinaryRootToWideRoot,
    cwbvhNodeCount: cwbvh.cwbvhNodeCount,
    bvhNodeCount: Math.floor(pack.bvhNodes.length / BVH_NODE_FLOATS),
    triangleCount: pack.triangleCount,
    primitiveTlasBindings: pack.primitiveTlasBindings,
    sceneCenter: bounds.center,
    sceneRadius: bounds.radius,
  };
  if (includeTlas) {
    Object.assign(next, {
      tlasNodes: pack.tlasNodes,
      tlasInstanceIndices: pack.tlasInstanceIndices,
      tlasBlasRoots: pack.tlasBlasRoots,
      tlasInstanceWorldToLocal: pack.tlasInstanceWorldToLocal,
      tlasInstanceLocalToWorld: pack.tlasInstanceLocalToWorld,
      tlasNodeCount: pack.tlasNodeCount,
    } satisfies SceneBufferMutationPatch);
  }
  return next;
}

/** Prepare a candidate-first BLAS/CWBVH replacement, optionally including TLAS. */
export function prepareScenePackGeometryMutation(
  device: GPUDevice,
  sb: UploadedSceneBuffers,
  pack: ScenePackResult,
  includeTlas: boolean,
): PreparedSceneBufferMutation {
  return prepareSceneBufferMutation(
    device,
    sb,
    scenePackGeometryMutationPatch(sb, pack, includeTlas),
  );
}

/** Build the TLAS + CWBVH-root mirror patch. */
export function scenePackTlasMutationPatch(
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
): SceneBufferMutationPatch {
  const roots = remapCwbvhTlasBlasRoots(
    pack.tlasBlasRoots,
    sb.cwbvhBinaryRootToWideRoot,
  );
  const bounds = sceneCenterRadiusFromPack({
    bvhNodes: sb.bvhNodes,
    tlasNodes: pack.tlasNodes,
  });
  return {
    tlasNodes: pack.tlasNodes,
    tlasInstanceIndices: pack.tlasInstanceIndices,
    tlasBlasRoots: pack.tlasBlasRoots,
    tlasInstanceWorldToLocal: pack.tlasInstanceWorldToLocal,
    tlasInstanceLocalToWorld: pack.tlasInstanceLocalToWorld,
    cwbvhTlasBlasRoots: roots,
    tlasNodeCount: pack.tlasNodeCount,
    primitiveTlasBindings: pack.primitiveTlasBindings,
    sceneCenter: bounds.center,
    sceneRadius: bounds.radius,
  };
}

/** Prepare a candidate-first TLAS + CWBVH-root replacement. */
export function prepareScenePackTlasMutation(
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
): PreparedSceneBufferMutation {
  return prepareSceneBufferMutation(
    device,
    sb,
    scenePackTlasMutationPatch(sb, pack),
  );
}


function sceneBufferResourceHandles(sb: UploadedSceneBuffers): GPUBuffer[] {
  return SCENE_BUFFER_REGISTRY
    .map((entry) => sb[entry.bufferField])
    .filter((resource): resource is GPUBuffer => resource != null);
}

/** All destroyable resources owned by one uploaded scene generation. */
export function uploadedSceneGpuResources(
  sb: UploadedSceneBuffers,
): readonly (GPUBuffer | GPUTexture)[] {
  return [
    ...sceneBufferResourceHandles(sb),
    sb.materialTexture,
    sb.materialLinearTexture,
    sb.materialEmissiveTexture,
  ].filter((resource): resource is GPUBuffer | GPUTexture => resource != null);
}

function assertCwbvhTlasRootIsDistinct(
  sb: UploadedSceneBuffers,
  protectedBuffers: readonly GPUBuffer[],
): void {
  const root = sb.cwbvhTlasBlasRootsBuffer as GPUBuffer | undefined;
  if (root != null && protectedBuffers.includes(root)) {
    throw new Error(
      '[pt-webgpu] live CWBVH TLAS-root buffer aliases a binary TLAS resource',
    );
  }
}

function replaceCwbvhBuffers(
  device: GPUDevice,
  sb: UploadedSceneBuffers,
  cwbvh: PackedCwbvhSceneData,
  beforeCommit: () => void = () => {},
  protectedBuffers: readonly GPUBuffer[] = [],
): void {
  if (
    sb.cwbvhNodeBoundsBuffer == null ||
    sb.cwbvhChildBoundsPackedBuffer == null ||
    sb.cwbvhChildMetaBuffer == null ||
    sb.cwbvhChildCountBuffer == null ||
    sb.cwbvhTlasBlasRootsBuffer == null
  ) {
    beforeCommit();
    return;
  }
  const mutable = asMutableSceneBuffers(sb);
  const entries = [
    { key: 'nodeBounds', current: sb.cwbvhNodeBoundsBuffer, label: 'vitrum.pt-webgpu.scene.cwbvhNodeBounds', data: cwbvh.cwbvhNodeBounds, previousData: sb.cwbvhNodeBounds, assign: (buffer: GPUBuffer) => { mutable.cwbvhNodeBoundsBuffer = buffer; } },
    { key: 'childBoundsPacked', current: sb.cwbvhChildBoundsPackedBuffer, label: 'vitrum.pt-webgpu.scene.cwbvhChildBoundsPacked', data: cwbvh.cwbvhChildBoundsPacked, previousData: sb.cwbvhChildBoundsPacked, assign: (buffer: GPUBuffer) => { mutable.cwbvhChildBoundsPackedBuffer = buffer; } },
    { key: 'childMeta', current: sb.cwbvhChildMetaBuffer, label: 'vitrum.pt-webgpu.scene.cwbvhChildMeta', data: cwbvh.cwbvhChildMeta, previousData: sb.cwbvhChildMeta, assign: (buffer: GPUBuffer) => { mutable.cwbvhChildMetaBuffer = buffer; } },
    { key: 'childCount', current: sb.cwbvhChildCountBuffer, label: 'vitrum.pt-webgpu.scene.cwbvhChildCount', data: cwbvh.cwbvhChildCount, previousData: sb.cwbvhChildCount, assign: (buffer: GPUBuffer) => { mutable.cwbvhChildCountBuffer = buffer; } },
    { key: 'tlasBlasRoots', current: sb.cwbvhTlasBlasRootsBuffer, label: 'vitrum.pt-webgpu.scene.cwbvhTlasBlasRoots', data: cwbvh.cwbvhTlasBlasRoots, previousData: sb.cwbvhTlasBlasRoots, assign: (buffer: GPUBuffer) => { mutable.cwbvhTlasBlasRootsBuffer = buffer; } },
  ] as const;
  const uniqueCurrentBuffers = new Set<GPUBuffer>();
  for (const entry of entries) {
    if (uniqueCurrentBuffers.has(entry.current)) {
      throw new Error(
        '[pt-webgpu] live CWBVH resource set aliases one GPUBuffer across multiple fields',
      );
    }
    uniqueCurrentBuffers.add(entry.current);
  }
  const protectedBufferSet = new Set(protectedBuffers);
  for (const entry of entries) {
    if (protectedBufferSet.has(entry.current)) {
      throw new Error(
        '[pt-webgpu] live CWBVH resource aliases a protected transactional GPUBuffer',
      );
    }
  }
  const needsReplacement = entries.filter((entry) => {
    const targetSize = Math.ceil(Math.max(
      entry.data.byteLength,
      MIN_STORAGE_BUFFER_BYTES_BY_LABEL[entry.label] ?? 16,
    ) / 4) * 4;
    // A few test/host GPUBuffer facades predate the WebGPU `size` property.
    // The authoritative CPU mirror has the exact live allocation payload in
    // that case, so use it rather than spuriously reallocating every CWBVH
    // buffer on otherwise in-place updates.
    const currentSize = entry.current.size ?? Math.ceil(Math.max(
      entry.previousData.byteLength,
      MIN_STORAGE_BUFFER_BYTES_BY_LABEL[entry.label] ?? 16,
    ) / 4) * 4;
    return currentSize !== targetSize;
  });
  const candidates = createStorageBufferCandidates(
    device,
    needsReplacement.map((entry) => ({
      key: entry.key,
      label: entry.label,
      data: entry.data,
    })),
    sceneBufferResourceHandles(sb),
  );
  const writtenInPlace: Array<(typeof entries)[number]> = [];
  try {
    beforeCommit();
    for (const entry of entries) {
      if (candidates[entry.key] != null) continue;
      writeBufferIfNonEmpty(entry.current, entry.data, device);
      writtenInPlace.push(entry);
    }
  } catch (error) {
    // queue.writeBuffer normally reports validation asynchronously, but injected
    // synchronous failures are rollbackable from the authoritative CPU mirrors.
    for (let i = writtenInPlace.length - 1; i >= 0; i -= 1) {
      const entry = writtenInPlace[i]!;
      try { writeBufferIfNonEmpty(entry.current, entry.previousData, device); } catch { /* best effort */ }
    }
    destroyResourcesBestEffort(
      Object.values(candidates),
      sceneBufferResourceHandles(sb),
    );
    throw error;
  }
  for (const entry of needsReplacement) {
    const candidate = candidates[entry.key]!;
    entry.assign(candidate);
  }
  destroyResourcesBestEffort(
    needsReplacement.map((entry) => entry.current),
    sceneBufferResourceHandles(sb),
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
  const currentSize = sb.cwbvhTlasBlasRootsBuffer.size ?? Math.ceil(Math.max(
    sb.cwbvhTlasBlasRoots.byteLength,
    MIN_STORAGE_BUFFER_BYTES_BY_LABEL[label] ?? 16,
  ) / 4) * 4;
  if (currentSize !== expectedSize) {
    const previous = sb.cwbvhTlasBlasRootsBuffer;
    const candidate = createStorageBuffer(
      device,
      label,
      roots,
      new Set(sceneBufferResourceHandles(sb)),
    );
    mutable.cwbvhTlasBlasRootsBuffer = candidate;
    destroyResourcesBestEffort([previous], sceneBufferResourceHandles(sb));
  } else {
    writeBufferIfNonEmpty(sb.cwbvhTlasBlasRootsBuffer, roots, device);
  }
  mutable.cwbvhTlasBlasRoots = roots;
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

export interface PackedLightTreeMutation {
  readonly lightTreeNodes: Float32Array;
  readonly lightTreeNodeCount: number;
  readonly lightTreeEnabled: boolean;
}

/** Pure light-tree pack used by both full uploads and prepared mutations. */
export function packLightTreeForScene(
  scene: Scene,
  precomputed?: { packed?: PackedEmitterArrays; envSummary?: EnvSummaryForTree },
): PackedLightTreeMutation {
  const input = buildLightTreeInputForScene(scene, precomputed);
  if (input.powers.length < 2) {
    return {
      lightTreeNodes: new Float32Array(0),
      lightTreeNodeCount: 0,
      lightTreeEnabled: false,
    };
  }
  const built = buildLightTree(input);
  return {
    lightTreeNodes: new Float32Array(packLightTreeForGPU(built.nodes)),
    lightTreeNodeCount: built.nodes.length,
    lightTreeEnabled: true,
  };
}

/** Rewrite environment fields after an in-place HDRI/sky upload. */
export function applyEnvironmentMutation(
  sb: UploadedSceneBuffers,
  next: {
    readonly environmentTint: readonly [number, number, number];
    readonly environmentSunDirection: readonly [number, number, number];
    readonly environmentSunStrength: number;
    readonly environmentLightTreePower: number;
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
  mutable.environmentLightTreePower = next.environmentLightTreePower;
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
    case 'texture-unsupported-layout':
      return 'pt-webgpu.material-texture-unsupported-layout';
  }
}

function materialTextureEngineWarnings(
  warnings: readonly MaterialTextureArrayWarning[],
  colorSpace: 'sRGB' | 'linear' | 'emissive',
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
      },
    };
  });
}

export function uploadPackedScene(
  device: GPUDevice,
  packed: PackedSceneData,
  forbiddenResources: readonly object[] = [],
): UploadedSceneBuffers {
  // Upload-time assertion: pt-webgpu uses stride-4 indices (vec4u, .w = 0).
  // byteLength must be a multiple of 4 u32 = 16 bytes.
  if (packed.indices.byteLength > 0 && packed.indices.byteLength % 16 !== 0) {
    throw new Error(
      `[pt-webgpu/uploadPackedScene] Index buffer byteLength (${packed.indices.byteLength}) ` +
        `is not aligned to BvhIndexStride 4 (16 bytes per triangle). ` +
        `pt-webgpu requires stride-4 indices — 3 vertex u32 + 1 zero-fill u32.`,
    );
  }
  const maxBufferSize = device.limits?.maxBufferSize ?? 256 * 1024 * 1024;
  const maxStorageBindingSize =
    device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const materialByteLimit = Math.min(maxBufferSize, maxStorageBindingSize);
  if (packed.materials.byteLength > materialByteLimit) {
    throw new Error(
      `[pt-webgpu/uploadPackedScene] Materials buffer byteLength ` +
      `(${packed.materials.byteLength}) exceeds the device storage-buffer ` +
      `limit (${materialByteLimit}; maxBufferSize=${maxBufferSize}, ` +
      `maxStorageBufferBindingSize=${maxStorageBindingSize}).`,
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
  const createdResourceSet = new Set<object>(forbiddenResources);
  const trackedCreateStorageBuffer = (
    dev: GPUDevice,
    label: string,
    data: ArrayBufferView,
  ): GPUBuffer => {
    const buf = createStorageBuffer(dev, label, data, createdResourceSet);
    createdResources.push(buf);
    createdResourceSet.add(buf);
    return buf;
  };
  const trackMaterialArray = (arr: MaterialTextureArray): MaterialTextureArray => {
    // GPUSampler has no destroy(); only the texture holds device memory.
    if (createdResourceSet.has(arr.texture)) {
      throw new Error(
        '[pt-webgpu] material texture candidate aliased an existing scene resource',
      );
    }
    createdResources.push(arr.texture);
    createdResourceSet.add(arr.texture);
    return arr;
  };
  try {
    return uploadPackedSceneInner(
      device,
      packed,
      trackedCreateStorageBuffer,
      trackMaterialArray,
      createdResourceSet,
    );
  } catch (err) {
    destroyResourcesBestEffort(createdResources);
    throw err;
  }
}

function uploadPackedSceneInner(
  device: GPUDevice,
  packed: PackedSceneData,
  createStorageBuffer: (dev: GPUDevice, label: string, data: ArrayBufferView) => GPUBuffer,
  trackMaterialArray: (arr: MaterialTextureArray) => MaterialTextureArray,
  forbiddenResources: ReadonlySet<object>,
): UploadedSceneBuffers {
  // P2 — the three material texture arrays + the UV-fit descriptor patch MUST run
  // before the `materialTexDescriptors` storage buffer is created, because
  // `applyMaterialTextureUvFitScales` mutates `packed.materialTexDescriptors`
  // in place (with the arrays' per-layer source rects) and the buffer is written
  // from that array at creation time. Everything else is order-independent, so
  // the storage buffers are created by a single registry-driven loop below.
  // Validate both array payloads and their combined resident/upload peak
  // before allocating the first texture. Per-array checks alone permit two
  // individually-valid arrays to exceed the process-level atlas budget.
  const aggregateMaterialTexturePeakBytes =
    estimateMaterialTextureArrayPeakBytes(
      device,
      packed.materialTextureSources,
      'rgba8unorm-srgb',
      forbiddenResources,
    ) +
    estimateMaterialTextureArrayPeakBytes(
      device,
      packed.materialTextureLinearSources,
      'rgba8unorm',
      forbiddenResources,
    ) +
    estimateMaterialTextureArrayPeakBytes(
      device,
      packed.materialTextureEmissiveSources,
      'rgba16float',
      forbiddenResources,
    );
  if (
    !Number.isSafeInteger(aggregateMaterialTexturePeakBytes) ||
    aggregateMaterialTexturePeakBytes > MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES
  ) {
    throw new RangeError(
      `[pt-webgpu] aggregate material-atlas peak ${aggregateMaterialTexturePeakBytes} bytes exceeds ` +
        `${MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES}-byte budget before GPU allocation.`,
    );
  }

  const materialTextureArray = trackMaterialArray(createMaterialTextureArray(
    device,
    packed.materialTextureSources,
    'rgba8unorm-srgb',
    packed.materialTextureSourceInfos,
    forbiddenResources,
  ));
  // Linear array (normal + ORM) — rgba8unorm so the sampler does NOT sRGB-decode.
  const materialLinearArray = trackMaterialArray(createMaterialTextureArray(
    device,
    packed.materialTextureLinearSources,
    'rgba8unorm',
    packed.materialTextureLinearSourceInfos,
    forbiddenResources,
  ));
  // Dedicated HDR emissive array. Integer/external sources are sRGB-decoded on
  // upload while raw-float sources retain linear radiance above one.
  const materialEmissiveArray = trackMaterialArray(createMaterialTextureArray(
    device,
    packed.materialTextureEmissiveSources,
    'rgba16float',
    packed.materialTextureEmissiveSourceInfos,
    forbiddenResources,
  ));
  applyMaterialTextureUvFitScales(
    packed.materialTexDescriptors,
    materialTextureArray.layerUvScales,
    materialLinearArray.layerUvScales,
    materialEmissiveArray.layerUvScales,
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
    ...materialEmissiveArray.warnings.map((w) => `material texture array (emissive): ${w}`),
  ];
  const materialTextureStructuredWarnings = [
    ...materialTextureEngineWarnings(materialTextureArray.structuredWarnings, 'sRGB'),
    ...materialTextureEngineWarnings(materialLinearArray.structuredWarnings, 'linear'),
    ...materialTextureEngineWarnings(materialEmissiveArray.structuredWarnings, 'emissive'),
  ];

  let uploadedDestroyed = false;
  const uploaded: UploadedSceneBuffers = {
    ...packed,
    warnings: [...packed.warnings, ...materialTextureWarnings],
    structuredWarnings: [...packed.structuredWarnings, ...materialTextureStructuredWarnings],
    bvhNodeCount: Math.floor(packed.bvhNodes.length / BVH_NODE_FLOATS),
    tlasNodeCount: Math.floor(packed.tlasNodes.length / BVH_NODE_FLOATS),
    materialCount: Math.floor(packed.materialTexDescriptors.length / MATERIAL_TEX_FLOAT_STRIDE),
    ...buffers,
    materialTexture: materialTextureArray.texture,
    materialTextureView: materialTextureArray.view,
    materialTextureSampler: materialTextureArray.sampler,
    materialLinearTexture: materialLinearArray.texture,
    materialLinearTextureView: materialLinearArray.view,
    materialEmissiveTexture: materialEmissiveArray.texture,
    materialEmissiveTextureView: materialEmissiveArray.view,
    // T2-A — every scene storage buffer is destroyed via a single registry-driven
    // loop reading `uploaded[bufferField]`. Resolving each handle LATE off
    // `uploaded` (not a captured local) is required for the realloc fast paths
    // that swap fresh handles onto the struct — {@link uploadScenePackTlasRealloc}
    // (instance-count change), {@link uploadScenePackGeometryRealloc} (mesh
    // vertex/index-count change), and prepared light-tree mutations (node-count
    // change) — so `destroy` never touches a stale (leaked/double-freed) handle.
    // Non-resized buffers (materials / analytic / environment / descriptors) are
    // never reassigned, so reading them off `uploaded` is identical to the
    // previously-captured locals. The two material texture arrays are destroyed
    // explicitly (textures are not in the buffer registry).
    destroy: () => {
      if (uploadedDestroyed) return;
      uploadedDestroyed = true;
      destroyResourcesBestEffort([
        ...SCENE_BUFFER_REGISTRY.map((entry) => uploaded[entry.bufferField]),
        materialTextureArray.texture,
        materialLinearArray.texture,
        materialEmissiveArray.texture,
      ]);
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
        // Bytes-per-texel by format: rgba8* = 4, rgba16float = 8 (T1-6 emissive).
        const bytesPerTexel = t.format === 'rgba16float' ? 8 : 4;
        const bytes = t.width * t.height * t.depthOrArrayLayers * bytesPerTexel;
        textureBytesByFormat[t.format] = (textureBytesByFormat[t.format] ?? 0) + bytes;
      };
      addTex(uploaded.materialTexture);
      addTex(uploaded.materialLinearTexture);
      addTex(uploaded.materialEmissiveTexture);
      return { bufferBytes, textureBytesByFormat };
    },
  };
  return uploaded;
}
