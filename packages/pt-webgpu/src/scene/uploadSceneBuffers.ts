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
  assertOpticalMediumTopology,
  lowerTransmissiveAnalyticPrimitives,
  mergeUv1FromCore,
  mergeWorldSpaceFromCore,
  materialSpecScalarEmissiveLe,
  materialSpecSkipEmitter,
  packMergedOpticalMediumBoundaryIds,
  packOpticalMediumBoundaryIds,
  packCwbvhBuildBoundsForWgsl,
  packRadianceRgbScaleF32,
  packSceneFromCore,
  refitTlasTransforms,
  type PrimitiveTlasBinding,
  type RadianceRgb,
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
  createMaterialInputSnapshotContext,
  type MaterialTextureLayerInfo,
  type MaterialInputSnapshotContext,
  MATERIAL_TEX_FLOAT_STRIDE,
} from './materialTextures.js';
import { resolvePtWebgpuSceneRadius } from './sceneScalePolicy.js';
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
  createMaterialTextureArrayFromStaged,
  stageMaterialTextureUploadPlan,
  MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES,
  type MaterialTextureArray,
  type MaterialTextureRadianceEnvelope,
  type MaterialTextureArrayWarning,
} from './materialTextureArray.js';
import { environmentParams } from './environmentPacking.js';
import { assertPtWebgpuEnvironmentMaterialEnvelopeF32 } from '../environmentRadianceScale.js';
import {
  buildLightTreeInputForScene,
  MESH_AREA_LIGHT_FLOAT_STRIDE,
  packEmitterArrays,
  type EnvSummaryForTree,
  type PackedEmitterArrays,
} from './emitterPacking.js';
import { applyDistantDirectProposalPmf } from './distantDirectProposal.js';
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

/**
 * True when at least one analytic is represented by a mesh in the render
 * snapshot. Incremental mutation routing uses this as a conservative layout
 * boundary: authored analytic records and generated mesh records do not share
 * buffer offsets or topology.
 */
export function sceneHasAnalyticRenderFallback(scene: Scene): boolean {
  return scene.primitives.some((primitive) => (
    primitive.kind === 'analytic' &&
    (
      analyticMaterialTextureFields(primitive).length > 0 ||
      (primitive.material.transmission ?? 0) > 0 ||
      (
        !materialSpecSkipEmitter(primitive.material) &&
        materialSpecScalarEmissiveLe(primitive.material) != null
      )
    )
  ));
}

/**
 * Build the geometry snapshot actually consumed by the renderer.
 *
 * Native analytic intersections have no surface-area proposal implementation.
 * An emissive analytic is therefore lowered once to the canonical mesh
 * tessellation, and that same mesh is consumed by both forward intersections
 * and every light-sampling estimator. Core validation deliberately requires an
 * explicit `mesh-area` emitter to reference mesh-like geometry, so this
 * fallback is for material-owned analytic emission rather than an alternate
 * way to bypass that scene-contract boundary.
 */
export function applyAnalyticRenderFallbacks(
  scene: Scene,
  warnings: string[],
): Scene {
  let changed = false;
  const primitives = scene.primitives.map((primitive) => {
    if (primitive.kind !== 'analytic') return primitive;
    const textureFields = analyticMaterialTextureFields(primitive);
    if (textureFields.length > 0) {
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
    }

    const hasImplicitMeshEmitter =
      !materialSpecSkipEmitter(primitive.material) &&
      materialSpecScalarEmissiveLe(primitive.material) != null;
    if (!hasImplicitMeshEmitter) return primitive;

    changed = true;
    warnings.push(
      `Analytic primitive "${primitive.id}" is emissive analytic geometry; ` +
        'rendering a mesh fallback so forward-hit geometry and light-sampling support share one surface.',
    );
    return analyticPrimitiveToMesh(primitive);
  });
  return lowerTransmissiveAnalyticPrimitives(
    changed ? { ...scene, primitives } : scene,
  );
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
  /** Dedup'd, upload-ordered sRGB texture handles (baseColor and extension
   *  color-tint maps; layer i = sources[i]); the GPU upload turns
   *  these into a texture_2d_array. */
  readonly materialTextureSources: readonly unknown[];
  /** Provenance for each sRGB material texture layer, for structured upload warnings. */
  readonly materialTextureSourceInfos: readonly MaterialTextureLayerInfo[];
  /** Dedup'd, upload-ordered LINEAR texture handles (normal, ORM, scalar maps,
   *  height/coverage data) → a second texture_2d_array sampled without
   *  sRGB decode. */
  readonly materialTextureLinearSources: readonly unknown[];
  /** Provenance for each LINEAR material texture layer, for structured upload warnings. */
  readonly materialTextureLinearSourceInfos: readonly MaterialTextureLayerInfo[];
  /** Dedup'd, upload-ordered outgoing-RADIANCE handles (emissiveMap + lightMap)
   *  → a dedicated rgba16float texture_2d_array. The historical field name is
   *  retained for ABI stability. Integer emissive layers are sRGB-decoded;
   *  integer light-map layers and every Float32 layer are linear. */
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
  /** GPU binding-5 payload: vec2u(materialId, representedInstanceIdPlusOne). */
  readonly triMaterialPayload: Uint32Array;
  readonly triangleRepresentedPrimitiveInstanceIds: Uint32Array;
  /** Source-triangle / source-primitive addresses parallel to the reordered stream. */
  readonly triangleSourceIndices: Uint32Array;
  readonly trianglePrimitiveIndices: Uint32Array;
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
  /** Host-visible reasons that one or more BLAS roots use binary fallback. */
  readonly cwbvhWarnings?: readonly string[];
  readonly tlasNodes: Uint32Array; // 8 u32 words (32 bytes) per node
  readonly tlasInstanceIndices: Uint32Array;
  readonly tlasBlasRoots: Uint32Array;
  /** World-to-local matrices, 16 floats per instance. */
  readonly tlasInstanceWorldToLocal: Float32Array;
  /** Local-to-world matrices, 16 floats per instance. */
  readonly tlasInstanceLocalToWorld: Float32Array;
  /** Source primitive/instance addresses parallel to canonical packed instance slots. */
  readonly instancePrimitiveIndices: Uint32Array;
  readonly instanceSourceIndices: Uint32Array;
  /** Encoded optical boundary-id base for each full-tier packed instance. */
  readonly opticalInstanceBoundaryIdBasePlusOne: Uint32Array;
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
  readonly environmentTint: readonly [number, number, number];
  /** CPU-only power proxy retained so emitter-only mutations rebuild the light
   * tree from the same environment integral without rebaking the map. */
  readonly environmentLightTreePower: number;
  /** Exact represented PMF for the optional environment candidate in medium NEE. */
  readonly environmentDistantProposalPmf: number;
  /**
   * H14-E: map-backed environment-radiance intensity multiplier.
   * Value = `scene.environment.intensity ?? 1` when a valid HDRI is present; 0 otherwise.
   * Uploaded to `params.environmentHdriIntensity`.
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
  readonly tlasNodesBuffer: GPUBuffer;
  readonly tlasInstanceIndicesBuffer: GPUBuffer;
  readonly tlasBlasRootsBuffer: GPUBuffer;
  readonly tlasInstanceWorldToLocalBuffer: GPUBuffer;
  readonly tlasInstanceLocalToWorldBuffer: GPUBuffer;
  readonly opticalInstanceBoundaryIdBasePlusOneBuffer: GPUBuffer;
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
   *  honour `GpuMemoryBreakdown`'s invariants: `bufferBytes` (all 32 scene
   *  STORAGE buffers) + `textureBytesByFormat` (the three material arrays, keyed
   *  by their actual `GPUTextureFormat`). Read off the CURRENT handles (not a
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
  { key: 'triMaterialPayload', bufferField: 'triMaterialIdsBuffer',     label: 'vitrum.pt-webgpu.scene.triMaterialIds' },
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
  // ── WS2 light tree ────────────────────────────────────────────────────────
  { key: 'lightTreeNodes', bufferField: 'lightTreeBuffer', label: 'vitrum.pt-webgpu.scene.lightTree' },
  // ── P2 per-vertex UVs/tangents/colors + material texture descriptors ─────
  { key: 'uvs',                    bufferField: 'uvsBuffer',                    label: 'vitrum.pt-webgpu.scene.uvs' },
  { key: 'tangents',               bufferField: 'tangentsBuffer',               label: 'vitrum.pt-webgpu.scene.tangents' },
  { key: 'colors',                 bufferField: 'colorsBuffer',                 label: 'vitrum.pt-webgpu.scene.colors' },
  { key: 'materialTexDescriptors', bufferField: 'materialTexDescriptorsBuffer', label: 'vitrum.pt-webgpu.scene.materialTexDescriptors' },
  // ── TLAS (must be contiguous at the END) ─────────────────────────────────
  { key: 'tlasNodes',                  bufferField: 'tlasNodesBuffer',                  label: 'vitrum.pt-webgpu.scene.tlasNodes' },
  { key: 'tlasInstanceIndices',        bufferField: 'tlasInstanceIndicesBuffer',        label: 'vitrum.pt-webgpu.scene.tlasInstanceIndices' },
  { key: 'tlasBlasRoots',             bufferField: 'tlasBlasRootsBuffer',             label: 'vitrum.pt-webgpu.scene.tlasBlasRoots' },
  { key: 'tlasInstanceWorldToLocal',   bufferField: 'tlasInstanceWorldToLocalBuffer',   label: 'vitrum.pt-webgpu.scene.tlasInstanceWorldToLocal' },
  { key: 'tlasInstanceLocalToWorld',   bufferField: 'tlasInstanceLocalToWorldBuffer',   label: 'vitrum.pt-webgpu.scene.tlasInstanceLocalToWorld' },
  { key: 'opticalInstanceBoundaryIdBasePlusOne', bufferField: 'opticalInstanceBoundaryIdBasePlusOneBuffer', label: 'vitrum.pt-webgpu.scene.opticalInstanceBoundaryIdBasePlusOne' },
] as const;


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
 * Pack the public shape-specific scalar tuples into the two-vec4 ABI decoded
 * by every pt-webgpu analytic consumer.
 *
 * Public tuples are intentionally compact (for example box is
 * [center.xyz, halfExtent.xyz]); the shader ABI groups geometric vectors
 * instead (box p0=center, p1=halfExtent; capsule p0=a, p1=[b,radius]).
 * Sequentially copying the public tuple therefore corrupts box/capsule.
 */
function packAnalyticParamVec4s(
  shape: Extract<ScenePrimitive, { kind: 'analytic' }>['shape'],
  p: Float32Array,
): readonly number[] {
  switch (shape) {
    case 'sphere':
      return [p[0]!, p[1]!, p[2]!, p[3]!, 0, 0, 0, 0];
    case 'box':
      return [p[0]!, p[1]!, p[2]!, 0, p[3]!, p[4]!, p[5]!, 0];
    case 'capsule':
      return [p[0]!, p[1]!, p[2]!, 0, p[3]!, p[4]!, p[5]!, p[6]!];
    case 'cylinder':
      return [p[0]!, p[1]!, p[2]!, p[3]!, p[4]!, 0, 0, 0];
    case 'h-channel-came':
      return [p[0]!, p[1]!, p[2]!, p[3]!, 0, 0, 0, 0];
  }
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
  /** Include the D65 spectral-emission expansion in environment envelopes. */
  readonly spectralEnabled?: boolean;

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
  /** Operation-scoped immutable material/TextureRef observations shared by the
   * MNEE gates, descriptor collection, and atlas-source metadata. */
  readonly materialInputSnapshotContext?: MaterialInputSnapshotContext;

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
    const foldedRadiance = packRadianceRgbScaleF32(
      emitter.color,
      emitter.intensity,
      `@vitrum/pt-webgpu emitter "${emitter.id}" camera-visible fold`,
    ).scaled;
    // Pre-multiply intensity so `emissive · emissiveIntensity` == NEE radiance.
    // Keep the primitive's emissiveMap: both the forward hit and mapped proposal
    // apply that same texture sample to this scalar/base radiance.
    const foldedMat = {
      ...primitive.material,
      emissive: cameraVisibleEmitters
        ? foldedRadiance
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
 * tangent, UV, and color morph deltas; solved tangents/UVs/colors are
 * preserved for material evaluation.
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
        ...(solved.colors ? { colors: solved.colors } : {}),
        ...(solved.colorSets ? { colorSets: solved.colorSets } : {}),
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

interface MergedScenePackAddressing {
  readonly triangleSourceIndices: Uint32Array;
  readonly trianglePrimitiveIndices: Uint32Array;
  readonly instancePrimitiveIndices: Uint32Array;
  readonly instanceSourceIndices: Uint32Array;
  /** Canonical merged-range instance slot for each BVH-reordered triangle. */
  readonly packedInstanceIndexByTriangle: Uint32Array;
}

export function packTriangleMaterialPayload(
  materialIds: Uint32Array,
  representedPrimitiveInstanceIds: Uint32Array,
): Uint32Array {
  if (materialIds.length !== representedPrimitiveInstanceIds.length) {
    throw new RangeError(
      '@vitrum/pt-webgpu: triangle material and represented-instance payloads must be parallel.',
    );
  }
  const payload = new Uint32Array(materialIds.length * 2);
  for (let triangle = 0; triangle < materialIds.length; triangle += 1) {
    const representedId = representedPrimitiveInstanceIds[triangle]!;
    if (representedId === 0) {
      throw new RangeError(
        `@vitrum/pt-webgpu: triangle ${triangle} has no represented primitive-instance identity.`,
      );
    }
    payload[triangle * 2] = materialIds[triangle]!;
    payload[triangle * 2 + 1] = representedId;
  }
  return payload;
}

/**
 * Reconstruct the exact source addresses that the shared optical-boundary
 * packer expects from the merged world-space stream. A merged range is one
 * represented primitive instance, while `bvhTriToMergedTri` supplies the
 * inverse permutation needed to recover the source triangle ordinal.
 */
function mergedScenePackAddressing(
  scene: Scene,
  merged: ReturnType<typeof mergeWorldSpaceFromCore>,
): MergedScenePackAddressing {
  const invalid = 0xffff_ffff;
  const primitiveIndexById = new Map(
    scene.primitives.map((primitive, index) => [primitive.id, index] as const),
  );
  const rangeByMergedTriangle = new Uint32Array(merged.triangleCount);
  rangeByMergedTriangle.fill(invalid);
  const instancePrimitiveIndices = new Uint32Array(merged.meshVertexRanges.length);
  const instanceSourceIndices = new Uint32Array(merged.meshVertexRanges.length);

  for (let rangeIndex = 0; rangeIndex < merged.meshVertexRanges.length; rangeIndex += 1) {
    const range = merged.meshVertexRanges[rangeIndex]!;
    const primitiveId = range.sourcePrimitiveId ?? range.name;
    const primitiveIndex = primitiveIndexById.get(primitiveId);
    if (primitiveIndex === undefined) {
      throw new RangeError(
        `@vitrum/pt-webgpu: merged range ${rangeIndex} references missing primitive "${primitiveId}".`,
      );
    }
    instancePrimitiveIndices[rangeIndex] = primitiveIndex;
    instanceSourceIndices[rangeIndex] = range.sourceInstanceIndex ?? 0;
    const end = range.triStart + range.triCount;
    if (
      !Number.isSafeInteger(range.triStart) || !Number.isSafeInteger(range.triCount) ||
      range.triStart < 0 || range.triCount < 0 || end > merged.triangleCount
    ) {
      throw new RangeError(
        `@vitrum/pt-webgpu: merged range ${rangeIndex} has an invalid triangle span.`,
      );
    }
    for (let mergedTriangle = range.triStart; mergedTriangle < end; mergedTriangle += 1) {
      if (rangeByMergedTriangle[mergedTriangle] !== invalid) {
        throw new RangeError(
          `@vitrum/pt-webgpu: merged triangle ${mergedTriangle} belongs to multiple source ranges.`,
        );
      }
      rangeByMergedTriangle[mergedTriangle] = rangeIndex;
    }
  }

  const triangleSourceIndices = new Uint32Array(merged.triangleCount);
  const trianglePrimitiveIndices = new Uint32Array(merged.triangleCount);
  const packedInstanceIndexByTriangle = new Uint32Array(merged.triangleCount);
  if (merged.bvhTriToMergedTri.length !== merged.triangleCount) {
    throw new RangeError(
      '@vitrum/pt-webgpu: merged BVH source-triangle map does not match triangleCount.',
    );
  }
  for (let triangle = 0; triangle < merged.triangleCount; triangle += 1) {
    const mergedTriangle = merged.bvhTriToMergedTri[triangle]!;
    const rangeIndex = rangeByMergedTriangle[mergedTriangle] ?? invalid;
    const range = rangeIndex === invalid ? undefined : merged.meshVertexRanges[rangeIndex];
    if (range == null) {
      throw new RangeError(
        `@vitrum/pt-webgpu: merged triangle ${mergedTriangle} has no source range.`,
      );
    }
    triangleSourceIndices[triangle] = mergedTriangle - range.triStart;
    trianglePrimitiveIndices[triangle] = instancePrimitiveIndices[rangeIndex]!;
    packedInstanceIndexByTriangle[triangle] = rangeIndex;
  }
  return {
    triangleSourceIndices,
    trianglePrimitiveIndices,
    instancePrimitiveIndices,
    instanceSourceIndices,
    packedInstanceIndexByTriangle,
  };
}

function packMergedMaterial(
  material: MaterialSpec & { readonly castShadow?: boolean },
): number[] {
  return materialToPackedVec4s(material, { castShadow: material.castShadow });
}

interface FiniteSceneAabb {
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
}

interface PackedAnalyticBounds {
  readonly analyticHeaders: ArrayLike<number>;
  readonly analyticParams: ArrayLike<number>;
  readonly analyticLocalToWorld: ArrayLike<number>;
}

function finiteRootAabbFromFloatWords(
  words: ArrayLike<number>,
): FiniteSceneAabb | null {
  if (words.length < BVH_NODE_FLOATS) return null;
  const minX = words[0] ?? 0;
  const minY = words[1] ?? 0;
  const minZ = words[2] ?? 0;
  const maxX = words[3] ?? 0;
  const maxY = words[4] ?? 0;
  const maxZ = words[5] ?? 0;
  if (![minX, minY, minZ, maxX, maxY, maxZ].every(Number.isFinite)) return null;
  if (minX > maxX || minY > maxY || minZ > maxZ) return null;
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

function mergeFiniteSceneAabbs(
  a: FiniteSceneAabb | null,
  b: FiniteSceneAabb | null,
): FiniteSceneAabb | null {
  if (a == null) return b;
  if (b == null) return a;
  return {
    min: [
      Math.min(a.min[0], b.min[0]),
      Math.min(a.min[1], b.min[1]),
      Math.min(a.min[2], b.min[2]),
    ],
    max: [
      Math.max(a.max[0], b.max[0]),
      Math.max(a.max[1], b.max[1]),
      Math.max(a.max[2], b.max[2]),
    ],
  };
}

function packedAnalyticLocalCenterHalfExtent(
  shapeId: number,
  p0: readonly [number, number, number, number],
  p1: readonly [number, number, number, number],
): {
  readonly center: readonly [number, number, number];
  readonly half: readonly [number, number, number];
} {
  switch (shapeId) {
    case 1:
      return {
        center: [p0[0], p0[1], p0[2]],
        half: [p0[3], p0[3], p0[3]],
      };
    case 2:
      return {
        center: [p0[0], p0[1], p0[2]],
        half: [p1[0], p1[1], p1[2]],
      };
    case 3: {
      const center: [number, number, number] = [
        p0[0] * 0.5 + p1[0] * 0.5,
        p0[1] * 0.5 + p1[1] * 0.5,
        p0[2] * 0.5 + p1[2] * 0.5,
      ];
      return {
        center,
        half: [
          Math.abs(p1[0] * 0.5 - p0[0] * 0.5) + p1[3],
          Math.abs(p1[1] * 0.5 - p0[1] * 0.5) + p1[3],
          Math.abs(p1[2] * 0.5 - p0[2] * 0.5) + p1[3],
        ],
      };
    }
    case 4:
      return {
        center: [p0[0], p0[1], p0[2]],
        half: [p0[3], p1[0], p0[3]],
      };
    case 5:
      return {
        center: [0, 0, 0],
        half: [p0[0] * 0.5, p0[2] * 0.5, p0[1] * 0.5],
      };
    default:
      throw new RangeError(
        `pt-webgpu packed analytic bounds received unknown shape id ${shapeId}.`,
      );
  }
}

function packedAnalyticSceneAabb(
  packed: PackedAnalyticBounds,
): FiniteSceneAabb | null {
  if (packed.analyticHeaders.length % 4 !== 0) {
    throw new RangeError('pt-webgpu analytic header buffer is not vec4-aligned.');
  }
  const analyticCount = packed.analyticHeaders.length / 4;
  if (packed.analyticLocalToWorld.length < analyticCount * 16) {
    throw new RangeError('pt-webgpu analytic transform buffer is truncated.');
  }
  let bounds: FiniteSceneAabb | null = null;
  for (let analyticIndex = 0; analyticIndex < analyticCount; analyticIndex += 1) {
    const headerBase = analyticIndex * 4;
    const shapeId = packed.analyticHeaders[headerBase] ?? 0;
    const paramVec4Offset = packed.analyticHeaders[headerBase + 2] ?? -1;
    if (
      !Number.isInteger(shapeId) ||
      !Number.isInteger(paramVec4Offset) ||
      paramVec4Offset < 0
    ) {
      throw new RangeError('pt-webgpu analytic header contains an invalid shape or parameter offset.');
    }
    const paramBase = paramVec4Offset * 4;
    if (paramBase + 8 > packed.analyticParams.length) {
      throw new RangeError('pt-webgpu analytic parameter buffer is truncated.');
    }
    const p0: [number, number, number, number] = [
      packed.analyticParams[paramBase] ?? Number.NaN,
      packed.analyticParams[paramBase + 1] ?? Number.NaN,
      packed.analyticParams[paramBase + 2] ?? Number.NaN,
      packed.analyticParams[paramBase + 3] ?? Number.NaN,
    ];
    const p1: [number, number, number, number] = [
      packed.analyticParams[paramBase + 4] ?? Number.NaN,
      packed.analyticParams[paramBase + 5] ?? Number.NaN,
      packed.analyticParams[paramBase + 6] ?? Number.NaN,
      packed.analyticParams[paramBase + 7] ?? Number.NaN,
    ];
    const local = packedAnalyticLocalCenterHalfExtent(shapeId, p0, p1);
    const matrixBase = analyticIndex * 16;
    const m = (index: number): number =>
      packed.analyticLocalToWorld[matrixBase + index] ?? Number.NaN;
    const worldCenter: [number, number, number] = [
      m(0) * local.center[0] + m(4) * local.center[1] +
        m(8) * local.center[2] + m(12),
      m(1) * local.center[0] + m(5) * local.center[1] +
        m(9) * local.center[2] + m(13),
      m(2) * local.center[0] + m(6) * local.center[1] +
        m(10) * local.center[2] + m(14),
    ];
    const worldHalf: [number, number, number] = [
      Math.abs(m(0)) * local.half[0] + Math.abs(m(4)) * local.half[1] +
        Math.abs(m(8)) * local.half[2],
      Math.abs(m(1)) * local.half[0] + Math.abs(m(5)) * local.half[1] +
        Math.abs(m(9)) * local.half[2],
      Math.abs(m(2)) * local.half[0] + Math.abs(m(6)) * local.half[1] +
        Math.abs(m(10)) * local.half[2],
    ];
    const analyticBounds: FiniteSceneAabb = {
      min: [
        worldCenter[0] - worldHalf[0],
        worldCenter[1] - worldHalf[1],
        worldCenter[2] - worldHalf[2],
      ],
      max: [
        worldCenter[0] + worldHalf[0],
        worldCenter[1] + worldHalf[1],
        worldCenter[2] + worldHalf[2],
      ],
    };
    const boundWords = [...analyticBounds.min, ...analyticBounds.max];
    if (
      !boundWords.every(Number.isFinite) ||
      !boundWords.every((value) => Number.isFinite(Math.fround(value)))
    ) {
      throw new RangeError(
        `pt-webgpu analytic ${analyticIndex} bounds are not representable as finite f32.`,
      );
    }
    bounds = mergeFiniteSceneAabbs(bounds, analyticBounds);
  }
  return bounds;
}

function centerRadiusFromFiniteSceneAabb(bounds: FiniteSceneAabb): {
  readonly center: readonly [number, number, number];
  readonly radius: number;
} {
  const center: [number, number, number] = [
    bounds.min[0] * 0.5 + bounds.max[0] * 0.5,
    bounds.min[1] * 0.5 + bounds.max[1] * 0.5,
    bounds.min[2] * 0.5 + bounds.max[2] * 0.5,
  ];
  const halfX = bounds.max[0] * 0.5 - bounds.min[0] * 0.5;
  const halfY = bounds.max[1] * 0.5 - bounds.min[1] * 0.5;
  const halfZ = bounds.max[2] * 0.5 - bounds.min[2] * 0.5;
  return {
    center,
    radius: resolvePtWebgpuSceneRadius(
      center,
      Math.hypot(halfX, halfY, halfZ),
    ),
  };
}

function sceneCenterRadiusFromPack(
  pack: Pick<ScenePackResult, 'bvhNodes'> &
    Partial<Pick<ScenePackResult, 'tlasNodes'>> &
    Partial<PackedAnalyticBounds>,
): {
  readonly center: readonly [number, number, number];
  readonly radius: number;
} {
  let meshBounds: FiniteSceneAabb | null = null;
  if (pack.tlasNodes && pack.tlasNodes.length >= BVH_NODE_FLOATS) {
    const tlasWords = new Float32Array(
      pack.tlasNodes.buffer,
      pack.tlasNodes.byteOffset,
      pack.tlasNodes.length,
    );
    meshBounds = finiteRootAabbFromFloatWords(tlasWords);
  }
  if (meshBounds == null) {
    meshBounds = finiteRootAabbFromFloatWords(pack.bvhNodes);
  }
  const analyticBounds = (
    pack.analyticHeaders != null &&
    pack.analyticParams != null &&
    pack.analyticLocalToWorld != null
  )
    ? packedAnalyticSceneAabb({
        analyticHeaders: pack.analyticHeaders,
        analyticParams: pack.analyticParams,
        analyticLocalToWorld: pack.analyticLocalToWorld,
      })
    : null;
  const combined = mergeFiniteSceneAabbs(meshBounds, analyticBounds);
  return combined == null
    ? { center: [0, 0, 0], radius: 1 }
    : centerRadiusFromFiniteSceneAabb(combined);
}

/**
 * Recompute the combined mesh + analytic scene sphere from the exact packed
 * arrays consumed by the GPU. Incremental analytic-transform mutations use
 * this before publication so pseudo-distant launch domains and SPPM never
 * retain mesh-only/stale bounds.
 */
export function sceneCenterRadiusForPackedGeometry(
  packed: Pick<
    PackedSceneData,
    | 'bvhNodes'
    | 'tlasNodes'
    | 'analyticHeaders'
    | 'analyticParams'
    | 'analyticLocalToWorld'
  >,
): {
  readonly center: readonly [number, number, number];
  readonly radius: number;
} {
  return sceneCenterRadiusFromPack(packed);
}

interface PackedCwbvhSceneData {
  readonly cwbvhNodeBounds: Float32Array;
  readonly cwbvhChildBoundsPacked: Uint32Array;
  readonly cwbvhChildMeta: Uint32Array;
  readonly cwbvhChildCount: Uint32Array;
  readonly cwbvhTlasBlasRoots: Uint32Array;
  readonly cwbvhBinaryRootToWideRoot: Uint32Array;
  readonly cwbvhNodeCount: number;
  readonly warnings: readonly string[];
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
  const warnings: string[] = [];
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
    if (cwbvh.cwbvhBuildStatus.traversal === 'binary-fallback') {
      warnings.push(
        `CWBVH BLAS root ${span.binaryRoot} requires binary traversal fallback: ` +
          `${cwbvh.cwbvhBuildStatus.reason ?? 'unknown'} ` +
          `(required stack ${cwbvh.cwbvhBuildStatus.maxTraversalStackEntries}, ` +
          `capacity ${cwbvh.cwbvhBuildStatus.traversalStackCapacity}).`,
      );
    }
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
    warnings,
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
  const materialInputSnapshotContext =
    options.materialInputSnapshotContext ?? createMaterialInputSnapshotContext();
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
  const renderFallbackScene = applyAnalyticRenderFallbacks(filteredScene, warnings);
  assertThinFilmLayerCapacity(renderFallbackScene);
  // Item 1 — apply CPU LBS to skinned-mesh primitives so packSceneFromCore uses
  // solved (deformed) positions instead of rest-pose. morphTargets are also
  // handled by solveSkin (blend applied before LBS).
  const scene = applySolveSkinToScene(renderFallbackScene);
  const opticalTopology = assertOpticalMediumTopology(scene, {
    maxNestedMedia: 8,
    analyticGeometry: 'generated-triangle',
    transformArithmetic: geometryMode === 'merged'
      ? 'merged-world-f64-to-f32'
      : 'tlas-shader-f32',
    backend: '@vitrum/pt-webgpu',
    method: options.warningMethod ?? 'setScene',
  });
  if (options.includeMneeFacetCandidates) {
    assertMneeInterfaceDomainSupported(scene, materialInputSnapshotContext);
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
      analyticParams.push(...packAnalyticParamVec4s(
        primitive.shape,
        primitive.params,
      ));
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
  let mergedOpticalGeometry: ReturnType<typeof mergeWorldSpaceFromCore> | undefined;
  const emitterMaterialIds = new Map(meshMaterialIds);
  const rawGeo = geometryMode === 'merged'
    ? (() => {
        const merged = mergeWorldSpaceFromCore(scene, {
          positionStride: 4,
          splitMaterialsByCastShadow: true,
          bakeConstantVertexColorIntoMaterial: true,
        });
        mergedOpticalGeometry = merged;
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
        const addressing = mergedScenePackAddressing(scene, merged);
        return {
          positions: merged.positions,
          normals: merged.normals,
          uvs: packMergedUvs(scene, merged),
          tangents: merged.tangents,
          colors: merged.colors,
          indices: padTriangleIndicesToVec4(merged.indices),
          triMaterialIds: merged.triMaterialId,
          triangleSourceIndices: addressing.triangleSourceIndices,
          trianglePrimitiveIndices: addressing.trianglePrimitiveIndices,
          bvhNodes: merged.bvhNodes,
          triangleCount: merged.triangleCount,
          tlasNodes: new Uint32Array(0),
          tlasInstanceIndices: new Uint32Array(0),
          tlasBlasRoots: new Uint32Array(0),
          tlasInstanceWorldToLocal: new Float32Array(0),
          tlasInstanceLocalToWorld: new Float32Array(0),
          instancePrimitiveIndices: addressing.instancePrimitiveIndices,
          instanceSourceIndices: addressing.instanceSourceIndices,
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
  const opticalBoundaryIds = geometryMode === 'merged'
    ? packMergedOpticalMediumBoundaryIds(
        scene,
        mergedOpticalGeometry!,
        opticalTopology,
      )
    : packOpticalMediumBoundaryIds(scene, rawGeo, opticalTopology);
  const opticalIndices = new Uint32Array(rawGeo.indices);
  for (let triangle = 0; triangle < rawGeo.triangleCount; triangle += 1) {
    const encodedBoundary =
      opticalBoundaryIds.triangleComponentIndexPlusOne[triangle] ?? 0;
    opticalIndices[triangle * 4 + 3] = encodedBoundary;
  }
  const geo: ScenePackResult = { ...rawGeo, indices: opticalIndices };
  const triangleRepresentedPrimitiveInstanceIds =
    opticalBoundaryIds.triangleRepresentedPrimitiveInstanceIds;
  const triMaterialPayload = packTriangleMaterialPayload(
    geo.triMaterialIds.subarray(0, geo.triangleCount),
    triangleRepresentedPrimitiveInstanceIds,
  );
  const opticalInstanceBoundaryIdBasePlusOne = geometryMode === 'merged'
    ? new Uint32Array(0)
    : opticalBoundaryIds.instanceBoundaryIdBasePlusOne;
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
      materialInputSnapshotContext,
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
  const packedAnalyticHeaders = new Float32Array(analyticHeaders);
  const packedAnalyticLocalToWorld = new Float32Array(analyticLocalToWorld);
  const packedAnalyticWorldToLocal = new Float32Array(analyticWorldToLocal);
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
  const texCollection = collectMaterialTextures(
    materialSpecs,
    materialInputSnapshotContext,
  );
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
  const environmentDistantProposalPmf = applyDistantDirectProposalPmf(
    emitArrays.directionalLightsData,
    emitArrays.directionalLightCount,
    environment.lightTreePower,
    environment.hasHdri,
  );
  assertPtWebgpuEnvironmentMaterialEnvelopeF32(
    environment.hdriTexels,
    environment.hdriIntensity,
    materialSpecs.map((material) => material.envMapIntensity ?? 1),
    options.spectralEnabled === true,
  );
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
  const sceneBounds = sceneCenterRadiusFromPack({
    ...geo,
    analyticHeaders: packedAnalyticHeaders,
    analyticParams: packedAnalyticParams,
    analyticLocalToWorld: packedAnalyticLocalToWorld,
  });
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
  warnings.push(...cwbvh.warnings);

  return {
    positions: geo.positions,
    normals: geo.normals,
    uvs: gpuUvs,
    uvSetTexCoords: texCollection.uvSetTexCoords,
    tangents: geo.tangents,
    colors: geo.colors,
    indices: geo.indices,
    triMaterialIds: geo.triMaterialIds,
    triMaterialPayload,
    triangleRepresentedPrimitiveInstanceIds,
    triangleSourceIndices: geo.triangleSourceIndices,
    trianglePrimitiveIndices: geo.trianglePrimitiveIndices,
    materials: new Float32Array(materials),
    cwbvhNodeBounds: cwbvh.cwbvhNodeBounds,
    cwbvhChildBoundsPacked: cwbvh.cwbvhChildBoundsPacked,
    cwbvhChildMeta: cwbvh.cwbvhChildMeta,
    cwbvhChildCount: cwbvh.cwbvhChildCount,
    cwbvhTlasBlasRoots: cwbvh.cwbvhTlasBlasRoots,
    cwbvhBinaryRootToWideRoot: cwbvh.cwbvhBinaryRootToWideRoot,
    cwbvhNodeCount: cwbvh.cwbvhNodeCount,
    cwbvhWarnings: cwbvh.warnings,
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
    instancePrimitiveIndices: geo.instancePrimitiveIndices,
    instanceSourceIndices: geo.instanceSourceIndices,
    opticalInstanceBoundaryIdBasePlusOne,
    primitiveTlasBindings: geo.primitiveTlasBindings,
    analyticHeaders: packedAnalyticHeaders,
    analyticParams: packedAnalyticParams,
    analyticLocalToWorld: packedAnalyticLocalToWorld,
    analyticWorldToLocal: packedAnalyticWorldToLocal,
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
    environmentTint: environment.tint,
    environmentLightTreePower: environment.lightTreePower,
    environmentDistantProposalPmf,
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
    triangleSourceIndices: packed.triangleSourceIndices,
    trianglePrimitiveIndices: packed.trianglePrimitiveIndices,
    bvhNodes: packed.bvhNodes,
    triangleCount: packed.triangleCount,
    tlasNodes: packed.tlasNodes,
    tlasInstanceIndices: packed.tlasInstanceIndices,
    tlasBlasRoots: packed.tlasBlasRoots,
    tlasInstanceWorldToLocal: packed.tlasInstanceWorldToLocal,
    tlasInstanceLocalToWorld: packed.tlasInstanceLocalToWorld,
    instancePrimitiveIndices: packed.instancePrimitiveIndices,
    instanceSourceIndices: packed.instanceSourceIndices,
    tlasNodeCount: Math.floor(packed.tlasNodes.length / BVH_NODE_FLOATS),
    primitiveTlasBindings: packed.primitiveTlasBindings,
    warnings: packed.warnings,
  };
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
  cwbvhWarnings?: readonly string[];
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
  opticalInstanceBoundaryIdBasePlusOneBuffer: GPUBuffer;
  tlasNodes: Uint32Array;
  tlasInstanceIndices: Uint32Array;
  tlasBlasRoots: Uint32Array;
  tlasInstanceWorldToLocal: Float32Array;
  tlasInstanceLocalToWorld: Float32Array;
  opticalInstanceBoundaryIdBasePlusOne: Uint32Array;
  instancePrimitiveIndices: Uint32Array;
  instanceSourceIndices: Uint32Array;

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
  triMaterialPayload: Uint32Array;
  triangleRepresentedPrimitiveInstanceIds: Uint32Array;
  triangleSourceIndices: Uint32Array;
  trianglePrimitiveIndices: Uint32Array;
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
  directionalLightsData: Float32Array;
  pointLightsData: Float32Array;
  spotLightsData: Float32Array;
  rectAreaLightsData: Float32Array;
  meshAreaLightsData: Float32Array;

  // ── Emitter counts (incremental emitter patches) ─────────────────────────
  directionalLightCount: number;
  pointLightCount: number;
  spotLightCount: number;
  rectAreaLightCount: number;
  meshAreaLightCount: number;

  // ── Environment fields (incremental environment patches) ─────────────────
  environmentTint: readonly [number, number, number];
  environmentLightTreePower: number;
  environmentDistantProposalPmf: number;
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
 * Stage every dirty range in allocations bounded by `maxBufferSize`, encode
 * every copy into one command buffer, and submit the whole mutation as one
 * ordered batch. No GPU work is created for a byte-identical set.
 */
function prepareInPlaceCopyBatch(
  device: GPUDevice,
  specs: readonly InPlaceCopySpec[],
  protectedResources: readonly object[],
): PreparedInPlaceCopyBatch {
  interface StagedCopy {
    readonly destination: GPUBuffer;
    readonly destinationByteOffset: number;
    readonly stagingByteOffset: number;
    readonly byteLength: number;
    readonly data: ArrayBufferView;
  }
  interface StagingChunk {
    byteLength: number;
    readonly copies: StagedCopy[];
  }

  const rawMaxBufferSize = Number(device.limits?.maxBufferSize);
  const maxStagingByteLength =
    Number.isFinite(rawMaxBufferSize) && rawMaxBufferSize > 0
      ? Math.floor(rawMaxBufferSize / 4) * 4
      : Math.floor(Number.MAX_SAFE_INTEGER / 4) * 4;
  if (maxStagingByteLength < 4) {
    throw new Error(
      `[pt-webgpu] device maxBufferSize (${String(rawMaxBufferSize)}) ` +
        'cannot hold one aligned incremental-copy word',
    );
  }

  const chunks: StagingChunk[] = [];
  let chunk: StagingChunk | undefined;
  const appendCopy = (
    spec: InPlaceCopySpec,
    destinationByteOffset: number,
    dataByteOffset: number,
    byteLength: number,
  ): void => {
    let remaining = byteLength;
    let destinationOffset = destinationByteOffset;
    let sourceOffset = dataByteOffset;
    while (remaining > 0) {
      if (chunk == null || chunk.byteLength === maxStagingByteLength) {
        chunk = { byteLength: 0, copies: [] };
        chunks.push(chunk);
      }
      const available = maxStagingByteLength - chunk.byteLength;
      const copyByteLength = Math.min(remaining, available);
      chunk.copies.push({
        destination: spec.destination,
        destinationByteOffset: destinationOffset,
        stagingByteOffset: chunk.byteLength,
        byteLength: copyByteLength,
        data: new Uint8Array(
          spec.data.buffer,
          spec.data.byteOffset + sourceOffset,
          copyByteLength,
        ),
      });
      chunk.byteLength += copyByteLength;
      destinationOffset += copyByteLength;
      sourceOffset += copyByteLength;
      remaining -= copyByteLength;
    }
  };

  for (const spec of specs) {
    for (const range of spec.ranges) {
      if (
        !Number.isSafeInteger(range.byteOffset) ||
        !Number.isSafeInteger(range.byteLength) ||
        range.byteOffset % 4 !== 0 ||
        range.byteLength <= 0 ||
        range.byteLength % 4 !== 0 ||
        range.byteOffset + range.byteLength > spec.data.byteLength
      ) {
        throw new Error(
          `[pt-webgpu] incremental copy range for ${spec.label} is outside its ` +
            'payload or is not WebGPU-aligned',
        );
      }
      appendCopy(spec, range.byteOffset, range.byteOffset, range.byteLength);
    }
  }

  if (chunks.length === 0) {
    return {
      submit: () => {},
      destroy: () => {},
    };
  }

  const protectedSet = new Set<object>(protectedResources);
  const stagingBuffers: GPUBuffer[] = [];
  try {
    for (const stagingChunk of chunks) {
      const staging = device.createBuffer({
        label: 'vitrum.pt-webgpu.scene.incremental-staging',
        size: stagingChunk.byteLength,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      if (protectedSet.has(staging)) {
        throw new Error(
          '[pt-webgpu] incremental staging allocation aliased a live GPU resource',
        );
      }
      protectedSet.add(staging);
      stagingBuffers.push(staging);

      const stagingBytes = new Uint8Array(stagingChunk.byteLength);
      for (const copy of stagingChunk.copies) {
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
    }

    const encoder = device.createCommandEncoder({
      label: 'vitrum.pt-webgpu.scene.incremental-copy',
    });
    for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex += 1) {
      const staging = stagingBuffers[chunkIndex]!;
      for (const copy of chunks[chunkIndex]!.copies) {
        encoder.copyBufferToBuffer(
          staging,
          copy.stagingByteOffset,
          copy.destination,
          copy.destinationByteOffset,
          copy.byteLength,
        );
      }
    }
    const commandBuffer = encoder.finish();
    let destroyed = false;
    return {
      submit: () => { device.queue.submit([commandBuffer]); },
      destroy: () => {
        if (destroyed) return;
        destroyed = true;
        destroyResourcesBestEffort(stagingBuffers, protectedResources);
      },
    };
  } catch (error) {
    destroyResourcesBestEffort(stagingBuffers, protectedResources);
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
  const triangleRepresentedPrimitiveInstanceIds = new Uint32Array(pack.triangleCount);
  for (let triangle = 0; triangle < pack.triangleCount; triangle += 1) {
    const primitiveIndex = pack.trianglePrimitiveIndices[triangle]!;
    if (primitiveIndex === 0xffff_ffff) {
      throw new RangeError(
        `[pt-webgpu] triangle ${triangle} has an unencodable represented primitive identity`,
      );
    }
    triangleRepresentedPrimitiveInstanceIds[triangle] = primitiveIndex + 1;
  }
  const bounds = sceneCenterRadiusFromPack({
    ...pack,
    analyticHeaders: sb.analyticHeaders,
    analyticParams: sb.analyticParams,
    analyticLocalToWorld: sb.analyticLocalToWorld,
  });
  const next: SceneBufferMutationPatch = {
    positions: pack.positions,
    normals: pack.normals,
    uvs: uvsOverride ?? pack.uvs,
    tangents: pack.tangents,
    colors: pack.colors,
    indices: pack.indices,
    triMaterialIds: pack.triMaterialIds,
    triMaterialPayload: packTriangleMaterialPayload(
      pack.triMaterialIds.subarray(0, pack.triangleCount),
      triangleRepresentedPrimitiveInstanceIds,
    ),
    triangleRepresentedPrimitiveInstanceIds,
    triangleSourceIndices: pack.triangleSourceIndices,
    trianglePrimitiveIndices: pack.trianglePrimitiveIndices,
    bvhNodes: pack.bvhNodes,
    cwbvhNodeBounds: cwbvh.cwbvhNodeBounds,
    cwbvhChildBoundsPacked: cwbvh.cwbvhChildBoundsPacked,
    cwbvhChildMeta: cwbvh.cwbvhChildMeta,
    cwbvhChildCount: cwbvh.cwbvhChildCount,
    cwbvhTlasBlasRoots: cwbvh.cwbvhTlasBlasRoots,
    cwbvhBinaryRootToWideRoot: cwbvh.cwbvhBinaryRootToWideRoot,
    cwbvhNodeCount: cwbvh.cwbvhNodeCount,
    cwbvhWarnings: cwbvh.warnings,
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
      instancePrimitiveIndices: pack.instancePrimitiveIndices,
      instanceSourceIndices: pack.instanceSourceIndices,
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
    | 'instancePrimitiveIndices'
    | 'instanceSourceIndices'
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
    analyticHeaders: sb.analyticHeaders,
    analyticParams: sb.analyticParams,
    analyticLocalToWorld: sb.analyticLocalToWorld,
  });
  return {
    tlasNodes: pack.tlasNodes,
    tlasInstanceIndices: pack.tlasInstanceIndices,
    tlasBlasRoots: pack.tlasBlasRoots,
    tlasInstanceWorldToLocal: pack.tlasInstanceWorldToLocal,
    tlasInstanceLocalToWorld: pack.tlasInstanceLocalToWorld,
    instancePrimitiveIndices: pack.instancePrimitiveIndices,
    instanceSourceIndices: pack.instanceSourceIndices,
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
    | 'instancePrimitiveIndices'
    | 'instanceSourceIndices'
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
  // the three material texture arrays register their textures explicitly.
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

function pushUniqueRadianceFactor(
  factors: RadianceRgb[],
  factor: RadianceRgb,
): void {
  if (factors.some((candidate) =>
    candidate[0] === factor[0] &&
    candidate[1] === factor[1] &&
    candidate[2] === factor[2]
  )) {
    return;
  }
  factors.push(factor);
}

function materialTextureRadianceEnvelope(
  packed: PackedSceneData,
): MaterialTextureRadianceEnvelope {
  const materialCount = Math.floor(
    packed.materialTexDescriptors.length / MATERIAL_TEX_FLOAT_STRIDE,
  );
  const emissiveMap: RadianceRgb[][] = Array.from(
    { length: materialCount },
    () => [],
  );
  const lightMap: RadianceRgb[][] = Array.from(
    { length: materialCount },
    () => [],
  );
  for (let materialId = 0; materialId < materialCount; materialId += 1) {
    const materialBase = materialId * MATERIAL_FLOAT_STRIDE;
    pushUniqueRadianceFactor(emissiveMap[materialId]!, [
      packed.materials[materialBase + 4] ?? 0,
      packed.materials[materialBase + 5] ?? 0,
      packed.materials[materialBase + 6] ?? 0,
    ]);
    const descriptorBase = materialId * MATERIAL_TEX_FLOAT_STRIDE;
    const lightMapIntensity = packed.materialTexDescriptors[descriptorBase + 17] ?? 0;
    pushUniqueRadianceFactor(lightMap[materialId]!, [
      lightMapIntensity,
      lightMapIntensity,
      lightMapIntensity,
    ]);
  }
  for (
    let base = 0;
    base + MESH_AREA_LIGHT_FLOAT_STRIDE <= packed.meshAreaLightsData.length;
    base += MESH_AREA_LIGHT_FLOAT_STRIDE
  ) {
    const materialIdPlusOne = packed.meshAreaLightsData[base + 22] ?? 0;
    if (
      !Number.isInteger(materialIdPlusOne) ||
      materialIdPlusOne <= 0 ||
      materialIdPlusOne > materialCount
    ) {
      continue;
    }
    const materialId = materialIdPlusOne - 1;
    pushUniqueRadianceFactor(emissiveMap[materialId]!, [
      packed.meshAreaLightsData[base + 24] ?? 0,
      packed.meshAreaLightsData[base + 25] ?? 0,
      packed.meshAreaLightsData[base + 26] ?? 0,
    ]);
  }
  return { emissiveMap, lightMap };
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
  const radianceEnvelope = materialTextureRadianceEnvelope(packed);
  const materialTextureUploadPlan = stageMaterialTextureUploadPlan(
    device,
    [
      {
        sources: packed.materialTextureSources,
        format: 'rgba8unorm-srgb',
        layerInfos: packed.materialTextureSourceInfos,
      },
      {
        sources: packed.materialTextureLinearSources,
        format: 'rgba8unorm',
        layerInfos: packed.materialTextureLinearSourceInfos,
      },
      {
        sources: packed.materialTextureEmissiveSources,
        format: 'rgba16float',
        layerInfos: packed.materialTextureEmissiveSourceInfos,
        radianceEnvelope,
      },
    ],
    forbiddenResources,
  );
  const aggregateMaterialTexturePeakBytes = materialTextureUploadPlan.estimatedPeakBytes;
  if (
    !Number.isSafeInteger(aggregateMaterialTexturePeakBytes) ||
    aggregateMaterialTexturePeakBytes > MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES
  ) {
    throw new RangeError(
      `[pt-webgpu] aggregate material-atlas peak ${aggregateMaterialTexturePeakBytes} bytes exceeds ` +
        `${MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES}-byte budget before GPU allocation.`,
    );
  }
  const materialTextureArray = trackMaterialArray(createMaterialTextureArrayFromStaged(
    device,
    materialTextureUploadPlan.arrays[0]!,
  ));
  // Linear array (normal + ORM) — rgba8unorm so the sampler does NOT sRGB-decode.
  const materialLinearArray = trackMaterialArray(createMaterialTextureArrayFromStaged(
    device,
    materialTextureUploadPlan.arrays[1]!,
  ));
  // Dedicated HDR outgoing-radiance array. Per-layer provenance selects sRGB
  // emissive versus linear light-map bytes; raw Float32 remains linear HDR.
  const materialEmissiveArray = trackMaterialArray(createMaterialTextureArrayFromStaged(
    device,
    materialTextureUploadPlan.arrays[2]!,
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
    // `uploaded` (not a captured local) is required for prepared scene-buffer
    // mutations that swap fresh handles onto the struct after size changes, so
    // `destroy` never touches a stale (leaked/double-freed) handle.
    // Non-resized buffers (materials / analytic / environment / descriptors) are
    // never reassigned, so reading them off `uploaded` is identical to the
    // previously-captured locals. The three material texture arrays are destroyed
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
    // handles included) + every retained mip of the three material texture
    // arrays. GPUTexture has no byte-size field, so derive the full mip-chain
    // footprint from its live dimensions, layer count, format, and mip count.
    gpuMemoryBytes: () => {
      let bufferBytes = 0;
      for (const entry of SCENE_BUFFER_REGISTRY) {
        bufferBytes += uploaded[entry.bufferField].size;
      }
      const textureBytesByFormat: Record<string, number> = {};
      const addTex = (t: GPUTexture): void => {
        // Bytes-per-texel by format: rgba8* = 4, rgba16float = 8 (T1-6 emissive).
        const bytesPerTexel = t.format === 'rgba16float' ? 8 : 4;
        let texelCount = 0;
        for (let level = 0; level < t.mipLevelCount; level += 1) {
          texelCount +=
            Math.max(1, Math.floor(t.width / 2 ** level)) *
            Math.max(1, Math.floor(t.height / 2 ** level)) *
            t.depthOrArrayLayers;
        }
        const bytes = texelCount * bytesPerTexel;
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
