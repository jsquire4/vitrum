import type {
  EngineWarning,
  MaterialSpec,
  MaterialSpecPatch,
  Scene,
  ScenePrimitive,
  ScenePrimitivePatch,
} from '@vitrum/core';
import {
  materialDefinesBulkOpticalMedium,
  type WorldSpaceMergeResult,
} from '@vitrum/shared-bvh';
import { packMaterialsTexture } from './materialsTexture.js';
import { LIGHT_PIXELS, packLightsTexture } from './lightsTexture.js';
import { hasMeshAreaLightForPrimitive, packMeshAreaLights, TRI_LIGHT_PIXELS } from './meshAreaLights.js';
import { foldMeshAreaEmittersIntoMaterials } from './foldEmissiveEmitters.js';
import { buildEquirectInfo } from './equirectHdrInfo.js';
import type { UploadedSceneTextures } from './sceneTextures.js';
import { retireTexturesIndependently } from '../gl/resourceRetirement.js';
import {
  buildSceneGeometryTextureData,
  buildRefitSceneGeometryTextures,
  expandAnalyticPrimitiveFallbacks,
  uploadRgba32f,
  uploadRgba32fRect,
  uploadRgba32fArray,
  uploadRgba32ui,
} from './uploadSceneTextures.js';
import {
  materialTextureAtlasLayerCapacities,
  preflightMaterialTextureAtlases,
  textureStorageClassForMapKey,
} from './texturesArray.js';
import type {
  MaterialTextureAtlasLayerMaps,
  TextureAtlasStorageClass,
  TextureSampleColorSpace,
} from './texturesArray.js';
import { packBvhTextureData, squareDim, type BvhTextureData } from './bvhTextureAdapter.js';
import {
  ATTR_LAYER_UV,
  ATTR_LAYER_UV1,
  sameUvAttributeLayout,
} from './uvAttributeLayout.js';
import { computeWebgl2TransportBounds } from './sceneScalePolicy.js';

export const TEXTURE_MAP_FIELDS: ReadonlySet<string> = new Set([
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
]);

const DISPLACEMENT_GEOMETRY_MATERIAL_FIELDS: ReadonlySet<string> = new Set([
  'displacementMap',
  'displacementScale',
  'displacementBias',
]);

const TEXTURE_MAP_COLOR_SPACE: Readonly<Record<string, TextureSampleColorSpace>> = Object.freeze({
  baseColorMap: 'srgb',
  normalMap: 'linear',
  roughnessMap: 'linear',
  metallicMap: 'linear',
  transmissionMap: 'linear',
  thicknessMap: 'linear',
  emissiveMap: 'srgb',
  alphaMap: 'linear',
  aoMap: 'linear',
  clearcoatMap: 'linear',
  clearcoatRoughnessMap: 'linear',
  clearcoatNormalMap: 'linear',
  sheenColorMap: 'srgb',
  sheenRoughnessMap: 'linear',
  iridescenceMap: 'linear',
  iridescenceThicknessMap: 'linear',
  anisotropyMap: 'linear',
  specularColorMap: 'srgb',
  specularIntensityMap: 'linear',
  bumpMap: 'linear',
  displacementMap: 'linear',
  lightMap: 'linear',
});

const LAYER_TEXTURE_MAP_FIELDS = [
  { layer: 'frontLayer', field: 'normalMap', path: 'frontLayer.normalMap', colorSpace: 'linear' },
  { layer: 'backLayer', field: 'normalMap', path: 'backLayer.normalMap', colorSpace: 'linear' },
] as const satisfies readonly {
  readonly layer: 'frontLayer' | 'backLayer';
  readonly field: 'normalMap';
  readonly path: string;
  readonly colorSpace: TextureSampleColorSpace;
}[];

const GEOMETRY_TEXTURE_REFRESH_FIELDS: ReadonlySet<string> = new Set([
  'transform',
  'positions',
  'normals',
  'indices',
  'uvs',
  'uv1',
  'uvSets',
  'tangents',
  'colors',
  'vertexColorSet',
  'instances',
  'bones',
  'boneInverses',
  'skinIndices',
  'skinWeights',
  'skinInfluencesPerVertex',
  'morphTargets',
  'morphTargetNormals',
  'morphTargetTangents',
  'morphTargetUvs',
  'morphTargetUv1s',
  'morphTargetUvSets',
  'morphTargetColors',
  'morphTargetColorSets',
  'morphWeights',
]);

export interface WebGl2MutationSwap {
  readonly textures: UploadedSceneTextures;
  readonly geoPack?: WorldSpaceMergeResult;
  readonly scene?: Scene;
  readonly deleteOldTextures: readonly (WebGLTexture | null)[];
  readonly structuredWarnings?: readonly EngineWarning[];
  readonly mutationFallback?: {
    readonly fallbackReason: string;
    readonly nativePatchMissing: string;
    readonly textureRefreshMode?: string;
  };
}

type RectFloatTextureData = { readonly data: Float32Array; readonly width: number; readonly height: number };

function requiresFullRepresentedEmitterProposalTransaction(): boolean {
  return true;
}

function isMeshLikePrimitive(p: ScenePrimitive | undefined): p is Extract<
  ScenePrimitive,
  { kind: 'mesh' | 'instanced-mesh' | 'skinned-mesh' }
> {
  return p?.kind === 'mesh' || p?.kind === 'instanced-mesh' || p?.kind === 'skinned-mesh';
}

function canFastPathMaterialPatch(patch: ScenePrimitivePatch): boolean {
  let sawMaterialLaneField = false;
  for (const key of Object.keys(patch)) {
    if (key === 'id' || key === 'kind') continue;
    if (key === 'material' || key === 'castShadow') {
      sawMaterialLaneField = true;
      continue;
    }
    return false;
  }
  if (patch.material != null) {
    for (const field of Object.keys(patch.material as unknown as Record<string, unknown>)) {
      if (DISPLACEMENT_GEOMETRY_MATERIAL_FIELDS.has(field)) return false;
    }
  }
  return sawMaterialLaneField;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value) && !ArrayBuffer.isView(value);
}

function hasOwn(value: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

interface TextureMapPatchEntry {
  readonly path: string;
  readonly colorSpace: TextureSampleColorSpace;
  readonly storageClass: TextureAtlasStorageClass;
  readonly value: unknown;
}

function textureMapPatchEntries(material: Record<string, unknown>): TextureMapPatchEntry[] {
  const entries: TextureMapPatchEntry[] = [];
  for (const field of Object.keys(material)) {
    if (!TEXTURE_MAP_FIELDS.has(field)) continue;
    entries.push({
      path: field,
      colorSpace: TEXTURE_MAP_COLOR_SPACE[field] ?? 'linear',
      storageClass: textureStorageClassForMapKey(field as keyof MaterialSpec),
      value: material[field],
    });
  }
  for (const descriptor of LAYER_TEXTURE_MAP_FIELDS) {
    if (!hasOwn(material, descriptor.layer)) continue;
    const layer = material[descriptor.layer];
    if (isRecord(layer)) {
      if (!hasOwn(layer, descriptor.field)) continue;
      entries.push({
        path: descriptor.path,
        colorSpace: descriptor.colorSpace,
        storageClass: 'ldr',
        value: layer[descriptor.field],
      });
    } else {
      entries.push({
        path: descriptor.path,
        colorSpace: descriptor.colorSpace,
        storageClass: 'ldr',
        value: undefined,
      });
    }
  }
  return entries;
}

function materialTextureValueAt(material: Record<string, unknown>, path: string): unknown {
  const dot = path.indexOf('.');
  if (dot < 0) return material[path];
  const parent = material[path.slice(0, dot)];
  if (!isRecord(parent)) return undefined;
  return parent[path.slice(dot + 1)];
}

export function materialTextureMapPatchFields(patch: ScenePrimitivePatch): string[] {
  if (patch.material == null) return [];
  return textureMapPatchEntries(patch.material as unknown as Record<string, unknown>)
    .map((entry) => entry.path)
    .sort();
}

function texturePatchNeedsAtlasRefresh(
  patch: ScenePrimitivePatch & { material: MaterialSpecPatch },
  materialLayerMap: MaterialTextureAtlasLayerMaps,
): boolean {
  const mat = patch.material as unknown as Record<string, unknown>;
  for (const entry of textureMapPatchEntries(mat)) {
    if (
      textureValueNeedsAtlasRefresh(
        entry.storageClass,
        entry.colorSpace,
        entry.value,
        materialLayerMap,
      )
    ) return true;
  }
  return false;
}

function effectiveUvLayerMap(current: UploadedSceneTextures): ReadonlyMap<number, number> {
  return current.uvLayerByTexCoord ?? new Map<number, number>([
    [0, ATTR_LAYER_UV],
    [1, ATTR_LAYER_UV1],
  ]);
}

function texturePatchNeedsUvLayoutRefresh(
  patch: ScenePrimitivePatch & { material: MaterialSpecPatch },
  current: UploadedSceneTextures,
): boolean {
  const layout = effectiveUvLayerMap(current);
  for (const entry of textureMapPatchEntries(
    patch.material,
  )) {
    if (entry.value == null || typeof entry.value !== 'object') continue;
    const texCoord = (entry.value as { readonly texCoord?: number }).texCoord ?? 0;
    if (!layout.has(texCoord)) return true;
  }
  return false;
}

function textureHandleOf(value: unknown): unknown {
  if (value == null || typeof value !== 'object') return null;
  return (value as { readonly handle?: unknown }).handle ?? null;
}

function texturePatchMayCompactAtlas(
  previousMaterial: MaterialSpec | undefined,
  patch: ScenePrimitivePatch & { material: MaterialSpecPatch },
): boolean {
  if (previousMaterial == null) return false;
  const previous = previousMaterial as unknown as Record<string, unknown>;
  const next = patch.material as unknown as Record<string, unknown>;
  for (const entry of textureMapPatchEntries(next)) {
    const previousHandle = textureHandleOf(materialTextureValueAt(previous, entry.path));
    if (previousHandle == null) continue;
    if (textureHandleOf(entry.value) !== previousHandle) return true;
  }
  return false;
}

function textureValueNeedsAtlasRefresh(
  storageClass: TextureAtlasStorageClass,
  colorSpace: TextureSampleColorSpace,
  value: unknown,
  materialLayerMap: MaterialTextureAtlasLayerMaps,
): boolean {
  if (value == null) return false;
  if (typeof value !== 'object') return false;
  const handle = textureHandleOf(value);
  if (handle == null) return false;
  return materialLayerMap[storageClass]?.[colorSpace].has(handle) !== true;
}

function canRefreshGeometryTextures(patch: ScenePrimitivePatch): boolean {
  let sawGeometryField = false;
  for (const key of Object.keys(patch)) {
    if (key === 'id' || key === 'kind') continue;
    if (!GEOMETRY_TEXTURE_REFRESH_FIELDS.has(key)) return false;
    sawGeometryField = true;
  }
  return sawGeometryField;
}

function sameNumberSet(a: ReadonlySet<number>, b: ReadonlySet<number>): boolean {
  if (a.size !== b.size) return false;
  for (const value of a) {
    if (!b.has(value)) return false;
  }
  return true;
}

function updateResidentMeshLightTexture(
  gl: WebGL2RenderingContext,
  current: UploadedSceneTextures,
  nextMeshLightsData: ReturnType<typeof packMeshAreaLights>,
  deleteOldTextures: (WebGLTexture | null)[],
): WebGLTexture | null {
  if (nextMeshLightsData.data == null) {
    if (current.meshLights != null) {
      deleteOldTextures.push(current.meshLights);
    }
    return null;
  }

  const currentMeshLightDim = squareDim((current.meshLightCount ?? 0) * TRI_LIGHT_PIXELS);
  if (current.meshLights != null && currentMeshLightDim === nextMeshLightsData.dim) {
    const replacement = uploadRgba32f(
      gl,
      nextMeshLightsData.data,
      nextMeshLightsData.dim,
      'mesh-area lights',
    );
    deleteOldTextures.push(current.meshLights);
    return replacement;
  }
  throw new Error(
    'pt-webgl2: mesh-light storage replacement requires a transactional scene rebuild',
  );
}

function updateResidentLightsTexture(
  gl: WebGL2RenderingContext,
  current: UploadedSceneTextures,
  nextLightsData: ReturnType<typeof packLightsTexture>,
): WebGLTexture {
  const currentLightsDim = squareDim(current.lightCount * LIGHT_PIXELS);
  const lights = nextLightsData.data as Float32Array;
  if (currentLightsDim === nextLightsData.dim) {
    return uploadRgba32f(gl, lights, nextLightsData.dim, 'scene lights');
  } else {
    throw new Error('pt-webgl2: light storage replacement requires a transactional scene rebuild');
  }
}

function updateNullableRectTexture(
  gl: WebGL2RenderingContext,
  currentTexture: WebGLTexture | null,
  currentWidth: number,
  currentHeight: number,
  nextData: RectFloatTextureData | null,
  resourceName: string,
  deleteOldTextures: (WebGLTexture | null)[],
): WebGLTexture | null {
  if (nextData == null) {
    if (currentTexture != null) {
      deleteOldTextures.push(currentTexture);
    }
    return null;
  }
  if (currentTexture == null) {
    throw new Error(`pt-webgl2: ${resourceName} allocation requires a transactional scene rebuild`);
  }
  if (currentWidth === nextData.width && currentHeight === nextData.height) {
    const replacement = uploadRgba32fRect(
      gl,
      nextData.data,
      nextData.width,
      nextData.height,
      resourceName,
    );
    deleteOldTextures.push(currentTexture);
    return replacement;
  } else {
    throw new Error(`pt-webgl2: ${resourceName} storage replacement requires a transactional scene rebuild`);
  }
}

function allocateMutationTransaction<T>(
  gl: WebGL2RenderingContext,
  build: (own: (texture: WebGLTexture) => WebGLTexture) => T,
): T {
  const allocated: WebGLTexture[] = [];
  const own = (texture: WebGLTexture): WebGLTexture => {
    allocated.push(texture);
    return texture;
  };
  try {
    return build(own);
  } catch (error) {
    for (const texture of allocated) {
      try {
        gl.deleteTexture(texture);
      } catch {
        // Preserve the upload error. A failed best-effort retirement must not
        // hide the operation that made the candidate transaction unusable.
      }
    }
    throw error;
  }
}

function materialWithCastShadow(primitive: Extract<ScenePrimitive, { material: MaterialSpec }>): MaterialSpec {
  return {
    ...primitive.material,
    castShadow: (primitive as { castShadow?: boolean }).castShadow ?? true,
  } as MaterialSpec;
}

function uniqueMaterialSlotForPrimitive(geoPack: WorldSpaceMergeResult, primitiveId: string): number | null {
  const ownSlots = new Set<number>();
  const otherSlots = new Set<number>();
  for (const range of geoPack.meshVertexRanges) {
    const slots = range.name === primitiveId ? ownSlots : otherSlots;
    for (let tri = range.triStart; tri < range.triStart + range.triCount; tri += 1) {
      slots.add(geoPack.mergedTriMaterialId[tri] ?? 0);
    }
  }
  if (ownSlots.size !== 1) return null;
  const slot = ownSlots.values().next().value;
  if (slot == null || otherSlots.has(slot)) return null;
  return slot;
}

function hasExplicitMeshAreaEmitterForPrimitive(scene: Scene, primitiveId: string): boolean {
  return scene.emitters.some((e) => e.kind === 'mesh-area' && String(e.meshId) === primitiveId);
}

function materialSlotsByPrimitive(
  geoPack: WorldSpaceMergeResult,
): Map<string, Set<number>> {
  const out = new Map<string, Set<number>>();
  for (const range of geoPack.meshVertexRanges) {
    let slots = out.get(range.name);
    if (slots == null) {
      slots = new Set<number>();
      out.set(range.name, slots);
    }
    for (let tri = range.triStart; tri < range.triStart + range.triCount; tri += 1) {
      slots.add(geoPack.mergedTriMaterialId[tri] ?? 0);
    }
  }
  return out;
}

function repackMeshAreaFoldedMaterials(
  geoPack: WorldSpaceMergeResult,
  nextScene: Scene,
): { nextMaterials: MaterialSpec[]; nextGeoPack: WorldSpaceMergeResult } {
  const foldedScene = foldMeshAreaEmittersIntoMaterials(nextScene);
  const foldedMaterialsByPrimitive = new Map<string, MaterialSpec>();
  for (const primitive of foldedScene.primitives) {
    if (isMeshLikePrimitive(primitive)) {
      foldedMaterialsByPrimitive.set(String(primitive.id), materialWithCastShadow(primitive));
    }
  }

  const nextMaterials = geoPack.materials.slice();
  for (const [primitiveId, slots] of materialSlotsByPrimitive(geoPack)) {
    const material = foldedMaterialsByPrimitive.get(primitiveId);
    if (material == null) continue;
    for (const slot of slots) {
      if (slot < nextMaterials.length) nextMaterials[slot] = material;
    }
  }

  return {
    nextMaterials,
    nextGeoPack: { ...geoPack, materials: nextMaterials },
  };
}

function withTextureReplacementsForGl(
  gl: WebGL2RenderingContext,
  current: UploadedSceneTextures,
  replacements: Partial<UploadedSceneTextures>,
): UploadedSceneTextures {
  let destroyed = false;
  const next: UploadedSceneTextures = {
    ...current,
    ...replacements,
    destroy(): void {
      if (destroyed) return;
      destroyed = true;
      retireTexturesIndependently(gl, [
        next.bvhBounds,
        next.bvhContents,
        next.bvhPosition,
        next.bvhIndex,
        next.materialIndex,
        next.materials,
        next.attributesArray,
        next.lights,
        next.meshLights,
        next.envMap,
        next.envMarginal,
        next.envConditional,
        next.textures2DArray,
        next.materialHdrTextures2DArray,
      ], 'pt-webgl2: one or more mutated scene textures failed to retire');
    },
  };
  return next;
}

function canRewriteMeshLightsResident(
  current: UploadedSceneTextures,
  meshLightsData: ReturnType<typeof packMeshAreaLights>,
): boolean {
  if (meshLightsData.data == null) return true;
  return (
    current.meshLights != null &&
    squareDim((current.meshLightCount ?? 0) * TRI_LIGHT_PIXELS) === meshLightsData.dim
  );
}

function bvhStorageDimsMatch(currentMerged: WorldSpaceMergeResult, nextBvh: BvhTextureData): boolean {
  const currentBvh = packBvhTextureData(currentMerged);
  return (
    currentBvh.boundsDim === nextBvh.boundsDim &&
    currentBvh.contentsDim === nextBvh.contentsDim &&
    currentBvh.positionDim === nextBvh.positionDim &&
    currentBvh.indexDim === nextBvh.indexDim &&
    currentBvh.materialIndexDim === nextBvh.materialIndexDim
  );
}

function canRewriteGeometryStorageResident(
  current: UploadedSceneTextures,
  currentMerged: WorldSpaceMergeResult,
  built: ReturnType<typeof buildSceneGeometryTextureData>,
): boolean {
  if (!bvhStorageDimsMatch(currentMerged, built.bvhData)) return false;
  if (squareDim(currentMerged.vertexCount) !== built.attrData.dim) return false;
  if ((current.attributeLayerCount ?? 5) !== built.attrData.layers) return false;
  if (!sameUvAttributeLayout(effectiveUvLayerMap(current), built.uvLayerByTexCoord)) return false;
  return canRewriteMeshLightsResident(current, built.meshLightsData);
}

function materialTextureDimMatches(
  current: UploadedSceneTextures,
  currentMerged: WorldSpaceMergeResult,
  materialsData: ReturnType<typeof packMaterialsTexture>,
): boolean {
  const currentMaterialsData = packMaterialsTexture(currentMerged.materials, current.materialLayerMap ?? undefined, {
    vertexColorMaterialIds: current.vertexColorMaterialIds,
    uvLayerByTexCoord: effectiveUvLayerMap(current),
  });
  return currentMaterialsData.dim === materialsData.dim;
}

function canRewriteGeometryResident(
  current: UploadedSceneTextures,
  currentMerged: WorldSpaceMergeResult,
  built: ReturnType<typeof buildSceneGeometryTextureData>,
  materialsData: ReturnType<typeof packMaterialsTexture>,
): boolean {
  if (!canRewriteGeometryStorageResident(current, currentMerged, built)) return false;
  return materialTextureDimMatches(current, currentMerged, materialsData);
}

function canRewritePrimitiveListResident(
  current: UploadedSceneTextures,
  currentMerged: WorldSpaceMergeResult,
  built: ReturnType<typeof buildSceneGeometryTextureData>,
): boolean {
  return canRewriteGeometryStorageResident(current, currentMerged, built);
}

export function tryFastPathMaterialMutation(
  gl: WebGL2RenderingContext,
  current: UploadedSceneTextures | null,
  geoPack: WorldSpaceMergeResult | null,
  nextScene: Scene,
  primitiveId: string,
  patch: ScenePrimitivePatch,
): WebGl2MutationSwap | null {
  if (current == null || geoPack == null || !canFastPathMaterialPatch(patch)) return null;
  const primitive = nextScene.primitives.find((p) => String(p.id) === primitiveId);
  if (!isMeshLikePrimitive(primitive)) return null;
  const slot = uniqueMaterialSlotForPrimitive(geoPack, primitiveId);
  if (slot == null || slot >= geoPack.materials.length) return null;
  const materialPatch = patch.material != null ? { material: patch.material } : null;
  const atlasRefreshNeeded = materialPatch != null && (
    texturePatchNeedsAtlasRefresh(materialPatch, current.materialLayerMap) ||
    texturePatchMayCompactAtlas(geoPack.materials[slot], materialPatch)
  );
  if (materialPatch != null && texturePatchNeedsUvLayoutRefresh(materialPatch, current)) {
    return null;
  }
  const structuredWarnings: EngineWarning[] = [];

  const explicitMeshArea = hasExplicitMeshAreaEmitterForPrimitive(nextScene, primitiveId);
  const foldedMaterials = explicitMeshArea
    ? repackMeshAreaFoldedMaterials(geoPack, nextScene)
    : null;

  let nextGeoPack: WorldSpaceMergeResult;
  let nextMaterials: MaterialSpec[];
  if (foldedMaterials != null) {
    nextGeoPack = foldedMaterials.nextGeoPack;
    nextMaterials = foldedMaterials.nextMaterials;
  } else {
    nextMaterials = geoPack.materials.slice();
    nextMaterials[slot] = materialWithCastShadow(primitive);
    nextGeoPack = { ...geoPack, materials: nextMaterials };
  }

  // materialIndex.y stores validated optical-component identity. A mutation
  // that activates or deactivates bulk transport changes that identity texture
  // as well as the scalar material payload, so the material-only transaction
  // cannot publish it safely. Defer to the complete staged scene rebuild where
  // materials and materialIndex are replaced atomically.
  if (
    materialDefinesBulkOpticalMedium(geoPack.materials[slot]!) !==
    materialDefinesBulkOpticalMedium(nextMaterials[slot]!)
  ) {
    return null;
  }

  if (atlasRefreshNeeded) {
    // Decode and validate both candidate atlases under the user-visible mutation
    // operation before returning to setScene's staged GPU transaction. This
    // preserves synchronous updatePrimitive diagnostics while guaranteeing that
    // malformed input cannot touch resident textures.
    const maxAtlasLayers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
    const candidateAtlases = preflightMaterialTextureAtlases(nextMaterials, {
      warningMethod: 'updatePrimitive',
      maxArrayTextureLayers: maxAtlasLayers,
    });
    materialTextureAtlasLayerCapacities(
      candidateAtlases.ldr,
      candidateAtlases.hdr,
      maxAtlasLayers,
    );
    return null;
  }

  const meshLightsData = hasMeshAreaLightForPrimitive(nextScene, primitiveId)
    || (current.meshLightCount ?? 0) > 0
    ? packMeshAreaLights(nextScene, nextGeoPack)
    : null;
  if (
    (current.meshLightCount ?? 0) > 0 ||
    (meshLightsData?.triLightCount ?? 0) > 0
  ) {
    // A mesh-family PMF change also changes every analytic/environment global
    // BDPT PMF lane. Let the complete scene transaction repack them together.
    return null;
  }
  const meshLightStorageMatches = meshLightsData == null
    || meshLightsData.data == null
    || (
      current.meshLights != null &&
      squareDim((current.meshLightCount ?? 0) * TRI_LIGHT_PIXELS) === meshLightsData.dim
    );
  if (!meshLightStorageMatches) {
    return null;
  }

  const materialData = packMaterialsTexture(
    nextMaterials,
    current.materialLayerMap,
    {
      vertexColorMaterialIds: current.vertexColorMaterialIds,
      uvLayerByTexCoord: effectiveUvLayerMap(current),
    },
  );
  const { materials, meshLights, deleteOldTextures } = allocateMutationTransaction(
    gl,
    (own) => {
      const retired: (WebGLTexture | null)[] = [current.materials];
      const nextMaterialsTexture = own(
        uploadRgba32f(
          gl,
          materialData.data,
          materialData.dim,
          'scene materials',
        ),
      );
      const nextMeshLights = meshLightsData != null
        ? updateResidentMeshLightTexture(gl, current, meshLightsData, retired)
        : null;
      if (nextMeshLights != null) own(nextMeshLights);
      return {
        materials: nextMaterialsTexture,
        meshLights: nextMeshLights,
        deleteOldTextures: retired,
      };
    },
  );
  return {
    textures: withTextureReplacementsForGl(gl, current, {
      materials,
      ...(meshLightsData != null
        ? {
            meshLights,
            meshLightCount: meshLightsData.triLightCount,
            totalEmissiveArea: meshLightsData.totalEmissiveArea,
            totalEmissivePower: meshLightsData.totalEmissivePower,
          }
        : {}),
    }),
    geoPack: nextGeoPack,
    deleteOldTextures,
    ...(structuredWarnings.length > 0 ? { structuredWarnings } : {}),
  };
}

export function tryFastPathGeometryMutation(
  gl: WebGL2RenderingContext,
  current: UploadedSceneTextures | null,
  currentMerged: WorldSpaceMergeResult | null,
  nextScene: Scene,
  patch: ScenePrimitivePatch,
  options: { readonly bdpt?: boolean } = {},
): WebGl2MutationSwap | null {
  if (current == null || !canRefreshGeometryTextures(patch)) return null;
  if (currentMerged != null) {
    const refit = buildRefitSceneGeometryTextures(nextScene, currentMerged, {
      warningPhase: 'mutation',
      warningMethod: 'updatePrimitive',
    });
    if (refit != null) {
      if (
        (current.meshLightCount ?? 0) > 0 ||
        refit.meshLightsData.triLightCount > 0
      ) {
        return null;
      }
      computeWebgl2TransportBounds(refit.merged, nextScene, options);
      const structuredWarnings: EngineWarning[] = [...refit.structuredWarnings];
      if (
        !canRewriteMeshLightsResident(current, refit.meshLightsData) ||
        (current.attributeLayerCount ?? 5) !== refit.attrData.layers ||
        !sameUvAttributeLayout(effectiveUvLayerMap(current), refit.uvLayerByTexCoord)
      ) {
        return null;
      }
      for (const warning of refit.warnings) {
        structuredWarnings.push({
          code: 'pt-webgl2.scene-upload-warning',
          backend: 'pt-webgl2',
          phase: 'mutation',
          method: 'updatePrimitive',
          message: `[vitrum/pt-webgl2] ${warning}`,
          details: {
            warning,
            operation: 'geometry-bvh-refit-staged-texture-replacement',
          },
        });
      }
      const vertexColorMaterialIdsChanged = !sameNumberSet(
        current.vertexColorMaterialIds,
        refit.vertexColorMaterialIds,
      );
      const materialsData = vertexColorMaterialIdsChanged
        ? packMaterialsTexture(
            refit.merged.materials,
            current.materialLayerMap ?? undefined,
            {
              vertexColorMaterialIds: refit.vertexColorMaterialIds,
              uvLayerByTexCoord: refit.uvLayerByTexCoord,
            },
          )
        : null;
      const nextMeshLightsData = refit.meshLightsData;
      const replacements = allocateMutationTransaction(gl, (own) => {
        const deleteOldTextures: (WebGLTexture | null)[] = [
          current.bvhBounds,
          current.bvhPosition,
          current.attributesArray,
        ];
        const bvhBounds = own(
          uploadRgba32f(
            gl,
            refit.bvhData.bounds,
            refit.bvhData.boundsDim,
            'scene BVH bounds',
          ),
        );
        const bvhPosition = own(
          uploadRgba32f(
            gl,
            refit.bvhData.position,
            refit.bvhData.positionDim,
            'scene BVH position',
          ),
        );
        const attributesArray = own(
          uploadRgba32fArray(
            gl,
            refit.attrData.data,
            refit.attrData.dim,
            refit.attrData.layers,
            'vertex attributes',
          ),
        );
        const materials = materialsData != null
          ? own(
              uploadRgba32f(
                gl,
                materialsData.data,
                materialsData.dim,
                'scene materials',
              ),
            )
          : current.materials;
        if (materialsData != null) deleteOldTextures.push(current.materials);
        const meshLights = updateResidentMeshLightTexture(
          gl,
          current,
          nextMeshLightsData,
          deleteOldTextures,
        );
        if (meshLights != null) own(meshLights);
        return {
          bvhBounds,
          bvhPosition,
          attributesArray,
          materials,
          meshLights,
          deleteOldTextures,
        };
      });

      return {
        textures: withTextureReplacementsForGl(gl, current, {
          bvhBounds: replacements.bvhBounds,
          bvhPosition: replacements.bvhPosition,
          attributesArray: replacements.attributesArray,
          materials: replacements.materials,
          meshLights: replacements.meshLights,
          meshLightCount: nextMeshLightsData.triLightCount,
          totalEmissiveArea: nextMeshLightsData.totalEmissiveArea,
          totalEmissivePower: nextMeshLightsData.totalEmissivePower,
          triangleCount: refit.merged.triangleCount,
          vertexColorMaterialIds: refit.vertexColorMaterialIds,
          uvLayerByTexCoord: refit.uvLayerByTexCoord,
          attributeLayerCount: refit.attrData.layers,
        }),
        geoPack: refit.merged,
        deleteOldTextures: replacements.deleteOldTextures,
        structuredWarnings,
      };
    }
  }

  if (currentMerged != null) {
    const built = buildSceneGeometryTextureData(nextScene, {
      warningPhase: 'mutation',
      warningMethod: 'updatePrimitive',
    });
    if (
      (current.meshLightCount ?? 0) > 0 ||
      built.meshLightsData.triLightCount > 0
    ) {
      return null;
    }
    computeWebgl2TransportBounds(built.merged, nextScene, options);
    const vertexColorMaterialIdsChanged = !sameNumberSet(
      current.vertexColorMaterialIds,
      built.vertexColorMaterialIds,
    );
    const materialsData = packMaterialsTexture(
      built.merged.materials,
      current.materialLayerMap ?? undefined,
      {
        vertexColorMaterialIds: built.vertexColorMaterialIds,
        uvLayerByTexCoord: built.uvLayerByTexCoord,
      },
    );
    if (canRewriteGeometryResident(current, currentMerged, built, materialsData)) {
      const structuredWarnings: EngineWarning[] = [...built.structuredWarnings];
      for (const warning of built.warnings) {
        structuredWarnings.push({
          code: 'pt-webgl2.scene-upload-warning',
          backend: 'pt-webgl2',
          phase: 'mutation',
          method: 'updatePrimitive',
          message: `[vitrum/pt-webgl2] ${warning}`,
          details: { warning, operation: 'geometry-topology-resident-update' },
        });
      }
      const replacements = allocateMutationTransaction(gl, (own) => {
        const deleteOldTextures: (WebGLTexture | null)[] = [
          current.bvhBounds,
          current.bvhContents,
          current.bvhPosition,
          current.bvhIndex,
          current.materialIndex,
          current.attributesArray,
        ];
        const bvhBounds = own(
          uploadRgba32f(gl, built.bvhData.bounds, built.bvhData.boundsDim, 'scene BVH bounds'),
        );
        const bvhContents = own(
          uploadRgba32ui(
            gl,
            built.bvhData.contents,
            built.bvhData.contentsDim,
            'scene BVH contents',
          ),
        );
        const bvhPosition = own(
          uploadRgba32f(
            gl,
            built.bvhData.position,
            built.bvhData.positionDim,
            'scene BVH position',
          ),
        );
        const bvhIndex = own(
          uploadRgba32ui(gl, built.bvhData.index, built.bvhData.indexDim, 'scene BVH index'),
        );
        const materialIndex = own(
          uploadRgba32ui(
            gl,
            built.bvhData.materialIndex,
            built.bvhData.materialIndexDim,
            'scene BVH material index',
          ),
        );
        const materials = vertexColorMaterialIdsChanged
          ? own(
              uploadRgba32f(
                gl,
                materialsData.data,
                materialsData.dim,
                'scene materials',
              ),
            )
          : current.materials;
        if (vertexColorMaterialIdsChanged) deleteOldTextures.push(current.materials);
        const attributesArray = own(
          uploadRgba32fArray(
            gl,
            built.attrData.data,
            built.attrData.dim,
            built.attrData.layers,
            'vertex attributes',
          ),
        );
        const meshLights = updateResidentMeshLightTexture(
          gl,
          current,
          built.meshLightsData,
          deleteOldTextures,
        );
        if (meshLights != null) own(meshLights);
        return {
          bvhBounds,
          bvhContents,
          bvhPosition,
          bvhIndex,
          materialIndex,
          materials,
          attributesArray,
          meshLights,
          deleteOldTextures,
        };
      });

      return {
        textures: withTextureReplacementsForGl(gl, current, {
          bvhBounds: replacements.bvhBounds,
          bvhContents: replacements.bvhContents,
          bvhPosition: replacements.bvhPosition,
          bvhIndex: replacements.bvhIndex,
          materialIndex: replacements.materialIndex,
          materials: replacements.materials,
          attributesArray: replacements.attributesArray,
          meshLights: replacements.meshLights,
          meshLightCount: built.meshLightsData.triLightCount,
          totalEmissiveArea: built.meshLightsData.totalEmissiveArea,
          totalEmissivePower: built.meshLightsData.totalEmissivePower,
          triangleCount: built.triangleCount,
          vertexColorMaterialIds: built.vertexColorMaterialIds,
          uvLayerByTexCoord: built.uvLayerByTexCoord,
          attributeLayerCount: built.attrData.layers,
        }),
        geoPack: built.merged,
        deleteOldTextures: replacements.deleteOldTextures,
        structuredWarnings,
      };
    }
  }

  return null;
}

export function tryFastPathPrimitiveListMutation(
  gl: WebGL2RenderingContext,
  current: UploadedSceneTextures | null,
  currentMerged: WorldSpaceMergeResult | null,
  nextScene: Scene,
  opts: {
    readonly method: 'addPrimitive' | 'removePrimitive';
    readonly primitiveId: string;
    readonly bdpt?: boolean;
  },
): WebGl2MutationSwap | null {
  if (current == null) return null;
  const analyticExpansion = expandAnalyticPrimitiveFallbacks(nextScene);
  const mutationScene = analyticExpansion.scene;
  if (mutationScene.primitives.some((primitive) => !isMeshLikePrimitive(primitive))) return null;

  const built = buildSceneGeometryTextureData(mutationScene, {
    warningPhase: 'mutation',
    warningMethod: opts.method,
  });
  if (
    (current.meshLightCount ?? 0) > 0 ||
    built.meshLightsData.triLightCount > 0
  ) {
    return null;
  }
  computeWebgl2TransportBounds(built.merged, mutationScene, opts);
  const structuredWarnings: EngineWarning[] = [...built.structuredWarnings];
  for (const warning of [...analyticExpansion.warnings, ...built.warnings]) {
    structuredWarnings.push({
      code: 'pt-webgl2.scene-upload-warning',
      backend: 'pt-webgl2',
      phase: 'mutation',
      method: opts.method,
      message: `[vitrum/pt-webgl2] ${warning}`,
      details: { warning, operation: 'primitive-list-texture-refresh' },
    });
  }

  const maxAtlasLayers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
  const materialAtlases = preflightMaterialTextureAtlases(built.merged.materials, {
    warningMethod: opts.method,
    maxArrayTextureLayers: maxAtlasLayers,
  });
  const atlas = materialAtlases.ldr;
  const hdrAtlas = materialAtlases.hdr;
  materialTextureAtlasLayerCapacities(atlas, hdrAtlas, maxAtlasLayers);
  // Atlas membership/order changes are staged by the full scene transaction.
  // Resident primitive-list rewrites remain available for texture-free scenes.
  if (
    current.textures2DArray != null ||
    current.materialHdrTextures2DArray != null ||
    atlas != null ||
    hdrAtlas != null
  ) {
    return null;
  }
  const materialsData = packMaterialsTexture(
    built.merged.materials,
    { ldr: null, hdr: null },
    {
      vertexColorMaterialIds: built.vertexColorMaterialIds,
      uvLayerByTexCoord: built.uvLayerByTexCoord,
    },
  );
  const materialStorageMatches = currentMerged != null
    && materialTextureDimMatches(current, currentMerged, materialsData);
  const meshLightStorageMatches = built.meshLightsData.data == null
    || (
      current.meshLights != null
      && squareDim((current.meshLightCount ?? 0) * TRI_LIGHT_PIXELS) === built.meshLightsData.dim
    );
  if (
    currentMerged != null &&
    canRewritePrimitiveListResident(
      current,
      currentMerged,
      built,
    ) &&
    materialStorageMatches &&
    meshLightStorageMatches
  ) {
    const replacements = allocateMutationTransaction(gl, (own) => {
      const deleteOldTextures: (WebGLTexture | null)[] = [
        current.bvhBounds,
        current.bvhContents,
        current.bvhPosition,
        current.bvhIndex,
        current.materialIndex,
        current.materials,
        current.attributesArray,
      ];
      const bvhBounds = own(
        uploadRgba32f(gl, built.bvhData.bounds, built.bvhData.boundsDim, 'scene BVH bounds'),
      );
      const bvhContents = own(
        uploadRgba32ui(
          gl,
          built.bvhData.contents,
          built.bvhData.contentsDim,
          'scene BVH contents',
        ),
      );
      const bvhPosition = own(
        uploadRgba32f(
          gl,
          built.bvhData.position,
          built.bvhData.positionDim,
          'scene BVH position',
        ),
      );
      const bvhIndex = own(
        uploadRgba32ui(gl, built.bvhData.index, built.bvhData.indexDim, 'scene BVH index'),
      );
      const materialIndex = own(
        uploadRgba32ui(
          gl,
          built.bvhData.materialIndex,
          built.bvhData.materialIndexDim,
          'scene BVH material index',
        ),
      );
      const materials = own(
        uploadRgba32f(
          gl,
          materialsData.data,
          materialsData.dim,
          'scene materials',
        ),
      );
      const attributesArray = own(
        uploadRgba32fArray(
          gl,
          built.attrData.data,
          built.attrData.dim,
          built.attrData.layers,
          'vertex attributes',
        ),
      );
      const meshLights = updateResidentMeshLightTexture(
        gl,
        current,
        built.meshLightsData,
        deleteOldTextures,
      );
      if (meshLights != null) own(meshLights);
      return {
        bvhBounds,
        bvhContents,
        bvhPosition,
        bvhIndex,
        materialIndex,
        materials,
        attributesArray,
        meshLights,
        deleteOldTextures,
      };
    });

    return {
      textures: withTextureReplacementsForGl(gl, current, {
        bvhBounds: replacements.bvhBounds,
        bvhContents: replacements.bvhContents,
        bvhPosition: replacements.bvhPosition,
        bvhIndex: replacements.bvhIndex,
        materialIndex: replacements.materialIndex,
        materials: replacements.materials,
        attributesArray: replacements.attributesArray,
        meshLights: replacements.meshLights,
        meshLightCount: built.meshLightsData.triLightCount,
        totalEmissiveArea: built.meshLightsData.totalEmissiveArea,
        totalEmissivePower: built.meshLightsData.totalEmissivePower,
        triangleCount: built.triangleCount,
        vertexColorMaterialIds: built.vertexColorMaterialIds,
        uvLayerByTexCoord: built.uvLayerByTexCoord,
        attributeLayerCount: built.attrData.layers,
      }),
      geoPack: built.merged,
      scene: mutationScene,
      deleteOldTextures: replacements.deleteOldTextures,
      structuredWarnings,
    };
  }

  return null;
}

export function tryFastPathEmitterMutation(
  gl: WebGL2RenderingContext,
  current: UploadedSceneTextures | null,
  geoPack: WorldSpaceMergeResult | null,
  nextScene: Scene,
  emitterId: string,
  options: { readonly bdpt?: boolean } = {},
): WebGl2MutationSwap | null {
  if (current == null || geoPack == null) return null;
  // Global BDPT PMFs span analytic, mesh, and environment candidates. An
  // emitter-only texture swap would leave the other domains stale.
  if (requiresFullRepresentedEmitterProposalTransaction()) return null;
  const changed = nextScene.emitters.find((e) => String(e.id) === emitterId);
  const isMeshAreaMutation = changed?.kind === 'mesh-area';
  const lightsData = packLightsTexture(nextScene.emitters);
  const structuredWarnings: EngineWarning[] = [];
  const meshLightsData = packMeshAreaLights(nextScene, geoPack);
  computeWebgl2TransportBounds(geoPack, nextScene, options);
  const lightsStorageMatches =
    squareDim(current.lightCount * LIGHT_PIXELS) === lightsData.dim;
  const meshLightStorageMatches = meshLightsData.data == null
    || (
      current.meshLights != null
      && squareDim((current.meshLightCount ?? 0) * TRI_LIGHT_PIXELS) === meshLightsData.dim
    );
  if (!lightsStorageMatches || !meshLightStorageMatches) {
    return null;
  }
  const foldedMaterials = isMeshAreaMutation
    ? repackMeshAreaFoldedMaterials(geoPack, nextScene)
    : null;
  const materialData = foldedMaterials != null
    ? packMaterialsTexture(
        foldedMaterials.nextMaterials,
        current.materialLayerMap ?? undefined,
        {
          vertexColorMaterialIds: current.vertexColorMaterialIds,
          uvLayerByTexCoord: effectiveUvLayerMap(current),
        },
      )
    : null;
  const replacements = allocateMutationTransaction(gl, (own) => {
    const deleteOldTextures: (WebGLTexture | null)[] = [current.lights];
    const lights = own(updateResidentLightsTexture(gl, current, lightsData));
    const meshLights = updateResidentMeshLightTexture(
      gl,
      current,
      meshLightsData,
      deleteOldTextures,
    );
    if (meshLights != null) own(meshLights);
    const materials = materialData != null
      ? own(
          uploadRgba32f(
            gl,
            materialData.data,
            materialData.dim,
            'scene materials',
          ),
        )
      : current.materials;
    if (materialData != null) deleteOldTextures.push(current.materials);
    return { lights, meshLights, materials, deleteOldTextures };
  });
  return {
    textures: withTextureReplacementsForGl(gl, current, {
      lights: replacements.lights,
      lightCount: lightsData.lightCount,
      meshLights: replacements.meshLights,
      materials: replacements.materials,
      meshLightCount: meshLightsData.triLightCount,
      totalEmissiveArea: meshLightsData.totalEmissiveArea,
      totalEmissivePower: meshLightsData.totalEmissivePower,
    }),
    ...(foldedMaterials != null ? { geoPack: foldedMaterials.nextGeoPack } : {}),
    deleteOldTextures: replacements.deleteOldTextures,
    structuredWarnings,
  };
}

export function fastPathEnvironmentMutation(
  gl: WebGL2RenderingContext,
  current: UploadedSceneTextures | null,
  nextScene: Scene,
): WebGl2MutationSwap | null {
  if (current == null) return null;
  // See tryFastPathEmitterMutation: environment changes must repack all global
  // emitter PMF lanes in one complete scene transaction.
  if (requiresFullRepresentedEmitterProposalTransaction()) return null;
  const structuredWarnings: EngineWarning[] = [];
  const env = buildEquirectInfo(nextScene.environment, {
    onWarning: (warning) => structuredWarnings.push(warning),
    warningPhase: 'mutation',
    warningMethod: 'updateEnvironment',
  });
  const deleteOldTextures: (WebGLTexture | null)[] = [];
  const canReplaceRectWithoutSceneRebuild = (
    currentTexture: WebGLTexture | null,
    currentWidth: number,
    currentHeight: number,
    next: RectFloatTextureData | null,
  ): boolean => next == null || (
    currentTexture != null &&
    currentWidth === next.width &&
    currentHeight === next.height
  );
  if (
    !canReplaceRectWithoutSceneRebuild(
      current.envMap,
      current.envWidth,
      current.envHeight,
      env.map,
    ) ||
    !canReplaceRectWithoutSceneRebuild(
      current.envConditional,
      current.envWidth,
      current.envHeight,
      env.conditional,
    )
  ) {
    return null;
  }
  const replacements = allocateMutationTransaction(gl, (own) => {
    const envMap = updateNullableRectTexture(
      gl,
      current.envMap,
      current.envWidth,
      current.envHeight,
      env.map,
      'environment map',
      deleteOldTextures,
    );
    if (envMap != null) own(envMap);
    const envConditional = updateNullableRectTexture(
      gl,
      current.envConditional,
      current.envWidth,
      current.envHeight,
      env.conditional,
      'environment CDF distribution',
      deleteOldTextures,
    );
    if (envConditional != null) own(envConditional);
    if (current.envMarginal != null) deleteOldTextures.push(current.envMarginal);
    return { envMap, envMarginal: null, envConditional };
  });
  return {
    textures: withTextureReplacementsForGl(gl, current, {
      envMap: replacements.envMap,
      envMarginal: replacements.envMarginal,
      envConditional: replacements.envConditional,
      envTotalSum: env.totalSum,
      envWidth: env.map?.width ?? 0,
      envHeight: env.map?.height ?? 0,
    }),
    deleteOldTextures,
    structuredWarnings,
  };
}
