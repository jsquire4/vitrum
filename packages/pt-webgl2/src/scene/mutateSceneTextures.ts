import type { EngineWarning, MaterialSpec, Scene, ScenePrimitive } from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import { packMaterialsTexture } from './materialsTexture.js';
import { packLightsTexture } from './lightsTexture.js';
import { hasMeshAreaLightForPrimitive, packMeshAreaLights, TRI_LIGHT_PIXELS } from './meshAreaLights.js';
import { foldMeshAreaEmittersIntoMaterials } from './foldEmissiveEmitters.js';
import { buildEquirectInfo } from './equirectHdrInfo.js';
import type { UploadedSceneTextures } from './sceneTextures.js';
import {
  buildSceneGeometryTextureData,
  buildRefitSceneGeometryTextures,
  buildSceneGeometryTextures,
  expandAnalyticPrimitiveFallbacks,
  updateRgba32f,
  updateRgba32fArray,
  updateRgba32ui,
  uploadRgba32f,
  uploadRgba32fArray,
  uploadRgba32fRect,
} from './uploadSceneTextures.js';
import {
  packTextureAtlas,
  refreshTextureAtlasStorage,
  textureAtlasLayerCapacity,
  updateTextureAtlasLayers,
  uploadTextureAtlas,
} from './texturesArray.js';
import type { TextureAtlasLayerMap, TextureSampleColorSpace } from './texturesArray.js';
import { packBvhTextureData, squareDim, uploadBvhTextures, type BvhTextureData } from './bvhTextureAdapter.js';

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

const UNSUPPORTED_DISPLACEMENT_FIELDS: ReadonlySet<string> = new Set([
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

const GEOMETRY_TEXTURE_REFRESH_FIELDS: ReadonlySet<string> = new Set([
  'transform',
  'positions',
  'normals',
  'indices',
  'uvs',
  'uv1',
  'tangents',
  'colors',
  'instances',
  'bones',
  'boneInverses',
  'skinIndices',
  'skinWeights',
  'morphTargets',
  'morphTargetNormals',
  'morphTargetTangents',
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
  };
}

function isMeshLikePrimitive(p: ScenePrimitive | undefined): p is Extract<
  ScenePrimitive,
  { kind: 'mesh' | 'instanced-mesh' | 'skinned-mesh' }
> {
  return p?.kind === 'mesh' || p?.kind === 'instanced-mesh' || p?.kind === 'skinned-mesh';
}

function canFastPathMaterialPatch(
  patch: Partial<ScenePrimitive>,
): patch is Partial<ScenePrimitive> & { material: MaterialSpec } {
  if (patch.material == null) return false;
  for (const key of Object.keys(patch)) {
    if (key !== 'material' && key !== 'id' && key !== 'kind') return false;
  }
  return true;
}

function texturePatchNeedsAtlasRefresh(
  patch: Partial<ScenePrimitive> & { material: MaterialSpec },
  materialLayerMap: TextureAtlasLayerMap | null,
): boolean {
  const mat = patch.material as unknown as Record<string, unknown>;
  for (const field of Object.keys(mat)) {
    if (!TEXTURE_MAP_FIELDS.has(field)) continue;
    if (textureValueNeedsAtlasRefresh(field, mat[field], materialLayerMap)) return true;
  }
  return false;
}

function textureHandleOf(value: unknown): unknown | null {
  if (value == null || typeof value !== 'object') return null;
  return (value as { readonly handle?: unknown }).handle ?? null;
}

function texturePatchMayCompactAtlas(
  previousMaterial: MaterialSpec | undefined,
  patch: Partial<ScenePrimitive> & { material: MaterialSpec },
): boolean {
  if (previousMaterial == null) return false;
  const previous = previousMaterial as unknown as Record<string, unknown>;
  const next = patch.material as unknown as Record<string, unknown>;
  for (const field of Object.keys(next)) {
    if (!TEXTURE_MAP_FIELDS.has(field)) continue;
    const previousHandle = textureHandleOf(previous[field]);
    if (previousHandle == null) continue;
    if (textureHandleOf(next[field]) !== previousHandle) return true;
  }
  return false;
}

function textureValueNeedsAtlasRefresh(
  field: string,
  value: unknown,
  materialLayerMap: TextureAtlasLayerMap | null,
): boolean {
  if (value == null) return false;
  if (typeof value !== 'object') return false;
  const handle = textureHandleOf(value);
  if (handle == null) return false;
  const colorSpace = TEXTURE_MAP_COLOR_SPACE[field] ?? 'linear';
  return materialLayerMap?.[colorSpace].has(handle) !== true;
}

function unsupportedDisplacementPatchFields(patch: Partial<ScenePrimitive>): readonly string[] {
  if (patch.material == null) return [];
  const material = patch.material as unknown as Record<string, unknown>;
  return Object.keys(material)
    .filter((field) => UNSUPPORTED_DISPLACEMENT_FIELDS.has(field))
    .sort();
}

function canRefreshGeometryTextures(patch: Partial<ScenePrimitive>): boolean {
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
  const slot = ownSlots.values().next().value as number | undefined;
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
      for (const t of [
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
      ]) {
        if (t != null) gl.deleteTexture(t);
      }
    },
  };
  return next;
}

function uploadAtlasWithCapacity(
  gl: WebGL2RenderingContext,
  atlas: NonNullable<ReturnType<typeof packTextureAtlas>>,
): { texture: WebGLTexture; capacity: number } {
  const capacity = textureAtlasLayerCapacity(
    atlas.layerCount,
    gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number,
  );
  return {
    texture: uploadTextureAtlas(gl, atlas, { layerCapacity: capacity }),
    capacity,
  };
}

type MaterialAtlasRefreshReason =
  | 'first-atlas'
  | 'atlas-removed'
  | 'dimension-change'
  | 'capacity-exhausted'
  | 'capacity-compaction';

function materialAtlasRefreshReason(
  current: UploadedSceneTextures,
  atlas: NonNullable<ReturnType<typeof packTextureAtlas>> | null,
  nextLayerCapacity: number,
): MaterialAtlasRefreshReason | null {
  if (atlas == null) return current.textures2DArray != null ? 'atlas-removed' : null;
  if (current.textures2DArray == null) return 'first-atlas';
  if (current.materialAtlasDim !== atlas.dim) return 'dimension-change';
  if (atlas.layerCount > current.materialAtlasLayerCapacity) return 'capacity-exhausted';
  if (nextLayerCapacity < current.materialAtlasLayerCapacity) return 'capacity-compaction';
  return null;
}

function pushMaterialAtlasRefreshWarning(
  warnings: EngineWarning[],
  context: {
    readonly method: 'updatePrimitive' | 'addPrimitive' | 'removePrimitive';
    readonly primitiveId: string;
  },
  current: UploadedSceneTextures,
  atlas: NonNullable<ReturnType<typeof packTextureAtlas>> | null,
  reason: MaterialAtlasRefreshReason,
  nextLayerCapacity: number,
): void {
  warnings.push({
    code: 'pt-webgl2.material-atlas-texture-refresh',
    backend: 'pt-webgl2',
    phase: 'mutation',
    method: context.method,
    message:
      `[vitrum/pt-webgl2] ${context.method}("${context.primitiveId}") ` +
      `refreshed the material texture atlas (${reason}); ` +
      'same-dimension resident layer growth patches stay in place, but this change requires a texture refresh.',
    details: {
      primitiveId: context.primitiveId,
      reason,
      previousDim: current.materialAtlasDim,
      nextDim: atlas?.dim ?? 0,
      previousLayerCount: current.materialAtlasLayerCount,
      nextLayerCount: atlas?.layerCount ?? 0,
      previousLayerCapacity: current.materialAtlasLayerCapacity,
      nextLayerCapacity,
    },
  });
}

function canUpdateAtlasInPlace(
  current: UploadedSceneTextures,
  atlas: NonNullable<ReturnType<typeof packTextureAtlas>>,
  nextLayerCapacity: number,
): boolean {
  return (
    current.textures2DArray != null &&
    current.materialAtlasDim === atlas.dim &&
    atlas.layerCount <= current.materialAtlasLayerCapacity &&
    nextLayerCapacity >= current.materialAtlasLayerCapacity
  );
}

function canRewriteMeshLightsResident(
  current: UploadedSceneTextures,
  meshLightsData: ReturnType<typeof packMeshAreaLights>,
): boolean {
  if (current.meshLights == null) return meshLightsData.data == null;
  if (meshLightsData.data == null) return true;
  return squareDim((current.meshLightCount ?? 0) * TRI_LIGHT_PIXELS) === meshLightsData.dim;
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
  return canRewriteMeshLightsResident(current, built.meshLightsData);
}

function materialTextureDimMatches(
  current: UploadedSceneTextures,
  currentMerged: WorldSpaceMergeResult,
  materialsData: ReturnType<typeof packMaterialsTexture>,
): boolean {
  const currentMaterialsData = packMaterialsTexture(currentMerged.materials, current.materialLayerMap ?? undefined, {
    vertexColorMaterialIds: current.vertexColorMaterialIds,
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
  patch: Partial<ScenePrimitive>,
): WebGl2MutationSwap | null {
  if (current == null || geoPack == null || !canFastPathMaterialPatch(patch)) return null;
  const primitive = nextScene.primitives.find((p) => String(p.id) === primitiveId);
  if (!isMeshLikePrimitive(primitive)) return null;
  const slot = uniqueMaterialSlotForPrimitive(geoPack, primitiveId);
  if (slot == null || slot >= geoPack.materials.length) return null;
  const atlasRefreshNeeded = texturePatchNeedsAtlasRefresh(patch, current.materialLayerMap) ||
    texturePatchMayCompactAtlas(geoPack.materials[slot], patch);

  const unsupportedDisplacementFields = unsupportedDisplacementPatchFields(patch);
  const structuredWarnings: EngineWarning[] = unsupportedDisplacementFields.length > 0
    ? [{
        code: 'pt-webgl2.unsupported-displacement-material',
        backend: 'pt-webgl2',
        phase: 'mutation',
        method: 'updatePrimitive',
        message:
          `[vitrum/pt-webgl2] updatePrimitive("${primitiveId}"): displacement material fields are supplied ` +
          `but not rendered by this backend: ${unsupportedDisplacementFields.join(', ')}.`,
        details: {
          fields: unsupportedDisplacementFields,
          primitiveIds: [primitiveId],
          primitiveFields: [{ primitiveId, fields: unsupportedDisplacementFields }],
        },
      }]
    : [];

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

  const atlas = atlasRefreshNeeded
    ? packTextureAtlas(nextMaterials, {
        onWarning: (warning) => structuredWarnings.push(warning),
        warningPhase: 'mutation',
        warningMethod: 'updatePrimitive',
      })
    : null;
  let atlasTexture = current.textures2DArray;
  let materialAtlasDim = current.materialAtlasDim;
  let materialAtlasLayerCount = current.materialAtlasLayerCount;
  let materialAtlasLayerCapacity = current.materialAtlasLayerCapacity;
  let deleteOldAtlas = false;
  const maxAtlasLayers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
  if (atlasRefreshNeeded) {
    if (atlas == null) {
      const reason = materialAtlasRefreshReason(current, null, 0);
      atlasTexture = null;
      materialAtlasDim = 0;
      materialAtlasLayerCount = 0;
      materialAtlasLayerCapacity = 0;
      deleteOldAtlas = current.textures2DArray != null;
      if (reason != null) {
        pushMaterialAtlasRefreshWarning(
          structuredWarnings,
          { method: 'updatePrimitive', primitiveId },
          current,
          null,
          reason,
          0,
        );
      }
    } else {
      const nextLayerCapacity = textureAtlasLayerCapacity(atlas.layerCount, maxAtlasLayers);
      if (canUpdateAtlasInPlace(current, atlas, nextLayerCapacity)) {
        updateTextureAtlasLayers(gl, current.textures2DArray as WebGLTexture, atlas);
        materialAtlasDim = atlas.dim;
        materialAtlasLayerCount = atlas.layerCount;
      } else {
        const reason = materialAtlasRefreshReason(current, atlas, nextLayerCapacity);
        const refreshedCapacity = current.textures2DArray != null
          ? refreshTextureAtlasStorage(
              gl,
              current.textures2DArray,
              atlas,
              { layerCapacity: nextLayerCapacity },
            )
          : null;
        const uploadedAtlas = refreshedCapacity == null ? uploadAtlasWithCapacity(gl, atlas) : null;
        atlasTexture = current.textures2DArray ?? uploadedAtlas?.texture ?? null;
        materialAtlasDim = atlas.dim;
        materialAtlasLayerCount = atlas.layerCount;
        materialAtlasLayerCapacity = refreshedCapacity ?? uploadedAtlas?.capacity ?? 0;
        deleteOldAtlas = false;
        if (reason != null) {
          pushMaterialAtlasRefreshWarning(
            structuredWarnings,
            { method: 'updatePrimitive', primitiveId },
            current,
            atlas,
            reason,
            materialAtlasLayerCapacity,
          );
        }
      }
    }
  }
  const materialLayerMap = atlasRefreshNeeded ? atlas?.layerOfByColorSpace ?? null : current.materialLayerMap;
  const materialData = packMaterialsTexture(
    nextMaterials,
    materialLayerMap ?? undefined,
    { vertexColorMaterialIds: current.vertexColorMaterialIds },
  );
  updateRgba32f(gl, current.materials, materialData.data as Float32Array, materialData.dim, 'scene materials');

  const meshLightsData = hasMeshAreaLightForPrimitive(nextScene, primitiveId)
    || (current.meshLightCount ?? 0) > 0
    ? packMeshAreaLights(nextScene, nextGeoPack)
    : null;
  const meshLights = meshLightsData?.data != null
    ? uploadRgba32f(gl, meshLightsData.data, meshLightsData.dim, 'mesh-area lights')
    : null;
  return {
    textures: withTextureReplacementsForGl(gl, current, {
      ...(atlasRefreshNeeded
        ? {
            textures2DArray: atlasTexture,
            materialAtlasDim,
            materialAtlasLayerCount,
            materialAtlasLayerCapacity,
            materialLayerMap,
          }
        : {}),
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
    deleteOldTextures: [
      ...(deleteOldAtlas ? [current.textures2DArray] : []),
      ...(meshLightsData != null ? [current.meshLights] : []),
    ],
    ...(structuredWarnings.length > 0 ? { structuredWarnings } : {}),
  };
}

export function tryFastPathGeometryMutation(
  gl: WebGL2RenderingContext,
  current: UploadedSceneTextures | null,
  currentMerged: WorldSpaceMergeResult | null,
  nextScene: Scene,
  patch: Partial<ScenePrimitive>,
): WebGl2MutationSwap | null {
  if (current == null || !canRefreshGeometryTextures(patch)) return null;
  if (currentMerged != null) {
    const refit = buildRefitSceneGeometryTextures(nextScene, currentMerged, {
      warningPhase: 'mutation',
      warningMethod: 'updatePrimitive',
    });
    if (refit != null) {
      const structuredWarnings: EngineWarning[] = [...refit.structuredWarnings];
      for (const warning of refit.warnings) {
        structuredWarnings.push({
          code: 'pt-webgl2.scene-upload-warning',
          backend: 'pt-webgl2',
          phase: 'mutation',
          method: 'updatePrimitive',
          message: `[vitrum/pt-webgl2] ${warning}`,
          details: { warning, operation: 'geometry-bvh-refit-subimage-update' },
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
            { vertexColorMaterialIds: refit.vertexColorMaterialIds },
          )
        : null;
      updateRgba32f(gl, current.bvhBounds, refit.bvhData.bounds, refit.bvhData.boundsDim, 'scene BVH bounds');
      updateRgba32f(gl, current.bvhPosition, refit.bvhData.position, refit.bvhData.positionDim, 'scene BVH position');
      updateRgba32fArray(
        gl,
        current.attributesArray,
        refit.attrData.data,
        refit.attrData.dim,
        refit.attrData.layers,
        'vertex attributes',
      );
      if (materialsData != null) {
        updateRgba32f(gl, current.materials, materialsData.data as Float32Array, materialsData.dim, 'scene materials');
      }

      const nextMeshLightsData = refit.meshLightsData;
      const currentMeshLightDim = squareDim((current.meshLightCount ?? 0) * TRI_LIGHT_PIXELS);
      let meshLights = current.meshLights;
      const deleteOldTextures: (WebGLTexture | null)[] = [];
      if (nextMeshLightsData.data == null) {
        if (current.meshLights != null) {
          meshLights = null;
          deleteOldTextures.push(current.meshLights);
        }
      } else if (current.meshLights != null && currentMeshLightDim === nextMeshLightsData.dim) {
        updateRgba32f(gl, current.meshLights, nextMeshLightsData.data, nextMeshLightsData.dim, 'mesh-area lights');
      } else {
        meshLights = uploadRgba32f(gl, nextMeshLightsData.data, nextMeshLightsData.dim, 'mesh-area lights');
        deleteOldTextures.push(current.meshLights);
      }

      return {
        textures: withTextureReplacementsForGl(gl, current, {
          meshLights,
          meshLightCount: nextMeshLightsData.triLightCount,
          totalEmissiveArea: nextMeshLightsData.totalEmissiveArea,
          totalEmissivePower: nextMeshLightsData.totalEmissivePower,
          triangleCount: refit.merged.triangleCount,
          vertexColorMaterialIds: refit.vertexColorMaterialIds,
        }),
        geoPack: refit.merged,
        deleteOldTextures,
        structuredWarnings,
      };
    }
  }

  if (currentMerged != null) {
    const built = buildSceneGeometryTextureData(nextScene, {
      warningPhase: 'mutation',
      warningMethod: 'updatePrimitive',
    });
    const vertexColorMaterialIdsChanged = !sameNumberSet(
      current.vertexColorMaterialIds,
      built.vertexColorMaterialIds,
    );
    const materialsData = packMaterialsTexture(
      built.merged.materials,
      current.materialLayerMap ?? undefined,
      { vertexColorMaterialIds: built.vertexColorMaterialIds },
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
      updateRgba32f(gl, current.bvhBounds, built.bvhData.bounds, built.bvhData.boundsDim, 'scene BVH bounds');
      updateRgba32ui(gl, current.bvhContents, built.bvhData.contents, built.bvhData.contentsDim, 'scene BVH contents');
      updateRgba32f(gl, current.bvhPosition, built.bvhData.position, built.bvhData.positionDim, 'scene BVH position');
      updateRgba32ui(gl, current.bvhIndex, built.bvhData.index, built.bvhData.indexDim, 'scene BVH index');
      updateRgba32ui(
        gl,
        current.materialIndex,
        built.bvhData.materialIndex,
        built.bvhData.materialIndexDim,
        'scene BVH material index',
      );
      if (vertexColorMaterialIdsChanged) {
        updateRgba32f(gl, current.materials, materialsData.data as Float32Array, materialsData.dim, 'scene materials');
      }
      updateRgba32fArray(
        gl,
        current.attributesArray,
        built.attrData.data,
        built.attrData.dim,
        built.attrData.layers,
        'vertex attributes',
      );

      let meshLights = current.meshLights;
      const deleteOldTextures: (WebGLTexture | null)[] = [];
      if (current.meshLights != null && built.meshLightsData.data == null) {
        meshLights = null;
        deleteOldTextures.push(current.meshLights);
      } else if (current.meshLights != null && built.meshLightsData.data != null) {
        updateRgba32f(gl, current.meshLights, built.meshLightsData.data, built.meshLightsData.dim, 'mesh-area lights');
      }

      return {
        textures: withTextureReplacementsForGl(gl, current, {
          meshLights,
          meshLightCount: built.meshLightsData.triLightCount,
          totalEmissiveArea: built.meshLightsData.totalEmissiveArea,
          totalEmissivePower: built.meshLightsData.totalEmissivePower,
          triangleCount: built.triangleCount,
          vertexColorMaterialIds: built.vertexColorMaterialIds,
        }),
        geoPack: built.merged,
        deleteOldTextures,
        structuredWarnings,
      };
    }
  }

  const built = buildSceneGeometryTextures(gl, nextScene, {
    warningPhase: 'mutation',
    warningMethod: 'updatePrimitive',
  });
  const structuredWarnings: EngineWarning[] = [...built.structuredWarnings];
  for (const warning of built.warnings) {
    structuredWarnings.push({
      code: 'pt-webgl2.scene-upload-warning',
      backend: 'pt-webgl2',
      phase: 'mutation',
      method: 'updatePrimitive',
      message: `[vitrum/pt-webgl2] ${warning}`,
      details: { warning, operation: 'geometry-texture-refresh' },
    });
  }
  const vertexColorMaterialIdsChanged = !sameNumberSet(
    current.vertexColorMaterialIds,
    built.vertexColorMaterialIds,
  );
  const materialsData = vertexColorMaterialIdsChanged
    ? packMaterialsTexture(
        built.merged.materials,
        current.materialLayerMap ?? undefined,
        { vertexColorMaterialIds: built.vertexColorMaterialIds },
      )
    : null;
  const materials = materialsData != null
    ? uploadRgba32f(gl, materialsData.data, materialsData.dim, 'scene materials')
    : null;

  return {
    textures: withTextureReplacementsForGl(gl, current, {
      bvhBounds: built.bvh.bounds,
      bvhContents: built.bvh.contents,
      bvhPosition: built.bvh.position,
      bvhIndex: built.bvh.index,
      materialIndex: built.bvh.materialIndex,
      ...(materials != null ? { materials } : {}),
      attributesArray: built.attributesArray,
      meshLights: built.meshLights,
      meshLightCount: built.meshLightCount,
      totalEmissiveArea: built.totalEmissiveArea,
      totalEmissivePower: built.totalEmissivePower,
      triangleCount: built.triangleCount,
      vertexColorMaterialIds: built.vertexColorMaterialIds,
    }),
    geoPack: built.merged,
    deleteOldTextures: [
      current.bvhBounds,
      current.bvhContents,
      current.bvhPosition,
      current.bvhIndex,
      current.materialIndex,
      ...(materials != null ? [current.materials] : []),
      current.attributesArray,
      current.meshLights,
    ],
    structuredWarnings,
    mutationFallback: {
      fallbackReason: 'geometry-bvh-texture-rebuild',
      nativePatchMissing: 'targeted-geometry-bvh-refit',
    },
  };
}

export function tryFastPathPrimitiveListMutation(
  gl: WebGL2RenderingContext,
  current: UploadedSceneTextures | null,
  currentMerged: WorldSpaceMergeResult | null,
  nextScene: Scene,
  opts: {
    readonly method: 'addPrimitive' | 'removePrimitive';
    readonly primitiveId: string;
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

  const atlas = packTextureAtlas(built.merged.materials, {
    onWarning: (warning) => structuredWarnings.push(warning),
    warningPhase: 'mutation',
    warningMethod: opts.method,
  });
  const materialsData = packMaterialsTexture(
    built.merged.materials,
    atlas?.layerOfByColorSpace,
    { vertexColorMaterialIds: built.vertexColorMaterialIds },
  );
  const maxAtlasLayers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
  const nextAtlasLayerCapacity = atlas != null
    ? textureAtlasLayerCapacity(atlas.layerCount, maxAtlasLayers)
    : 0;
  if (
    currentMerged != null &&
    canRewritePrimitiveListResident(
      current,
      currentMerged,
      built,
    )
  ) {
    const deleteOldTextures: (WebGLTexture | null)[] = [];
    updateRgba32f(gl, current.bvhBounds, built.bvhData.bounds, built.bvhData.boundsDim, 'scene BVH bounds');
    updateRgba32ui(gl, current.bvhContents, built.bvhData.contents, built.bvhData.contentsDim, 'scene BVH contents');
    updateRgba32f(gl, current.bvhPosition, built.bvhData.position, built.bvhData.positionDim, 'scene BVH position');
    updateRgba32ui(gl, current.bvhIndex, built.bvhData.index, built.bvhData.indexDim, 'scene BVH index');
    updateRgba32ui(
      gl,
      current.materialIndex,
      built.bvhData.materialIndex,
        built.bvhData.materialIndexDim,
        'scene BVH material index',
      );
    let materials = current.materials;
    if (materialTextureDimMatches(current, currentMerged, materialsData)) {
      updateRgba32f(gl, current.materials, materialsData.data as Float32Array, materialsData.dim, 'scene materials');
    } else {
      materials = uploadRgba32f(gl, materialsData.data as Float32Array, materialsData.dim, 'scene materials');
      deleteOldTextures.push(current.materials);
    }
    updateRgba32fArray(
      gl,
      current.attributesArray,
      built.attrData.data,
      built.attrData.dim,
      built.attrData.layers,
      'vertex attributes',
    );
    let meshLights = current.meshLights;
    if (current.meshLights != null && built.meshLightsData.data == null) {
      meshLights = null;
      deleteOldTextures.push(current.meshLights);
    } else if (current.meshLights != null && built.meshLightsData.data != null) {
      updateRgba32f(gl, current.meshLights, built.meshLightsData.data, built.meshLightsData.dim, 'mesh-area lights');
    }

    let textures2DArray = current.textures2DArray;
    let materialAtlasDim = 0;
    let materialAtlasLayerCount = 0;
    let materialAtlasLayerCapacity = 0;
    let materialLayerMap: TextureAtlasLayerMap | null = null;
    const atlasRefreshReason = materialAtlasRefreshReason(current, atlas, nextAtlasLayerCapacity);
    if (atlas == null) {
      if (current.textures2DArray != null) {
        textures2DArray = null;
        deleteOldTextures.push(current.textures2DArray);
      }
    } else {
      materialAtlasDim = atlas.dim;
      materialAtlasLayerCount = atlas.layerCount;
      materialLayerMap = atlas.layerOfByColorSpace;
      if (canUpdateAtlasInPlace(current, atlas, nextAtlasLayerCapacity)) {
        updateTextureAtlasLayers(gl, current.textures2DArray as WebGLTexture, atlas);
        materialAtlasLayerCapacity = current.materialAtlasLayerCapacity;
      } else if (current.textures2DArray != null) {
        materialAtlasLayerCapacity = refreshTextureAtlasStorage(
          gl,
          current.textures2DArray,
          atlas,
          { layerCapacity: nextAtlasLayerCapacity },
        );
      } else {
        const uploadedAtlas = uploadAtlasWithCapacity(gl, atlas);
        textures2DArray = uploadedAtlas.texture;
        materialAtlasLayerCapacity = uploadedAtlas.capacity;
      }
    }
    if (atlasRefreshReason != null) {
      pushMaterialAtlasRefreshWarning(
        structuredWarnings,
        { method: opts.method, primitiveId: opts.primitiveId },
        current,
        atlas,
        atlasRefreshReason,
        nextAtlasLayerCapacity,
      );
    }

    return {
      textures: withTextureReplacementsForGl(gl, current, {
        materials,
        textures2DArray,
        materialAtlasDim,
        materialAtlasLayerCount,
        materialAtlasLayerCapacity,
        materialLayerMap,
        meshLights,
        meshLightCount: built.meshLightsData.triLightCount,
        totalEmissiveArea: built.meshLightsData.totalEmissiveArea,
        totalEmissivePower: built.meshLightsData.totalEmissivePower,
        triangleCount: built.triangleCount,
        vertexColorMaterialIds: built.vertexColorMaterialIds,
      }),
      geoPack: built.merged,
      scene: mutationScene,
      deleteOldTextures,
      structuredWarnings,
    };
  }

  const bvh = uploadBvhTextures(gl, built.bvhData);
  const attributesArray = uploadRgba32fArray(
    gl,
    built.attrData.data,
    built.attrData.dim,
    built.attrData.layers,
    'vertex attributes',
  );
  const meshLights = built.meshLightsData.data != null
    ? uploadRgba32f(gl, built.meshLightsData.data, built.meshLightsData.dim, 'mesh-area lights')
    : null;
  const uploadedAtlas = atlas != null ? uploadAtlasWithCapacity(gl, atlas) : null;
  const textures2DArray = uploadedAtlas?.texture ?? null;
  const materials = uploadRgba32f(gl, materialsData.data, materialsData.dim, 'scene materials');

  return {
    textures: withTextureReplacementsForGl(gl, current, {
      bvhBounds: bvh.bounds,
      bvhContents: bvh.contents,
      bvhPosition: bvh.position,
      bvhIndex: bvh.index,
      materialIndex: bvh.materialIndex,
      materials,
      attributesArray,
      meshLights,
      textures2DArray,
      materialAtlasDim: atlas?.dim ?? 0,
      materialAtlasLayerCount: atlas?.layerCount ?? 0,
      materialAtlasLayerCapacity: uploadedAtlas?.capacity ?? 0,
      materialLayerMap: atlas?.layerOfByColorSpace ?? null,
      meshLightCount: built.meshLightsData.triLightCount,
      totalEmissiveArea: built.meshLightsData.totalEmissiveArea,
      totalEmissivePower: built.meshLightsData.totalEmissivePower,
      triangleCount: built.triangleCount,
      vertexColorMaterialIds: built.vertexColorMaterialIds,
    }),
    geoPack: built.merged,
    scene: mutationScene,
    deleteOldTextures: [
      current.bvhBounds,
      current.bvhContents,
      current.bvhPosition,
      current.bvhIndex,
      current.materialIndex,
      current.materials,
      current.attributesArray,
      current.meshLights,
      current.textures2DArray,
    ],
    structuredWarnings,
    mutationFallback: {
      fallbackReason: 'primitive-list-texture-refresh',
      nativePatchMissing: 'targeted-primitive-list-splice',
    },
  };
}

export function tryFastPathEmitterMutation(
  gl: WebGL2RenderingContext,
  current: UploadedSceneTextures | null,
  geoPack: WorldSpaceMergeResult | null,
  nextScene: Scene,
  emitterId: string,
): WebGl2MutationSwap | null {
  if (current == null || geoPack == null) return null;
  const changed = nextScene.emitters.find((e) => String(e.id) === emitterId);
  const isMeshAreaMutation = changed?.kind === 'mesh-area';
  const lightsData = packLightsTexture(nextScene.emitters);
  const structuredWarnings: EngineWarning[] = [];
  const lights = uploadRgba32f(gl, lightsData.data, lightsData.dim, 'scene lights');
  const meshLightsData = packMeshAreaLights(nextScene, geoPack);
  const meshLights = meshLightsData.data != null
    ? uploadRgba32f(gl, meshLightsData.data, meshLightsData.dim, 'mesh-area lights')
    : null;
  const foldedMaterials = isMeshAreaMutation
    ? repackMeshAreaFoldedMaterials(geoPack, nextScene)
    : null;
  if (foldedMaterials != null) {
    const materialData = packMaterialsTexture(
      foldedMaterials.nextMaterials,
      current.materialLayerMap ?? undefined,
      { vertexColorMaterialIds: current.vertexColorMaterialIds },
    );
    updateRgba32f(
      gl,
      current.materials,
      materialData.data as Float32Array,
      materialData.dim,
      'scene materials',
    );
  }
  return {
    textures: withTextureReplacementsForGl(gl, current, {
      lights,
      lightCount: lightsData.lightCount,
      meshLights,
      meshLightCount: meshLightsData.triLightCount,
      totalEmissiveArea: meshLightsData.totalEmissiveArea,
      totalEmissivePower: meshLightsData.totalEmissivePower,
    }),
    ...(foldedMaterials != null ? { geoPack: foldedMaterials.nextGeoPack } : {}),
    deleteOldTextures: [
      current.lights,
      current.meshLights,
    ],
    structuredWarnings,
  };
}

export function fastPathEnvironmentMutation(
  gl: WebGL2RenderingContext,
  current: UploadedSceneTextures | null,
  nextScene: Scene,
): WebGl2MutationSwap | null {
  if (current == null) return null;
  const structuredWarnings: EngineWarning[] = [];
  const env = buildEquirectInfo(nextScene.environment, {
    onWarning: (warning) => structuredWarnings.push(warning),
    warningPhase: 'mutation',
    warningMethod: 'updateEnvironment',
  });
  const envMap = env.map
    ? uploadRgba32fRect(gl, env.map.data, env.map.width, env.map.height, 'environment map')
    : null;
  const envMarginal = env.marginal
    ? uploadRgba32fRect(gl, env.marginal.data, env.marginal.width, env.marginal.height, 'environment marginal CDF')
    : null;
  const envConditional = env.conditional
    ? uploadRgba32fRect(gl, env.conditional.data, env.conditional.width, env.conditional.height, 'environment conditional CDF')
    : null;
  return {
    textures: withTextureReplacementsForGl(gl, current, {
      envMap,
      envMarginal,
      envConditional,
      envTotalSum: env.totalSum,
      envWidth: env.map?.width ?? 0,
      envHeight: env.map?.height ?? 0,
    }),
    deleteOldTextures: [current.envMap, current.envMarginal, current.envConditional],
    structuredWarnings,
  };
}
