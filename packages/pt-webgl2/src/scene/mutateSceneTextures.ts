import type { EngineWarning, MaterialSpec, Scene, ScenePrimitive } from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import { MATERIAL_PIXELS, packMaterialsTexture } from './materialsTexture.js';
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
  updateRgba32f,
  updateRgba32fRect,
  updateRgba32fArray,
  updateRgba32ui,
} from './uploadSceneTextures.js';
import {
  packTextureAtlas,
  textureAtlasLayerCapacity,
  updateTextureAtlasLayers,
} from './texturesArray.js';
import type { TextureAtlasLayerMap, TextureSampleColorSpace } from './texturesArray.js';
import { packBvhTextureData, squareDim, type BvhTextureData } from './bvhTextureAdapter.js';
import {
  ATTR_LAYER_UV,
  ATTR_LAYER_UV1,
  sameUvAttributeLayout,
} from './uvAttributeLayout.js';

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

function isMeshLikePrimitive(p: ScenePrimitive | undefined): p is Extract<
  ScenePrimitive,
  { kind: 'mesh' | 'instanced-mesh' | 'skinned-mesh' }
> {
  return p?.kind === 'mesh' || p?.kind === 'instanced-mesh' || p?.kind === 'skinned-mesh';
}

function canFastPathMaterialPatch(patch: Partial<ScenePrimitive>): boolean {
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
  readonly value: unknown;
}

function textureMapPatchEntries(material: Record<string, unknown>): TextureMapPatchEntry[] {
  const entries: TextureMapPatchEntry[] = [];
  for (const field of Object.keys(material)) {
    if (!TEXTURE_MAP_FIELDS.has(field)) continue;
    entries.push({
      path: field,
      colorSpace: TEXTURE_MAP_COLOR_SPACE[field] ?? 'linear',
      value: material[field],
    });
  }
  for (const descriptor of LAYER_TEXTURE_MAP_FIELDS) {
    if (!hasOwn(material, descriptor.layer)) continue;
    const layer = material[descriptor.layer];
    if (isRecord(layer)) {
      if (!hasOwn(layer, descriptor.field)) continue;
      entries.push({ path: descriptor.path, colorSpace: descriptor.colorSpace, value: layer[descriptor.field] });
    } else {
      entries.push({ path: descriptor.path, colorSpace: descriptor.colorSpace, value: undefined });
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

export function materialTextureMapPatchFields(patch: Partial<ScenePrimitive>): string[] {
  if (patch.material == null) return [];
  return textureMapPatchEntries(patch.material as unknown as Record<string, unknown>)
    .map((entry) => entry.path)
    .sort();
}

function texturePatchNeedsAtlasRefresh(
  patch: Partial<ScenePrimitive> & { material: MaterialSpec },
  materialLayerMap: TextureAtlasLayerMap | null,
): boolean {
  const mat = patch.material as unknown as Record<string, unknown>;
  for (const entry of textureMapPatchEntries(mat)) {
    if (textureValueNeedsAtlasRefresh(entry.colorSpace, entry.value, materialLayerMap)) return true;
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
  patch: Partial<ScenePrimitive> & { material: MaterialSpec },
  current: UploadedSceneTextures,
): boolean {
  const layout = effectiveUvLayerMap(current);
  for (const entry of textureMapPatchEntries(
    patch.material as unknown as Record<string, unknown>,
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
  patch: Partial<ScenePrimitive> & { material: MaterialSpec },
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
  colorSpace: TextureSampleColorSpace,
  value: unknown,
  materialLayerMap: TextureAtlasLayerMap | null,
): boolean {
  if (value == null) return false;
  if (typeof value !== 'object') return false;
  const handle = textureHandleOf(value);
  if (handle == null) return false;
  return materialLayerMap?.[colorSpace].has(handle) !== true;
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
    updateRgba32f(gl, current.meshLights, nextMeshLightsData.data, nextMeshLightsData.dim, 'mesh-area lights');
    return current.meshLights;
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
    updateRgba32f(gl, current.lights, lights, nextLightsData.dim, 'scene lights');
  } else {
    throw new Error('pt-webgl2: light storage replacement requires a transactional scene rebuild');
  }
  return current.lights;
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
    updateRgba32fRect(gl, currentTexture, nextData.data, nextData.width, nextData.height, resourceName);
  } else {
    throw new Error(`pt-webgl2: ${resourceName} storage replacement requires a transactional scene rebuild`);
  }
  return currentTexture;
}

function updateResidentMaterialSlotTexture(
  gl: WebGL2RenderingContext,
  texture: WebGLTexture,
  data: Float32Array,
  dim: number,
  slot: number,
): void {
  if (gl.isContextLost()) {
    throw new Error('pt-webgl2: WebGL context lost — cannot update scene materials texture');
  }
  gl.bindTexture(gl.TEXTURE_2D, texture);
  let texel = slot * MATERIAL_PIXELS;
  let remaining = MATERIAL_PIXELS;
  while (remaining > 0) {
    const x = texel % dim;
    const y = Math.floor(texel / dim);
    const width = Math.min(remaining, dim - x);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      x,
      y,
      width,
      1,
      gl.RGBA,
      gl.FLOAT,
      data.subarray(texel * 4, (texel + width) * 4),
    );
    texel += width;
    remaining -= width;
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
      ], 'pt-webgl2: one or more mutated scene textures failed to retire');
    },
  };
  return next;
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
  patch: Partial<ScenePrimitive>,
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

  const maxAtlasLayers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
  const atlas = atlasRefreshNeeded
    ? packTextureAtlas(nextMaterials, {
        onWarning: (warning) => structuredWarnings.push(warning),
        warningPhase: 'mutation',
        warningMethod: 'updatePrimitive',
        maxArrayTextureLayers: maxAtlasLayers,
      })
    : null;
  const nextAtlasLayerCapacity = atlas != null
    ? textureAtlasLayerCapacity(atlas.layerCount, maxAtlasLayers)
    : 0;
  const meshLightsData = hasMeshAreaLightForPrimitive(nextScene, primitiveId)
    || (current.meshLightCount ?? 0) > 0
    ? packMeshAreaLights(nextScene, nextGeoPack)
    : null;
  const meshLightStorageMatches = meshLightsData == null
    || meshLightsData.data == null
    || (
      current.meshLights != null &&
      squareDim((current.meshLightCount ?? 0) * TRI_LIGHT_PIXELS) === meshLightsData.dim
    );
  const atlasStorageMatches = !atlasRefreshNeeded
    || atlas == null
    || canUpdateAtlasInPlace(current, atlas, nextAtlasLayerCapacity);
  if (!meshLightStorageMatches || !atlasStorageMatches) {
    return null;
  }

  let atlasTexture = current.textures2DArray;
  let materialAtlasDim = current.materialAtlasDim;
  let materialAtlasLayerCount = current.materialAtlasLayerCount;
  let materialAtlasLayerCapacity = current.materialAtlasLayerCapacity;
  let deleteOldAtlas = false;
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
      updateTextureAtlasLayers(gl, current.textures2DArray as WebGLTexture, atlas);
      materialAtlasDim = atlas.dim;
      materialAtlasLayerCount = atlas.layerCount;
    }
  }
  const materialLayerMap = atlasRefreshNeeded ? atlas?.layerOfByColorSpace ?? null : current.materialLayerMap;
  const materialData = packMaterialsTexture(
    nextMaterials,
    materialLayerMap ?? undefined,
    {
      vertexColorMaterialIds: current.vertexColorMaterialIds,
      uvLayerByTexCoord: effectiveUvLayerMap(current),
    },
  );
  if (
    foldedMaterials == null &&
    !atlasRefreshNeeded &&
    materialTextureDimMatches(current, geoPack, materialData)
  ) {
    updateResidentMaterialSlotTexture(gl, current.materials, materialData.data as Float32Array, materialData.dim, slot);
  } else {
    updateRgba32f(gl, current.materials, materialData.data as Float32Array, materialData.dim, 'scene materials');
  }

  const deleteOldTextures: (WebGLTexture | null)[] = [];
  const meshLights = meshLightsData != null
    ? updateResidentMeshLightTexture(gl, current, meshLightsData, deleteOldTextures)
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
      ...deleteOldTextures,
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
            {
              vertexColorMaterialIds: refit.vertexColorMaterialIds,
              uvLayerByTexCoord: refit.uvLayerByTexCoord,
            },
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

      const deleteOldTextures: (WebGLTexture | null)[] = [];
      const nextMeshLightsData = refit.meshLightsData;
      const meshLights = updateResidentMeshLightTexture(gl, current, nextMeshLightsData, deleteOldTextures);

      return {
        textures: withTextureReplacementsForGl(gl, current, {
          meshLights,
          meshLightCount: nextMeshLightsData.triLightCount,
          totalEmissiveArea: nextMeshLightsData.totalEmissiveArea,
          totalEmissivePower: nextMeshLightsData.totalEmissivePower,
          triangleCount: refit.merged.triangleCount,
          vertexColorMaterialIds: refit.vertexColorMaterialIds,
          uvLayerByTexCoord: refit.uvLayerByTexCoord,
          attributeLayerCount: refit.attrData.layers,
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

      const deleteOldTextures: (WebGLTexture | null)[] = [];
      const meshLights = updateResidentMeshLightTexture(gl, current, built.meshLightsData, deleteOldTextures);

      return {
        textures: withTextureReplacementsForGl(gl, current, {
          meshLights,
          meshLightCount: built.meshLightsData.triLightCount,
          totalEmissiveArea: built.meshLightsData.totalEmissiveArea,
          totalEmissivePower: built.meshLightsData.totalEmissivePower,
          triangleCount: built.triangleCount,
          vertexColorMaterialIds: built.vertexColorMaterialIds,
          uvLayerByTexCoord: built.uvLayerByTexCoord,
          attributeLayerCount: built.attrData.layers,
        }),
        geoPack: built.merged,
        deleteOldTextures,
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

  const maxAtlasLayers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
  const atlas = packTextureAtlas(built.merged.materials, {
    onWarning: (warning) => structuredWarnings.push(warning),
    warningPhase: 'mutation',
    warningMethod: opts.method,
    maxArrayTextureLayers: maxAtlasLayers,
  });
  const materialsData = packMaterialsTexture(
    built.merged.materials,
    atlas?.layerOfByColorSpace,
    {
      vertexColorMaterialIds: built.vertexColorMaterialIds,
      uvLayerByTexCoord: built.uvLayerByTexCoord,
    },
  );
  const nextAtlasLayerCapacity = atlas != null
    ? textureAtlasLayerCapacity(atlas.layerCount, maxAtlasLayers)
    : 0;
  const materialStorageMatches = currentMerged != null
    && materialTextureDimMatches(current, currentMerged, materialsData);
  const meshLightStorageMatches = built.meshLightsData.data == null
    || (
      current.meshLights != null
      && squareDim((current.meshLightCount ?? 0) * TRI_LIGHT_PIXELS) === built.meshLightsData.dim
    );
  const atlasStorageMatches = atlas == null
    || canUpdateAtlasInPlace(current, atlas, nextAtlasLayerCapacity);
  if (
    currentMerged != null &&
    canRewritePrimitiveListResident(
      current,
      currentMerged,
      built,
    ) &&
    materialStorageMatches &&
    meshLightStorageMatches &&
    atlasStorageMatches
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
    const materials = current.materials;
    updateRgba32f(gl, current.materials, materialsData.data as Float32Array, materialsData.dim, 'scene materials');
    updateRgba32fArray(
      gl,
      current.attributesArray,
      built.attrData.data,
      built.attrData.dim,
      built.attrData.layers,
      'vertex attributes',
    );
    const meshLights = updateResidentMeshLightTexture(gl, current, built.meshLightsData, deleteOldTextures);

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
      updateTextureAtlasLayers(gl, current.textures2DArray as WebGLTexture, atlas);
      materialAtlasLayerCapacity = current.materialAtlasLayerCapacity;
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
        uvLayerByTexCoord: built.uvLayerByTexCoord,
        attributeLayerCount: built.attrData.layers,
      }),
      geoPack: built.merged,
      scene: mutationScene,
      deleteOldTextures,
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
): WebGl2MutationSwap | null {
  if (current == null || geoPack == null) return null;
  const changed = nextScene.emitters.find((e) => String(e.id) === emitterId);
  const isMeshAreaMutation = changed?.kind === 'mesh-area';
  const lightsData = packLightsTexture(nextScene.emitters);
  const structuredWarnings: EngineWarning[] = [];
  const meshLightsData = packMeshAreaLights(nextScene, geoPack);
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
  const lights = updateResidentLightsTexture(gl, current, lightsData);
  const deleteOldTextures: (WebGLTexture | null)[] = [];
  const meshLights = updateResidentMeshLightTexture(gl, current, meshLightsData, deleteOldTextures);
  const foldedMaterials = isMeshAreaMutation
    ? repackMeshAreaFoldedMaterials(geoPack, nextScene)
    : null;
  if (foldedMaterials != null) {
    const materialData = packMaterialsTexture(
      foldedMaterials.nextMaterials,
      current.materialLayerMap ?? undefined,
      {
        vertexColorMaterialIds: current.vertexColorMaterialIds,
        uvLayerByTexCoord: effectiveUvLayerMap(current),
      },
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
    deleteOldTextures,
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
  const deleteOldTextures: (WebGLTexture | null)[] = [];
  const canUpdateRectWithoutAllocation = (
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
    !canUpdateRectWithoutAllocation(current.envMap, current.envWidth, current.envHeight, env.map) ||
    !canUpdateRectWithoutAllocation(
      current.envMarginal,
      current.envHeight,
      current.envMarginal != null ? 1 : 0,
      env.marginal,
    ) ||
    !canUpdateRectWithoutAllocation(current.envConditional, current.envWidth, current.envHeight, env.conditional)
  ) {
    return null;
  }
  const envMap = updateNullableRectTexture(
    gl,
    current.envMap,
    current.envWidth,
    current.envHeight,
    env.map,
    'environment map',
    deleteOldTextures,
  );
  const envMarginal = updateNullableRectTexture(
    gl,
    current.envMarginal,
    current.envHeight,
    current.envMarginal != null ? 1 : 0,
    env.marginal,
    'environment marginal CDF',
    deleteOldTextures,
  );
  const envConditional = updateNullableRectTexture(
    gl,
    current.envConditional,
    current.envWidth,
    current.envHeight,
    env.conditional,
    'environment conditional CDF',
    deleteOldTextures,
  );
  return {
    textures: withTextureReplacementsForGl(gl, current, {
      envMap,
      envMarginal,
      envConditional,
      envTotalSum: env.totalSum,
      envWidth: env.map?.width ?? 0,
      envHeight: env.map?.height ?? 0,
    }),
    deleteOldTextures,
    structuredWarnings,
  };
}
