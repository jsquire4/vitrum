import type { EngineWarning, MaterialSpec, Scene, ScenePrimitive } from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import { packMaterialsTexture } from './materialsTexture.js';
import { packLightsTexture } from './lightsTexture.js';
import { hasMeshAreaLightForPrimitive, packMeshAreaLights, TRI_LIGHT_PIXELS } from './meshAreaLights.js';
import { foldMeshAreaEmittersIntoMaterials } from './foldEmissiveEmitters.js';
import { buildEquirectInfo } from './equirectHdrInfo.js';
import type { UploadedSceneTextures } from './sceneTextures.js';
import {
  buildRefitSceneGeometryTextures,
  buildSceneGeometryTextures,
  expandAnalyticPrimitiveFallbacks,
  updateRgba32f,
  updateRgba32fArray,
  uploadRgba32f,
  uploadRgba32fRect,
} from './uploadSceneTextures.js';
import { packTextureAtlas, uploadTextureAtlas } from './texturesArray.js';
import type { TextureAtlasLayerMap, TextureSampleColorSpace } from './texturesArray.js';
import { squareDim } from './bvhTextureAdapter.js';

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

function textureValueNeedsAtlasRefresh(
  field: string,
  value: unknown,
  materialLayerMap: TextureAtlasLayerMap | null,
): boolean {
  if (value == null) return false;
  if (typeof value !== 'object') return false;
  const handle = (value as { readonly handle?: unknown }).handle;
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
  const atlasRefreshNeeded = texturePatchNeedsAtlasRefresh(patch, current.materialLayerMap);

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
  const atlasTexture = atlasRefreshNeeded && atlas != null ? uploadTextureAtlas(gl, atlas) : null;
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
      ...(atlasRefreshNeeded ? [current.textures2DArray] : []),
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
  nextScene: Scene,
  opts: {
    readonly method: 'addPrimitive' | 'removePrimitive';
  },
): WebGl2MutationSwap | null {
  if (current == null) return null;
  const analyticExpansion = expandAnalyticPrimitiveFallbacks(nextScene);
  const mutationScene = analyticExpansion.scene;
  if (mutationScene.primitives.some((primitive) => !isMeshLikePrimitive(primitive))) return null;

  const built = buildSceneGeometryTextures(gl, mutationScene, {
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
  const textures2DArray = atlas != null ? uploadTextureAtlas(gl, atlas) : null;
  const materialsData = packMaterialsTexture(
    built.merged.materials,
    atlas?.layerOfByColorSpace,
    { vertexColorMaterialIds: built.vertexColorMaterialIds },
  );
  const materials = uploadRgba32f(gl, materialsData.data, materialsData.dim, 'scene materials');

  return {
    textures: withTextureReplacementsForGl(gl, current, {
      bvhBounds: built.bvh.bounds,
      bvhContents: built.bvh.contents,
      bvhPosition: built.bvh.position,
      bvhIndex: built.bvh.index,
      materialIndex: built.bvh.materialIndex,
      materials,
      attributesArray: built.attributesArray,
      meshLights: built.meshLights,
      textures2DArray,
      materialLayerMap: atlas?.layerOfByColorSpace ?? null,
      meshLightCount: built.meshLightCount,
      totalEmissiveArea: built.totalEmissiveArea,
      totalEmissivePower: built.totalEmissivePower,
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
