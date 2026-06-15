import type { EngineWarning, MaterialSpec, Scene, ScenePrimitive } from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import { packMaterialsTexture } from './materialsTexture.js';
import { packLightsTexture } from './lightsTexture.js';
import { packMeshAreaLights } from './meshAreaLights.js';
import { foldMeshAreaEmittersIntoMaterials } from './foldEmissiveEmitters.js';
import { buildEquirectInfo } from './equirectHdrInfo.js';
import type { UploadedSceneTextures } from './sceneTextures.js';
import {
  uploadRgba32f,
  uploadRgba32fRect,
} from './uploadSceneTextures.js';

const TEXTURE_MAP_FIELDS: ReadonlySet<string> = new Set([
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

export interface WebGl2MutationSwap {
  readonly textures: UploadedSceneTextures;
  readonly geoPack?: WorldSpaceMergeResult;
  readonly deleteOldTextures: readonly (WebGLTexture | null)[];
  readonly structuredWarnings?: readonly EngineWarning[];
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
  const mat = patch.material as unknown as Record<string, unknown>;
  for (const field of Object.keys(mat)) {
    if (TEXTURE_MAP_FIELDS.has(field)) return false;
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
  gl: WebGL2RenderingContext,
  current: UploadedSceneTextures,
  geoPack: WorldSpaceMergeResult,
  nextScene: Scene,
): { materials: WebGLTexture; nextGeoPack: WorldSpaceMergeResult } {
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

  const data = packMaterialsTexture(
    nextMaterials,
    current.materialLayerMap ?? undefined,
    { vertexColorMaterialIds: current.vertexColorMaterialIds },
  );
  return {
    materials: uploadRgba32f(gl, data.data, data.dim, 'scene materials'),
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

  const nextMaterials = geoPack.materials.slice();
  nextMaterials[slot] = materialWithCastShadow(primitive);
  const data = packMaterialsTexture(
    nextMaterials,
    current.materialLayerMap ?? undefined,
    { vertexColorMaterialIds: current.vertexColorMaterialIds },
  );
  const materials = uploadRgba32f(gl, data.data, data.dim, 'scene materials');
  return {
    textures: withTextureReplacementsForGl(gl, current, { materials }),
    geoPack: { ...geoPack, materials: nextMaterials },
    deleteOldTextures: [current.materials],
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
  const lights = uploadRgba32f(gl, lightsData.data, lightsData.dim, 'scene lights');
  const meshLightsData = packMeshAreaLights(nextScene, geoPack);
  const meshLights = meshLightsData.data != null
    ? uploadRgba32f(gl, meshLightsData.data, meshLightsData.dim, 'mesh-area lights')
    : null;
  const foldedMaterials = isMeshAreaMutation
    ? repackMeshAreaFoldedMaterials(gl, current, geoPack, nextScene)
    : null;
  return {
    textures: withTextureReplacementsForGl(gl, current, {
      ...(foldedMaterials != null ? { materials: foldedMaterials.materials } : {}),
      lights,
      lightCount: lightsData.lightCount,
      meshLights,
      meshLightCount: meshLightsData.triLightCount,
      totalEmissiveArea: meshLightsData.totalEmissiveArea,
    }),
    ...(foldedMaterials != null ? { geoPack: foldedMaterials.nextGeoPack } : {}),
    deleteOldTextures: [
      current.lights,
      current.meshLights,
      ...(foldedMaterials != null ? [current.materials] : []),
    ],
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
