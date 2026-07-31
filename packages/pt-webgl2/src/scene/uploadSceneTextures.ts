// uploadSceneTextures.ts — assemble the full `UploadedSceneTextures` bundle a
// pt-webgl2 path-trace program binds (plan/three-removal/03-scene-bvh-packers.md §8).
//
// Pipeline (THREE-free, from a `@vitrum/core` `Scene`):
//   1. partitionSceneBySupport(scene, caps)        → supported subset + warnings
//   2. mergeWorldSpaceFromCore(supported, …)       → merged world-space tri stream + BVH
//   3. packBvhTextureData + uploadBvhTextures       → the 4 BVH data textures + materialIndex
//   4. packMaterialTextureAtlases(…)                → RGBA8 + RGBA16F sampler2DArray pair
//      packMaterialsTexture(merged.materials)       → 136px/material RGBA32F sampler2D
//   5. packLightsTexture(scene.emitters)            → 6px/light RGBA32F sampler2D
//   6. buildEquirectInfo(scene.environment)         → equirect map + marginal/conditional CDFs
//   7. packAttributesArray(merged)                  → 5-layer RGBA32F sampler2DArray
//   8. assemble the bundle with a destroy() that deletes every owned texture.
//
// NOTE: `packMaterialsTexture` / `packLightsTexture` / `buildEquirectInfo` are
// authored by parallel agents (materialsTexture.ts / lightsTexture.ts /
// equirectHdrInfo.ts); they are imported here by the names the plan fixes and
// will exist at integration. Their return shapes are the `sceneTextures.ts`
// contracts (`MaterialsTextureData` / `LightsTextureData` / `EnvTextureData`).

import type { EngineCapabilities, EngineWarning, Scene, ScenePrimitive } from '@vitrum/core';
import {
  analyticPrimitiveToMesh,
  getPrimitiveActiveColorSet,
  partitionSceneBySupport,
} from '@vitrum/core';
import {
  mergeWorldSpaceFromCore,
  refitBvhBounds,
  type WorldSpaceMergeResult,
} from '@vitrum/shared-bvh';
import { packBvhTextureData, uploadBvhTextures, type BvhTextureData } from './bvhTextureAdapter.js';
import { allocGlTexture } from '../gl/texAlloc.js';
import { retireIndependently } from '../gl/resourceRetirement.js';
import { foldMeshAreaEmittersIntoMaterials } from './foldEmissiveEmitters.js';
import { packAttributesArray } from './attributesTextureArray.js';
import { packMaterialsTexture } from './materialsTexture.js';
import {
  materialTextureAtlasLayerCapacities,
  packMaterialTextureAtlases,
  uploadTextureAtlas,
} from './texturesArray.js';
import { packLightsTexture } from './lightsTexture.js';
import { assertSceneEmissiveMapsCpuReadable, packMeshAreaLights } from './meshAreaLights.js';
import { buildEquirectInfo } from './equirectHdrInfo.js';
import {
  computeWebgl2TransportBounds,
  type Webgl2TransportBounds,
} from './sceneScalePolicy.js';
import { solveSkinPrimitives } from './solveSkinPrimitives.js';
import type { UploadedSceneTextures } from './sceneTextures.js';
import { buildUvAttributeLayout, type UvAttributeLayout } from './uvAttributeLayout.js';

export interface SceneTexturesBuild {
  readonly textures: UploadedSceneTextures;
  readonly merged: WorldSpaceMergeResult;
  readonly warnings: string[];
  readonly structuredWarnings: readonly EngineWarning[];
  readonly transportBounds: Webgl2TransportBounds;
  /** The capability-filtered scene (H7: returned so callers reuse this single
   *  partition instead of running `partitionSceneBySupport` a second time). */
  readonly supported: Scene;
}

export interface SceneGeometryTextureDataBuild {
  readonly bvhData: BvhTextureData;
  readonly attrData: ReturnType<typeof packAttributesArray>;
  readonly meshLightsData: ReturnType<typeof packMeshAreaLights>;
  readonly triangleCount: number;
  readonly merged: WorldSpaceMergeResult;
  readonly vertexColorMaterialIds: ReadonlySet<number>;
  readonly uvLayerByTexCoord: ReadonlyMap<number, number>;
  readonly warnings: readonly string[];
  readonly structuredWarnings: readonly EngineWarning[];
}

export interface RefitSceneGeometryTexturesBuild {
  readonly bvhData: BvhTextureData;
  readonly attrData: ReturnType<typeof packAttributesArray>;
  readonly meshLightsData: ReturnType<typeof packMeshAreaLights>;
  readonly merged: WorldSpaceMergeResult;
  readonly vertexColorMaterialIds: ReadonlySet<number>;
  readonly uvLayerByTexCoord: ReadonlyMap<number, number>;
  readonly warnings: readonly string[];
  readonly structuredWarnings: readonly EngineWarning[];
}

interface GeometryBuildInputs {
  readonly skinnedScene: Scene;
  readonly merged: WorldSpaceMergeResult;
  readonly attrData: ReturnType<typeof packAttributesArray>;
  readonly uvLayout: UvAttributeLayout;
}

function vertexDisplacementWarningDetails(message: string): Readonly<Record<string, unknown>> {
  const details: Record<string, unknown> = {
    source: 'mergeWorldSpaceFromCore',
    warning: message,
  };
  const match =
    / displacementMap at (.+?)(?: handle | requests | has | displacementSubdivisions| triangle )/.exec(
      message,
    );
  if (match?.[1] !== undefined) details.sourcePath = match[1];
  return details;
}

function buildGeometryInputs(
  scene: Scene,
  warningOptions: {
    readonly onWarning: (warning: EngineWarning) => void;
    readonly warningPhase: string;
    readonly warningMethod: string;
  },
): GeometryBuildInputs {
  const ptScene = foldMeshAreaEmittersIntoMaterials(scene);
  const skinnedScene = solveSkinPrimitives(ptScene, warningOptions);
  const merged = mergeWorldSpaceFromCore(skinnedScene, {
    positionStride: 4,
    splitMaterialsByCastShadow: true,
    onWarning: (message) => {
      warningOptions.onWarning({
        code: 'pt-webgl2.vertex-displacement-warning',
        message: `[pt-webgl2] ${message}`,
        backend: 'pt-webgl2',
        phase: warningOptions.warningPhase,
        method: warningOptions.warningMethod,
        details: vertexDisplacementWarningDetails(message),
      });
    },
  });
  const uvLayout = buildUvAttributeLayout(skinnedScene, merged, merged.materials);
  const mergedTangents = mergeTangentsFromCore(
    skinnedScene,
    merged.meshVertexRanges,
    merged.vertexCount,
  );
  const mergedColors = mergeColorsFromCore(
    skinnedScene,
    merged.meshVertexRanges,
    merged.vertexCount,
  );
  const attrData = packAttributesArray({
    ...merged,
    uv1: uvLayout.mergedByTexCoord.get(1)!,
    extraUvLayers: uvLayout.extraUvLayers,
    ...(mergedTangents != null ? { tangents: mergedTangents } : {}),
    ...(mergedColors != null ? { colors: mergedColors } : {}),
  });
  return { skinnedScene, merged, attrData, uvLayout };
}

function assertAttributeLayerBudget(gl: WebGL2RenderingContext, requiredLayers: number): void {
  const limit = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
  if (Number.isFinite(limit) && limit > 0 && requiredLayers > limit) {
    throw new RangeError(
      `pt-webgl2: vertex attributes require ${requiredLayers} texture-array layers, ` +
        `but this device exposes MAX_ARRAY_TEXTURE_LAYERS=${limit}`,
    );
  }
}

export function buildSceneGeometryTextureData(
  scene: Scene,
  opts?: {
    readonly warningPhase?: string;
    readonly warningMethod?: string;
  },
): SceneGeometryTextureDataBuild {
  const structuredWarnings: EngineWarning[] = [];
  const warningOptions = {
    onWarning: (warning: EngineWarning) => structuredWarnings.push(warning),
    warningPhase: opts?.warningPhase ?? 'setScene',
    warningMethod: opts?.warningMethod ?? 'setScene',
  };
  const geometry = buildGeometryInputs(scene, warningOptions);
  const bvhData = packBvhTextureData(geometry.merged);
  const meshLightsData = packMeshAreaLights(
    scene,
    geometry.merged,
    geometry.uvLayout.mergedByTexCoord,
  );
  const vertexColorMaterialIds = collectVertexColorMaterialIds(
    geometry.skinnedScene,
    geometry.merged,
  );
  return {
    bvhData,
    attrData: geometry.attrData,
    meshLightsData,
    triangleCount: geometry.merged.triangleCount,
    merged: geometry.merged,
    vertexColorMaterialIds,
    uvLayerByTexCoord: geometry.uvLayout.layerByTexCoord,
    warnings: geometry.merged.warnings,
    structuredWarnings,
  };
}

function sameUint32Array(a: Uint32Array, b: Uint32Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function sameMeshRanges(
  a: WorldSpaceMergeResult['meshVertexRanges'],
  b: WorldSpaceMergeResult['meshVertexRanges'],
): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    const ai = a[i];
    const bi = b[i];
    if (
      ai == null ||
      bi == null ||
      ai.name !== bi.name ||
      ai.vertexStart !== bi.vertexStart ||
      ai.vertexCount !== bi.vertexCount ||
      ai.triStart !== bi.triStart ||
      ai.triCount !== bi.triCount ||
      (ai.windingFlipped === true) !== (bi.windingFlipped === true)
    ) {
      return false;
    }
  }
  return true;
}

function canRefitAgainstCurrentTopology(
  current: WorldSpaceMergeResult,
  next: WorldSpaceMergeResult,
): boolean {
  return (
    current.vertexCount === next.vertexCount &&
    current.triangleCount === next.triangleCount &&
    current.positionStrideFloats === next.positionStrideFloats &&
    current.bvhIndexStride === next.bvhIndexStride &&
    current.bvhNodes.length === next.bvhNodes.length &&
    sameMeshRanges(current.meshVertexRanges, next.meshVertexRanges) &&
    sameUint32Array(current.indices, next.indices) &&
    sameUint32Array(current.triMaterialId, next.triMaterialId) &&
    sameUint32Array(current.bvhTriToMergedTri, next.bvhTriToMergedTri) &&
    sameUint32Array(current.mergedIndices, next.mergedIndices) &&
    sameUint32Array(current.mergedTriMaterialId, next.mergedTriMaterialId)
  );
}

export function buildRefitSceneGeometryTextures(
  scene: Scene,
  currentMerged: WorldSpaceMergeResult,
  opts?: {
    readonly warningPhase?: string;
    readonly warningMethod?: string;
  },
): RefitSceneGeometryTexturesBuild | null {
  const structuredWarnings: EngineWarning[] = [];
  const warningOptions = {
    onWarning: (warning: EngineWarning) => structuredWarnings.push(warning),
    warningPhase: opts?.warningPhase ?? 'mutation',
    warningMethod: opts?.warningMethod ?? 'updatePrimitive',
  };
  const geometry = buildGeometryInputs(scene, warningOptions);
  if (!canRefitAgainstCurrentTopology(currentMerged, geometry.merged)) return null;

  const refitNodes = new Float32Array(currentMerged.bvhNodes);
  refitBvhBounds(
    refitNodes,
    currentMerged.indices,
    geometry.merged.positions,
    geometry.merged.positionStrideFloats,
    currentMerged.bvhIndexStride,
  );
  const refitMerged: WorldSpaceMergeResult = {
    ...geometry.merged,
    bvhNodes: refitNodes,
    indices: currentMerged.indices,
    triMaterialId: currentMerged.triMaterialId,
    bvhTriToMergedTri: currentMerged.bvhTriToMergedTri,
    mergedIndices: currentMerged.mergedIndices,
    mergedTriMaterialId: currentMerged.mergedTriMaterialId,
  };

  const bvhData = packBvhTextureData(refitMerged);
  const meshLightsData = packMeshAreaLights(scene, refitMerged, geometry.uvLayout.mergedByTexCoord);
  const vertexColorMaterialIds = collectVertexColorMaterialIds(geometry.skinnedScene, refitMerged);

  return {
    bvhData,
    attrData: geometry.attrData,
    meshLightsData,
    merged: refitMerged,
    vertexColorMaterialIds,
    uvLayerByTexCoord: geometry.uvLayout.layerByTexCoord,
    warnings: geometry.merged.warnings,
    structuredWarnings,
  };
}

/**
 * Build the full uploaded scene-texture bundle for `scene`, partitioned to what
 * `caps` declares this backend can ingest. The returned `merged` stream is
 * retained by the caller for incremental scene patches; `warnings` lists every
 * dropped/unsupported node. The bundle owns every texture — call
 * `textures.destroy()` to release them.
 */
export function buildSceneTextures(
  gl: WebGL2RenderingContext,
  scene: Scene,
  caps: EngineCapabilities,
  options: { readonly bdpt?: boolean } = {},
): SceneTexturesBuild {
  const analyticExpansion = expandAnalyticPrimitiveFallbacks(scene);
  // (1) capability filter
  const { supported, warnings } = partitionSceneBySupport(analyticExpansion.scene, caps);
  warnings.unshift(...analyticExpansion.warnings);
  // This must precede every GL allocation below. An opaque emissive map would
  // otherwise be sampled by forward hits while mesh-light NEE used scalar Le.
  assertSceneEmissiveMapsCpuReadable(supported);
  const structuredWarnings: EngineWarning[] = [];
  const warningOptions = {
    onWarning: (warning: EngineWarning) => structuredWarnings.push(warning),
    warningPhase: 'setScene',
    warningMethod: 'setScene',
  };

  // (1b) fold `mesh-area` emitter radiance back onto its surface material's
  //      emissive — three-bindings strips it for NEE backends, but the fork
  //      integrator lights area sources by HITTING the emissive surface
  //      (surf.emission). Without this the Cornell light renders black.
  // (1c/2/7 CPU side) Skinning pre-pass + merged world-space tri stream +
  //     attribute payload inputs. The geometry-only mutation path reuses this
  //     exact helper so setScene and primitive geometry patches cannot drift.
  const geometry = buildGeometryInputs(supported, warningOptions);
  const skinnedScene = geometry.skinnedScene;
  const merged = geometry.merged;
  assertAttributeLayerBudget(gl, geometry.attrData.layers);
  const transportBounds = computeWebgl2TransportBounds(
    merged,
    supported,
    options,
  );

  // Complete every CPU pack and authored-input validation before the first GL
  // allocation. A malformed material map or HDRI must leave no candidate GPU
  // writes for the caller to clean up.
  // (3) BVH data textures (+ per-tri materialIndex), CPU payload.
  const bvhData = packBvhTextureData(merged);

  // (4a) material-map atlas — gather every readable map texture into a sampler2DArray
  //      and a handle→layer map (null when the scene has no usable textures).
  const maxAtlasLayers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
  const materialAtlases = packMaterialTextureAtlases(merged.materials, {
    warningMethod: warningOptions.warningMethod,
    maxArrayTextureLayers: maxAtlasLayers,
  });
  const atlas = materialAtlases.ldr;
  const hdrAtlas = materialAtlases.hdr;
  const atlasCapacities = materialTextureAtlasLayerCapacities(atlas, hdrAtlas, maxAtlasLayers);
  const materialAtlasLayerCapacity = atlasCapacities.ldr;
  const materialHdrAtlasLayerCapacity = atlasCapacities.hdr;

  // (4b) materials — the merged result already dedups the scene's unique
  //      MaterialSpecs in first-seen order (triMaterialId indexes into it), so it
  //      IS the unique-material list the materials texture packs. The role-aware
  //      atlas maps turn each material's `<map>` ref into the GLSL's layer index
  //      without confusing sRGB color maps and linear data maps sharing a handle.
  const vertexColorMaterialIds = collectVertexColorMaterialIds(skinnedScene, merged);
  const materialLayerMap = {
    ldr: atlas?.layerOfByColorSpace ?? null,
    hdr: hdrAtlas?.layerOfByColorSpace ?? null,
  };
  const materialsData = packMaterialsTexture(merged.materials, materialLayerMap, {
    vertexColorMaterialIds,
    uvLayerByTexCoord: geometry.uvLayout.layerByTexCoord,
    ...warningOptions,
  });

  // (5) lights (6px/light) — driven from the original scene's emitters.
  const lightsData = packLightsTexture(supported.emitters);

  // (5b) B4 — mesh-area triangle lights for NEE, built from explicit mesh-area
  //      emitters plus implicit emissive-material meshes over the merged world-
  //      space geometry. null when the scene has no emissive mesh triangles.
  const meshLightsData = packMeshAreaLights(supported, merged, geometry.uvLayout.mergedByTexCoord);

  // (6) environment importance-sampling (null for non-HDRI scenes).
  const env = buildEquirectInfo(supported.environment, warningOptions);

  // (7) vertex-attribute array (normal / tangent / uv0 / color / uv1), 5 layers.
  // Build a merged uv1 array from the scene primitives using the same vertex-range
  // ordering that mergeWorldSpaceFromCore used. Falls back to uv0 per vertex when a
  // primitive carries no uv1 (see packAttributesArray).
  // D10.7: uses mergeUv1FromCore from @vitrum/shared-bvh, colocated with worldSpaceMerge.ts.
  const attrData = geometry.attrData;

  // CPU preflight is complete. GPU failures below remain recoverable through
  // the BVH uploader's local transaction and this bundle-level owned list.
  const bvh = uploadBvhTextures(gl, bvhData);
  const allocated: WebGLTexture[] = [];
  const own = (texture: WebGLTexture): WebGLTexture => {
    allocated.push(texture);
    return texture;
  };
  try {
    const textures2DArray =
      atlas != null
        ? own(uploadTextureAtlas(gl, atlas, { layerCapacity: materialAtlasLayerCapacity }))
        : null;
    const materialHdrTextures2DArray =
      hdrAtlas != null
        ? own(uploadTextureAtlas(gl, hdrAtlas, { layerCapacity: materialHdrAtlasLayerCapacity }))
        : null;
    const materials = own(
      uploadRgba32f(gl, materialsData.data, materialsData.dim, 'scene materials'),
    );
    const lights = own(uploadRgba32f(gl, lightsData.data, lightsData.dim, 'scene lights'));
    const meshLights =
      meshLightsData.data != null
        ? own(uploadRgba32f(gl, meshLightsData.data, meshLightsData.dim, 'mesh-area lights'))
        : null;
    const envMap = env.map
      ? own(uploadRgba32fRect(gl, env.map.data, env.map.width, env.map.height, 'environment map'))
      : null;
    // The conditional payload also carries the marginal forward CDF in .g, so
    // the production graph owns one distribution texture and stays within the
    // WebGL2-minimum 16 active fragment samplers.
    const envMarginal = null;
    const envConditional = env.conditional
      ? own(
          uploadRgba32fRect(
            gl,
            env.conditional.data,
            env.conditional.width,
            env.conditional.height,
            'environment CDF distribution',
          ),
        )
      : null;
    const attributesArray = own(
      uploadRgba32fArray(gl, attrData.data, attrData.dim, attrData.layers, 'vertex attributes'),
    );

    // (8) assemble the bundle.
    let destroyed = false;
    const textures: UploadedSceneTextures = {
      bvhBounds: bvh.bounds,
      bvhContents: bvh.contents,
      bvhPosition: bvh.position,
      bvhIndex: bvh.index,
      materialIndex: bvh.materialIndex,
      materials,
      attributesArray,
      lights,
      lightCount: lightsData.lightCount,
      meshLights,
      meshLightCount: meshLightsData.triLightCount,
      totalEmissiveArea: meshLightsData.totalEmissiveArea,
      totalEmissivePower: meshLightsData.totalEmissivePower,
      envMap,
      envMarginal,
      envConditional,
      envTotalSum: env.totalSum,
      envWidth: env.map?.width ?? 0,
      envHeight: env.map?.height ?? 0,
      textures2DArray,
      materialAtlasDim: atlas?.dim ?? 0,
      materialAtlasLayerCount: atlas?.layerCount ?? 0,
      materialAtlasLayerCapacity,
      materialHdrTextures2DArray,
      materialHdrAtlasDim: hdrAtlas?.dim ?? 0,
      materialHdrAtlasLayerCount: hdrAtlas?.layerCount ?? 0,
      materialHdrAtlasLayerCapacity,
      materialLayerMap,
      vertexColorMaterialIds,
      uvLayerByTexCoord: geometry.uvLayout.layerByTexCoord,
      attributeLayerCount: attrData.layers,
      triangleCount: merged.triangleCount,
      destroy(): void {
        if (destroyed) return;
        destroyed = true;
        retireIndependently(
          [
            () => bvh.destroy(),
            ...[
              materials,
              attributesArray,
              lights,
              meshLights,
              envMap,
              envMarginal,
              envConditional,
              textures2DArray,
              materialHdrTextures2DArray,
            ]
              .filter((texture): texture is WebGLTexture => texture != null)
              .map((texture) => () => gl.deleteTexture(texture)),
          ],
          'pt-webgl2: one or more scene textures failed to retire',
        );
      },
    };

    return {
      textures,
      merged,
      warnings: [...warnings, ...merged.warnings],
      structuredWarnings,
      transportBounds,
      supported,
    };
  } catch (error) {
    bvh.destroy();
    for (const texture of allocated) gl.deleteTexture(texture);
    throw error;
  }
}

export function expandAnalyticPrimitiveFallbacks(scene: Scene): {
  readonly scene: Scene;
  readonly warnings: string[];
} {
  let changed = false;
  const warnings: string[] = [];
  const primitives = scene.primitives.map((primitive): ScenePrimitive => {
    if (primitive.kind !== 'analytic') return primitive;
    changed = true;
    warnings.push(
      `Scene primitive "${primitive.id}" (analytic ${primitive.shape}) is tessellated to a generated MeshPrimitive ` +
        `fallback for @vitrum/pt-webgl2.`,
    );
    return analyticPrimitiveToMesh(primitive);
  });
  if (!changed) return { scene, warnings };
  return {
    scene: {
      ...scene,
      primitives,
    },
    warnings,
  };
}

type MeshLikePrimitive = Extract<
  ScenePrimitive,
  { positions: Float32Array; tangents?: Float32Array }
>;

const IDENTITY_MAT4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

function isMeshLikePrimitive(p: ScenePrimitive): p is MeshLikePrimitive {
  return p.kind === 'mesh' || p.kind === 'instanced-mesh' || p.kind === 'skinned-mesh';
}

function determinant4(m: ArrayLike<number>): number {
  const n11 = m[0] ?? 0,
    n12 = m[4] ?? 0,
    n13 = m[8] ?? 0,
    n14 = m[12] ?? 0;
  const n21 = m[1] ?? 0,
    n22 = m[5] ?? 0,
    n23 = m[9] ?? 0,
    n24 = m[13] ?? 0;
  const n31 = m[2] ?? 0,
    n32 = m[6] ?? 0,
    n33 = m[10] ?? 0,
    n34 = m[14] ?? 0;
  const n41 = m[3] ?? 0,
    n42 = m[7] ?? 0,
    n43 = m[11] ?? 0,
    n44 = m[15] ?? 0;
  return (
    n41 *
      (+n14 * n23 * n32 -
        n13 * n24 * n32 -
        n14 * n22 * n33 +
        n12 * n24 * n33 +
        n13 * n22 * n34 -
        n12 * n23 * n34) +
    n42 *
      (+n11 * n23 * n34 -
        n11 * n24 * n33 +
        n14 * n21 * n33 -
        n13 * n21 * n34 +
        n13 * n24 * n31 -
        n14 * n23 * n31) +
    n43 *
      (+n11 * n24 * n32 -
        n11 * n22 * n34 -
        n14 * n21 * n32 +
        n12 * n21 * n34 +
        n14 * n22 * n31 -
        n12 * n24 * n31) +
    n44 *
      (-n13 * n22 * n31 -
        n11 * n23 * n32 +
        n11 * n22 * n33 +
        n13 * n21 * n32 -
        n12 * n21 * n33 +
        n12 * n23 * n31)
  );
}

function transformDirection(
  m: ArrayLike<number>,
  x: number,
  y: number,
  z: number,
): [number, number, number] {
  return [
    (m[0] ?? 0) * x + (m[4] ?? 0) * y + (m[8] ?? 0) * z,
    (m[1] ?? 0) * x + (m[5] ?? 0) * y + (m[9] ?? 0) * z,
    (m[2] ?? 0) * x + (m[6] ?? 0) * y + (m[10] ?? 0) * z,
  ];
}

function mergeTangentsFromCore(
  scene: Scene,
  ranges: WorldSpaceMergeResult['meshVertexRanges'],
  totalVertexCount: number,
): Float32Array | undefined {
  const meshLike = scene.primitives.filter(isMeshLikePrimitive);
  if (!meshLike.some((p) => p.tangents != null && p.tangents.length > 0)) return undefined;

  const out = new Float32Array(totalVertexCount * 4);
  const primitiveById = new Map(
    meshLike.map((primitive) => [String(primitive.id), primitive] as const),
  );
  const legacyInstanceCursor = new Map<string, number>();
  for (const range of ranges) {
    const sourceId = String(range.sourcePrimitiveId ?? range.name);
    const prim = primitiveById.get(sourceId);
    if (prim == null) continue;
    const localVertexCount = Math.floor(prim.positions.length / 3);
    const src = prim.tangents;
    if (localVertexCount < 1 || src == null || src.length === 0) continue;

    const legacyInstanceIndex = legacyInstanceCursor.get(sourceId) ?? 0;
    legacyInstanceCursor.set(sourceId, legacyInstanceIndex + 1);
    const sourceInstanceIndex = range.sourceInstanceIndex ?? legacyInstanceIndex;
    const transform =
      prim.kind === 'instanced-mesh'
        ? (prim.instances[sourceInstanceIndex] ?? IDENTITY_MAT4)
        : (prim.transform ?? IDENTITY_MAT4);
    const handednessScale = determinant4(transform) < 0 ? -1 : 1;
    for (let v = 0; v < range.vertexCount; v += 1) {
      const local = Math.min(v, localVertexCount - 1);
      const sx = src[local * 4] ?? 0;
      const sy = src[local * 4 + 1] ?? 0;
      const sz = src[local * 4 + 2] ?? 0;
      const sw = (src[local * 4 + 3] ?? 1) < 0 ? -1 : 1;
      const [tx, ty, tz] = transformDirection(transform, sx, sy, sz);
      const o = (range.vertexStart + v) * 4;
      out[o] = tx;
      out[o + 1] = ty;
      out[o + 2] = tz;
      out[o + 3] = sw * handednessScale;
    }
  }

  return out;
}

function mergeColorsFromCore(
  scene: Scene,
  ranges: WorldSpaceMergeResult['meshVertexRanges'],
  totalVertexCount: number,
): Float32Array | undefined {
  const meshLike = scene.primitives.filter(isMeshLikePrimitive);
  if (
    !meshLike.some((p) => {
      const colors = getPrimitiveActiveColorSet(p);
      return colors != null && colors.length > 0;
    })
  )
    return undefined;

  const out = new Float32Array(totalVertexCount * 4);
  for (let i = 0; i < totalVertexCount; i += 1) {
    const o = i * 4;
    out[o] = 1;
    out[o + 1] = 1;
    out[o + 2] = 1;
    out[o + 3] = 1;
  }

  const primitiveById = new Map(
    meshLike.map((primitive) => [String(primitive.id), primitive] as const),
  );
  for (const range of ranges) {
    const prim = primitiveById.get(String(range.sourcePrimitiveId ?? range.name));
    if (prim == null) continue;
    const localVertexCount = Math.floor(prim.positions.length / 3);
    if (localVertexCount < 1) continue;
    const src = getPrimitiveActiveColorSet(prim);
    const colorStride =
      src == null || src.length === 0
        ? 4
        : Math.max(3, Math.min(4, Math.floor(src.length / Math.max(1, localVertexCount))));

    if (src == null || src.length === 0) continue;

    for (let v = 0; v < range.vertexCount; v += 1) {
      const local = Math.min(v, localVertexCount - 1);
      const so = local * colorStride;
      const o = (range.vertexStart + v) * 4;
      out[o] = src[so] ?? 1;
      out[o + 1] = src[so + 1] ?? 1;
      out[o + 2] = src[so + 2] ?? 1;
      out[o + 3] = colorStride >= 4 ? (src[so + 3] ?? 1) : 1;
    }
  }

  return out;
}

export function collectVertexColorMaterialIds(
  scene: Scene,
  merged: WorldSpaceMergeResult,
): ReadonlySet<number> {
  const ids = new Set<number>();
  const meshLike = scene.primitives.filter(isMeshLikePrimitive);
  const primitiveById = new Map(
    meshLike.map((primitive) => [String(primitive.id), primitive] as const),
  );
  for (const range of merged.meshVertexRanges) {
    const prim = primitiveById.get(String(range.sourcePrimitiveId ?? range.name));
    if (prim == null) continue;
    const colors = getPrimitiveActiveColorSet(prim);
    const hasColors = colors != null && colors.length > 0;
    if (!hasColors) continue;
    for (let t = range.triStart; t < range.triStart + range.triCount; t += 1) {
      const materialId = merged.mergedTriMaterialId[t];
      if (materialId !== undefined) ids.add(materialId);
    }
  }
  return ids;
}

// ──────────────────────────────────────────────────────────────────────────
// GL upload helpers — RGBA32F sampler2D / sampler2DArray, NEAREST + ClampToEdge.
// D10.14: delegate to allocGlTexture (gl/texAlloc.ts) — the shared helper that
// also covers bvhTextureAdapter.ts's `makeTex`. This removes the duplicated
// guard+create+bind+sampling+upload sequence.
// ──────────────────────────────────────────────────────────────────────────

/** Square RGBA32F sampler2D (dim×dim), NEAREST/ClampToEdge. Accepts the
 *  `TexelGrid.data` union; the materials/lights grids are `kind: 'rgba32f'`, so
 *  the float view is the correct upload type. */
export function uploadRgba32f(
  gl: WebGL2RenderingContext,
  data: Float32Array | Uint32Array,
  dim: number,
  resourceName: string,
): WebGLTexture {
  return allocGlTexture(gl, {
    kind: '2d',
    dim,
    internalFormat: gl.RGBA32F,
    format: gl.RGBA,
    type: gl.FLOAT,
    data,
    resourceName,
  });
}

/** Square RGBA32UI sampler2D (dim×dim), NEAREST/ClampToEdge. */
export function uploadRgba32ui(
  gl: WebGL2RenderingContext,
  data: Uint32Array,
  dim: number,
  resourceName: string,
): WebGLTexture {
  return allocGlTexture(gl, {
    kind: '2d',
    dim,
    internalFormat: gl.RGBA32UI,
    format: gl.RGBA_INTEGER,
    type: gl.UNSIGNED_INT,
    data,
    resourceName,
  });
}

/** Non-square RGBA32F sampler2D (width×height) — for the equirect map / CDF slabs. */
export function uploadRgba32fRect(
  gl: WebGL2RenderingContext,
  data: Float32Array,
  width: number,
  height: number,
  resourceName: string,
): WebGLTexture {
  return allocGlTexture(gl, {
    kind: 'rect',
    width,
    height,
    internalFormat: gl.RGBA32F,
    format: gl.RGBA,
    type: gl.FLOAT,
    data,
    resourceName,
  });
}

/** RGBA32F TEXTURE_2D_ARRAY (dim×dim × `layers`), NEAREST/ClampToEdge — the
 *  5-layer vertex-attribute array (normal/tangent/uv0/color/uv1). */
export function uploadRgba32fArray(
  gl: WebGL2RenderingContext,
  data: Float32Array,
  dim: number,
  layers: number,
  resourceName: string,
): WebGLTexture {
  return allocGlTexture(gl, {
    kind: 'array',
    dim,
    layers,
    internalFormat: gl.RGBA32F,
    format: gl.RGBA,
    type: gl.FLOAT,
    data,
    resourceName,
  });
}
