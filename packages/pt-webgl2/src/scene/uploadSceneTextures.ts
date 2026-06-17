// uploadSceneTextures.ts — assemble the full `UploadedSceneTextures` bundle a
// pt-webgl2 path-trace program binds (plan/three-removal/03-scene-bvh-packers.md §8).
//
// Pipeline (THREE-free, from a `@vitrum/core` `Scene`):
//   1. partitionSceneBySupport(scene, caps)        → supported subset + warnings
//   2. mergeWorldSpaceFromCore(supported, …)       → merged world-space tri stream + BVH
//   3. packBvhTextureData + uploadBvhTextures       → the 4 BVH data textures + materialIndex
//   4. packMaterialsTexture(merged.materials)       → 85px/material RGBA32F sampler2D
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
import { analyticPrimitiveToMesh, partitionSceneBySupport } from '@vitrum/core';
import { mergeWorldSpaceFromCore, mergeUv1FromCore, type WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import { packBvhTextureData, uploadBvhTextures } from './bvhTextureAdapter.js';
import { allocGlTexture } from '../gl/texAlloc.js';
import { foldMeshAreaEmittersIntoMaterials } from './foldEmissiveEmitters.js';
import { packAttributesArray } from './attributesTextureArray.js';
import { packMaterialsTexture } from './materialsTexture.js';
import { packTextureAtlas, uploadTextureAtlas } from './texturesArray.js';
import { directionalAngularDiameterWarnings, packLightsTexture } from './lightsTexture.js';
import { packMeshAreaLights } from './meshAreaLights.js';
import { buildEquirectInfo } from './equirectHdrInfo.js';
import { solveSkinPrimitives } from './solveSkinPrimitives.js';
import type { UploadedSceneTextures } from './sceneTextures.js';

export interface SceneTexturesBuild {
  readonly textures: UploadedSceneTextures;
  readonly merged: WorldSpaceMergeResult;
  readonly warnings: string[];
  readonly structuredWarnings: readonly EngineWarning[];
  /** The capability-filtered scene (H7: returned so callers reuse this single
   *  partition instead of running `partitionSceneBySupport` a second time). */
  readonly supported: Scene;
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
): SceneTexturesBuild {
  const analyticExpansion = expandAnalyticPrimitiveFallbacks(scene);
  // (1) capability filter
  const { supported, warnings } = partitionSceneBySupport(analyticExpansion.scene, caps);
  warnings.unshift(...analyticExpansion.warnings);
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
  const ptScene = foldMeshAreaEmittersIntoMaterials(supported);

  // (1c) Skinning pre-pass: replace each skinned-mesh's rest-pose
  //      positions/normals with CPU-solved posed geometry so the BVH +
  //      attribute packers see the actual deformed mesh.  Fast-path: if no
  //      skinned-mesh primitives exist, ptScene is returned unchanged.
  //      When a host later calls updatePrimitive(id, { bones: newBones })
  //      the full setScene rebuild re-runs this pass — no separate incremental
  //      path required (pt-webgl2 updatePrimitive always rebuilds wholesale).
  const skinnedScene = solveSkinPrimitives(ptScene);

  // (2) merged world-space tri stream + single-root BVH (stride 4 = the form the
  //     BVH texture adapter and attribute array both index).
  const merged = mergeWorldSpaceFromCore(skinnedScene, {
    positionStride: 4,
    splitMaterialsByCastShadow: true,
  });

  // (3) BVH data textures (+ per-tri materialIndex)
  const bvhData = packBvhTextureData(merged);
  const bvh = uploadBvhTextures(gl, bvhData);

  // (4a) material-map atlas — gather every readable map texture into a sampler2DArray
  //      and a handle→layer map (null when the scene has no usable textures).
  const atlas = packTextureAtlas(merged.materials, warningOptions);
  const textures2DArray = atlas != null ? uploadTextureAtlas(gl, atlas) : null;

  // (4b) materials — the merged result already dedups the scene's unique
  //      MaterialSpecs in first-seen order (triMaterialId indexes into it), so it
  //      IS the unique-material list the materials texture packs. The role-aware
  //      atlas maps turn each material's `<map>` ref into the GLSL's layer index
  //      without confusing sRGB color maps and linear data maps sharing a handle.
  const vertexColorMaterialIds = collectVertexColorMaterialIds(skinnedScene, merged);
  const materialsData = packMaterialsTexture(merged.materials, atlas?.layerOfByColorSpace, { vertexColorMaterialIds });
  const materials = uploadRgba32f(gl, materialsData.data, materialsData.dim, 'scene materials');

  // (5) lights (6px/light) — driven from the original scene's emitters.
  const lightsData = packLightsTexture(supported.emitters);
  structuredWarnings.push(...directionalAngularDiameterWarnings(supported.emitters, {
    phase: 'setScene',
    method: 'setScene',
  }));
  const lights = uploadRgba32f(gl, lightsData.data, lightsData.dim, 'scene lights');

  // (5b) B4 — mesh-area triangle lights for NEE, built from explicit mesh-area
  //      emitters plus implicit emissive-material meshes over the merged world-
  //      space geometry. null when the scene has no emissive mesh triangles.
  const meshLightsData = packMeshAreaLights(supported, merged);
  const meshLights =
    meshLightsData.data != null ? uploadRgba32f(gl, meshLightsData.data, meshLightsData.dim, 'mesh-area lights') : null;

  // (6) environment importance-sampling (null for non-HDRI scenes).
  const env = buildEquirectInfo(supported.environment, warningOptions);
  const envMap = env.map ? uploadRgba32fRect(gl, env.map.data, env.map.width, env.map.height, 'environment map') : null;
  const envMarginal = env.marginal
    ? uploadRgba32fRect(gl, env.marginal.data, env.marginal.width, env.marginal.height, 'environment marginal CDF')
    : null;
  const envConditional = env.conditional
    ? uploadRgba32fRect(gl, env.conditional.data, env.conditional.width, env.conditional.height, 'environment conditional CDF')
    : null;

  // (7) vertex-attribute array (normal / tangent / uv0 / color / uv1), 5 layers.
  // Build a merged uv1 array from the scene primitives using the same vertex-range
  // ordering that mergeWorldSpaceFromCore used. Falls back to uv0 per vertex when a
  // primitive carries no uv1 (see packAttributesArray).
  // D10.7: uses mergeUv1FromCore from @vitrum/shared-bvh, colocated with worldSpaceMerge.ts.
  const mergedUv1 = mergeUv1FromCore(skinnedScene, merged.meshVertexRanges, merged.vertexCount);
  const mergedTangents = mergeTangentsFromCore(skinnedScene, merged.meshVertexRanges, merged.vertexCount);
  const mergedColors = mergeColorsFromCore(skinnedScene, merged.meshVertexRanges, merged.vertexCount);
  const attrData = packAttributesArray(
    {
      ...merged,
      ...(mergedUv1 != null ? { uv1: mergedUv1 } : {}),
      ...(mergedTangents != null ? { tangents: mergedTangents } : {}),
      ...(mergedColors != null ? { colors: mergedColors } : {}),
    },
  );
  const attributesArray = uploadRgba32fArray(gl, attrData.data, attrData.dim, attrData.layers, 'vertex attributes');

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
    envMap,
    envMarginal,
    envConditional,
    envTotalSum: env.totalSum,
    envWidth: env.map?.width ?? 0,
    envHeight: env.map?.height ?? 0,
    textures2DArray,
    materialLayerMap: atlas?.layerOfByColorSpace ?? null,
    vertexColorMaterialIds,
    triangleCount: merged.triangleCount,
    destroy(): void {
      if (destroyed) return;
      destroyed = true;
      bvh.destroy();
      for (const t of [
        materials,
        attributesArray,
        lights,
        meshLights,
        envMap,
        envMarginal,
        envConditional,
        textures2DArray,
      ]) {
        if (t != null) gl.deleteTexture(t);
      }
    },
  };

  return {
    textures,
    merged,
    warnings: meshLightsData.warnings.length > 0 ? [...warnings, ...meshLightsData.warnings] : warnings,
    structuredWarnings,
    supported,
  };
}

function expandAnalyticPrimitiveFallbacks(scene: Scene): { readonly scene: Scene; readonly warnings: string[] } {
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

const IDENTITY_MAT4 = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

function isMeshLikePrimitive(p: ScenePrimitive): p is MeshLikePrimitive {
  return p.kind === 'mesh' || p.kind === 'instanced-mesh' || p.kind === 'skinned-mesh';
}

function determinant4(m: ArrayLike<number>): number {
  const n11 = m[0] ?? 0, n12 = m[4] ?? 0, n13 = m[8] ?? 0, n14 = m[12] ?? 0;
  const n21 = m[1] ?? 0, n22 = m[5] ?? 0, n23 = m[9] ?? 0, n24 = m[13] ?? 0;
  const n31 = m[2] ?? 0, n32 = m[6] ?? 0, n33 = m[10] ?? 0, n34 = m[14] ?? 0;
  const n41 = m[3] ?? 0, n42 = m[7] ?? 0, n43 = m[11] ?? 0, n44 = m[15] ?? 0;
  return (
    n41 * (
      +n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 +
      n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34
    ) +
    n42 * (
      +n11 * n23 * n34 - n11 * n24 * n33 + n14 * n21 * n33 -
      n13 * n21 * n34 + n13 * n24 * n31 - n14 * n23 * n31
    ) +
    n43 * (
      +n11 * n24 * n32 - n11 * n22 * n34 - n14 * n21 * n32 +
      n12 * n21 * n34 + n14 * n22 * n31 - n12 * n24 * n31
    ) +
    n44 * (
      -n13 * n22 * n31 - n11 * n23 * n32 + n11 * n22 * n33 +
      n13 * n21 * n32 - n12 * n21 * n33 + n12 * n23 * n31
    )
  );
}

function transformDirection(m: ArrayLike<number>, x: number, y: number, z: number): [number, number, number] {
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
  let rangeIdx = 0;
  for (const prim of meshLike) {
    const localVertexCount = Math.floor(prim.positions.length / 3);
    if (localVertexCount < 3) continue;
    const localIndexCount = prim.indices?.length ?? localVertexCount;
    if (Math.floor(localIndexCount / 3) === 0) continue;

    const instanceCount = prim.kind === 'instanced-mesh' ? prim.instances.length : 1;
    for (let inst = 0; inst < instanceCount; inst += 1) {
      const range = ranges[rangeIdx];
      if (range == null) break;
      rangeIdx += 1;

      const src = prim.tangents;
      if (src == null || src.length === 0) continue;

      const transform = prim.kind === 'instanced-mesh'
        ? (prim.instances[inst] ?? IDENTITY_MAT4)
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
  }

  return out;
}

function mergeColorsFromCore(
  scene: Scene,
  ranges: WorldSpaceMergeResult['meshVertexRanges'],
  totalVertexCount: number,
): Float32Array | undefined {
  const meshLike = scene.primitives.filter(isMeshLikePrimitive);
  if (!meshLike.some((p) => p.colors != null && p.colors.length > 0)) return undefined;

  const out = new Float32Array(totalVertexCount * 4);
  for (let i = 0; i < totalVertexCount; i += 1) {
    const o = i * 4;
    out[o] = 1;
    out[o + 1] = 1;
    out[o + 2] = 1;
    out[o + 3] = 1;
  }

  let rangeIdx = 0;
  for (const prim of meshLike) {
    const localVertexCount = Math.floor(prim.positions.length / 3);
    if (localVertexCount < 1) continue;
    const src = prim.colors;
    const colorStride = src == null || src.length === 0
      ? 4
      : Math.max(3, Math.min(4, Math.floor(src.length / Math.max(1, localVertexCount))));

    const instanceCount = prim.kind === 'instanced-mesh' ? prim.instances.length : 1;
    for (let inst = 0; inst < instanceCount; inst += 1) {
      const range = ranges[rangeIdx];
      if (range == null) break;
      rangeIdx += 1;
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
  }

  return out;
}

function collectVertexColorMaterialIds(
  scene: Scene,
  merged: WorldSpaceMergeResult,
): ReadonlySet<number> {
  const ids = new Set<number>();
  const meshLike = scene.primitives.filter(isMeshLikePrimitive);
  let rangeIdx = 0;
  for (const prim of meshLike) {
    const hasColors = prim.colors != null && prim.colors.length > 0;
    const instanceCount = prim.kind === 'instanced-mesh' ? prim.instances.length : 1;
    for (let inst = 0; inst < instanceCount; inst += 1) {
      const range = merged.meshVertexRanges[rangeIdx];
      if (range == null) break;
      rangeIdx += 1;
      if (!hasColors) continue;
      for (let t = range.triStart; t < range.triStart + range.triCount; t += 1) {
        const materialId = merged.mergedTriMaterialId[t];
        if (materialId !== undefined) ids.add(materialId);
      }
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
    kind: '2d', dim,
    internalFormat: gl.RGBA32F, format: gl.RGBA, type: gl.FLOAT,
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
    kind: 'rect', width, height,
    internalFormat: gl.RGBA32F, format: gl.RGBA, type: gl.FLOAT,
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
    kind: 'array', dim, layers,
    internalFormat: gl.RGBA32F, format: gl.RGBA, type: gl.FLOAT,
    data,
    resourceName,
  });
}
