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
//   7. packAttributesArray(merged)                  → 4-layer RGBA32F sampler2DArray
//   8. assemble the bundle with a destroy() that deletes every owned texture.
//
// NOTE: `packMaterialsTexture` / `packLightsTexture` / `buildEquirectInfo` are
// authored by parallel agents (materialsTexture.ts / lightsTexture.ts /
// equirectHdrInfo.ts); they are imported here by the names the plan fixes and
// will exist at integration. Their return shapes are the `sceneTextures.ts`
// contracts (`MaterialsTextureData` / `LightsTextureData` / `EnvTextureData`).

import type { EngineCapabilities, Scene } from '@vitrum/core';
import { partitionSceneBySupport } from '@vitrum/core';
import { mergeWorldSpaceFromCore, type WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import { packBvhTextureData, uploadBvhTextures } from './bvhTextureAdapter.js';
import { foldMeshAreaEmittersIntoMaterials } from './foldEmissiveEmitters.js';
import { packAttributesArray } from './attributesTextureArray.js';
import { packMaterialsTexture } from './materialsTexture.js';
import { packTextureAtlas, uploadTextureAtlas } from './texturesArray.js';
import { packLightsTexture } from './lightsTexture.js';
import { packMeshAreaLights } from './meshAreaLights.js';
import { buildEquirectInfo } from './equirectHdrInfo.js';
import { solveSkinPrimitives } from './solveSkinPrimitives.js';
import type { UploadedSceneTextures } from './sceneTextures.js';

export interface SceneTexturesBuild {
  readonly textures: UploadedSceneTextures;
  readonly merged: WorldSpaceMergeResult;
  readonly warnings: string[];
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
  // (1) capability filter
  const { supported, warnings } = partitionSceneBySupport(scene, caps);

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
  const merged = mergeWorldSpaceFromCore(skinnedScene, { positionStride: 4 });

  // (3) BVH data textures (+ per-tri materialIndex)
  const bvhData = packBvhTextureData(merged);
  const bvh = uploadBvhTextures(gl, bvhData);

  // (4a) material-map atlas — gather every readable map texture into a sampler2DArray
  //      and a handle→layer map (null when the scene has no usable textures).
  const atlas = packTextureAtlas(merged.materials);
  const textures2DArray = atlas != null ? uploadTextureAtlas(gl, atlas) : null;

  // (4b) materials — the merged result already dedups the scene's unique
  //      MaterialSpecs in first-seen order (triMaterialId indexes into it), so it
  //      IS the unique-material list the materials texture packs. The atlas layer
  //      map turns each material's `<map>` ref into the GLSL's layer index.
  const materialsData = packMaterialsTexture(merged.materials, atlas?.layerOf);
  const materials = uploadRgba32f(gl, materialsData.data, materialsData.dim);

  // (5) lights (6px/light) — driven from the original scene's emitters.
  const lightsData = packLightsTexture(supported.emitters);
  const lights = uploadRgba32f(gl, lightsData.data, lightsData.dim);

  // (5b) B4 — mesh-area triangle lights for NEE, built from the emissive mesh-area
  //      emitters + the merged world-space geometry. null when the scene has none.
  const meshLightsData = packMeshAreaLights(supported, merged);
  const meshLights =
    meshLightsData.data != null ? uploadRgba32f(gl, meshLightsData.data, meshLightsData.dim) : null;

  // (6) environment importance-sampling (null for non-HDRI scenes).
  const env = buildEquirectInfo(supported.environment);
  const envMap = env.map ? uploadRgba32fRect(gl, env.map.data, env.map.width, env.map.height) : null;
  const envMarginal = env.marginal
    ? uploadRgba32fRect(gl, env.marginal.data, env.marginal.width, env.marginal.height)
    : null;
  const envConditional = env.conditional
    ? uploadRgba32fRect(gl, env.conditional.data, env.conditional.width, env.conditional.height)
    : null;

  // (7) vertex-attribute array (normal / tangent / uv / color), 4 layers.
  const attrData = packAttributesArray(merged);
  const attributesArray = uploadRgba32fArray(gl, attrData.data, attrData.dim, attrData.layers);

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
    iesProfiles: null,
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

  return { textures, merged, warnings, supported };
}

// ──────────────────────────────────────────────────────────────────────────
// GL upload helpers — RGBA32F sampler2D / sampler2DArray, NEAREST + ClampToEdge.
// ──────────────────────────────────────────────────────────────────────────

function setSampling2D(gl: WebGL2RenderingContext, target: number): void {
  gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

/** Square RGBA32F sampler2D (dim×dim), NEAREST/ClampToEdge. Accepts the
 *  `TexelGrid.data` union; the materials/lights grids are `kind: 'rgba32f'`, so
 *  the float view is the correct upload type. */
function uploadRgba32f(
  gl: WebGL2RenderingContext,
  data: Float32Array | Uint32Array,
  dim: number,
): WebGLTexture {
  const tex = gl.createTexture();
  if (tex == null) throw new Error('pt-webgl2: failed to create RGBA32F texture');
  gl.bindTexture(gl.TEXTURE_2D, tex);
  setSampling2D(gl, gl.TEXTURE_2D);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, dim, dim, 0, gl.RGBA, gl.FLOAT, data);
  return tex;
}

/** Non-square RGBA32F sampler2D (width×height) — for the equirect map / CDF slabs. */
function uploadRgba32fRect(
  gl: WebGL2RenderingContext,
  data: Float32Array,
  width: number,
  height: number,
): WebGLTexture {
  const tex = gl.createTexture();
  if (tex == null) throw new Error('pt-webgl2: failed to create RGBA32F rect texture');
  gl.bindTexture(gl.TEXTURE_2D, tex);
  setSampling2D(gl, gl.TEXTURE_2D);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, data);
  return tex;
}

/** RGBA32F TEXTURE_2D_ARRAY (dim×dim × `layers`), NEAREST/ClampToEdge — the
 *  4-layer vertex-attribute array (normal/tangent/uv/color). */
function uploadRgba32fArray(
  gl: WebGL2RenderingContext,
  data: Float32Array,
  dim: number,
  layers: number,
): WebGLTexture {
  const tex = gl.createTexture();
  if (tex == null) throw new Error('pt-webgl2: failed to create RGBA32F array texture');
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
  setSampling2D(gl, gl.TEXTURE_2D_ARRAY);
  gl.texImage3D(
    gl.TEXTURE_2D_ARRAY,
    0,
    gl.RGBA32F,
    dim,
    dim,
    layers,
    0,
    gl.RGBA,
    gl.FLOAT,
    data,
  );
  return tex;
}
