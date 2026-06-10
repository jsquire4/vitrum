// Shared scene-texture contract — the seam between the CPU packers (materialsTexture,
// lightsTexture, equirectHdrInfo, attributesTextureArray) and the GL consumer
// (uploadSceneTextures builds the bundle; GlResources binds it to the program).

/** A square CPU texel grid ready for `gl.texImage2D` (dim×dim, RGBA-strided). */
export interface TexelGrid {
  readonly data: Float32Array | Uint32Array;
  readonly dim: number;
  readonly kind: 'rgba32f' | 'rgba32ui';
}

/** A layered (sampler2DArray) CPU payload: `layers` RGBA32F grids stacked. */
export interface LayeredTexelGrid {
  readonly data: Float32Array;
  readonly dim: number;
  readonly layers: number;
}

/** Output of the materials packer (85px/material RGBA32F square; plan 03 §4). */
export interface MaterialsTextureData extends TexelGrid {
  readonly kind: 'rgba32f';
  readonly materialCount: number;
}

/** Output of the lights packer (6px/light RGBA32F square; plan 03 §5). */
export interface LightsTextureData extends TexelGrid {
  readonly kind: 'rgba32f';
  readonly lightCount: number;
}

/** Output of the equirect env-map importance-sampling build (plan 03 §6). */
export interface EnvTextureData {
  readonly map: { data: Float32Array; width: number; height: number } | null;
  readonly marginal: { data: Float32Array; width: number; height: number } | null;
  readonly conditional: { data: Float32Array; width: number; height: number } | null;
  readonly totalSum: number;
}

/**
 * The uploaded GL texture bundle the path-trace program binds. Built by
 * `uploadSceneTextures(gl, scene, caps)` (plan 03 §8). All `WebGLTexture`s are owned
 * by this bundle; `destroy()` deletes them.
 */
export interface UploadedSceneTextures {
  // BVH (from bvhTextureAdapter.uploadBvhTextures)
  readonly bvhBounds: WebGLTexture;
  readonly bvhContents: WebGLTexture;
  readonly bvhPosition: WebGLTexture;
  readonly bvhIndex: WebGLTexture;
  readonly materialIndex: WebGLTexture;
  // materials (sampler2D, 85px/material)
  readonly materials: WebGLTexture;
  // vertex attributes (sampler2DArray: layer 0=normal, 1=tangent, 2=uv, 3=color)
  readonly attributesArray: WebGLTexture;
  // lights (sampler2D, 6px/light)
  readonly lights: WebGLTexture;
  readonly lightCount: number;
  // B4 — mesh-area triangle lights for NEE (sampler2D, 6px/triangle). null when the
  // scene has no emissive mesh-area triangles; meshLightCount/totalEmissiveArea drive
  // the GLSL mesh-NEE branch + forward-hit MIS weight.
  readonly meshLights: WebGLTexture | null;
  readonly meshLightCount: number;
  readonly totalEmissiveArea: number;
  // environment importance-sampling (optional — null for non-HDRI scenes)
  readonly envMap: WebGLTexture | null;
  readonly envMarginal: WebGLTexture | null;
  readonly envConditional: WebGLTexture | null;
  readonly envTotalSum: number;
  readonly envWidth: number;
  readonly envHeight: number;
  // material texture atlas + IES profiles (optional)
  readonly textures2DArray: WebGLTexture | null;
  readonly iesProfiles: WebGLTexture | null;
  readonly triangleCount: number;
  destroy(): void;
}
