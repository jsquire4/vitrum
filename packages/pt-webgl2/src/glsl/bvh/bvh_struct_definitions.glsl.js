// Ported verbatim from three-mesh-bvh (gkjohnson/three-mesh-bvh),
// src/webgl/glsl/bvh_struct_definitions.glsl.js. MIT License, (c) Garrett Johnson.
// Copied to drop the three-mesh-bvh runtime dependency for the THREE-free
// @vitrum/pt-webgl2 backend. See CREDITS.md.

// Note that a struct cannot be used for the hit record including faceIndices, faceNormal, barycoord,
// side, and dist because on some mobile GPUS (such as Adreno) numbers are afforded less precision specifically
// when in a struct leading to inaccurate hit results. See KhronosGroup/WebGL#3351 for more details.

/**
 * Set of shader structs and defined constants used for interacting with the packed BVH in a
 * shader. See [src/webgl/glsl/bvh_struct_definitions.glsl.js](https://github.com/gkjohnson/three-mesh-bvh/blob/master/src/webgl/glsl/bvh_struct_definitions.glsl.js)
 * for full implementations and declarations.
 *
 * Accessed as `BVHShaderGLSL.bvh_struct_definitions`.
 *
 * @section Shader and Texture Packing API
 * @type {string}
 */
export const bvh_struct_definitions = /* glsl */`
struct BVH {

	usampler2D index;
	sampler2D position;

	sampler2D bvhBounds;
	usampler2D bvhContents;

};
`;
