// Fullscreen pass-through vertex shader — the GLSL-layer mirror of the fork's
// PhysicalPathTracingMaterial vertex shader (PhysicalPathTracingMaterial.js:212-225),
// WITHOUT the `#version 300 es`/precision preamble (the GlProgram preamble prepends those).
//
// This is the faithful byte-for-byte mirror of the fork VS, which renders a fullscreen
// quad geometry: it reads the per-vertex `position`/`uv` attributes and the
// `modelViewMatrix`/`projectionMatrix` uniforms that THREE auto-injected, and writes the
// `vUv` the fragment kernels read. The host (GlProgram) must declare those attributes
// (`in vec3 position; in vec2 uv;`) and uniforms (`uniform mat4 modelViewMatrix,
// projectionMatrix;`) in the preamble — they were implicit in THREE's ShaderMaterial.
//
// (An attribute-less "big triangle" variant that synthesises positions from gl_VertexID
//  lives in `src/gl/fullscreenQuad.ts`; this module is the literal fork transcription.)

/**
 * The pass-through vertex shader BODY (no `#version`/precision preamble — GlProgram
 * prepends those). Transcribes PhysicalPathTracingMaterial.js:212-225 verbatim.
 */
export const FULLSCREEN_VERT_BODY: string = /* glsl */ `
varying vec2 vUv;
void main() {

	vec4 mvPosition = vec4( position, 1.0 );
	mvPosition = modelViewMatrix * mvPosition;
	gl_Position = projectionMatrix * mvPosition;

	vUv = uv;

}
`;
