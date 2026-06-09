// THREE `<common>` shim — the small set of GLSL helpers the fork kernels reference
// from THREE's auto-injected `#include <common>` (plan/three-removal/04-glsl-kernels.md §3).
//
// The fork's PhysicalPathTracingMaterial fragment shader does `#include <common>` (line 235),
// which THREE expands into its ShaderChunk/common.glsl.js. The kernels copied verbatim into
// this package therefore assume those symbols exist. We have no THREE preprocessor, so we
// inline ONLY the symbols the kernels actually use. The used set was found by grepping the
// copied shader/** + render/** kernels for `<common>`-provided identifiers that have no local
// definition:
//   - PI         (12 kernel files; no local #define)
//   - EPSILON    (3 kernel files; no local #define)
//   - saturate   (light_sampling, sheen, camera_util; no local def)
//   - pow2       (light_sampling)
//   - pow4       (light_sampling)
//   - luminance  (equirect_sampling)
//
// Definitions transcribed from three's ShaderChunk/common.glsl.js + WebGLProgram.js
// (`getLuminanceCoefficients` resolves to the default linear-sRGB / Rec.709 weights
//  (0.2126, 0.7152, 0.0722) for the engine's working color space). This is the only
// genuinely hand-authored GLSL in the port. Symbols NOT used by the kernels (PI2,
// RECIPROCAL_PI, whiteComplement, pow3, max3, average, the THREE `rand(vec2)`, the
// IncidentLight/ReflectedLight structs, etc.) are intentionally omitted to keep the
// shim minimal and avoid clashing with the kernels' own `rand(...)` macros.

export const THREE_COMMON_SHIM: string = /* glsl */ `
// --- inlined subset of three's <common> ShaderChunk (see threeCommonShim.ts header) ---
#ifndef PI
#define PI 3.141592653589793
#endif
#ifndef EPSILON
#define EPSILON 1e-6
#endif
#ifndef saturate
// <tonemapping_pars_fragment> may have defined saturate() already
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif

float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }

float luminance( const in vec3 rgb ) {

	const vec3 weights = vec3( 0.2126, 0.7152, 0.0722 );
	return dot( weights, rgb );

}
// --- end inlined <common> subset ---
`;
