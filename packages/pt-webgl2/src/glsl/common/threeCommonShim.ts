// THREE `<common>` shim — the small set of GLSL helpers the fork kernels reference
// from THREE's auto-injected `#include <common>` (plan/three-removal/04-glsl-kernels.md §3).
//
// The fork's PhysicalPathTracingMaterial fragment shader does `#include <common>` (line 235),
// which THREE expands into its ShaderChunk/common.glsl.js. The kernels copied verbatim into
// this package therefore assume those symbols exist. We have no THREE preprocessor, so we
// inline those symbols plus backend-wide finite-RNG representation helpers. The
// THREE-provided set was found by grepping the
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
//  (0.2126, 0.7152, 0.0722) for the engine's working color space). Symbols NOT
// used by the kernels (PI2,
// RECIPROCAL_PI, whiteComplement, pow3, max3, average, the THREE `rand(vec2)`, the
// IncidentLight/ReflectedLight structs, etc.) are intentionally omitted to keep the
// shim small and avoid clashing with the kernels' own `rand(...)` macros.

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

// PCG and Owen-scrambled Sobol both expose exactly 2^24 uniformly spaced
// values in [0, 1). A threshold that is not on that lattice has a different
// realised branch probability from the float used by the estimator. Snap all
// dynamic Bernoulli probabilities to a reachable threshold, retaining both
// branches for every finite interior probability.
float representedBernoulliProbabilityF32( float probability ) {

	if ( ! ( probability > 0.0 ) ) return 0.0;
	if ( probability >= 1.0 ) return 1.0;
	float bucket = clamp(
		floor( probability * 16777216.0 + 0.5 ),
		1.0,
		16777215.0
	);
	return bucket / 16777216.0;

}

// Turn a four-way categorical distribution into exact integer widths on the
// same 2^24-point RNG lattice. Every positive input retains at least one
// outcome; zero inputs retain none; and the last positive category consumes
// the exact remainder so the represented probabilities sum to one.
void representedCategoricalProbabilities4(
	inout float p0,
	inout float p1,
	inout float p2,
	inout float p3
) {

	p0 = p0 > 0.0 && ! isnan( p0 ) && ! isinf( p0 ) ? p0 : 0.0;
	p1 = p1 > 0.0 && ! isnan( p1 ) && ! isinf( p1 ) ? p1 : 0.0;
	p2 = p2 > 0.0 && ! isnan( p2 ) && ! isinf( p2 ) ? p2 : 0.0;
	p3 = p3 > 0.0 && ! isnan( p3 ) && ! isinf( p3 ) ? p3 : 0.0;
	float total = p0 + p1 + p2 + p3;
	if ( ! ( total > 0.0 ) || isnan( total ) || isinf( total ) ) {

		p0 = 1.0;
		p1 = 0.0;
		p2 = 0.0;
		p3 = 0.0;
		return;

	}

	p0 /= total;
	p1 /= total;
	p2 /= total;
	p3 /= total;
	float raw0 = p0;
	float raw1 = p1;
	float raw2 = p2;
	float raw3 = p3;
	float previousCutoff = 0.0;
	float cutoff0 = 0.0;
	float cutoff1 = 0.0;
	float cutoff2 = 0.0;
	float cutoff3 = 0.0;

	if ( raw0 > 0.0 ) {

		float positiveLater =
			( raw1 > 0.0 ? 1.0 : 0.0 ) +
			( raw2 > 0.0 ? 1.0 : 0.0 ) +
			( raw3 > 0.0 ? 1.0 : 0.0 );
		cutoff0 = positiveLater == 0.0
			? 16777216.0
			: clamp(
				floor( raw0 * 16777216.0 + 0.5 ),
				previousCutoff + 1.0,
				16777216.0 - positiveLater
			);
		previousCutoff = cutoff0;

	}

	cutoff1 = previousCutoff;
	if ( raw1 > 0.0 ) {

		float positiveLater =
			( raw2 > 0.0 ? 1.0 : 0.0 ) +
			( raw3 > 0.0 ? 1.0 : 0.0 );
		cutoff1 = positiveLater == 0.0
			? 16777216.0
			: clamp(
				floor( ( raw0 + raw1 ) * 16777216.0 + 0.5 ),
				previousCutoff + 1.0,
				16777216.0 - positiveLater
			);
		previousCutoff = cutoff1;

	}

	cutoff2 = previousCutoff;
	if ( raw2 > 0.0 ) {

		float positiveLater = raw3 > 0.0 ? 1.0 : 0.0;
		cutoff2 = positiveLater == 0.0
			? 16777216.0
			: clamp(
				floor( ( raw0 + raw1 + raw2 ) * 16777216.0 + 0.5 ),
				previousCutoff + 1.0,
				16777216.0 - positiveLater
			);
		previousCutoff = cutoff2;

	}

	cutoff3 = raw3 > 0.0 ? 16777216.0 : previousCutoff;
	p0 = cutoff0 / 16777216.0;
	p1 = ( cutoff1 - cutoff0 ) / 16777216.0;
	p2 = ( cutoff2 - cutoff1 ) / 16777216.0;
	p3 = ( cutoff3 - cutoff2 ) / 16777216.0;

}

vec3 representedEqualThreeWayProbabilities() {

	float p0 = 1.0;
	float p1 = 1.0;
	float p2 = 1.0;
	float unused = 0.0;
	representedCategoricalProbabilities4( p0, p1, p2, unused );
	return vec3( p0, p1, p2 );

}

void representedThreeSlotStrategyProbabilitiesF32(
	float denominator,
	float slots0,
	float slots1,
	float slots2,
	out float p0,
	out float p1,
	out float p2
) {

	if ( ! ( denominator > 0.0 ) ) {

		p0 = 0.0;
		p1 = 0.0;
		p2 = 0.0;
		return;

	}
	p0 = max( slots0, 0.0 ) / denominator;
	p1 = max( slots1, 0.0 ) / denominator;
	p2 = max( slots2, 0.0 ) / denominator;
	float unused = 0.0;
	representedCategoricalProbabilities4( p0, p1, p2, unused );

}
// --- end inlined <common> subset ---
`;
