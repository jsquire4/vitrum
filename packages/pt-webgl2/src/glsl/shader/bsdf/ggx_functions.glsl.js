/** @public — dynamic-access test-load-bearing; accessed via namespace import in b9Multiscatter.test.ts */
export const ggx_functions = /* glsl */`

	// The GGX functions provide sampling and distribution information for normals as output so
	// in order to get probability of scatter direction the half vector must be computed and provided.
	// [0] https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
	// [1] https://hal.archives-ouvertes.fr/hal-01509746/document
	// [2] http://jcgt.org/published/0007/04/01/
	// [4] http://jcgt.org/published/0003/02/03/

	// trowbridge-reitz === GGX === GTR

	vec3 ggxDirection( vec3 incidentDir, vec2 roughness, vec2 uv ) {

		// Fork-compatible implementation from reference [1]. The WebGPU backend
		// has a VNDF-specific path; this WebGL2 port keeps the paired sampler/PDF
		// convention used by the original fork for numeric A/B stability.
		// stretch view
		vec3 V = normalize( vec3( roughness * incidentDir.xy, incidentDir.z ) );

		// orthonormal basis
		vec3 T1 = ( V.z < 0.9999 ) ? normalize( cross( V, vec3( 0.0, 0.0, 1.0 ) ) ) : vec3( 1.0, 0.0, 0.0 );
		vec3 T2 = cross( T1, V );

		// sample point with polar coordinates (r, phi)
		float a = 1.0 / ( 1.0 + V.z );
		float r = sqrt( uv.x );
		float phi = ( uv.y < a ) ? uv.y / a * PI : PI + ( uv.y - a ) / ( 1.0 - a ) * PI;
		float P1 = r * cos( phi );
		float P2 = r * sin( phi ) * ( ( uv.y < a ) ? 1.0 : V.z );

		// compute normal
		vec3 N = P1 * T1 + P2 * T2 + V * sqrt( max( 0.0, 1.0 - P1 * P1 - P2 * P2 ) );

		// unstretch
		N = normalize( vec3( roughness * N.xy, max( 0.0, N.z ) ) );

		return N;

	}

	// Below are PDF and related functions for use in a Monte Carlo path tracer
	// as specified in Appendix B of the following paper
	// See equation (34) from reference [0]
	float ggxLamda( float theta, float roughness ) {

		float tanTheta = tan( theta );
		float tanTheta2 = tanTheta * tanTheta;
		float alpha2 = roughness * roughness;

		float numerator = - 1.0 + sqrt( 1.0 + alpha2 * tanTheta2 );
		return numerator / 2.0;

	}

	// See equation (34) from reference [0]
	float ggxShadowMaskG1( float theta, float roughness ) {

		return 1.0 / ( 1.0 + ggxLamda( theta, roughness ) );

	}

	// See equation (125) from reference [4]
	float ggxShadowMaskG2( vec3 wi, vec3 wo, float roughness ) {

		float incidentTheta = acos( wi.z );
		float scatterTheta = acos( wo.z );
		return 1.0 / ( 1.0 + ggxLamda( incidentTheta, roughness ) + ggxLamda( scatterTheta, roughness ) );

	}

	// See equation (33) from reference [0]
	float ggxDistribution( vec3 halfVector, float roughness ) {

		float a2 = roughness * roughness;
		a2 = max( EPSILON, a2 );
		float cosTheta = halfVector.z;
		float cosTheta4 = pow( cosTheta, 4.0 );

		if ( cosTheta == 0.0 ) return 0.0;

		float theta = acosSafe( halfVector.z );
		float tanTheta = tan( theta );
		float tanTheta2 = pow( tanTheta, 2.0 );

		float denom = PI * cosTheta4 * pow( a2 + tanTheta2, 2.0 );
		return ( a2 / denom );

	}

	// See equation (3) from reference [2]
	float ggxPDF( vec3 wi, vec3 halfVector, float roughness ) {

		float incidentTheta = acos( wi.z );
		float D = ggxDistribution( halfVector, roughness );
		float G1 = ggxShadowMaskG1( incidentTheta, roughness );

		return D * G1 * max( 0.0, dot( wi, halfVector ) ) / wi.z;

	}

	// ── B9 — Kulla-Conty multiscatter energy compensation ────────────────────
	// A single-scatter GGX lobe loses energy as roughness rises (multi-bounce
	// microfacet inter-reflections are dropped), so rough metals/speculars read
	// dark. Kulla & Conty 2017 ("Revisiting Physically Based Shading at Imageworks")
	// add a multiscatter lobe that recovers exactly the missing energy. The fit
	// below is shared CONVENTION-coordinate with the walkaround-hybrid B9 fit
	// (ggxBrdf.wgsl.ts) — same ggxDirectionalAlbedo / ggxAverageAlbedo analytic
	// approximations and the same Fms reciprocity, so all backends compensate
	// identically. mu is |cosθ| of the view (or light) direction; rough is the
	// linear (perceptual) roughness.

	// Directional albedo E(μ) of the single-scatter GGX lobe (analytic fit).
	float ggxDirectionalAlbedo( float mu, float rough ) {

		float a = clamp( rough, 0.0, 1.0 );
		float c = clamp( mu, 0.0, 1.0 );
		float a2 = a * a;
		return clamp( 1.0 - a2 * ( 1.0 - c ) * ( 0.75 + 0.25 * c ), 0.0, 1.0 );

	}

	// Cosine-weighted hemispherical average albedo E_avg of the GGX lobe.
	float ggxAverageAlbedo( float rough ) {

		float a = clamp( rough, 0.0, 1.0 );
		float a2 = a * a;
		return clamp( 1.0 - a2 * ( 7.0 / 24.0 ), 0.0, 1.0 );

	}

	// Multiscatter BRDF lobe value (Kulla-Conty). Adds the energy the single-
	// scatter lobe drops. Favg is the cosine-weighted average Fresnel of the
	// specular f0 (per Fdez-Agüera 2019: Favg ≈ f0 + (1 − f0)/21). Returns the
	// per-component lobe value to ADD to the single-scatter color (already carries
	// the wi.z cosine, matching specularEval's color convention).
	vec3 ggxMultiscatter( float rough, float NdotV, float NdotL, vec3 Favg ) {

		float Eo = ggxDirectionalAlbedo( NdotV, rough );
		float Ei = ggxDirectionalAlbedo( NdotL, rough );
		float Eavg = ggxAverageAlbedo( rough );
		float oneMinusEavg = 1.0 - Eavg;
		if ( oneMinusEavg < 1e-4 ) return vec3( 0.0 );
		float fms = ( 1.0 - Eo ) * ( 1.0 - Ei ) / ( PI * oneMinusEavg );
		// Fms = Favg² · Eavg / (1 − Favg·(1−Eavg)) — the geometric series of
		// repeated Fresnel-weighted bounces (Kulla-Conty eq. 9 / Fdez-Agüera).
		vec3 Fms = ( Favg * Favg * Eavg ) / max( vec3( 1.0 ) - Favg * oneMinusEavg, vec3( 1e-4 ) );
		// Carry the wi.z (NdotL) cosine to match specularEval's color term.
		return fms * Fms * NdotL;

	}

`;
