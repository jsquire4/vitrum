// volume_march.glsl.js — homogeneous medium march for Sprint 7.
//
// Scope: uniform density, single scatter, equi-angular NEE.
// Per-region density volumes are out of scope (Decision 7).
//
// GLSL mirror of @vitrum/shared-samplers/src/hgPhase.ts (evaluateHG, sampleHG)
// and @vitrum/shared-samplers/src/equiAngular.ts (sampleEquiAngular).
//
// References:
//   Henyey & Greenstein 1941, "Diffuse radiation in the galaxy"
//   Pharr, Jakob, Humphreys "Physically Based Rendering" 4th ed., §11.4
//   Sprint 7 fork patch spec: plan/sprint-7-pt-fork-patch.md
//
// Uniforms provided by PhysicalPathTracingMaterial.js (Sprint 7 additions):
//   u_volumeDensity  : float  — extinction coefficient σ_t (0 = no medium)
//   u_scatterAlbedo  : vec3   — σ_s / σ_t per channel (0 = absorb only, 1 = scatter only)
//   u_anisotropyG    : float  — HG anisotropy g ∈ (-1, 1)
//   u_sssSigmaT      : float  — per-material scatter distance reciprocal
//   u_sssAlbedo      : vec3   — single-scatter albedo per channel
//   u_sssAnisotropyG : float  — HG anisotropy g for SSS

export const volume_march = /* glsl */`

	// Sample scatter distance using exponential distribution.
	// Returns sampled t in [0, maxT]; pdf = density * exp(-density * t).
	// If density <= 0, returns maxT + 1.0 (no medium).
	float sampleExponential( float u, float density, float maxT ) {
		if ( density <= 0.0 ) return maxT + 1.0;
		float t = - log( max( 1e-10, 1.0 - u ) ) / density;
		return min( t, maxT );
	}

	// Equi-angular PDF for a scatter at distance t along the ray,
	// given light at perpendicular distance D and closest-approach t_c.
	// GLSL mirror of @vitrum/shared-samplers/src/equiAngular.ts.
	float equiAngularPdf( float t, float tC, float D, float thetaRange ) {
		if ( D < 1e-6 || thetaRange < 1e-6 ) return 1.0 / max( 1e-6, t );
		float ratio = ( t - tC ) / D;
		return 1.0 / ( D * thetaRange * ( 1.0 + ratio * ratio ) );
	}

	// Henyey-Greenstein phase function — GLSL mirror of hgPhase.ts::evaluateHG.
	// p(cosTheta, g) = (1 - g²) / (4π (1 + g² - 2g·cosTheta)^(3/2))
	// Normalized to integrate to 1 over the sphere.
	float hg_phase( float cosTheta, float g ) {
		float g2 = g * g;
		float denom = 1.0 + g2 - 2.0 * g * cosTheta;
		return ( 1.0 - g2 ) / ( 4.0 * PI * denom * sqrt( denom ) );
	}

	// Sample a direction from the Henyey-Greenstein distribution.
	// GLSL mirror of hgPhase.ts::sampleHG.
	// Returns a direction in world space given the incident direction wi.
	// u1, u2: independent uniform random variates in [0, 1).
	vec3 sampleHG_glsl( float u1, float u2, float g, vec3 wi ) {
		float cosTheta;
		if ( abs( g ) < 1e-4 ) {
			// Isotropic: uniform sphere
			cosTheta = 1.0 - 2.0 * u2;
		} else {
			// HG inversion (Pharr et al. §11.4)
			float sqrtTerm = ( 1.0 - g * g ) / ( 1.0 - g + 2.0 * g * u2 );
			cosTheta = ( 1.0 + g * g - sqrtTerm * sqrtTerm ) / ( 2.0 * g );
		}
		cosTheta = clamp( cosTheta, -1.0, 1.0 );
		float sinTheta = sqrt( max( 0.0, 1.0 - cosTheta * cosTheta ) );
		float phi = 2.0 * PI * u1;

		// Local frame: build orthonormal basis around wi
		vec3 up = abs( wi.y ) < 0.999 ? vec3( 0.0, 1.0, 0.0 ) : vec3( 1.0, 0.0, 0.0 );
		vec3 tangent = normalize( cross( up, wi ) );
		vec3 bitangent = cross( wi, tangent );

		// Direction in local HG frame (z = wi)
		return sinTheta * ( cos( phi ) * tangent + sin( phi ) * bitangent ) + cosTheta * wi;
	}

	// Main volume march: given ray origin/direction and surface hit at tSurface,
	// return the scatter distance tScatter.
	// If tScatter >= tSurface, no scatter occurred — proceed to surface shading.
	float volumeMarch( vec3 ro, vec3 rd, float tSurface, float u ) {
		return sampleExponential( u, u_volumeDensity, tSurface );
	}

`;
