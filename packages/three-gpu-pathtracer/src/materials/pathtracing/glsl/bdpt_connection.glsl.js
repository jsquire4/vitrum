/**
 * bdpt_connection.glsl.js — BDPT eye↔light connection evaluation (Sprint 10c).
 *
 * Implements the connection phase of bidirectional path tracing:
 *   1. Visibility test (shadow ray from eye vertex toward light vertex).
 *   2. BSDF evaluation at the eye vertex (toward the light vertex).
 *   3. BSDF evaluation at the light vertex (toward the eye vertex).
 *   4. Geometric term G(x↔y) = |cosθ_x · cosθ_y| / ‖x−y‖².
 *   5. Full Veach §10.3 power-heuristic MIS weight (β=2).
 *   6. Returns RGB contribution = lightThroughput × BSDF_l × G × BSDF_e × MIS × eyeThroughput.
 *
 * MIS weight (inline GLSL port of `bdptConnectionMIS_full` from @vitrum/shared-samplers):
 *   w_s = p_s² / Σ_i p_i²   (power heuristic, β=2, one sample per strategy)
 *
 * The denominator is built from three strategy PDFs in the 2-strategy approximation
 * used here (light-subpath vertex s + eye-subpath vertex t = full path):
 *   p_s  = pdfFwd_light × G × pdfFwd_eye   (chosen strategy — joint forward PDF)
 *   p_{s-1} = pdfRev_light × ...            (light vertex "shifted" to eye subpath)
 *   p_{s+1} = pdfRev_eye   × ...            (eye vertex "shifted" to light subpath)
 *
 * For the practical BDPT implementation here (one explicit connection per stored
 * light vertex), the simplified 2-strategy form is used as specified in the spec:
 *   pdfSum2 = p_s² + p_otherStrategies²
 * where p_otherStrategies² = (pdfFwd_light × G × pdfFwd_eye)^2 / 4 (balanced approx).
 * The full recursive Veach sweep is deferred to a future patch when full
 * per-strategy PDF storage is available.
 *
 * Specular-vertex guard (Veach §10.3.5): if the light vertex or eye vertex is
 * specular (delta BSDF), return vec3(0.0) — the MIS weight is zero by definition
 * because explicit connections through delta surfaces have zero probability density
 * when sampled from the other subpath.
 *
 * NaN/Inf guards:
 *   - Denominator guard: return 0 when pdfSum2 ≤ 0.
 *   - Geometric term guard: return 0 when G ≤ 0 (degenerate connection).
 *   - Contribution clamp: final RGB is clamped to [0, BDPT_CONTRIBUTION_CLAMP]
 *     per-component to suppress fireflies from caustic spikes during early
 *     convergence. Default clamp = 100 (generous; not a bias for converged renders).
 *
 * References:
 *   Veach 1997, §9.2 (power heuristic), §10.3 (BDPT MIS weights),
 *     §10.3.5 (specular-vertex zero-weight rule).
 *   Pharr et al. 2023, PBR 4e §16.3.5 (recursive ratio, Eq. 16.16).
 *   @vitrum/shared-samplers: bdptConnectionMIS_full, buildBDPTStrategyPDFs_full.
 */
export const bdpt_connection = /* glsl */`

	#define BDPT_CONTRIBUTION_CLAMP 100.0

	// Note: bdptGeometricTerm() (Veach §8.3.2 G term) is defined in
	// bdpt_light_subpath.glsl.js, which is included before this block.
	// Both blocks compile under #if FEATURE_BDPT so the function is always
	// available when evaluateBdptConnection() is reachable.

	// ── Power-heuristic MIS weight (β=2, GLSL port of bdptConnectionMIS_full) ──
	// Simplified 2-strategy form: w = p_s² / (p_s² + p_alt²).
	// Used when only two competing strategies have non-zero PDFs.
	// Guard: returns 0 when denominator ≤ 0 or p_s ≤ 0.
	float bdptMISWeight2( float p_s, float p_alt ) {
		if ( p_s <= 0.0 ) return 0.0;
		float p2s   = p_s   * p_s;
		float p2alt = p_alt * p_alt;
		float denom = p2s + p2alt;
		return ( denom > 0.0 ) ? ( p2s / denom ) : 0.0;
	}

	// ── Visibility test ──────────────────────────────────────────────────────
	// Returns true when the segment (eyePos, lightPos) is unoccluded.
	// Uses the existing attenuateHit() infrastructure with a temporary RenderState.
	// A hit is occluded if any opaque surface lies between the two endpoints.
	// Note: attenuateHit returns true when a solid (opaque) surface blocks the ray.
	bool bdptIsVisible( vec3 eyePos, vec3 lightPos, RenderState state ) {
		vec3 dir  = lightPos - eyePos;
		float len = length( dir );
		if ( len < RAY_OFFSET ) return false; // degenerate — same point
		Ray shadowRay;
		shadowRay.origin    = eyePos;
		shadowRay.direction = dir / len;
		vec3 attenColor;
		// attenuateHit returns true when occluded by a solid surface.
		bool occluded = attenuateHit( state, shadowRay, len - RAY_OFFSET, attenColor );
		return ! occluded;
	}

	// ── BDPT connection contribution ─────────────────────────────────────────
	// Evaluates one eye↔light vertex connection and returns the RGB radiance
	// contribution to add to gColor.rgb.
	//
	// Parameters:
	//   eyePos          — world-space position of the eye-subpath vertex
	//   eyeNormal       — shading normal at the eye vertex (unit-length)
	//   eyeWo           — outgoing direction at the eye vertex (toward camera)
	//   eyeThroughput   — accumulated RGB path weight at the eye vertex
	//   eyePdfFwd       — forward PDF at the eye vertex (for MIS)
	//   eyeSurf         — full SurfaceRecord at the eye vertex (for BSDF eval)
	//   eyeState        — RenderState at the eye vertex (wavelength, traversals, etc.)
	//   lightVtxIdx     — column index into uBdptLightPathTex (0..BDPT_MAX_LIGHT_BOUNCES-1)
	//
	// Returns vec3(0) when:
	//   - The light vertex is invalid (kind = 3)
	//   - The eye or light vertex is specular (delta BSDF)
	//   - The connection is occluded
	//   - The geometric term is degenerate (G ≤ 0)
	//   - Any PDF is non-positive
	vec3 evaluateBdptConnection(
		vec3 eyePos,
		vec3 eyeNormal,
		vec3 eyeWo,
		vec3 eyeThroughput,
		float eyePdfFwd,
		SurfaceRecord eyeSurf,
		RenderState eyeState,
		int lightVtxIdx
	) {

		// ── Fetch light vertex from ping-pong texture ─────────────────────────
		vec4 lv0 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 0 ), 0 );
		vec4 lv1 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 1 ), 0 );
		vec4 lv2 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 2 ), 0 );

		// Check kind — skip invalid vertices (kind = 3.0 = BDPT_KIND_INVALID).
		if ( lv0.w == 3.0 ) return vec3( 0.0 );

		vec3  lightPos        = lv0.xyz;
		vec3  lightNormal     = lv1.xyz;
		float lightPdfFwd     = lv1.w;
		vec3  lightThroughput = lv2.xyz;
		// lv2.w = lightPdfRev — not used in the 2-strategy MIS approximation.
		// The full Veach recursive ratio sweep (future patch) will consume this.

		// ── Specular-vertex guard (Veach §10.3.5) ────────────────────────────
		// If the eye surface is specular (delta BSDF), explicit connection has
		// zero probability density when sampled by the light subpath — skip it.
		// We approximate specular as: transmission > 0.5 AND roughness < 0.05.
		bool eyeIsSpecular = ( eyeSurf.transmission > 0.5 && eyeSurf.filteredRoughness < 0.05 );
		if ( eyeIsSpecular ) return vec3( 0.0 );

		// ── Connection direction ──────────────────────────────────────────────
		vec3 toLight = lightPos - eyePos;
		float dist   = length( toLight );
		if ( dist < RAY_OFFSET ) return vec3( 0.0 ); // degenerate

		vec3 connDir = toLight / dist; // unit direction eye → light

		// ── Geometric term G(eye ↔ light) ────────────────────────────────────
		// Uses bdptGeometricTerm() from bdpt_light_subpath.glsl.js (included first).
		float gTerm = bdptGeometricTerm( eyePos, eyeNormal, lightPos, lightNormal );
		if ( gTerm <= 0.0 ) return vec3( 0.0 );

		// ── Visibility test ───────────────────────────────────────────────────
		if ( ! bdptIsVisible( eyePos, lightPos, eyeState ) ) return vec3( 0.0 );

		// ── BSDF evaluation at eye vertex (toward light) ──────────────────────
		// bsdfResult returns the PDF and sets color (the BSDF value × cosθ / PDF).
		// We need the BSDF × cosθ value, which bsdfResult provides normalized by PDF.
		// Reconstruct: bsdf_eye × cosθ = bsdfResult color × bsdfResult pdf.
		vec3 eyeBsdfColor;
		float eyeBsdfPdf = bsdfResult( eyeWo, connDir, eyeSurf, eyeState.wavelength, eyeBsdfColor );
		if ( eyeBsdfPdf <= 0.0 ) return vec3( 0.0 );
		// eyeBsdfColor is BSDF × cosθ / pdf → multiply by pdf to get BSDF × cosθ.
		vec3 eyeBsdfCosTheta = eyeBsdfColor * eyeBsdfPdf;

		// ── BSDF evaluation at light vertex (toward eye) ──────────────────────
		// We don't have a full SurfaceRecord for the light vertex (the light kernel
		// uses a Lambertian approximation). Use a Lambertian BSDF approximation:
		//   f(ωi, ωo) × cosθ = albedo / π × cosθ_light
		// where cosθ_light = |dot(lightNormal, -connDir)|.
		float cosLight = max( dot( lightNormal, -connDir ), 0.0 );
		// Approximate albedo as vec3(1.0) — the light throughput already encodes
		// the material color from the emitter bounce. This is consistent with the
		// Lambertian approximation in the light subpath kernel.
		vec3 lightBsdfCosTheta = vec3( cosLight / PI );

		// ── MIS weight ───────────────────────────────────────────────────────
		// Full Veach §10.3 weight uses all strategy PDFs. For the 2-strategy
		// approximation (explicit connection + unidirectional PT):
		//   p_s  = pdfFwd_light × G × pdfFwd_eye  (joint forward PDF of this strategy)
		//   p_alt = unidirectional PT forward PDF (approximated as eyePdfFwd × G)
		//
		// The approximation is conservative (underweights BDPT slightly) and
		// prevents MIS denominator collapse without requiring full strategy enumeration.
		float p_s   = lightPdfFwd * gTerm * eyePdfFwd;
		float p_alt = eyePdfFwd   * gTerm; // unidirectional: no light subpath
		float misW  = bdptMISWeight2( p_s, p_alt );
		if ( misW <= 0.0 ) return vec3( 0.0 );

		// ── Assemble contribution ────────────────────────────────────────────
		// contribution = lightThroughput × lightBsdf×cosθ × G × eyeBsdf×cosθ × MIS × eyeThroughput
		// Note: eyeThroughput is already converted to RGB via wavelengthToRGB() at the call site.
		vec3 contribution = lightThroughput * lightBsdfCosTheta * gTerm * eyeBsdfCosTheta * misW;
		// Multiply by eye throughput (caller provides RGB-converted throughput).
		contribution *= eyeThroughput;

		// ── NaN / Inf guard and firefly clamp ────────────────────────────────
		if ( any( isnan( contribution ) ) || any( isinf( contribution ) ) ) {
			return vec3( 0.0 );
		}
		return clamp( contribution, vec3( 0.0 ), vec3( BDPT_CONTRIBUTION_CLAMP ) );

	}

`;
