/**
 * bdpt_connection.glsl.js — BDPT eye↔light connection, full Veach §10.3 MIS.
 *
 * Computes the power-heuristic (β=2) MIS weight for ONE explicit connection
 * (current eye-subpath bounce `E_e` → one stored light-subpath vertex `L_c`) by
 * enumerating ALL Veach §10.3 strategy path-pdfs over the merged path
 *
 *   v[0]=L_0(emitter) … v[c]=L_c | v[c+1]=E_e … v[c+1+e]=E_0 | v[n-1]=camera
 *
 * with `n = c+e+3` and `selectedS = c+1`. This is the canonical PBRT-v4
 * `MISWeight` recurrence — a pure ratio of AREA-measure forward/reverse densities
 * walked over the actual vertices, with each stored solid-angle pdf converted to
 * area on the fly via ConvertDensity (PBRT `Vertex::ConvertDensity`, a
 * DESTINATION-cosine-only "half-G"). The four pdfs straddling the connection edge
 * are recomputed here from the connection geometry (PBRT pt/ptMinus/qs/qsMinus
 * pdfRev overrides); the eye-side overrides use `bsdfResult` with wo/wi as
 * required (PBRT-correct, non-symmetric reverse density). The eye prefix
 * (E_0…E_e construction-time SA pdfs + pos/normal/specular) is supplied by the
 * caller in local per-invocation arrays threaded through the eye loop. The light
 * chain is Lambertian (cosθ/π) throughout, matching the light-subpath kernel.
 *
 * This mirrors the WebGPU port (`@vitrum/pt-webgpu` bdptConnection.wgsl.ts),
 * which is pinned 1:1 to `@vitrum/shared-samplers`'s `bdptConnectionMIS_full` /
 * `buildBDPTStrategyPDFs_full` oracle to ~1e-12. Compiled only under
 * FEATURE_BDPT; the main loop calls evaluateBdptConnection only for indirect
 * bounces (`!state.firstRay`), so the BDPT-off path is bit-identical.
 *
 * Specular-vertex guard (Veach §10.3.5): a hypothetical strategy whose
 * connection edge touches a delta-BSDF vertex has zero weight; the sweep breaks.
 *
 * References:
 *   Veach 1997 §10.3 (BDPT MIS), §9.2 (power heuristic β=2), §10.3.5, §8.3.2.
 *   Pharr et al. 2023, PBR 4e §16.3.5 Eq. 16.16; integrators.cpp MISWeight.
 *   @vitrum/shared-samplers: bdptConnectionMIS_full, buildBDPTStrategyPDFs_full.
 */
export const bdpt_connection = /* glsl */`

	#define BDPT_CONTRIBUTION_CLAMP 100.0
	#define BDPT_MAX_EYE_DEPTH 8
	#define BDPT_MAX_MERGED 19

	// Note: bdptGeometricTerm() (Veach §8.3.2 G term) is defined in
	// bdpt_light_subpath.glsl.js, which is included before this block.

	// ── PBRT Vertex::ConvertDensity — SA pdf → area pdf, destination-cosine only ──
	float bdptConvertDensitySAtoArea( float pdfSA, vec3 fromPos, vec3 destPos, vec3 destNorm ) {
		vec3 d = destPos - fromPos;
		float dist2 = dot( d, d );
		if ( dist2 <= 0.0 ) return pdfSA; // coincident → unit Jacobian (endpoint)
		float invDist = inversesqrt( dist2 );
		float cosDest = abs( dot( destNorm, d * invDist ) );
		return ( pdfSA * cosDest ) / dist2;
	}

	// Lambertian outgoing SA density at a light-subpath vertex along dir: |cosθ|/π.
	float bdptLambertDirPdf( vec3 n, vec3 dir ) {
		return abs( dot( n, normalize( dir ) ) ) * ( 1.0 / PI );
	}

	// ── Visibility test (unchanged) ──────────────────────────────────────────
	bool bdptIsVisible( vec3 eyePos, vec3 lightPos, RenderState state ) {
		vec3 dir  = lightPos - eyePos;
		float len = length( dir );
		if ( len < RAY_OFFSET ) return false;
		Ray shadowRay;
		shadowRay.origin    = eyePos;
		shadowRay.direction = dir / len;
		vec3 attenColor;
		bool occluded = attenuateHit( state, shadowRay, len - RAY_OFFSET, attenColor );
		return ! occluded;
	}

	// ── Full §10.3 power-heuristic MIS weight ─────────────────────────────────
	// Assembles the merged path from: the light texture (v[0..c]); the eye-stack
	// arrays (v[c+1..c+1+e], reverse order); the camera endpoint (v[n-1]); and the
	// four connection-induced straddle overrides. Mirrors buildBDPTStrategyPDFs_full
	// + bdptConnectionMIS_full exactly.
	//
	// Eye-stack arrays are indexed by eye depth d (0 = primary hit … e = current
	// bounce). Merged index i in [c+1, c+1+e] maps to eye depth d = e - (i-(c+1)).
	float bdptMISWeightFull(
		int c, int e, int n, int selectedS, float pRef,
		// connection-induced straddle overrides (PBRT pt/ptMinus/qs/qsMinus pdfRev)
		float fwdEe, float fwdEeMinus, float revLc, float revLcMinus,
		// eye stack (light chain + camera read inside)
		vec3 eyePos[ BDPT_MAX_EYE_DEPTH ], vec3 eyeNrm[ BDPT_MAX_EYE_DEPTH ],
		float eyePdfFwd[ BDPT_MAX_EYE_DEPTH ], float eyePdfRev[ BDPT_MAX_EYE_DEPTH ],
		bool eyeSpec[ BDPT_MAX_EYE_DEPTH ],
		vec3 camPos, vec3 camNrm
	) {
		if ( pRef <= 0.0 || n <= 0 || selectedS >= n ) return 0.0;

		// Materialise the merged path into fixed-size local arrays — far simpler
		// (and branch-light) than re-deriving each vertex inside the sweep.
		vec3  mPos[ BDPT_MAX_MERGED ];
		vec3  mNrm[ BDPT_MAX_MERGED ];
		float mFwd[ BDPT_MAX_MERGED ];
		float mRev[ BDPT_MAX_MERGED ];
		bool  mSpec[ BDPT_MAX_MERGED ];

		for ( int i = 0; i < BDPT_MAX_MERGED; i ++ ) {
			if ( i >= n ) break;
			if ( i <= c ) {
				// Light side — texelFetch column i.
				vec4 l0 = texelFetch( uBdptLightPathTex, ivec2( i, 0 ), 0 );
				vec4 l1 = texelFetch( uBdptLightPathTex, ivec2( i, 1 ), 0 );
				vec4 l2 = texelFetch( uBdptLightPathTex, ivec2( i, 2 ), 0 );
				mPos[ i ] = l0.xyz;
				mNrm[ i ] = l1.xyz;
				mFwd[ i ] = l1.w;       // stored SA pdfFwd (no baked-in G)
				mRev[ i ] = l2.w;       // stored SA pdfRev (Lambertian)
				mSpec[ i ] = false;     // light subpath terminates specular (D4)
				if ( i == c ) mRev[ i ] = revLc;
				else if ( c >= 1 && i == c - 1 ) mRev[ i ] = revLcMinus;
			} else if ( i == n - 1 ) {
				mPos[ i ] = camPos;
				mNrm[ i ] = camNrm;
				mFwd[ i ] = 1.0;
				mRev[ i ] = 1.0;
				mSpec[ i ] = false;
			} else {
				int off = i - ( c + 1 );
				int d = e - off;        // eye depth
				mPos[ i ] = eyePos[ d ];
				mNrm[ i ] = eyeNrm[ d ];
				mFwd[ i ] = eyePdfFwd[ d ];
				mRev[ i ] = eyePdfRev[ d ];
				mSpec[ i ] = eyeSpec[ d ];
				if ( off == 0 ) mFwd[ i ] = fwdEe;          // E_e   override
				else if ( off == 1 ) mFwd[ i ] = fwdEeMinus; // E_{e-1} override
			}
		}

		float pdfs[ BDPT_MAX_MERGED ];
		for ( int k = 0; k < BDPT_MAX_MERGED; k ++ ) pdfs[ k ] = 0.0;
		pdfs[ selectedS ] = pRef;

		// Left sweep (decrement s): flip v[s-1]; p_{s-1} = p_s · pRev(s-1)/pFwd(s-1).
		{
			float p = pRef;
			for ( int s = selectedS; s > 0; s -- ) {
				bool flipSpec = mSpec[ s - 1 ];
				bool nbSpec = ( s >= 2 ) ? mSpec[ s - 2 ] : false;
				if ( flipSpec || nbSpec ) break;
				float pFwd = ( s - 1 == 0 )
					? mFwd[ 0 ]
					: bdptConvertDensitySAtoArea( mFwd[ s - 1 ], mPos[ s - 2 ], mPos[ s - 1 ], mNrm[ s - 1 ] );
				float pRev = ( s - 1 == n - 1 )
					? mRev[ s - 1 ]
					: bdptConvertDensitySAtoArea( mRev[ s - 1 ], mPos[ s ], mPos[ s - 1 ], mNrm[ s - 1 ] );
				if ( pFwd <= 0.0 || pRev <= 0.0 ) break;
				p = p * ( pRev / pFwd );
				pdfs[ s - 1 ] = p;
			}
		}
		// Right sweep (increment s): flip v[s]; p_{s+1} = p_s · pFwd(s)/pRev(s).
		{
			float p = pRef;
			for ( int s = selectedS; s < n - 1; s ++ ) {
				bool flipSpec = mSpec[ s ];
				bool nbSpec = mSpec[ s + 1 ];
				if ( flipSpec || nbSpec ) break;
				float pFwd = ( s == 0 )
					? mFwd[ 0 ]
					: bdptConvertDensitySAtoArea( mFwd[ s ], mPos[ s - 1 ], mPos[ s ], mNrm[ s ] );
				float pRev = ( s == n - 1 )
					? mRev[ s ]
					: bdptConvertDensitySAtoArea( mRev[ s ], mPos[ s + 1 ], mPos[ s ], mNrm[ s ] );
				if ( pFwd <= 0.0 || pRev <= 0.0 ) break;
				p = p * ( pFwd / pRev );
				pdfs[ s + 1 ] = p;
			}
		}

		// Power heuristic (β=2).
		float denom = 0.0;
		for ( int k = 0; k < BDPT_MAX_MERGED; k ++ ) {
			if ( k >= n ) break;
			float pk = pdfs[ k ];
			if ( pk > 0.0 ) denom += pk * pk;
		}
		if ( denom <= 0.0 ) return 0.0;
		float ps = pdfs[ selectedS ];
		if ( ps <= 0.0 ) return 0.0;
		return ( ps * ps ) / denom;
	}

	// ── BDPT connection contribution ─────────────────────────────────────────
	// eyeDepth = current bounce's eye depth e (0 = primary hit — never connected).
	// The eye-stack arrays carry E_0…E_e construction-time data (see main loop).
	vec3 evaluateBdptConnection(
		vec3 eyePos,
		vec3 eyeNormal,
		vec3 eyeWo,
		vec3 eyeThroughput,
		SurfaceRecord eyeSurf,
		RenderState eyeState,
		int eyeDepth,
		vec3 bdptEyePos[ BDPT_MAX_EYE_DEPTH ],
		vec3 bdptEyeNrm[ BDPT_MAX_EYE_DEPTH ],
		float bdptEyePdfFwd[ BDPT_MAX_EYE_DEPTH ],
		float bdptEyePdfRev[ BDPT_MAX_EYE_DEPTH ],
		bool bdptEyeSpec[ BDPT_MAX_EYE_DEPTH ],
		int lightVtxIdx
	) {

		vec4 lv0 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 0 ), 0 );
		vec4 lv1 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 1 ), 0 );
		vec4 lv2 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 2 ), 0 );
		if ( lv0.w == 3.0 ) return vec3( 0.0 ); // BDPT_KIND_INVALID

		vec3  lightPos        = lv0.xyz;
		vec3  lightNormal     = lv1.xyz;
		float lightPdfFwd     = lv1.w;
		vec3  lightThroughput = lv2.xyz;

		// Connection-edge specular guard at the eye vertex (Veach §10.3.5).
		bool eyeIsSpecular = ( eyeSurf.transmission > 0.5 && eyeSurf.filteredRoughness < 0.05 );
		if ( eyeIsSpecular ) return vec3( 0.0 );

		vec3 toLight = lightPos - eyePos;
		float dist   = length( toLight );
		if ( dist < RAY_OFFSET ) return vec3( 0.0 );
		vec3 connDir = toLight / dist; // E_e → L_c

		float gTerm = bdptGeometricTerm( eyePos, eyeNormal, lightPos, lightNormal );
		if ( gTerm <= 0.0 ) return vec3( 0.0 );
		if ( ! bdptIsVisible( eyePos, lightPos, eyeState ) ) return vec3( 0.0 );

		// eye BSDF × cosθ toward the light (bsdfResult returns BSDF×cosθ/pdf).
		vec3 eyeBsdfColor;
		float eyeBsdfPdf = bsdfResult( eyeWo, connDir, eyeSurf, eyeState.wavelength, eyeBsdfColor );
		if ( eyeBsdfPdf <= 0.0 ) return vec3( 0.0 );
		vec3 eyeBsdfCosTheta = eyeBsdfColor * eyeBsdfPdf;

		// light vertex Lambertian BSDF × cosθ toward the eye.
		float cosLight = max( dot( lightNormal, -connDir ), 0.0 );
		vec3 lightBsdfCosTheta = vec3( cosLight / PI );

		// ── Full §10.3 MIS weight ────────────────────────────────────────────
		int c = lightVtxIdx;
		int e = eyeDepth;
		int n = c + e + 3;
		int selectedS = c + 1;
		if ( n > BDPT_MAX_MERGED ) return vec3( 0.0 );

		vec3 camPos = ( cameraWorldMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
		vec3 camNrm = normalize( camPos - eyePos );

		// Connection-induced straddle overrides (PBRT MISWeight remapping).
		vec3 lcToE = -connDir;                       // L_c → E_e
		float fwdEe = bdptLambertDirPdf( lightNormal, lcToE );

		vec3 eeMinusPos = camPos;
		if ( e >= 1 ) eeMinusPos = bdptEyePos[ e - 1 ];
		vec3 eeToPrev = normalize( eeMinusPos - eyePos ); // E_e → E_{e-1} (or camera)

		vec3 dummyColor;
		float revLc = bsdfResult( eeToPrev, connDir, eyeSurf, eyeState.wavelength, dummyColor );
		float fwdEeMinus = 0.0;
		if ( e >= 1 ) {
			fwdEeMinus = bsdfResult( connDir, eeToPrev, eyeSurf, eyeState.wavelength, dummyColor );
		}
		float revLcMinus = 0.0;
		if ( c >= 1 ) {
			vec4 lcm0 = texelFetch( uBdptLightPathTex, ivec2( c - 1, 0 ), 0 );
			vec3 lcToLcMinus = normalize( lcm0.xyz - lightPos );
			revLcMinus = bdptLambertDirPdf( lightNormal, lcToLcMinus );
		}

		// pRef cancels in the power-heuristic ratio (scale-invariant); use the
		// area-forward of the selected strategy's connection edge for conditioning.
		float pRef = max( lightPdfFwd, 1e-12 ) * max( fwdEe, 1e-12 ) + 1e-30;
		float misW = bdptMISWeightFull(
			c, e, n, selectedS, pRef,
			fwdEe, fwdEeMinus, revLc, revLcMinus,
			bdptEyePos, bdptEyeNrm, bdptEyePdfFwd, bdptEyePdfRev, bdptEyeSpec,
			camPos, camNrm
		);
		if ( misW <= 0.0 ) return vec3( 0.0 );

		vec3 contribution = lightThroughput * lightBsdfCosTheta * gTerm * eyeBsdfCosTheta * misW;
		contribution *= eyeThroughput;
		if ( any( isnan( contribution ) ) || any( isinf( contribution ) ) ) {
			return vec3( 0.0 );
		}
		return clamp( contribution, vec3( 0.0 ), vec3( BDPT_CONTRIBUTION_CLAMP ) );

	}

`;
