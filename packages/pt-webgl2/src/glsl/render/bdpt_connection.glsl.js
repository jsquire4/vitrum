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
 * caller in local per-invocation arrays threaded through the eye loop. Light-chain
 * surface vertices reconstruct their material payload and evaluate the same BSDF as
 * the light-subpath kernel; emitter sentinel vertices keep the diffuse emission profile.
 *
 * This mirrors the WebGPU port (`@vitrum/pt-webgpu` bdptConnection.wgsl.ts),
 * which is pinned 1:1 to `@vitrum/shared-samplers`'s `bdptConnectionMIS_full` /
 * `buildBDPTStrategyPDFs_full` oracle to ~1e-12. Compiled only under
 * FEATURE_BDPT; the main loop connects every eye vertex, including the primary
 * vertex, and starts at the finite emitter endpoint (c=0). Infinite endpoints
 * are launch-disk constructs: their primary c=0 strategy stays in distant NEE,
 * while their first real scattering vertex begins the c>=1 BDPT partition.
 *
 * Specular-vertex guard (Veach §10.3.5): a hypothetical strategy whose
 * connection edge touches a delta-BSDF vertex has zero weight. The density
 * recurrence still crosses that edge, using PBRT's Remap0, because a farther
 * strategy may connect two finite vertices and remain eligible.
 *
 * References:
 *   Veach 1997 §10.3 (BDPT MIS), §9.2 (power heuristic β=2), §10.3.5, §8.3.2.
 *   Pharr et al. 2023, PBR 4e §16.3.5 Eq. 16.16; integrators.cpp MISWeight.
 *   @vitrum/shared-samplers: bdptConnectionMIS_full, buildBDPTStrategyPDFs_full.
 */
export const bdpt_connection = /* glsl */`

        #ifndef BDPT_MAX_EYE_DEPTH
        #define BDPT_MAX_EYE_DEPTH 8
        #endif
	#define BDPT_MAX_MERGED 19

	// The connection inverse-distance Jacobian is defined in
	// bdpt_light_subpath.glsl.js, which is included before this block.

	// ── PBRT Vertex::ConvertDensity — SA pdf → area pdf, destination-cosine only ──
	float bdptConvertDensitySAtoArea(
		float pdfSA, vec3 fromPos, vec3 destPos, vec3 destNorm, bool destIsMedium
	) {
		vec3 d = destPos - fromPos;
		if (
			! ( pdfSA >= 0.0 ) ||
			! bdptFiniteFloat( pdfSA ) ||
			! bdptFiniteVec3( d ) ||
			( ! destIsMedium && ! vitrumFiniteNonZeroVec3( destNorm ) )
		) return 0.0;
		if ( all( equal( d, vec3( 0.0 ) ) ) ) {
			return pdfSA; // coincident endpoint: unit Jacobian by convention
		}
		float distance = vitrumLengthVec3( d );
		if ( ! ( distance > 0.0 ) ) return 0.0;
		vec3 edgeDirection = vitrumNormalizeVec3( d, vec3( 0.0 ) );
		vec3 destinationNormal = destIsMedium
			? vec3( 0.0 )
			: vitrumNormalizeVec3( destNorm, vec3( 0.0 ) );
		float cosDest =
			destIsMedium ? 1.0 : abs( dot( destinationNormal, edgeDirection ) );
		return vitrumPositiveProductOverSquare( pdfSA, cosDest, distance );
	}

	float bdptRemapZeroDensity( float pdf ) {
		return pdf == 0.0 ? 1.0 : pdf;
	}


	// ── Visibility test ──────────────────────────────────────────────────────
	bool bdptVisibilityAttenuation(
		vec3 eyePos,
		vec3 lightPos,
		RenderState state,
		bool skipOcclusion,
		bool hasTargetFace,
		uint targetFaceIndex,
		out vec3 attenColor
	) {
		attenColor = vec3( 1.0 );
		if ( skipOcclusion ) return true;
		vec3 dir  = lightPos - eyePos;
		if ( ! bdptFiniteVec3( dir ) ) return false;
		float len = vitrumLengthVec3( dir );
		if ( ! ( len > RAY_OFFSET ) ) return false;
		Ray shadowRay;
		shadowRay.origin    = eyePos;
		shadowRay.direction =
			vitrumNormalizeVec3( dir, vec3( 0.0 ) );
			bool occluded = attenuateHit(
				state,
				shadowRay,
				len,
				hasTargetFace,
				targetFaceIndex,
				attenColor
			);
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
                bool infiniteRoot,
                bool infiniteEnvironmentRoot,
                vec3 infiniteSourceDirection,
                float infiniteNeePdf,
                float infiniteLaunchPdf,
                float infiniteEyeEscapePdf,
                bool infiniteEyeEscapeDelta,
		// connection-induced straddle overrides (PBRT pt/ptMinus/qs/qsMinus pdfRev)
		float fwdEe, float fwdEeMinus, float revLc, float revLcMinus,
		// eye stack (light chain + camera read inside)
		vec3 eyePos[ BDPT_MAX_EYE_DEPTH ], vec3 eyeNrm[ BDPT_MAX_EYE_DEPTH ],
		float eyePdfFwd[ BDPT_MAX_EYE_DEPTH ], float eyePdfRev[ BDPT_MAX_EYE_DEPTH ],
		bool eyeSpec[ BDPT_MAX_EYE_DEPTH ], bool eyeMedium[ BDPT_MAX_EYE_DEPTH ],
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
		bool  mMedium[ BDPT_MAX_MERGED ];

		for ( int i = 0; i < BDPT_MAX_MERGED; i ++ ) {
			if ( i >= n ) break;
			if ( i <= c ) {
				// Light side — texelFetch column i.
				vec4 l0 = texelFetch( uBdptLightPathTex, ivec2( i, 0 ), 0 );
				vec4 l1 = texelFetch( uBdptLightPathTex, ivec2( i, 1 ), 0 );
				vec4 l2 = texelFetch( uBdptLightPathTex, ivec2( i, 2 ), 0 );
				vec4 l3 = texelFetch( uBdptLightPathTex, ivec2( i, 3 ), 0 );
				mPos[ i ] = l0.xyz;
				mNrm[ i ] = l1.xyz;
				mFwd[ i ] = l1.w;       // stored SA pdfFwd (no baked-in G)
				mRev[ i ] = l2.w;       // stored SA pdfRev placeholder / patched straddle value
				mMedium[ i ] = l3.w == BDPT_LV_MEDIUM_MATID;
				mSpec[ i ] = l0.w == BDPT_KIND_DELTA;
				if ( i == c && l3.w >= 0.0 ) {
					mSpec[ i ] = fwdEe <= 0.0 || revLc <= 0.0;
				}
				if ( i == c ) mRev[ i ] = revLc;
				else if ( c >= 1 && i == c - 1 ) mRev[ i ] = revLcMinus;
			} else if ( i == n - 1 ) {
				mPos[ i ] = camPos;
				mNrm[ i ] = camNrm;
				mFwd[ i ] = 1.0;
				mRev[ i ] = 1.0;
				mSpec[ i ] = false;
				mMedium[ i ] = false;
			} else {
				int off = i - ( c + 1 );
				int d = e - off;        // eye depth
				mPos[ i ] = eyePos[ d ];
				mNrm[ i ] = eyeNrm[ d ];
				mFwd[ i ] = eyePdfFwd[ d ];
				mRev[ i ] = eyePdfRev[ d ];
				mSpec[ i ] = eyeSpec[ d ];
				mMedium[ i ] = eyeMedium[ d ];
				if ( off == 0 ) mSpec[ i ] = fwdEe <= 0.0 || revLc <= 0.0;
				if ( off == 0 ) mFwd[ i ] = fwdEe;          // E_e   override
				else if ( off == 1 ) mFwd[ i ] = fwdEeMinus; // E_{e-1} override
			}
		}

		// Work in ratios relative to the selected strategy (log pRef = 0).
		// This is the same Veach recurrence as the linear form, but remains finite
		// across long/grazing paths where products of area densities overflow.
		float logPdfs[ BDPT_MAX_MERGED ];
		bool knownPdfs[ BDPT_MAX_MERGED ];
		bool validPdfs[ BDPT_MAX_MERGED ];
		for ( int k = 0; k < BDPT_MAX_MERGED; k ++ ) {
			logPdfs[ k ] = 0.0;
			knownPdfs[ k ] = false;
			validPdfs[ k ] = false;
		}
		knownPdfs[ selectedS ] = true;
		validPdfs[ selectedS ] = pRef > 0.0;

		// Left sweep (decrement s): flip v[s-1]; p_{s-1} = p_s · pRev(s-1)/pFwd(s-1).
		{
			float logP = 0.0;
			for ( int s = selectedS; s > 0; s -- ) {
				bool flipSpec = mSpec[ s - 1 ];
				bool nbSpec = ( s >= 2 ) ? mSpec[ s - 2 ] : false;
				bool connectionIsDelta = flipSpec || nbSpec;
				float pFwd = ( s - 1 == 0 )
					? mFwd[ 0 ]
					: bdptConvertDensitySAtoArea( mFwd[ s - 1 ], mPos[ s - 2 ], mPos[ s - 1 ], mNrm[ s - 1 ], mMedium[ s - 1 ] );
				float pRev = ( s - 1 == n - 1 )
					? mRev[ s - 1 ]
					: bdptConvertDensitySAtoArea( mRev[ s - 1 ], mPos[ s ], mPos[ s - 1 ], mNrm[ s - 1 ], mMedium[ s - 1 ] );
				if (
					pFwd < 0.0 || pRev < 0.0 ||
					isnan( pFwd ) || isnan( pRev ) ||
					isinf( pFwd ) || isinf( pRev )
				) break;
				pFwd = bdptRemapZeroDensity( pFwd );
				pRev = bdptRemapZeroDensity( pRev );
				logP += log( pRev ) - log( pFwd );
				logPdfs[ s - 1 ] = logP;
				knownPdfs[ s - 1 ] = true;
				validPdfs[ s - 1 ] = ! connectionIsDelta;
			}
		}
		// Right sweep (increment s): flip v[s]; p_{s+1} = p_s · pFwd(s)/pRev(s).
		{
			float logP = 0.0;
			for ( int s = selectedS; s < n - 1; s ++ ) {
				bool flipSpec = mSpec[ s ];
				bool nbSpec = mSpec[ s + 1 ];
				bool connectionIsDelta = flipSpec || nbSpec;
				float pFwd = ( s == 0 )
					? mFwd[ 0 ]
					: bdptConvertDensitySAtoArea( mFwd[ s ], mPos[ s - 1 ], mPos[ s ], mNrm[ s ], mMedium[ s ] );
				float pRev = ( s == n - 1 )
					? mRev[ s ]
					: bdptConvertDensitySAtoArea( mRev[ s ], mPos[ s + 1 ], mPos[ s ], mNrm[ s ], mMedium[ s ] );
				if (
					pFwd < 0.0 || pRev < 0.0 ||
					isnan( pFwd ) || isnan( pRev ) ||
					isinf( pFwd ) || isinf( pRev )
				) break;
				pFwd = bdptRemapZeroDensity( pFwd );
				pRev = bdptRemapZeroDensity( pRev );
				logP += log( pFwd ) - log( pRev );
				logPdfs[ s + 1 ] = logP;
				knownPdfs[ s + 1 ] = true;
				validPdfs[ s + 1 ] = ! connectionIsDelta;
			}
                }

                if ( infiniteRoot ) {
                        // The generic finite-endpoint recurrence uses a solid-angle
                        // 1/r² conversion at v1. Infinite roots instead launch
                        // parallel rays from the bounding disk, and s=1 uses the
                        // separately sampled distant-NEE density.
                        validPdfs[ 0 ] = false;
                        validPdfs[ 1 ] = false;
                        if (
                                n > 3 &&
                                knownPdfs[ 2 ] &&
                                infiniteNeePdf > 0.0 &&
                                infiniteLaunchPdf > 0.0
                        ) {
                                float launchAreaPdf =
                                        bdptInfiniteLaunchDensityToArea(
                                                infiniteLaunchPdf,
                                                mNrm[ 1 ],
                                                infiniteSourceDirection,
                                                mMedium[ 1 ]
                                        );
                                float eyeAreaPdf = bdptConvertDensitySAtoArea(
                                        mRev[ 1 ],
                                        mPos[ 2 ],
                                        mPos[ 1 ],
                                        mNrm[ 1 ],
                                        mMedium[ 1 ]
                                );
                                if ( launchAreaPdf > 0.0 && eyeAreaPdf > 0.0 ) {
                                        // The shared source-direction factor is
                                        // either an SA density or (for a hard
                                        // directional emitter) a unit discrete
                                        // mass.  Cancel it before taking logs;
                                        // only receiver-area densities remain.
                                        float neeToLaunchAreaRatio =
                                                infiniteNeePdf / launchAreaPdf;
                                        logPdfs[ 1 ] =
                                                logPdfs[ 2 ] +
                                                log( neeToLaunchAreaRatio ) +
                                                log( eyeAreaPdf );
                                        knownPdfs[ 1 ] = true;
                                        // s=1 connects the distant source to
                                        // v1. A delta at v2 invalidates s=2,
                                        // but does not invalidate this edge.
                                        validPdfs[ 1 ] = ! mSpec[ 1 ];
                                        if (
                                                infiniteEnvironmentRoot &&
                                                ! infiniteEyeEscapeDelta &&
                                                infiniteEyeEscapePdf > 0.0
                                        ) {
                                                logPdfs[ 0 ] =
                                                        logPdfs[ 1 ] +
                                                        log( infiniteEyeEscapePdf ) -
                                                        log( infiniteNeePdf );
                                                knownPdfs[ 0 ] = true;
                                                validPdfs[ 0 ] = true;
                                        }
                                }
                        }
                }

                // Retain only strategies sampled by this bounded implementation. s=0
		// (pure-eye) and s=n-1 (pure-light) are separate estimator families.
		for ( int k = 0; k < BDPT_MAX_MERGED; k ++ ) {
			if ( k >= n ) break;
                        bool explicitStrategy = k >= 1 && k <= n - 2;
                        if ( infiniteRoot ) {
                                explicitStrategy =
                                        ( k == 0 && infiniteEnvironmentRoot ) ||
                                        k == 1 ||
                                        ( k >= 2 && k <= n - 2 );
                        }
			int lightVertices = k;
			int eyeVertices = n - k - 1;
                        bool exceedsLightBound =
                                k >= 2 && lightVertices > uBdptMaxLightBounces;
                        if (
                                ! explicitStrategy ||
                                exceedsLightBound ||
                                eyeVertices > BDPT_MAX_EYE_DEPTH
                        ) validPdfs[ k ] = false;
		}
		if ( ! validPdfs[ selectedS ] ) return 0.0;

		// Power heuristic (β=2) via max-shifted log-sum-exp.
		float maxPowerLog = 2.0 * logPdfs[ selectedS ];
		for ( int k = 0; k < BDPT_MAX_MERGED; k ++ ) {
			if ( k >= n ) break;
			if ( validPdfs[ k ] ) maxPowerLog = max( maxPowerLog, 2.0 * logPdfs[ k ] );
		}
		float denom = 0.0;
		for ( int k = 0; k < BDPT_MAX_MERGED; k ++ ) {
			if ( k >= n ) break;
			if ( validPdfs[ k ] ) denom += exp( 2.0 * logPdfs[ k ] - maxPowerLog );
		}
		if ( denom <= 0.0 ) return 0.0;
		return exp( 2.0 * logPdfs[ selectedS ] - maxPowerLog ) / denom;
	}

	// ── BDPT connection contribution ─────────────────────────────────────────
	// eyeDepth = current bounce's eye depth e (0 = primary hit).
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
		bool bdptEyeMedium[ BDPT_MAX_EYE_DEPTH ],
		int lightVtxIdx
	) {

                // c light-side scattering vertices plus e+1 eye vertices must
                // fit the renderer's total accepted-scattering-vertex budget.
                // Keep this defensive check local as well as in the caller's
                // loop so future call sites cannot evaluate a longer path.
                if ( lightVtxIdx + eyeDepth >= bounces ) return vec3( 0.0 );

		vec4 lv0 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 0 ), 0 );
		vec4 lv1 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 1 ), 0 );
		vec4 lv2 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 2 ), 0 );
		vec4 lv3 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 3 ), 0 );
                vec4 lv4 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 4 ), 0 );
                vec4 lv5 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 5 ), 0 );
                vec4 lv6 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 6 ), 0 );
                vec4 lv7 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 7 ), 0 );
		if ( lv0.w == 3.0 ) return vec3( 0.0 ); // BDPT_KIND_INVALID
		if (
			! bdptStoredVertexRowsValid(
				lv0, lv1, lv2, lv3, lv4, lv5, lv6, lv7
			) ||
			! bdptFiniteVec3( eyePos ) ||
			( ! eyeSurf.volumeParticle &&
				( ! bdptFiniteVec3( eyeNormal ) ||
					! vitrumFiniteNonZeroVec3( eyeNormal ) ) ) ||
			! bdptFiniteVec3( eyeWo ) ||
			! vitrumFiniteNonZeroVec3( eyeWo ) ||
			! bdptFiniteVec3( eyeThroughput ) ||
			any( lessThan( eyeThroughput, vec3( 0.0 ) ) )
		) return vec3( 0.0 );

		vec3  lightPos        = lv0.xyz;
		vec3  lightNormal     = lv1.xyz;
		float lightPdfFwd     = lv1.w;
		vec3  lightThroughput = lv2.xyz;
		vec3  lightWoPrev     = lv3.xyz;
		float lightMatId      = lv3.w;
		bool lightIsMedium    = lightMatId == BDPT_LV_MEDIUM_MATID;
		bool lightIsEndpoint  = lightVtxIdx == 0;
		bool pointEndpoint = lightIsEndpoint && lightMatId == BDPT_LV_POINT_EMITTER_MATID;
		bool spotEndpoint = lightIsEndpoint && lightMatId == BDPT_LV_SPOT_EMITTER_MATID;
		bool areaEndpoint = lightIsEndpoint && lightMatId == BDPT_LV_AREA_EMITTER_MATID;
		bool infiniteEndpoint = lightIsEndpoint && (
			lightMatId == BDPT_LV_DIRECTIONAL_EMITTER_MATID ||
			lightMatId == BDPT_LV_ENVIRONMENT_EMITTER_MATID
		);
		// Infinite endpoints are scene-bounding-disk launch constructs, not
			// finite points. Their primary c=0 strategy stays in distant NEE while
			// their first real scattering vertices (c>=1) remain connectable.
		if ( infiniteEndpoint ) return vec3( 0.0 );
		if ( ! lightIsEndpoint && lightMatId < 0.0 && ! lightIsMedium ) {
			return vec3( 0.0 );
		}
			bool meshAreaEndpointHasTarget =
				areaEndpoint && meshLightSourceFaceWordsValid( lv4.zw );
			bool surfaceVertexHasTarget = ! lightIsEndpoint && ! lightIsMedium;
			if (
				surfaceVertexHasTarget &&
				! meshLightSourceFaceWordsValid( lv7.zw )
			) return vec3( 0.0 );
		bool hasTargetFace =
			meshAreaEndpointHasTarget || surfaceVertexHasTarget;
		uint targetFaceIndex = 0u;
			if ( meshAreaEndpointHasTarget ) {
				targetFaceIndex = meshLightSourceFaceIndex( lv4.zw );
			} else if ( surfaceVertexHasTarget ) {
				targetFaceIndex = meshLightSourceFaceIndex( lv7.zw );
			}


		vec3 toLight = lightPos - eyePos;
		if ( ! bdptFiniteVec3( toLight ) ) return vec3( 0.0 );
		float dist = vitrumLengthVec3( toLight );
		if ( ! ( dist > RAY_OFFSET ) ) return vec3( 0.0 );
			vec3 connDir =
				vitrumNormalizeVec3( toLight, vec3( 0.0 ) ); // E_e → L_c
			vec3 visibilityOffset = connDir;
			if ( ! eyeSurf.volumeParticle ) {
				if (
					! bdptFiniteVec3( eyeSurf.faceNormal ) ||
					! vitrumFiniteNonZeroVec3( eyeSurf.faceNormal )
				) return vec3( 0.0 );
				vec3 geometricNormal =
					vitrumNormalizeVec3( eyeSurf.faceNormal, vec3( 0.0 ) );
				float visibilitySide =
					dot( connDir, geometricNormal ) < 0.0 ? -1.0 : 1.0;
				visibilityOffset = geometricNormal * visibilitySide;
			}
			// Keep the BDPT vertex and its Jacobians at the exact geometric point.
			// Only the visibility ray receives a connection-specific numerical step;
			// bdptVisibilityAttenuation then rebuilds its direction and distance to
			// the exact sampled endpoint from this stepped origin.
			vec3 visibilityOrigin = stepRayOrigin(
				eyePos, vec3( 0.0 ), visibilityOffset, 0.0
			);
			if ( ! bdptFiniteVec3( visibilityOrigin ) ) return vec3( 0.0 );

			float gTerm = bdptConnectionInverseDistanceSquared( eyePos, lightPos );
		if ( pointEndpoint ) {
			gTerm = getDistanceAttenuation( dist, lv4.y, lv4.z );
		} else if ( spotEndpoint ) {
			float spotCos = dot(
				vitrumNormalizeVec3( lightNormal, vec3( 0.0, 1.0, 0.0 ) ),
				-connDir
			);
			float spotAttenuation = getSpotAttenuation( lv4.w, lv3.x, spotCos );
			if ( spotAttenuation <= 0.0 ) return vec3( 0.0 );
			gTerm = spotAttenuation * getDistanceAttenuation( dist, lv4.y, lv4.z );
		}
		if ( gTerm <= 0.0 || ! bdptFiniteFloat( gTerm ) ) return vec3( 0.0 );
		vec3 bdptVisibilityColor;
		if (
				! bdptVisibilityAttenuation(
					visibilityOrigin,
					lightPos,
				eyeState,
				lightIsEndpoint && lv4.x > 0.5,
				hasTargetFace,
				targetFaceIndex,
				bdptVisibilityColor
			)
		) return vec3( 0.0 );

		// eye BSDF x cos(theta) toward the light. bsdfResult writes BSDF*cos to
		// eyeBsdfColor and returns the directional PDF separately.
                bool eyeConnectionDelta = false;
                float eyeConnectionPdf = bsdfPdfResult(
                        eyeWo,
                        connDir,
                        eyeSurf,
                        eyeState.wavelength,
                        eyeConnectionDelta
                );
                if ( eyeConnectionDelta || eyeConnectionPdf <= 0.0 ) {
                        return vec3( 0.0 );
                }
                vec3 eyeBsdfColor;
                float eyeBsdfPdf = bsdfResult(
                        eyeWo,
                        connDir,
                        eyeSurf,
                        eyeState.wavelength,
                        eyeBsdfColor
                );
                if ( eyeBsdfPdf <= 0.0 ) return vec3( 0.0 );
		vec3 eyeBsdfCosTheta = eyeBsdfColor;

                SurfaceRecord lightSurf;
                if ( lightIsMedium ) {
                        if ( lv5.x <= 0.0 || lv7.y < 0.0 ) return vec3( 0.0 );
                        FogMaterial lightFog = readFogMaterialInfo(
                                materials, uint( round( lv7.y ) )
			);
			lightFog.fogVolume = true;
			setFogSurfaceRecord( lightFog, lightSurf );
			} else if ( ! lightIsEndpoint ) {
				if ( ! bdptLoadSurfaceRecord(
					lightMatId, lv4, lv7.zw,
					lightNormal, eyeState.wavelength, lightSurf
				) ) return vec3( 0.0 );
		}

		// Surface bsdfResult returns f*cos; medium bsdfResult returns phase value.
		// Inverse distance squared is applied separately exactly once in both measures.
		vec3 lightBsdfCosTheta = vec3( 1.0 );
			float lightBsdfPdfToEye = 1.0;
			if ( areaEndpoint ) {
				bool twoSidedEndpoint = lv4.y > 0.5;
				if ( ! vitrumFiniteNonZeroVec3( lightNormal ) ) {
					return vec3( 0.0 );
				}
				float signedCosLight = dot(
					vitrumNormalizeVec3( lightNormal, vec3( 0.0 ) ),
					-connDir
				);
				float cosLight = twoSidedEndpoint
					? abs( signedCosLight )
					: max( signedCosLight, 0.0 );
				if ( cosLight <= 0.0 ) return vec3( 0.0 );
				lightBsdfCosTheta = vec3( cosLight );
				lightBsdfPdfToEye = cosLight / ( twoSidedEndpoint ? 2.0 * PI : PI );
		} else if ( pointEndpoint ) {
			lightBsdfPdfToEye = 1.0 / ( 4.0 * PI );
                } else if ( spotEndpoint ) {
                        float spotSolidAngle = 2.0 * PI * ( 1.0 - lv4.w );
                        if ( spotSolidAngle <= 0.0 ) return vec3( 0.0 );
                        lightBsdfPdfToEye = 1.0 / spotSolidAngle;
                } else {
			if ( ! vitrumFiniteNonZeroVec3( lightWoPrev ) ) {
				return vec3( 0.0 );
			}
                        lightWoPrev =
				vitrumNormalizeVec3( lightWoPrev, vec3( 0.0 ) );
                        if ( ! lightIsMedium && dot( lightNormal, -connDir ) <= 0.0 ) {
                                return vec3( 0.0 );
                        }
                        bool lightConnectionDelta = false;
                        float lightConnectionPdf = bsdfPdfResult(
                                lightWoPrev,
                                -connDir,
                                lightSurf,
                                eyeState.wavelength,
                                lightConnectionDelta
                        );
                        if ( lightConnectionDelta || lightConnectionPdf <= 0.0 ) {
                                return vec3( 0.0 );
                        }
                        lightBsdfPdfToEye = bsdfResult(
				lightWoPrev, -connDir, lightSurf, eyeState.wavelength, lightBsdfCosTheta
			);
		}

		if (
			lightBsdfPdfToEye <= 0.0 ||
			! bdptFiniteFloat( lightBsdfPdfToEye )
		) return vec3( 0.0 );
		// ── Full §10.3 MIS weight ────────────────────────────────────────────
		int c = lightVtxIdx;
		int e = eyeDepth;
		int n = c + e + 3;
		int selectedS = c + 1;
		if ( n > BDPT_MAX_MERGED ) return vec3( 0.0 );

		vec3 camPos = ( cameraWorldMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
		vec3 cameraEdge = camPos - eyePos;
		if (
			! bdptFiniteVec3( cameraEdge ) ||
			! vitrumFiniteNonZeroVec3( cameraEdge )
		) return vec3( 0.0 );
		vec3 camNrm =
			vitrumNormalizeVec3( cameraEdge, vec3( 0.0 ) );

		// Connection-induced straddle overrides (PBRT MISWeight remapping).
		vec3 lcToE = -connDir;                       // L_c → E_e
		float fwdEe = lightBsdfPdfToEye;

			vec3 eeToPrev =
				vitrumNormalizeVec3( eyeWo, vec3( 0.0 ) );
			if ( e >= 1 ) {
				vec3 eyePredecessorEdge = bdptEyePos[ e - 1 ] - eyePos;
				if (
					! bdptFiniteVec3( eyePredecessorEdge ) ||
					! vitrumFiniteNonZeroVec3( eyePredecessorEdge )
				) {
					return vec3( 0.0 );
				}
				eeToPrev = vitrumNormalizeVec3(
					eyePredecessorEdge, vec3( 0.0 )
				);
			}
			// At E_0, eyeWo is the actual camera-ray reverse direction. Using the
			// camera transform's centre is wrong for orthographic and thin-lens rays.

                bool alternateDelta = false;
                float revLc = bsdfPdfResult(
                        eeToPrev, connDir, eyeSurf, eyeState.wavelength, alternateDelta
                );
                if ( alternateDelta ) revLc = 0.0;
                float fwdEeMinus = 0.0;
                if ( e >= 1 ) {
                        fwdEeMinus = bsdfPdfResult(
                                connDir, eeToPrev, eyeSurf, eyeState.wavelength, alternateDelta
                        );
                        if ( alternateDelta ) fwdEeMinus = 0.0;
                }
		float revLcMinus = 0.0;
		if ( c >= 1 ) {
			vec4 lcm0 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx - 1, 0 ), 0 );
			vec3 lightPredecessorEdge = lcm0.xyz - lightPos;
			if (
				! bdptFiniteVec3( lightPredecessorEdge ) ||
				! vitrumFiniteNonZeroVec3( lightPredecessorEdge )
			) {
				return vec3( 0.0 );
			}
			vec3 lcToLcMinus = vitrumNormalizeVec3(
				lightPredecessorEdge, vec3( 0.0 )
			);
                        revLcMinus = bsdfPdfResult(
                                lcToE, lcToLcMinus, lightSurf, eyeState.wavelength, alternateDelta
                        );
                        if ( alternateDelta ) revLcMinus = 0.0;
                }

		// pRef cancels in the power-heuristic ratio (scale-invariant); use the
		// area-forward of the selected strategy's connection edge for conditioning.
                    float pRef = 1.0;
                        vec4 root1 = texelFetch( uBdptLightPathTex, ivec2( 0, 1 ), 0 );
                        vec4 root3 = texelFetch( uBdptLightPathTex, ivec2( 0, 3 ), 0 );
                        vec4 root4 = texelFetch( uBdptLightPathTex, ivec2( 0, 4 ), 0 );
                        bool infiniteRoot =
                                root3.w == BDPT_LV_DIRECTIONAL_EMITTER_MATID ||
                                root3.w == BDPT_LV_ENVIRONMENT_EMITTER_MATID;
                        bool infiniteEnvironmentRoot =
                                root3.w == BDPT_LV_ENVIRONMENT_EMITTER_MATID;
                        vec3 infiniteSourceDirection = vec3( 0.0 );
			if ( infiniteRoot ) {
				if (
					! bdptFiniteVec3( root1.xyz ) ||
					! vitrumFiniteNonZeroVec3( root1.xyz )
				) {
					return vec3( 0.0 );
				}
				infiniteSourceDirection = vitrumNormalizeVec3(
					root1.xyz, vec3( 0.0 )
				);
			}
                        float infiniteNeePdf = infiniteRoot ? root4.z : 0.0;
                        float infiniteLaunchPdf = infiniteRoot
                                ? root1.w * root4.y
                                : 0.0;
                        float infiniteEyeEscapePdf = 0.0;
                        bool infiniteEyeEscapeDelta = false;
                        if ( infiniteRoot && c >= 1 ) {
                                vec4 first0 = texelFetch(
                                        uBdptLightPathTex, ivec2( 1, 0 ), 0
                                );
                                vec4 first1 = texelFetch(
                                        uBdptLightPathTex, ivec2( 1, 1 ), 0
                                );
                                vec4 first3 = texelFetch(
                                        uBdptLightPathTex, ivec2( 1, 3 ), 0
                                );
                                vec4 first4 = texelFetch(
                                        uBdptLightPathTex, ivec2( 1, 4 ), 0
                                );
                                vec4 first7 = texelFetch(
                                        uBdptLightPathTex, ivec2( 1, 7 ), 0
                                );
                                bool firstIsMedium =
                                        first3.w == BDPT_LV_MEDIUM_MATID;
                                SurfaceRecord firstSurface;
                                bool firstSurfaceValid = false;
                                if ( firstIsMedium ) {
                                        if ( first7.y >= 0.0 ) {
                                                FogMaterial firstFog = readFogMaterialInfo(
                                                        materials,
                                                        uint( round( first7.y ) )
                                                );
                                                firstFog.fogVolume = true;
                                                setFogSurfaceRecord(
                                                        firstFog, firstSurface
                                                );
                                                firstSurfaceValid = true;
                                        }
                                } else if ( first3.w >= 0.0 ) {
                                        firstSurfaceValid = bdptLoadSurfaceRecord(
                                                first3.w,
                                                first4,
						first7.zw,
                                                first1.xyz,
                                                eyeState.wavelength,
                                                firstSurface
                                        );
                                }
                                if ( firstSurfaceValid ) {
                                        vec3 firstCamerawardPosition = eyePos;
                                        if ( c >= 2 ) {
                                                firstCamerawardPosition = texelFetch(
                                                        uBdptLightPathTex,
                                                        ivec2( 2, 0 ),
                                                        0
                                                ).xyz;
                                        }
					vec3 firstCamerawardEdge =
						firstCamerawardPosition - first0.xyz;
					if (
						bdptFiniteVec3( firstCamerawardEdge ) &&
						vitrumFiniteNonZeroVec3( firstCamerawardEdge )
					) {
						vec3 firstCamerawardDirection =
							vitrumNormalizeVec3(
								firstCamerawardEdge, vec3( 0.0 )
							);
						infiniteEyeEscapePdf = bsdfPdfResult(
							firstCamerawardDirection,
							infiniteSourceDirection,
							firstSurface,
							eyeState.wavelength,
							infiniteEyeEscapeDelta
						);
					}
                                }
                        }
                        float misW = bdptMISWeightFull(
                                c, e, n, selectedS, pRef,
                                infiniteRoot,
                                infiniteEnvironmentRoot,
                                infiniteSourceDirection,
                                infiniteNeePdf,
                                infiniteLaunchPdf,
                                infiniteEyeEscapePdf,
                                infiniteEyeEscapeDelta,
                        fwdEe, fwdEeMinus, revLc, revLcMinus,
			bdptEyePos, bdptEyeNrm, bdptEyePdfFwd, bdptEyePdfRev,
			bdptEyeSpec, bdptEyeMedium,
			camPos, camNrm
		);
		if ( misW <= 0.0 ) return vec3( 0.0 );

		vec3 monochromaticContribution =
			eyeThroughput *
			pathThroughputFromRgb( bdptVisibilityColor, eyeState.wavelength ) *
			pathThroughputFromRgb( lightThroughput, eyeState.wavelength ) *
			pathThroughputFromRgb( lightBsdfCosTheta, eyeState.wavelength ) *
			pathThroughputFromRgb( eyeBsdfCosTheta, eyeState.wavelength ) *
			gTerm * misW;
		vec3 contribution = wavelengthToRGB( eyeState.wavelength, monochromaticContribution, eyeState.wavelengthPdf );
		if ( any( isnan( contribution ) ) || any( isinf( contribution ) ) ) {
			return vec3( 0.0 );
		}
		return contribution;

	}

`;
