/**
 * bdpt_light_subpath.glsl.js — BDPT light-subpath kernel (Sprint 10c).
 *
 * Included when `FEATURE_BDPT == 1`. The host light-subpath draw uses
 * `uBdptLightSubpathPass` in PhysicalPathTracingMaterial (one fullscreen draw
 * per vertex column) via PathTracingRenderer.renderBdptLightSubpathPass().
 *
 * Ping-pong vertex texture layout (RGBA32F, width=BDPT_MAX_LIGHT_BOUNCES=3, height=5):
 *   Texel(col, 0):  position.xyz | kind    (0=light vertex, 3=invalid/empty)
 *   Texel(col, 1):  normal.xyz   | pdfFwd  (forward PDF, SOLID-ANGLE measure)
 *   Texel(col, 2):  throughput.rgb | pdfRev (radiance weight; reverse SA PDF)
 *   Texel(col, 3):  woTowardPrev.xyz | materialId (-1=emitter profile)
 *   Texel(col, 4):  triangleIndex | barycentric.xy | side
 *
 * Each draw call renders into a single 5-row × N-column render target at column
 * uBdptVertexCol (0…BDPT_MAX_LIGHT_BOUNCES-1). Five fragments (one per row) at
 * the same column cooperate: all five trace the SAME subpath (RNG seeded with
 * vec2(gl_FragCoord.x, 0.0) — row-independent) and each writes one row of the
 * vertex. The host ping-pongs: "write" target = current frame's texture; "read"
 * target (uBdptLightPathTex) = previous frame's texture. For bounce k=0 the read
 * texture is irrelevant (emitter vertex; no prior bounce).
 *
 * pdfFwd / pdfRev are stored in SOLID-ANGLE measure with NO baked-in geometry
 * term: the full Veach §10.3 connection sweep (bdpt_connection.glsl.js) converts
 * SA→area on the fly via ConvertDensity (PBRT Vertex::ConvertDensity), so baking
 * G here would double-apply the Jacobian. Emitter vertices use the cosine-emission
 * density; surface vertices use the same BSDF pdf reported by bsdfSample().
 *
 * Geometry term: G(x↔y) = |cosθ_x · cosθ_y| / ‖x−y‖²  (Veach §8.3.2, Eq. 8.10),
 * still used for visibility/throughput bookkeeping in the connection pass.
 *
 * Throughput model:
 *   T_0 = Le × cosθ_emit / (p_light × p_hemisphere)
 *   T_k = T_{k-1} × f_k(wo,wi)·cosθ / p_bsdf
 * Surface light vertices now use the same getSurfaceRecord + bsdfSample path as
 * the eye path, so maps/layer lobes participate in the generated light chain.
 *
 * Seed isolation: eye-path rand() uses seeds 0–30 (established by prior sprints).
 *   Light subpath bounce 0 uses seeds 50–52.
 *   Light subpath bounce k uses seeds 53 + k*3 … 55 + k*3.
 *
 * References:
 *   Veach 1997, §10.3 (BDPT), §8.3.2 (geometric term).
 *   Pharr et al. 2023, PBR 4e §16.3 (vertex formulation).
 */
export const bdpt_light_subpath = /* glsl */`

	// ── Geometry term G(x↔y) ────────────────────────────────────────────────
	// Returns 0 on degenerate connections (coincident points or near-tangent
	// incidence — both produce near-zero or negative cosines).
	float bdptGeometricTerm( vec3 posX, vec3 nX, vec3 posY, vec3 nY ) {
		vec3 d    = posY - posX;
		float dist2 = dot( d, d );
		if ( dist2 <= 1e-12 ) return 0.0;
		vec3 w    = d * inversesqrt( dist2 );
		float cosX = abs( dot( nX, w ) );
		float cosY = abs( dot( nY, -w ) ); // opposite direction
		return ( cosX * cosY ) / dist2;
	}

	const float BDPT_LV_EMITTER_MATID = -1.0;

	vec4 bdptSurfacePayload( SurfaceHit hit ) {
		return vec4(
			float( hit.faceIndices.w ),
			hit.barycoord.x,
			hit.barycoord.y,
			hit.side
		);
	}

	bool bdptLoadSurfaceRecord(
		float materialId,
		vec4 payload,
		vec3 fallbackFaceNormal,
		float heroWavelength,
		out SurfaceRecord surf
	) {
		uint triIndex = uint( max( floor( payload.x + 0.5 ), 0.0 ) );
		uvec3 indices = uTexelFetch1D( bvh.index, triIndex ).xyz;
		SurfaceHit hit;
		hit.faceIndices = uvec4( indices, triIndex );
		hit.barycoord = vec3( payload.y, payload.z, max( 0.0, 1.0 - payload.y - payload.z ) );
		hit.faceNormal = fallbackFaceNormal;
		hit.side = payload.w < 0.0 ? -1.0 : 1.0;
		hit.dist = 0.0;

		uint matIdx = uint( max( floor( materialId + 0.5 ), 0.0 ) );
		Material mat = readMaterialInfo( materials, matIdx );
		return getSurfaceRecord( mat, matIdx, hit, attributesArray, 0.0, 0, heroWavelength, surf ) == HIT_SURFACE;
	}

	// ── Write a fully invalid (empty) vertex ─────────────────────────────────
	// Called when sampling fails or the subpath terminates early.
	// kind = 3.0 = BDPT_KIND_INVALID — the connection pass skips these.
	void writeBdptInvalidVertex(
		out vec4 v0, out vec4 v1, out vec4 v2, out vec4 v3, out vec4 v4
	) {
		v0 = vec4( 0.0, 0.0, 0.0, 3.0 ); // kind = BDPT_KIND_INVALID
		v1 = vec4( 0.0 );
		v2 = vec4( 0.0 );
		v3 = vec4( 0.0, 0.0, 0.0, BDPT_LV_EMITTER_MATID );
		v4 = vec4( 0.0 );
	}

	// ── Main light-subpath vertex writer ─────────────────────────────────────
	// Writes one vertex per call; called from the BDPT light-subpath pass main().
	//
	// Parameters:
	//   vertexCol         — bounce index (0 = emitter vertex).
	//   maxLightBounces   — BDPT_MAX_LIGHT_BOUNCES uniform value.
	//   lightPathTex      — ping-pong texture (read = previous frame's texture).
	//   fogMat            — current fog material state (from host uniform).
	//
	// Outputs: writes to gBdptVertex0/1/2/3 light-path row layout.
	void writeLightSubpathVertex(
		int vertexCol,
		int maxLightBounces,
		sampler2D lightPathTex,
		Material fogMat,
		out vec4 gBdptVertex0,
		out vec4 gBdptVertex1,
		out vec4 gBdptVertex2,
		out vec4 gBdptVertex3,
		out vec4 gBdptVertex4
	) {

		// Bounds guard.
		if ( vertexCol < 0 || vertexCol >= maxLightBounces || lights.count == 0u ) {
			writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2, gBdptVertex3, gBdptVertex4 );
			return;
		}

		if ( vertexCol == 0 ) {

			// ── Bounce 0: sample emitter surface ─────────────────────────────
			// Pick a random area light / emitter via the existing light sampling CDF.
			// Use seeds 50–52 (isolated from eye-path seeds 0–30).
			LightRecord lightRec = randomLightSample(
				lights.tex, lights.count,
				vec3( 0.0 ),   // origin is irrelevant for emitter-surface sampling
				rand3( 50 )
			);

			if ( lightRec.pdf <= 0.0 || lightRec.emission == vec3( 0.0 ) ) {
				writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2, gBdptVertex3, gBdptVertex4 );
				return;
			}

			// Emitter position and normal from the light sampler.
			// Area lights use the sampled surface point and geometric normal;
			// punctual/directional lights provide synthetic stable vertices.
			vec3 emitPos    = lightRec.point;
			vec3 emitNormal = normalize( lightRec.normal );

			// Cosine-weighted hemisphere scatter direction from the emitter surface.
			// This gives the first scattered ray direction from the light.
			// Seed 51 (isolated from bounce k>0 seeds 53+).
			vec3 scatterDir = sampleHemisphere( emitNormal, rand2( 51 ) );
			float cosEmit   = max( dot( emitNormal, scatterDir ), 0.0 );
			float pdfHemi   = cosEmit / PI; // cosine-weighted hemisphere PDF = cosθ/π

			// Joint PDF = p_light × p_hemisphere.
			float pdfJoint = lightRec.pdf * pdfHemi;
			if ( pdfJoint <= 0.0 ) {
				writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2, gBdptVertex3, gBdptVertex4 );
				return;
			}

			// Throughput at emitter: Le × cosθ / pdfJoint.
			vec3 emitThroughput = lightRec.emission * cosEmit / pdfJoint;

			// pdfFwd = joint PDF of choosing this emitter surface point + direction.
			float pdfFwd = pdfJoint;
			// pdfRev: approximated as the cosine-hemisphere PDF for the reverse direction.
			float pdfRev = pdfHemi;

			gBdptVertex0 = vec4( emitPos,        0.0 );    // kind = BDPT_KIND_LIGHT
			gBdptVertex1 = vec4( emitNormal,     pdfFwd );
			gBdptVertex2 = vec4( emitThroughput, pdfRev );
			gBdptVertex3 = vec4( emitNormal, BDPT_LV_EMITTER_MATID );
			gBdptVertex4 = vec4( lightRec.castShadowDisabled, 0.0, 0.0, 0.0 );

		} else {

			// ── Bounce k>0: read prior vertex, extend subpath ─────────────────
			// Read prior vertex from the ping-pong "read" texture.
			// The "read" texture holds the previous frame's or the prior-bounce result.
			// Host must ensure: write target ≠ read source (WebGL2 requirement).
			int prevCol = vertexCol - 1;
			vec4 v0prev = texelFetch( lightPathTex, ivec2( prevCol, 0 ), 0 );
			vec4 v1prev = texelFetch( lightPathTex, ivec2( prevCol, 1 ), 0 );
			vec4 v2prev = texelFetch( lightPathTex, ivec2( prevCol, 2 ), 0 );
			vec4 v3prev = texelFetch( lightPathTex, ivec2( prevCol, 3 ), 0 );
			vec4 v4prev = texelFetch( lightPathTex, ivec2( prevCol, 4 ), 0 );

			// Check kind — skip if the prior vertex is invalid.
			if ( v0prev.w == 3.0 ) { // BDPT_KIND_INVALID
				writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2, gBdptVertex3, gBdptVertex4 );
				return;
			}

			vec3 prevPos        = v0prev.xyz;
			vec3 prevNormal     = v1prev.xyz;
			// v1prev.w = prevPdfFwd — not needed for scatter direction or throughput update.
			vec3 prevThroughput = v2prev.xyz;
			// v2prev.w = prevPdfRev — not needed for scatter; pdfRev is recomputed at this vertex.
			vec3 woAtPrev       = normalize( v3prev.xyz );
			float prevMatId     = v3prev.w;

			// Seed isolation: 53 + vertexCol*3 (covers bounces 1, 2).
			int seedBase = 53 + vertexCol * 3;
			vec3 scatterDir;
			float pdfScatter;
			vec3 segmentThroughput;

			if ( prevMatId < 0.0 ) {

				// Emitter profile: cosine-weighted emission about the stored source normal.
				scatterDir = sampleHemisphere( prevNormal, rand2( seedBase ) );
				float cosScatter = max( dot( prevNormal, scatterDir ), 0.0 );
				pdfScatter = cosScatter / PI;
				segmentThroughput = vec3( cosScatter / PI );

			} else {

				// Surface vertex: reuse the same material/texture BSDF sampler as the eye path.
				SurfaceRecord prevSurf;
				if ( ! bdptLoadSurfaceRecord( prevMatId, v4prev, prevNormal, 550.0, prevSurf ) ) {
					writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2, gBdptVertex3, gBdptVertex4 );
					return;
				}
				ScatterRecord scatterRec = bsdfSample( woAtPrev, prevSurf, 550.0 );
				scatterDir = scatterRec.direction;
				pdfScatter = scatterRec.pdf;
				segmentThroughput = scatterRec.throughput;

			}

			if ( pdfScatter <= 0.0 ) {
				writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2, gBdptVertex3, gBdptVertex4 );
				return;
			}

			// Trace ray from prior vertex into the scene.
			Ray scatterRay;
			scatterRay.origin    = prevPos + prevNormal * RAY_OFFSET;
			scatterRay.direction = scatterDir;

			SurfaceHit scatterHit;
			int hitType = traceScene( scatterRay, fogMat, scatterHit );

			if ( hitType != SURFACE_HIT ) {
				writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2, gBdptVertex3, gBdptVertex4 );
				return;
			}

			// Fetch material at the new hit.
			uint matIdx  = uTexelFetch1D( materialIndexAttribute, scatterHit.faceIndices.w ).r;
			Material mat = readMaterialInfo( materials, matIdx );
			SurfaceRecord newSurf;
			if ( getSurfaceRecord( mat, matIdx, scatterHit, attributesArray, 0.0, vertexCol, 550.0, newSurf ) != HIT_SURFACE ) {
				writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2, gBdptVertex3, gBdptVertex4 );
				return;
			}

			// Skip specular / delta-BSDF surfaces — MIS weight would be zero for
			// explicit connections through them (Veach §10.3.5).
			bool isSpecular = ( newSurf.transmission > 0.5 && newSurf.filteredRoughness < 0.05 );
			if ( isSpecular ) {
				writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2, gBdptVertex3, gBdptVertex4 );
				return;
			}

			// New vertex geometry.
			vec3 newPos    = scatterRay.origin + scatterRay.direction * scatterHit.dist;
			vec3 newNormal = normalize( newSurf.normal );

			// Throughput: prior × f(prev, scatterDir)·cos / pdfScatter. For real
			// surface vertices segmentThroughput comes from bsdfSample(); emitter
			// vertices keep the cosine emission profile used at bounce 0.
			vec3 newThroughput = prevThroughput * segmentThroughput / pdfScatter;

			// Store SOLID-ANGLE pdfs (NO baked-in geometry term). The full Veach
			// §10.3 connection sweep converts SA→area on the fly via ConvertDensity
			// (PBRT Vertex::ConvertDensity, destination-cosine only), so baking the
			// full G here would double-apply the Jacobian and bias the MIS weights.
			float pdfFwd = pdfScatter;                                  // SA forward generation density
			vec3 woToPrev = normalize( prevPos - newPos );
			float cosRev = max( dot( newNormal, woToPrev ), 0.0 );
			float pdfRev = cosRev / PI;                                 // SA reverse (cosθ/π)

			gBdptVertex0 = vec4( newPos,        0.0 );   // kind = BDPT_KIND_LIGHT
			gBdptVertex1 = vec4( newNormal,     pdfFwd );
			gBdptVertex2 = vec4( newThroughput, pdfRev );
			gBdptVertex3 = vec4( woToPrev, float( matIdx ) );
			gBdptVertex4 = bdptSurfacePayload( scatterHit );

		}

	}

`;
