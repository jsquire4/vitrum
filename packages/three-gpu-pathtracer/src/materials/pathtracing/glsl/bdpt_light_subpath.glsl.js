/**
 * bdpt_light_subpath.glsl.js — BDPT light-subpath kernel (Sprint 10c).
 *
 * This GLSL block is included via `#ifdef BDPT_LIGHT_SUBPATH_PASS` into the
 * light-subpath draw pass — a separate fullscreen-quad draw call that the host
 * issues BDPT_MAX_LIGHT_BOUNCES times (once per bounce) before the main eye-ray
 * accumulation pass.
 *
 * Ping-pong vertex texture layout (RGBA32F, width=BDPT_MAX_LIGHT_BOUNCES=3, height=3):
 *   Texel(col, 0):  position.xyz | kind    (0=light vertex, 3=invalid/empty)
 *   Texel(col, 1):  normal.xyz   | pdfFwd  (forward PDF in area measure × cosθ)
 *   Texel(col, 2):  throughput.rgb | pdfRev (accumulated radiance weight; reverse PDF)
 *
 * Each draw call renders into a 3-attachment MRT target at column uBdptVertexCol
 * (0…BDPT_MAX_LIGHT_BOUNCES-1). The host ping-pongs: "write" target = current
 * frame's texture; "read" target (uBdptLightPathTex) = previous frame's texture.
 * For bounce k=0 the read texture is irrelevant (emitter vertex; no prior bounce).
 *
 * Geometry term: G(x↔y) = |cosθ_x · cosθ_y| / ‖x−y‖²  (Veach §8.3.2, Eq. 8.10).
 *
 * Throughput model (approximation — see Risk §4 in sprint-10c-pt-fork-patch.md):
 *   T_0 = Le × cosθ_emit / (p_light × p_hemisphere)
 *   T_k = T_{k-1} × albedo_k × cosθ_k / p_hemisphere_k
 * Full BSDF evaluation is deferred to the connection pass; the light subpath uses
 * Lambertian approximation (cosine hemisphere) to keep per-bounce cost bounded.
 *
 * pdfRev approximation: symmetric Lambertian model — pdfRev = cosθ_rev / π.
 * This is an intentional approximation documented in the sprint spec (Risk §4).
 * It will bias the MIS weight slightly but is visually acceptable for caustic
 * convergence verification. Track as known gap in IMPLEMENTATION-STATUS.md.
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

	// ── Write a fully invalid (empty) vertex ─────────────────────────────────
	// Called when sampling fails or the subpath terminates early.
	// kind = 3.0 = BDPT_KIND_INVALID — the connection pass skips these.
	void writeBdptInvalidVertex(
		out vec4 v0, out vec4 v1, out vec4 v2
	) {
		v0 = vec4( 0.0, 0.0, 0.0, 3.0 ); // kind = BDPT_KIND_INVALID
		v1 = vec4( 0.0 );
		v2 = vec4( 0.0 );
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
	// Outputs: writes to gBdptVertex0/1/2 MRT layout.
	void writeLightSubpathVertex(
		int vertexCol,
		int maxLightBounces,
		sampler2D lightPathTex,
		Material fogMat,
		out vec4 gBdptVertex0,
		out vec4 gBdptVertex1,
		out vec4 gBdptVertex2
	) {

		// Bounds guard.
		if ( vertexCol < 0 || vertexCol >= maxLightBounces || lights.count == 0u ) {
			writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2 );
			return;
		}

		if ( vertexCol == 0 ) {

			// ── Bounce 0: sample emitter surface ─────────────────────────────
			// Pick a random area light / emitter via the existing light sampling CDF.
			// Use seeds 50–52 (isolated from eye-path seeds 0–30).
			LightRecord lightRec = randomLightSample(
				lights.tex, iesProfiles, lights.count,
				vec3( 0.0 ),   // origin is irrelevant for emitter-surface sampling
				rand3( 50 )
			);

			if ( lightRec.pdf <= 0.0 || lightRec.emission == vec3( 0.0 ) ) {
				writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2 );
				return;
			}

			// Emitter position and normal.
			// lightRec.point = surface point on the emitter.
			// lightRec.direction = direction FROM receiver TO emitter (world space).
			// Emitter normal ≈ -lightRec.direction (emission is toward the scene).
			vec3 emitPos    = lightRec.point;
			vec3 emitNormal = normalize( -lightRec.direction );

			// Cosine-weighted hemisphere scatter direction from the emitter surface.
			// This gives the first scattered ray direction from the light.
			// Seed 51 (isolated from bounce k>0 seeds 53+).
			vec3 scatterDir = sampleHemisphere( emitNormal, rand2( 51 ) );
			float cosEmit   = max( dot( emitNormal, scatterDir ), 0.0 );
			float pdfHemi   = cosEmit / PI; // cosine-weighted hemisphere PDF = cosθ/π

			// Joint PDF = p_light × p_hemisphere.
			float pdfJoint = lightRec.pdf * pdfHemi;
			if ( pdfJoint <= 0.0 ) {
				writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2 );
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

		} else {

			// ── Bounce k>0: read prior vertex, extend subpath ─────────────────
			// Read prior vertex from the ping-pong "read" texture.
			// The "read" texture holds the previous frame's or the prior-bounce result.
			// Host must ensure: write target ≠ read source (WebGL2 requirement).
			int prevCol = vertexCol - 1;
			vec4 v0prev = texelFetch( lightPathTex, ivec2( prevCol, 0 ), 0 );
			vec4 v1prev = texelFetch( lightPathTex, ivec2( prevCol, 1 ), 0 );
			vec4 v2prev = texelFetch( lightPathTex, ivec2( prevCol, 2 ), 0 );

			// Check kind — skip if the prior vertex is invalid.
			if ( v0prev.w == 3.0 ) { // BDPT_KIND_INVALID
				writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2 );
				return;
			}

			vec3 prevPos        = v0prev.xyz;
			vec3 prevNormal     = v1prev.xyz;
			// v1prev.w = prevPdfFwd — not needed for scatter direction or throughput update.
			vec3 prevThroughput = v2prev.xyz;
			// v2prev.w = prevPdfRev — not needed for scatter; pdfRev is recomputed at this vertex.

			// Scatter from the prior vertex using cosine-weighted hemisphere.
			// Seed isolation: 53 + vertexCol*3 (covers bounces 1, 2).
			int seedBase    = 53 + vertexCol * 3;
			vec3 scatterDir = sampleHemisphere( prevNormal, rand2( seedBase ) );
			float cosScatter = max( dot( prevNormal, scatterDir ), 0.0 );
			float pdfScatter = cosScatter / PI;

			if ( pdfScatter <= 0.0 ) {
				writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2 );
				return;
			}

			// Trace ray from prior vertex into the scene.
			Ray scatterRay;
			scatterRay.origin    = prevPos + prevNormal * RAY_OFFSET;
			scatterRay.direction = scatterDir;

			SurfaceHit scatterHit;
			int hitType = traceScene( scatterRay, fogMat, scatterHit );

			if ( hitType != SURFACE_HIT ) {
				writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2 );
				return;
			}

			// Fetch material at the new hit.
			uint matIdx  = uTexelFetch1D( materialIndexAttribute, scatterHit.faceIndices.x ).r;
			Material mat = readMaterialInfo( materials, matIdx );

			// Skip specular / delta-BSDF surfaces — MIS weight would be zero for
			// explicit connections through them (Veach §10.3.5).
			bool isSpecular = ( mat.transmission > 0.5 && mat.roughness < 0.05 );
			if ( isSpecular ) {
				writeBdptInvalidVertex( gBdptVertex0, gBdptVertex1, gBdptVertex2 );
				return;
			}

			// New vertex geometry.
			vec3 newPos    = scatterRay.origin + scatterRay.direction * scatterHit.dist;
			vec3 newNormal = normalize( scatterHit.faceNormal * scatterHit.side );

			// Throughput: prior × albedo × cosθ / pdfScatter.
			// Albedo = mat.color (Lambertian approximation; full BSDF in connection pass).
			vec3 newThroughput = prevThroughput * mat.color * cosScatter / pdfScatter;

			// Geometric term for PDF conversion (area→solid-angle).
			float gTerm = bdptGeometricTerm( prevPos, prevNormal, newPos, newNormal );

			// pdfFwd = pdfScatter (solid-angle) × G(prev↔new).
			float pdfFwd = pdfScatter * max( gTerm, 0.0 );
			// pdfRev: cosine-hemisphere from the new vertex toward the prior vertex.
			float cosRev = max( dot( newNormal, normalize( prevPos - newPos ) ), 0.0 );
			float pdfRevScatter = cosRev / PI;
			float pdfRev = pdfRevScatter * max( gTerm, 0.0 );

			gBdptVertex0 = vec4( newPos,        0.0 );   // kind = BDPT_KIND_LIGHT
			gBdptVertex1 = vec4( newNormal,     pdfFwd );
			gBdptVertex2 = vec4( newThroughput, pdfRev );

		}

	}

`;
