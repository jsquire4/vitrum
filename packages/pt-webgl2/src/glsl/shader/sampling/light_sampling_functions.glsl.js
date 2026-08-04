export const light_sampling_functions = /* glsl */`

        float finitePositiveLightPower( float power ) {

                return power > 0.0 && ! isnan( power ) && ! isinf( power )
                        ? power
                        : 0.0;

        }

	float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {

		if (
			isnan( coneCosine ) || isinf( coneCosine ) ||
			isnan( penumbraCosine ) || isinf( penumbraCosine ) ||
			isnan( angleCosine ) || isinf( angleCosine ) ||
			penumbraCosine < coneCosine
		) return 0.0;
		if ( penumbraCosine == coneCosine ) {

			return angleCosine >= coneCosine ? 1.0 : 0.0;

		}
		return smoothstep( coneCosine, penumbraCosine, angleCosine );

	}

	float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {

			// KHR_lights_punctual range window:
			//   clamp(1 - (distance / range)^4, 0, 1)
			// https://registry.khronos.org/glTF/extensions/2.0/Khronos/KHR_lights_punctual/
		if (
			! ( lightDistance > 0.0 ) ||
			isnan( lightDistance ) || isinf( lightDistance )
		) return 0.0;
		float attenuationDenominator = pow( lightDistance, decayExponent );
		if (
			! ( attenuationDenominator > 0.0 ) ||
			isnan( attenuationDenominator ) || isinf( attenuationDenominator )
		) return 0.0;
		float distanceFalloff = 1.0 / attenuationDenominator;

		if ( cutoffDistance > 0.0 ) {

				distanceFalloff *= saturate( 1.0 - pow4( lightDistance / cutoffDistance ) );

		}

		return distanceFalloff > 0.0 &&
			! isnan( distanceFalloff ) && ! isinf( distanceFalloff )
			? distanceFalloff
			: 0.0;

	}

	vec3 finitePunctualRadiance(
		vec3 color,
		float intensity,
		float distanceAttenuation,
		float angularAttenuation
	) {

		if (
			any( lessThan( color, vec3( 0.0 ) ) ) ||
			any( isnan( color ) ) || any( isinf( color ) ) ||
			! ( intensity >= 0.0 ) ||
			! ( distanceAttenuation >= 0.0 ) ||
			! ( angularAttenuation >= 0.0 ) ||
			isnan( intensity ) || isinf( intensity ) ||
			isnan( distanceAttenuation ) || isinf( distanceAttenuation ) ||
			isnan( angularAttenuation ) || isinf( angularAttenuation )
		) return vec3( 0.0 );

		vec3 sourceRadiance = color * intensity;
		if (
			any( isnan( sourceRadiance ) ) ||
			any( isinf( sourceRadiance ) )
		) return vec3( 0.0 );

		vec3 realizedRadiance =
			sourceRadiance * distanceAttenuation * angularAttenuation;
		return (
			any( lessThan( realizedRadiance, vec3( 0.0 ) ) ) ||
			any( isnan( realizedRadiance ) ) ||
			any( isinf( realizedRadiance ) )
		) ? vec3( 0.0 ) : realizedRadiance;

	}

	struct LightRecord {

		vec3 point;
		vec3 normal;
		float dist;
		vec3 direction;
		float pdf;
		vec3 emission;
		int type;
		// P(chosen light | NEE chose the discrete lights branch). Uniform sampling uses 1/count.
		float discretePdf;
		// SHADOW-01 — 1.0 when the source emitter set castShadow:false (skip the
		// NEE shadow test for this light); 0.0 default.
		float castShadowDisabled;
		// 1.0 for singular point/spot/hard-directional samples that cannot be
		// reached by BSDF sampling; 0.0 for finite-area/finite-cone samples.
		float delta;
		// Exact triangle identity for finite mesh-light visibility. Analytic and
		// infinite emitters leave this disabled.
		bool hasTargetFace;
		uint targetFaceIndex;

	};

	bool intersectLightAtIndex( sampler2D lights, vec3 rayOrigin, vec3 rayDirection, uint l, inout LightRecord lightRec ) {

		bool didHit = false;
		Light light = readLightInfo( lights, l );

		vec3 u = light.u;
		vec3 v = light.v;
		if (
			light.type != RECT_AREA_LIGHT_TYPE &&
			light.type != CIRC_AREA_LIGHT_TYPE
		) return false;

			// Core analytic area emitters are one-sided along cross(u,v). A forward
			// ray sees the emitting face only when it approaches against the normal.
			VitrumAreaVectorMeasure areaMeasure =
				vitrumMeasureAreaVector(
					u, v,
					light.type == CIRC_AREA_LIGHT_TYPE ? PI / 4.0 : 1.0
				);
		if ( ! areaMeasure.valid ) return false;
		vec3 normal = areaMeasure.normal;
		if ( dot( normal, rayDirection ) < 0.0 ) {

			float dist;

			// MIS / light intersection is not supported for punctual lights.
			if(
				( light.type == RECT_AREA_LIGHT_TYPE && intersectsRectangle( light.position, normal, u, v, rayOrigin, rayDirection, dist ) ) ||
				( light.type == CIRC_AREA_LIGHT_TYPE && intersectsCircle( light.position, normal, u, v, rayOrigin, rayDirection, dist ) )
			) {

				float cosTheta = dot( - rayDirection, normal );
				didHit = true;
				lightRec.dist = dist;
				lightRec.point = rayOrigin + rayDirection * dist;
				lightRec.normal = normal;
				lightRec.pdf = vitrumAreaToSolidAnglePdf(
					dist, cosTheta, areaMeasure
				);
				lightRec.emission = light.color * light.intensity;
				lightRec.direction = rayDirection;
				lightRec.type = light.type;
				lightRec.discretePdf = 1.0;
				lightRec.castShadowDisabled = light.castShadowDisabled;
				lightRec.delta = 0.0;
				lightRec.hasTargetFace = false;
				lightRec.targetFaceIndex = 0u;

			}

		}

		return didHit;

	}

	LightRecord randomAreaLightSample( Light light, vec3 rayOrigin, vec2 ruv ) {

		vec3 randomPos;
		if( light.type == RECT_AREA_LIGHT_TYPE ) {

			// rectangular area light
			randomPos = light.position + light.u * ( ruv.x - 0.5 ) + light.v * ( ruv.y - 0.5 );

		} else if( light.type == CIRC_AREA_LIGHT_TYPE ) {

			// circular area light
			float r = 0.5 * sqrt( ruv.x );
			float theta = ruv.y * 2.0 * PI;
			float x = r * cos( theta );
			float y = r * sin( theta );

			randomPos = light.position + light.u * x + light.v * y;

		}

		vec3 toLight = randomPos - rayOrigin;
		float dist = vitrumLengthVec3( toLight );
		vec3 direction = dist > 0.0 ? toLight / dist : vec3( 0.0 );
			VitrumAreaVectorMeasure areaMeasure =
				vitrumMeasureAreaVector(
					light.u, light.v,
					light.type == CIRC_AREA_LIGHT_TYPE ? PI / 4.0 : 1.0
				);
		vec3 lightNormal = areaMeasure.normal;

		LightRecord lightRec;
		lightRec.type = light.type;
		lightRec.emission = light.color * light.intensity;
		lightRec.dist = dist;
		lightRec.point = randomPos;
		lightRec.normal = lightNormal;
		lightRec.direction = direction;

		float cosLight = dot( lightNormal, - direction );
		lightRec.pdf = vitrumAreaToSolidAnglePdf(
			dist, cosLight, areaMeasure
		);
		lightRec.discretePdf = 1.0;
		lightRec.castShadowDisabled = light.castShadowDisabled;
		lightRec.delta = 0.0;
		lightRec.hasTargetFace = false;
		lightRec.targetFaceIndex = 0u;

		return lightRec;

	}

	LightRecord randomSpotLightSample( Light light, vec3 rayOrigin ) {

		// Core SpotEmitter is a delta-position source. cross(u,v) is its backward
		// axis, so a receiver inside the authored forward cone sees a positive
		// dot(direction-to-source, backwardAxis).
		VitrumAreaVectorMeasure axisMeasure =
			vitrumMeasureAreaVector( light.u, light.v, 1.0 );
		vec3 normal = axisMeasure.normal;
		vec3 toLight = light.position - rayOrigin;
		float dist = vitrumLengthVec3( toLight );
		vec3 direction = dist > 0.0 ? toLight / dist : vec3( 0.0 );
		float cosTheta = dot( direction, normal );

		float spotAttenuation = getSpotAttenuation( light.coneCos, light.penumbraCos, cosTheta );
		float distanceAttenuation = getDistanceAttenuation( dist, light.distance, light.decay );
		LightRecord lightRec;
		lightRec.type = light.type;
		lightRec.dist = dist;
		lightRec.point = light.position;
		lightRec.normal = normal;
		lightRec.direction = direction;
		lightRec.emission = finitePunctualRadiance(
			light.color,
			light.intensity,
			distanceAttenuation,
			spotAttenuation
		);
		lightRec.pdf = axisMeasure.valid && dist > 0.0 ? 1.0 : 0.0;
		lightRec.discretePdf = 1.0;
		lightRec.castShadowDisabled = light.castShadowDisabled;
		lightRec.delta = 1.0;
		lightRec.hasTargetFace = false;
		lightRec.targetFaceIndex = 0u;

		return lightRec;

	}

	// ── B4: mesh-area triangle lights (NEE) ──────────────────────────────────────
	// A triangle light's 6-texel slot (meshAreaLights.ts):
	//   s0 = (v0.xyz, type=TRI_AREA_LIGHT_TYPE=5)
	//   s1 = (radiance.rgb, 0)
	//   s2 = (v1.xyz, 0)
	//   s3 = (v2.xyz, triArea)
	//   s4.r = selectionPower = luminance(radiance) * triArea
	//   s5 = (sourceFaceLo16, castShadowDisabled, materialDoubleSided, sourceFaceHi16)
	// The face id uses two exact 16-bit float words; one f32 integer would lose
	// identity above 2^24.
	bool meshLightSourceFaceWordsValid( vec2 words ) {
		return all( greaterThanEqual( words, vec2( 0.0 ) ) ) &&
			all( lessThanEqual( words, vec2( 65535.0 ) ) ) &&
			all( equal( words, floor( words ) ) );
	}

	uint meshLightSourceFaceIndex( vec2 words ) {
		uvec2 integerWords = uvec2( round( words ) );
		return integerWords.x | ( integerWords.y << 16u );
	}

	struct MeshTriLight {
		vec3 v0;
		vec3 v1;
		vec3 v2;
		vec3 radiance;
		float area;
		float power;
		float proposalPmf;
		float bdptProposalPmf;
		float castShadowDisabled;
		float twoSided;
		vec2 sourceFaceWords;
		uint sourceFaceIndex;
	};

	MeshTriLight readMeshTriLight( sampler2D tex, uint index ) {
		uint i = index * 6u;
		vec4 s0 = texelFetch1D( tex, i + 0u );
		vec4 s1 = texelFetch1D( tex, i + 1u );
		vec4 s2 = texelFetch1D( tex, i + 2u );
		vec4 s3 = texelFetch1D( tex, i + 3u );
		vec4 s4 = texelFetch1D( tex, i + 4u );
		vec4 s5 = texelFetch1D( tex, i + 5u );
		MeshTriLight t;
		t.v0 = s0.xyz;
		t.radiance = s1.rgb;
		t.v1 = s2.xyz;
		t.v2 = s3.xyz;
		t.area = s3.a;
		t.power = s4.r;
		t.proposalPmf = max( s4.g, 0.0 );
		t.bdptProposalPmf = max( s4.b, 0.0 );
		t.castShadowDisabled = s5.g;
		t.twoSided = s5.b;
		t.sourceFaceWords = s5.ra;
		t.sourceFaceIndex = meshLightSourceFaceIndex( t.sourceFaceWords );
		return t;
	}

	// Sample a point uniformly on the chosen emissive triangle. The host converts
	// physical f64 weights into the exact represented PMF stored in s4.g.
	LightRecord sampleMeshAreaLight(
		sampler2D meshLights, uint meshLightCount, vec3 rayOrigin, vec3 ruv
	) {
		LightRecord rec;
		rec.point = vec3( 0.0 );
		rec.normal = vec3( 0.0 );
		rec.dist = 0.0;
		rec.direction = vec3( 0.0 );
		rec.pdf = 0.0;
		rec.emission = vec3( 0.0 );
		rec.discretePdf = 1.0;
		rec.type = TRI_AREA_LIGHT_TYPE;
		rec.castShadowDisabled = 0.0;
		rec.delta = 0.0;
		rec.hasTargetFace = false;
		rec.targetFaceIndex = 0u;
		if ( meshLightCount == 0u ) return rec;

		// The host stores an exact 24-bit represented PMF in s4.g.
		float uPick = ruv.x;
		uint chosen = 0u;
		bool foundPositive = false;
		float cum = 0.0;
		for ( uint ii = 0u; ii < meshLightCount; ii ++ ) {
			float representedPmf = readMeshTriLight( meshLights, ii ).proposalPmf;
			if ( representedPmf <= 0.0 ) continue;
			foundPositive = true;
			chosen = ii;
			cum += representedPmf;
			if ( uPick < cum ) break;
		}
		if ( ! foundPositive ) return rec;

                MeshTriLight tri = readMeshTriLight( meshLights, chosen );
                rec.castShadowDisabled = tri.castShadowDisabled;
                if ( tri.proposalPmf <= 0.0 || tri.area <= 0.0 ) return rec;
		rec.hasTargetFace = true;
		rec.targetFaceIndex = tri.sourceFaceIndex;

		// Uniform-area barycentric sample on the chosen triangle.
		float su = sqrt( max( ruv.y, 0.0 ) );
		float b0 = 1.0 - su;
		float b1 = ruv.z * su;
		float b2 = 1.0 - b0 - b1;
		vec3 pos = tri.v0 * b0 + tri.v1 * b1 + tri.v2 * b2;
		VitrumAreaVectorMeasure areaMeasure = vitrumMeasureAreaVector(
			tri.v1 - tri.v0, tri.v2 - tri.v0, 0.5
		);
		if ( ! areaMeasure.valid ) return rec;
		vec3 triNormal = areaMeasure.normal;

		vec3 toLight = pos - rayOrigin;
		float dist = vitrumLengthVec3( toLight );
		if ( ! ( dist > 0.0 ) ) return rec;
		vec3 direction = toLight / dist;

		// Material-owned sidedness: only double-sided surfaces may emit toward
		// a receiver behind their authored winding.
                float cosLight = dot( triNormal, -direction );
                if ( tri.twoSided > 0.5 && cosLight < 0.0 ) {
			triNormal = -triNormal;
			cosLight = -cosLight;
		}
                if ( cosLight <= 0.0 ) return rec;

		rec.point = pos;
		rec.normal = triNormal;
		rec.dist = dist;
		rec.direction = direction;
		rec.emission = tri.radiance;
		float selectionPdf = tri.proposalPmf;
		rec.pdf = selectionPdf * vitrumAreaToSolidAnglePdf(
			dist, cosLight, areaMeasure
		);
		return rec;
	}

	bool meshTriangleContainsPoint( MeshTriLight tri, vec3 point ) {
		vec3 edge0 = tri.v1 - tri.v0;
		vec3 edge1 = tri.v2 - tri.v0;
		vec3 relative = point - tri.v0;
		float d00 = dot( edge0, edge0 );
		float d01 = dot( edge0, edge1 );
		float d11 = dot( edge1, edge1 );
		float d20 = dot( relative, edge0 );
		float d21 = dot( relative, edge1 );
		float denominator = d00 * d11 - d01 * d01;
		if ( ! ( denominator > 0.0 ) || isnan( denominator ) || isinf( denominator ) ) {
			return false;
		}
		float b1 = ( d11 * d20 - d01 * d21 ) / denominator;
		float b2 = ( d00 * d21 - d01 * d20 ) / denominator;
		float b0 = 1.0 - b1 - b2;
		return b0 >= 0.0 && b1 >= 0.0 && b2 >= 0.0;
	}

	// SOLID-ANGLE NEE pdf of a forward emissive hit under the represented mesh
	// proposal. Face identity plus point containment recovers the exact texel-cell
	// sub-triangle PMF used by sampleMeshAreaLight.
	float meshAreaLightForwardPdf(
		sampler2D meshLights, uint meshLightCount, uint sourceFaceIndex,
		vec3 hitPoint, float distance, float cosLight
	) {
		float cosine = abs( cosLight );
		if (
			meshLightCount == 0u || ! ( distance > 0.0 ) || cosine <= 0.0 ||
			isnan( distance ) || isinf( distance )
		) return 0.0;
		float selectionPdf = 0.0;
		float selectedArea = 0.0;
		for ( uint i = 0u; i < meshLightCount; i ++ ) {
			MeshTriLight tri = readMeshTriLight( meshLights, i );
			if (
				tri.sourceFaceIndex == sourceFaceIndex && tri.proposalPmf > 0.0 &&
				tri.area > 0.0 && meshTriangleContainsPoint( tri, hitPoint )
			) {
				selectionPdf = tri.proposalPmf;
				selectedArea = tri.area;
				break;
			}
		}
		if ( selectionPdf <= 0.0 || selectedArea <= 0.0 ) return 0.0;
		// Evaluate the area-to-solid-angle density in log space so neither
		// distance² nor a compensating tiny represented PMF can overflow early.
		float logPdf = log2( selectionPdf ) - log2( selectedArea ) +
			2.0 * log2( distance ) -
			log2( cosine );
		float result = exp2( logPdf );
		return result > 0.0 && ! isnan( result ) && ! isinf( result )
			? result
			: 0.0;
	}

	vec3 sampleDirectionalCone( vec3 axis, float angularDiameter, vec2 uv, out float pdf ) {

		// Avoid subtracting two values near one. For a cone half-angle d/2:
		//   1 - cos(d/2) = 2 sin²(d/4)
		// The host rejects positive diameters whose resulting term is below the
		// normal-f32 range, so every accepted finite cone has a finite PDF.
		float quarterAngle = angularDiameter * 0.25;
		float sinQuarter = quarterAngle < 1e-3
			? quarterAngle
			: sin( quarterAngle );
		float sinQuarterSquared = sinQuarter * sinQuarter;
		float oneMinusCosHalf = 2.0 * sinQuarterSquared;
		// Preserve the inherited uv.x convention: zero samples the rim and one
		// samples the axis. Even when cosTheta rounds to one, sinTheta retains
		// the representable transverse angular displacement.
		float oneMinusCosTheta =
			( 1.0 - clamp( uv.x, 0.0, 1.0 ) ) * oneMinusCosHalf;
		float cosTheta = 1.0 - oneMinusCosTheta;
		float sinTheta = sqrt(
			max( 0.0, oneMinusCosTheta * ( 2.0 - oneMinusCosTheta ) )
		);
		float phi = 2.0 * PI * uv.y;
		vec3 localDir = vec3( cos( phi ) * sinTheta, sin( phi ) * sinTheta, cosTheta );
		float solidAngle = 2.0 * PI * oneMinusCosHalf;
		pdf = solidAngle > 0.0 ? 1.0 / solidAngle : 0.0;
		if ( isnan( pdf ) || isinf( pdf ) ) pdf = 0.0;
		return normalize( getBasisFromNormal( normalize( axis ) ) * localDir );

	}

	LightRecord randomLightSample( sampler2D lights, uint lightCount, vec3 rayOrigin, vec3 ruv ) {

		LightRecord result;

		uint l = 0u;
                float discretePdf = 0.0;

                if ( lightCount > 0u ) {

				float uPick = ruv.x;
                        float cum = 0.0;
                        for ( uint ii = 0u; ii < lightCount; ii ++ ) {

                                Light tmpLight = readLightInfo( lights, ii );
						float w = tmpLight.proposalPmf;
                                if ( w <= 0.0 ) continue;
                                // Also serves as the exact final-bin fallback
                                // if a generator ever returns u == 1.
                                l = ii;
						discretePdf = w;
						cum += w;
						if ( uPick < cum ) {
                                        break;

                                }

                        }

                }

		Light light = readLightInfo( lights, l );

		if ( light.type == SPOT_LIGHT_TYPE ) {

			result = randomSpotLightSample( light, rayOrigin );

		} else if ( light.type == POINT_LIGHT_TYPE ) {

			vec3 lightRay = light.u - rayOrigin;
			float lightDist = vitrumLengthVec3( lightRay );
			float cutoffDistance = light.distance;
                        float distanceFalloff = getDistanceAttenuation(
                                lightDist,
                                cutoffDistance,
                                light.decay
                        );

			LightRecord rec;
			rec.point = light.u;
			rec.direction =
				lightDist > 0.0 ? lightRay / lightDist : vec3( 0.0 );
			rec.dist = lightDist;
			rec.normal = - rec.direction;
			rec.pdf = lightDist > 0.0 ? 1.0 : 0.0;
			rec.emission = finitePunctualRadiance(
				light.color,
				light.intensity,
				distanceFalloff,
				1.0
			);
			rec.type = light.type;
			rec.discretePdf = 1.0;
			rec.castShadowDisabled = light.castShadowDisabled;
			rec.delta = 1.0;
			rec.hasTargetFace = false;
			rec.targetFaceIndex = 0u;
			result = rec;

		} else if ( light.type == DIR_LIGHT_TYPE ) {

			LightRecord rec;
			rec.dist = INFINITY;
			if ( light.angularDiameter > 0.0 ) {
				float conePdf;
				rec.direction = sampleDirectionalCone( light.u, light.angularDiameter, ruv.yz, conePdf );
				rec.pdf = conePdf;
				rec.delta = 0.0;
			} else {
				rec.direction = light.u;
				rec.pdf = 1.0;
				rec.delta = 1.0;
			}
			// Infinite/directional samples never own a finite endpoint. INFINITY
			// is the renderer's max-f32 sentinel, so multiplying it by a basis
			// direction can overflow and must not manufacture a position.
			rec.point = vec3( 0.0 );
			rec.normal = rec.direction;
			rec.emission = light.color * light.intensity;
			rec.type = light.type;
			rec.discretePdf = 1.0;
			rec.castShadowDisabled = light.castShadowDisabled;
			rec.hasTargetFace = false;
			rec.targetFaceIndex = 0u;

			result = rec;

		} else {

			// sample the light
			result = randomAreaLightSample( light, rayOrigin, ruv.yz );

		}

		result.discretePdf = discretePdf;
		return result;

	}

`;
