export const light_sampling_functions = /* glsl */`

	float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {

		return smoothstep( coneCosine, penumbraCosine, angleCosine );

	}

	float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {

		// based upon Frostbite 3 Moving to Physically-based Rendering
		// page 32, equation 26: E[window1]
		// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
		float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), EPSILON );

		if ( cutoffDistance > 0.0 ) {

			distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );

		}

		return distanceFalloff;

	}

	float getPhotometricAttenuation( sampler2DArray iesProfiles, int iesProfile, vec3 posToLight, vec3 lightDir, vec3 u, vec3 v ) {

		float cosTheta = dot( posToLight, lightDir );
		float angle = acos( cosTheta ) / PI;

		return texture2D( iesProfiles, vec3( angle, 0.0, iesProfile ) ).r;

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

	};

	bool intersectLightAtIndex( sampler2D lights, vec3 rayOrigin, vec3 rayDirection, uint l, inout LightRecord lightRec ) {

		bool didHit = false;
		Light light = readLightInfo( lights, l );

		vec3 u = light.u;
		vec3 v = light.v;

		// check for backface
		vec3 normal = normalize( cross( u, v ) );
		if ( dot( normal, rayDirection ) > 0.0 ) {

			u *= 1.0 / dot( u, u );
			v *= 1.0 / dot( v, v );

			float dist;

			// MIS / light intersection is not supported for punctual lights.
			if(
				( light.type == RECT_AREA_LIGHT_TYPE && intersectsRectangle( light.position, normal, u, v, rayOrigin, rayDirection, dist ) ) ||
				( light.type == CIRC_AREA_LIGHT_TYPE && intersectsCircle( light.position, normal, u, v, rayOrigin, rayDirection, dist ) )
			) {

				float cosTheta = dot( rayDirection, normal );
				didHit = true;
				lightRec.dist = dist;
				lightRec.point = rayOrigin + rayDirection * dist;
				lightRec.normal = normal;
				// Guard against grazing angles / degenerate area terms causing
				// divide-by-zero or negative PDFs in MIS weights.
				float denom = max( abs( light.area * cosTheta ), EPSILON );
				lightRec.pdf = max( ( dist * dist ) / denom, EPSILON );
				lightRec.emission = light.color * light.intensity;
				lightRec.direction = rayDirection;
				lightRec.type = light.type;
				lightRec.discretePdf = 1.0;

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
		float lightDistSq = dot( toLight, toLight );
		float dist = sqrt( lightDistSq );
		vec3 direction = toLight / dist;
		vec3 lightNormal = normalize( cross( light.u, light.v ) );

		LightRecord lightRec;
		lightRec.type = light.type;
		lightRec.emission = light.color * light.intensity;
		lightRec.dist = dist;
		lightRec.point = randomPos;
		lightRec.normal = lightNormal;
		lightRec.direction = direction;

		// Guard against grazing-angle and zero-area degeneracies so MIS weights
		// never see NaN/Inf PDFs from area-light sampling.
		float denom = max( abs( light.area * dot( direction, lightNormal ) ), EPSILON );
		lightRec.pdf = max( lightDistSq / denom, EPSILON );
		lightRec.discretePdf = 1.0;

		return lightRec;

	}

	LightRecord randomSpotLightSample( Light light, sampler2DArray iesProfiles, vec3 rayOrigin, vec2 ruv ) {

		float radius = light.radius * sqrt( ruv.x );
		float theta = ruv.y * 2.0 * PI;
		float x = radius * cos( theta );
		float y = radius * sin( theta );

		vec3 u = light.u;
		vec3 v = light.v;
		vec3 normal = normalize( cross( u, v ) );

		float angle = acos( light.coneCos );
		float angleTan = tan( angle );
		float startDistance = light.radius / max( angleTan, EPSILON );

		vec3 randomPos = light.position - normal * startDistance + u * x + v * y;
		vec3 toLight = randomPos - rayOrigin;
		float lightDistSq = dot( toLight, toLight );
		float dist = sqrt( lightDistSq );

		vec3 direction = toLight / max( dist, EPSILON );
		float cosTheta = dot( direction, normal );

		float spotAttenuation = light.iesProfile != - 1 ?
			getPhotometricAttenuation( iesProfiles, light.iesProfile, direction, normal, u, v ) :
			getSpotAttenuation( light.coneCos, light.penumbraCos, cosTheta );

		float distanceAttenuation = getDistanceAttenuation( dist, light.distance, light.decay );
		LightRecord lightRec;
		lightRec.type = light.type;
		lightRec.dist = dist;
		lightRec.point = randomPos;
		lightRec.normal = normal;
		lightRec.direction = direction;
		lightRec.emission = light.color * light.intensity * distanceAttenuation * spotAttenuation;
		lightRec.pdf = 1.0;
		lightRec.discretePdf = 1.0;

		return lightRec;

	}

	// ── B4: mesh-area triangle lights (NEE) ──────────────────────────────────────
	// A triangle light's 6-texel slot (meshAreaLights.ts):
	//   s0 = (v0.xyz, type=TRI_AREA_LIGHT_TYPE=5)
	//   s1 = (radiance.rgb, 0)
	//   s2 = (v1.xyz, 0)
	//   s3 = (v2.xyz, triArea)
	struct MeshTriLight { vec3 v0; vec3 v1; vec3 v2; vec3 radiance; float area; };

	MeshTriLight readMeshTriLight( sampler2D tex, uint index ) {
		uint i = index * 6u;
		vec4 s0 = texelFetch1D( tex, i + 0u );
		vec4 s1 = texelFetch1D( tex, i + 1u );
		vec4 s2 = texelFetch1D( tex, i + 2u );
		vec4 s3 = texelFetch1D( tex, i + 3u );
		MeshTriLight t;
		t.v0 = s0.xyz;
		t.radiance = s1.rgb;
		t.v1 = s2.xyz;
		t.v2 = s3.xyz;
		t.area = s3.a;
		return t;
	}

	// Sample a point uniformly on the union of emissive triangles, AREA-proportional
	// over triangles. Returns a LightRecord whose .pdf is the SOLID-ANGLE density at
	// rayOrigin. Because triangle selection is area-proportional, that pdf reduces to
	//   dist² / ( totalEmissiveArea · |cosθ_light| )
	// — INDEPENDENT of which triangle was chosen (so the forward emissive hit can
	// recompute the identical pdf without a triangle→index map; see meshAreaLights.ts).
	LightRecord sampleMeshAreaLight(
		sampler2D meshLights, uint meshLightCount, float totalEmissiveArea, vec3 rayOrigin, vec3 ruv
	) {
		LightRecord rec;
		rec.pdf = 0.0;
		rec.discretePdf = 1.0;
		rec.type = TRI_AREA_LIGHT_TYPE;
		if ( meshLightCount == 0u || totalEmissiveArea <= 0.0 ) return rec;

		// Area-proportional triangle selection by cumulative area (uPick in [0,area]).
		float uPick = ruv.x * totalEmissiveArea;
		uint chosen = meshLightCount - 1u;
		float cum = 0.0;
		for ( uint ii = 0u; ii < meshLightCount; ii ++ ) {
			cum += max( readMeshTriLight( meshLights, ii ).area, 0.0 );
			if ( uPick <= cum ) { chosen = ii; break; }
		}

		MeshTriLight tri = readMeshTriLight( meshLights, chosen );

		// Uniform-area barycentric sample on the chosen triangle.
		float su = sqrt( max( ruv.y, 0.0 ) );
		float b0 = 1.0 - su;
		float b1 = ruv.z * su;
		float b2 = 1.0 - b0 - b1;
		vec3 pos = tri.v0 * b0 + tri.v1 * b1 + tri.v2 * b2;
		vec3 triNormal = normalize( cross( tri.v1 - tri.v0, tri.v2 - tri.v0 ) );

		vec3 toLight = pos - rayOrigin;
		float distSq = dot( toLight, toLight );
		float dist = sqrt( max( distSq, 1e-20 ) );
		vec3 direction = toLight / dist;

		// Two-sided emitter: face the normal toward the receiver.
		float cosLight = dot( triNormal, -direction );
		if ( cosLight < 0.0 ) { triNormal = -triNormal; cosLight = -cosLight; }

		rec.point = pos;
		rec.normal = triNormal;
		rec.dist = dist;
		rec.direction = direction;
		rec.emission = tri.radiance;
		// SA pdf, triangle-area term cancels with the area-proportional selection:
		//   (distSq / (area·cosLight)) · (area / totalEmissiveArea)
		rec.pdf = max( distSq / ( totalEmissiveArea * max( cosLight, EPSILON ) ), EPSILON );
		return rec;
	}

	// SOLID-ANGLE NEE pdf of a FORWARD emissive hit (the same area-proportional
	// density, used to MIS-weight the forward surf.emission accumulation). cosLight
	// is |n_light · -ω|. Triangle-independent (only totalEmissiveArea needed).
	float meshAreaLightForwardPdf( float distSq, float cosLight, float totalEmissiveArea ) {
		if ( totalEmissiveArea <= 0.0 ) return 0.0;
		return distSq / ( totalEmissiveArea * max( abs( cosLight ), EPSILON ) );
	}

	LightRecord randomLightSample( sampler2D lights, sampler2DArray iesProfiles, uint lightCount, vec3 rayOrigin, vec3 ruv ) {

		LightRecord result;

		float invCount = 1.0 / max( float( lightCount ), 1.0 );
		float sumPower = 0.0;
		for ( uint ii = 0u; ii < lightCount; ii ++ ) {

			Light tmpLight = readLightInfo( lights, ii );
			sumPower += max( tmpLight.power, 1e-20 );

		}

		uint l = 0u;
		float discretePdf = invCount;

		if ( lightCount > 0u && sumPower > 1e-30 ) {

			Light defaultPick = readLightInfo( lights, lightCount - 1u );
			l = lightCount - 1u;
			discretePdf = max( defaultPick.power, 1e-20 ) / sumPower;

			float uPick = ruv.x * sumPower;
			float cum = 0.0;
			for ( uint ii = 0u; ii < lightCount; ii ++ ) {

				Light tmpLight = readLightInfo( lights, ii );
				float w = max( tmpLight.power, 1e-20 );
				cum += w;
				if ( uPick <= cum ) {

					l = ii;
					discretePdf = w / sumPower;
					break;

				}

			}

		} else if ( lightCount > 0u ) {

			l = uint( ruv.x * float( lightCount ) );
			discretePdf = invCount;

		}

		Light light = readLightInfo( lights, l );

		if ( light.type == SPOT_LIGHT_TYPE ) {

			result = randomSpotLightSample( light, iesProfiles, rayOrigin, ruv.yz );

		} else if ( light.type == POINT_LIGHT_TYPE ) {

			vec3 lightRay = light.u - rayOrigin;
			float lightDist = length( lightRay );
			float cutoffDistance = light.distance;
			float distanceFalloff = 1.0 / max( pow( lightDist, light.decay ), 0.01 );
			if ( cutoffDistance > 0.0 ) {

				distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDist / cutoffDistance ) ) );

			}

			LightRecord rec;
			rec.point = light.u;
			rec.direction = normalize( lightRay );
			rec.dist = length( lightRay );
			rec.normal = - rec.direction;
			rec.pdf = 1.0;
			rec.emission = light.color * light.intensity * distanceFalloff;
			rec.type = light.type;
			rec.discretePdf = 1.0;
			result = rec;

		} else if ( light.type == DIR_LIGHT_TYPE ) {

			LightRecord rec;
			rec.dist = 1e10;
			rec.direction = light.u;
			rec.point = - light.u * rec.dist;
			rec.normal = light.u;
			rec.pdf = 1.0;
			rec.emission = light.color * light.intensity;
			rec.type = light.type;
			rec.discretePdf = 1.0;

			result = rec;

		} else {

			// sample the light
			result = randomAreaLightSample( light, rayOrigin, ruv.yz );

		}

		result.discretePdf = discretePdf;
		return result;

	}

`;
