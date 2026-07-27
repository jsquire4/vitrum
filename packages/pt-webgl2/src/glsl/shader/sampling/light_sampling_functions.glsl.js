export const light_sampling_functions = /* glsl */`

        float finitePositiveLightPower( float power ) {

                return power > 0.0 && ! isnan( power ) && ! isinf( power )
                        ? power
                        : 0.0;

        }

	float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {

		return smoothstep( coneCosine, penumbraCosine, angleCosine );

	}

	float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {

		// based upon Frostbite 3 Moving to Physically-based Rendering
		// page 32, equation 26: E[window1]
		// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
                float distanceFalloff = 1.0 / pow( lightDistance, decayExponent );

		if ( cutoffDistance > 0.0 ) {

			distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );

		}

		return distanceFalloff;

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
                                float denom = abs( light.area * cosTheta );
                                lightRec.pdf = denom > 0.0
                                        ? ( dist * dist ) / denom
                                        : 0.0;
				lightRec.emission = light.color * light.intensity;
				lightRec.direction = rayDirection;
				lightRec.type = light.type;
				lightRec.discretePdf = 1.0;
				lightRec.castShadowDisabled = light.castShadowDisabled;
				lightRec.delta = 0.0;

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
                vec3 direction = dist > 0.0 ? toLight / dist : vec3( 0.0 );
		vec3 lightNormal = normalize( cross( light.u, light.v ) );

		LightRecord lightRec;
		lightRec.type = light.type;
		lightRec.emission = light.color * light.intensity;
		lightRec.dist = dist;
		lightRec.point = randomPos;
		lightRec.normal = lightNormal;
		lightRec.direction = direction;

                float denom = abs( light.area * dot( direction, lightNormal ) );
                lightRec.pdf = dist > 0.0 && denom > 0.0
                        ? lightDistSq / denom
                        : 0.0;
		lightRec.discretePdf = 1.0;
		lightRec.castShadowDisabled = light.castShadowDisabled;
		lightRec.delta = 0.0;

		return lightRec;

	}

	LightRecord randomSpotLightSample( Light light, vec3 rayOrigin ) {

		// Core SpotEmitter is a delta-position source. cross(u,v) is its backward
		// axis, so a receiver inside the authored forward cone sees a positive
		// dot(direction-to-source, backwardAxis).
		vec3 normal = normalize( cross( light.u, light.v ) );
		vec3 toLight = light.position - rayOrigin;
		float lightDistSq = dot( toLight, toLight );
		float dist = sqrt( lightDistSq );

		vec3 direction = toLight / max( dist, EPSILON );
		float cosTheta = dot( direction, normal );

		float spotAttenuation = getSpotAttenuation( light.coneCos, light.penumbraCos, cosTheta );
		float distanceAttenuation = getDistanceAttenuation( dist, light.distance, light.decay );
		LightRecord lightRec;
		lightRec.type = light.type;
		lightRec.dist = dist;
		lightRec.point = light.position;
		lightRec.normal = normal;
		lightRec.direction = direction;
		lightRec.emission = light.color * light.intensity * distanceAttenuation * spotAttenuation;
		lightRec.pdf = 1.0;
		lightRec.discretePdf = 1.0;
		lightRec.castShadowDisabled = light.castShadowDisabled;
		lightRec.delta = 1.0;

		return lightRec;

	}

	// ── B4: mesh-area triangle lights (NEE) ──────────────────────────────────────
	// A triangle light's 6-texel slot (meshAreaLights.ts):
	//   s0 = (v0.xyz, type=TRI_AREA_LIGHT_TYPE=5)
	//   s1 = (radiance.rgb, 0)
	//   s2 = (v1.xyz, 0)
	//   s3 = (v2.xyz, triArea)
	//   s4.r = selectionPower = luminance(radiance) * triArea
	//   s5.g = castShadowDisabled
	struct MeshTriLight { vec3 v0; vec3 v1; vec3 v2; vec3 radiance; float area; float power; float castShadowDisabled; };

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
		t.castShadowDisabled = s5.g;
		return t;
	}

	// Sample a point uniformly on the chosen emissive triangle, selecting triangles
	// by emitted power (luminance(radiance) * area). The resulting area-density is
	// luminance(radiance) / totalEmissivePower, which the forward emissive hit can
	// recover from surf.emission without a triangle→index map.
	LightRecord sampleMeshAreaLight(
		sampler2D meshLights, uint meshLightCount, float totalEmissivePower, vec3 rayOrigin, vec3 ruv
	) {
		LightRecord rec;
		rec.pdf = 0.0;
		rec.discretePdf = 1.0;
		rec.type = TRI_AREA_LIGHT_TYPE;
		rec.castShadowDisabled = 0.0;
		rec.delta = 0.0;
		if ( meshLightCount == 0u || totalEmissivePower <= 0.0 ) return rec;

		// Power-proportional triangle selection by cumulative emitted power.
		float uPick = ruv.x * totalEmissivePower;
		uint chosen = meshLightCount - 1u;
		float cum = 0.0;
		for ( uint ii = 0u; ii < meshLightCount; ii ++ ) {
                        cum += finitePositiveLightPower(
                                readMeshTriLight( meshLights, ii ).power
                        );
			if ( uPick <= cum ) { chosen = ii; break; }
		}

                MeshTriLight tri = readMeshTriLight( meshLights, chosen );
                rec.castShadowDisabled = tri.castShadowDisabled;
                float triPower = finitePositiveLightPower( tri.power );
                if ( triPower <= 0.0 || tri.area <= 0.0 ) return rec;

		// Uniform-area barycentric sample on the chosen triangle.
		float su = sqrt( max( ruv.y, 0.0 ) );
		float b0 = 1.0 - su;
		float b1 = ruv.z * su;
		float b2 = 1.0 - b0 - b1;
		vec3 pos = tri.v0 * b0 + tri.v1 * b1 + tri.v2 * b2;
		vec3 triNormal = normalize( cross( tri.v1 - tri.v0, tri.v2 - tri.v0 ) );

		vec3 toLight = pos - rayOrigin;
		float distSq = dot( toLight, toLight );
                if ( distSq <= 0.0 ) return rec;
                float dist = sqrt( distSq );
		vec3 direction = toLight / dist;

		// Two-sided emitter: face the normal toward the receiver.
                float cosLight = dot( triNormal, -direction );
                if ( cosLight < 0.0 ) { triNormal = -triNormal; cosLight = -cosLight; }
                if ( cosLight <= 0.0 ) return rec;

		rec.point = pos;
		rec.normal = triNormal;
		rec.dist = dist;
		rec.direction = direction;
		rec.emission = tri.radiance;
                float areaDensity = triPower / ( tri.area * totalEmissivePower );
                rec.pdf = areaDensity * distSq / cosLight;
		return rec;
	}

	// SOLID-ANGLE NEE pdf of a FORWARD emissive hit under the emitted-power mesh
	// strategy. The area density is luminance(surface emission) / totalPower.
	float meshAreaLightForwardPdf( float distSq, float cosLight, float totalEmissivePower, vec3 emission ) {
		if ( totalEmissivePower <= 0.0 ) return 0.0;
                float cosine = abs( cosLight );
                float areaDensity = finitePositiveLightPower(
                        luminance( emission )
                ) / totalEmissivePower;
                return cosine > 0.0
                        ? areaDensity * distSq / cosine
                        : 0.0;
	}

	vec3 sampleDirectionalCone( vec3 axis, float angularDiameter, vec2 uv, out float pdf ) {

		float cosHalfAngle = cos( angularDiameter * 0.5 );
		float cosTheta = mix( cosHalfAngle, 1.0, uv.x );
		float sinTheta = sqrt( max( 0.0, 1.0 - cosTheta * cosTheta ) );
		float phi = 2.0 * PI * uv.y;
		vec3 localDir = vec3( cos( phi ) * sinTheta, sin( phi ) * sinTheta, cosTheta );
                float solidAngle = 2.0 * PI * ( 1.0 - cosHalfAngle );
                pdf = solidAngle > 0.0 ? 1.0 / solidAngle : 0.0;
		return normalize( getBasisFromNormal( normalize( axis ) ) * localDir );

	}

	LightRecord randomLightSample( sampler2D lights, uint lightCount, vec3 rayOrigin, vec3 ruv ) {

		LightRecord result;

                float sumPower = 0.0;
		for ( uint ii = 0u; ii < lightCount; ii ++ ) {

			Light tmpLight = readLightInfo( lights, ii );
                        sumPower += finitePositiveLightPower( tmpLight.power );

		}

		uint l = 0u;
                float discretePdf = 0.0;

                if ( lightCount > 0u && sumPower > 0.0 ) {

                        float uPick = ruv.x * sumPower;
                        float cum = 0.0;
                        for ( uint ii = 0u; ii < lightCount; ii ++ ) {

                                Light tmpLight = readLightInfo( lights, ii );
                                float w = finitePositiveLightPower( tmpLight.power );
                                if ( w <= 0.0 ) continue;
                                // Also serves as the exact final-bin fallback
                                // if a generator ever returns u == 1.
                                l = ii;
                                discretePdf = w / sumPower;
                                cum += w;
                                if ( uPick <= cum ) {
                                        break;

                                }

                        }

                }

		Light light = readLightInfo( lights, l );

		if ( light.type == SPOT_LIGHT_TYPE ) {

			result = randomSpotLightSample( light, rayOrigin );

		} else if ( light.type == POINT_LIGHT_TYPE ) {

			vec3 lightRay = light.u - rayOrigin;
			float lightDist = length( lightRay );
			float cutoffDistance = light.distance;
                        float distanceFalloff = 1.0 / pow( lightDist, light.decay );
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
			rec.castShadowDisabled = light.castShadowDisabled;
			rec.delta = 1.0;
			result = rec;

		} else if ( light.type == DIR_LIGHT_TYPE ) {

			LightRecord rec;
			rec.dist = 1e10;
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
			rec.point = - rec.direction * rec.dist;
			rec.normal = rec.direction;
			rec.emission = light.color * light.intensity;
			rec.type = light.type;
			rec.discretePdf = 1.0;
			rec.castShadowDisabled = light.castShadowDisabled;

			result = rec;

		} else {

			// sample the light
			result = randomAreaLightSample( light, rayOrigin, ruv.yz );

		}

		result.discretePdf = discretePdf;
		return result;

	}

`;
