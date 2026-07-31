import {
	MATERIAL_ALPHA_TRANSFORM_TEXEL,
	MATERIAL_THICKNESS_TRANSFORM_TEXEL,
	MATERIAL_TRANSFORM_TEXEL,
	MATERIAL_WRAP_TEXEL_OFFSET,
} from '../shader/structs/materialStride.js';

export const attenuate_hit_function = /* glsl */`

	// step through multiple surface hits and accumulate color attenuation based on transmissive surfaces
	// returns true if a solid surface was hit
	bool attenuateHit(
		RenderState state,
		Ray ray, float rayDist,
		bool hasTargetFace,
		uint targetFaceIndex,
		out vec3 color
	) {

		// store the original bounce index so we can reset it after
		uint originalBounceIndex = sobolBounceIndex;

		int traversals = state.traversals;
		int transmissiveTraversals = state.transmissiveTraversals;
                bool isShadowRay = state.isShadowRay;
                FogMaterial fogMaterial = state.fogMaterial;
                MediumStack mediumStack = state.mediumStack;
                bool finiteRayDistance = ! vitrumIsInfiniteDistance( rayDist );

		float traveledDistance = 0.0;

		// hit results
		SurfaceHit surfaceHit;

                color = vec3( 1.0 );

                bool result = true;
                int remainingTraversals = max( traversals, 0 );
                // A monotonic, statically bounded loop is required here. This helper is
                // called from inside the path-bounce loop; nesting a uniform-bounded loop
                // caused ANGLE/SwiftShader to emit a non-terminating shader. The engine
                // caps each of the ordinary and transmissive budgets at 32, so 64 steps
                // preserves the full traversal budget without relying on loop-counter
                // rewinds that some GLSL compilers miscompile.
                for ( int attenuationStep = 0; attenuationStep < 64; attenuationStep ++ ) {

                        if ( remainingTraversals <= 0 ) break;
                        remainingTraversals --;

			sobolBounceIndex ++;

                        bool surfaceFound = bvhIntersectFirstHit(
                                bvh,
                                ray.origin,
                                ray.direction,
                                surfaceHit.faceIndices,
                                surfaceHit.faceNormal,
                                surfaceHit.barycoord,
                                surfaceHit.side,
                                surfaceHit.dist
                        );
                        float remainingDistance = finiteRayDistance
                                ? max( rayDist - traveledDistance, 0.0 )
                                : INFINITY;
                        float segmentDistance = surfaceFound
                                ? min( max( surfaceHit.dist, 0.0 ), remainingDistance )
                                : remainingDistance;
                        #if FEATURE_FOG
                        if ( fogMaterial.fogVolume ) {
                                color *= fogSegmentTransmittance(
                                        materials,
                                        fogMaterial,
                                        segmentDistance,
                                        state.wavelength
                                );
                        }
                        #endif
                        if ( ! surfaceFound ) {
                                result = false;
                                break;
                        }
                        if (
                                hasTargetFace &&
                                surfaceHit.faceIndices.w == targetFaceIndex
                        ) {
                                result = false;
                                break;
                        }
                        if (
                                finiteRayDistance &&
                                surfaceHit.dist >= remainingDistance
                        ) {
                                result = false;
                                break;
                        }

                        {

				// Shadow visibility through transmissive layers is intentionally a
				// bounded attenuation approximation here: this helper answers whether
				// a direct-light/BDPT connection remains visible and returns tint/medium
				// throughput. It does not add emissive surface radiance or try to make
				// the shadow ray a second full BSDF path; those are handled by hit/NEE
				// estimators rather than this visibility predicate.

				uint materialIndex = uTexelFetch1D( materialIndexAttribute, surfaceHit.faceIndices.w ).r;
				Material material;
				readMaterialInfo( materials, materialIndex, material );

					// adjust the ray to the new surface
					bool isEntering = surfaceHit.side == 1.0;
					if ( finiteRayDistance ) {
						traveledDistance += max( surfaceHit.dist, 0.0 );
					}
					ray.origin = stepRayOrigin( ray.origin, ray.direction, - surfaceHit.faceNormal, surfaceHit.dist );

				#if FEATURE_FOG

                                if ( material.fogVolume ) {
                                        bool stackValid = surfaceHit.side == 1.0
                                                ? enterMedium(
                                                        mediumStack, materialIndex,
                                                        materials, fogMaterial
                                                )
                                                : leaveMedium(
                                                        mediumStack, materialIndex,
                                                        materials, fogMaterial
                                                );
                                        if ( ! stackValid ) {
                                                result = true;
                                                break;
                                        }
                                        if ( transmissiveTraversals > 0 ) {

                                                remainingTraversals ++;
                                                transmissiveTraversals --;

                                        }
                                        continue;

				}

				#endif

				if ( ! material.castShadow && isShadowRay ) {

					continue;

				}

				vec4 vertexColor = textureSampleBarycoord( attributesArray, ATTR_COLOR, surfaceHit.barycoord, surfaceHit.faceIndices.xyz );

				#define ATTENUATE_MAP_UV(mapIndex) textureSampleBarycoord( attributesArray, readMaterialMapUvLayer( materials, materialIndex, mapIndex ), surfaceHit.barycoord, surfaceHit.faceIndices.xyz ).xy
				#define ATTENUATE_MAP_SAMPLE(layer,transformOffset,policyOffset,uvCoord) sampleMappedMaterialTexture( materials, textures, materialIndex, layer, transformOffset, policyOffset, uvCoord )
				#define ATTENUATE_SRGB_MAP_SAMPLE(layer,transformOffset,policyOffset,uvCoord) sampleMappedSrgbMaterialTexture( materials, textures, materialIndex, layer, transformOffset, policyOffset, uvCoord )

				// albedo
				vec4 albedo = vec4( material.color, material.opacity );
				if ( material.map != - 1 ) {

					albedo *= ATTENUATE_SRGB_MAP_SAMPLE(
						material.map, ${MATERIAL_TRANSFORM_TEXEL.baseColorMap}u,
						${MATERIAL_WRAP_TEXEL_OFFSET + 0}u, ATTENUATE_MAP_UV( 0u )
					);

				}

				if ( material.vertexColors ) {

					albedo *= vertexColor;

				}

				// alphaMap
				if ( material.alphaMap != - 1 ) {

					albedo.a *= ATTENUATE_MAP_SAMPLE(
						material.alphaMap, ${MATERIAL_ALPHA_TRANSFORM_TEXEL}u,
						${MATERIAL_WRAP_TEXEL_OFFSET + 6}u, ATTENUATE_MAP_UV( 6u )
					).x;

				}

				// transmission
				float transmission = material.transmission;
				if ( material.transmissionMap != - 1 ) {

					transmission *= ATTENUATE_MAP_SAMPLE(
						material.transmissionMap, ${MATERIAL_TRANSFORM_TEXEL.transmissionMap}u,
						${MATERIAL_WRAP_TEXEL_OFFSET + 3}u, ATTENUATE_MAP_UV( 3u )
					).r;

				}

				// metalness
				float metalness = material.metalness;
				if ( material.metalnessMap != - 1 ) {

					metalness *= ATTENUATE_MAP_SAMPLE(
						material.metalnessMap, ${MATERIAL_TRANSFORM_TEXEL.metallicMap}u,
						${MATERIAL_WRAP_TEXEL_OFFSET + 1}u, ATTENUATE_MAP_UV( 1u )
					).b;

				}

				float alphaTest = material.alphaTest;
				bool useAlphaTest = alphaTest != 0.0;
				float transmissionFactor = ( 1.0 - metalness ) * transmission;
				if (
					transmissionFactor < rand( 9 ) && ! (
						// material sidedness
						material.side != 0.0 && surfaceHit.side == material.side

						// alpha test
						|| useAlphaTest && albedo.a < alphaTest

						// opacity
						|| material.transparent && ! useAlphaTest && albedo.a < rand( 10 )
					)
				) {

					result = true;
					break;

				}

				if ( surfaceHit.side == 1.0 && isEntering ) {

					// only attenuate by surface color on the way in
					vec3 surfaceTransmission = mix( vec3( 1.0 ), albedo.rgb, transmissionFactor );
					color *= pathThroughputFromRgb( surfaceTransmission, state.wavelength );

				} else if ( surfaceHit.side == - 1.0 ) {

					float attenuationDist = surfaceHit.dist;
					if ( material.thickness > 0.0 || material.thicknessMap != - 1 ) {

						float attenuationThickness = material.thickness;
						if ( material.thicknessMap != - 1 ) {

							attenuationThickness *= ATTENUATE_MAP_SAMPLE(
								material.thicknessMap, ${MATERIAL_THICKNESS_TRANSFORM_TEXEL}u,
								${MATERIAL_WRAP_TEXEL_OFFSET + 20}u, ATTENUATE_MAP_UV( 20u )
							).g;

						}
						attenuationDist = min( attenuationDist, max( attenuationThickness, 0.0 ) );

					}

					// attenuate by medium once we hit the opposite side of the model.
					// Sprint 12 Gap §5: use hero-wavelength attenuation when spectral data exists.
					vec3 attenuation = transmissionAttenuationThroughput(
						materials,
						attenuationDist,
						material.attenuationColor,
						material.attenuationDistance,
						material.hasSpectralAttenuation,
						materialIndex,
						state.wavelength
					);
					color *= attenuation;

				}

				#undef ATTENUATE_MAP_SAMPLE
				#undef ATTENUATE_SRGB_MAP_SAMPLE
				#undef ATTENUATE_MAP_UV

				bool isTransmissiveRay = dot( ray.direction, surfaceHit.faceNormal * surfaceHit.side ) < 0.0;
                                if ( ( isTransmissiveRay || isEntering ) && transmissiveTraversals > 0 ) {

                                        remainingTraversals ++;
                                        transmissiveTraversals --;

				}

                        }

		}

		// reset the bounce index
		sobolBounceIndex = originalBounceIndex;
		return result;

	}

`;
