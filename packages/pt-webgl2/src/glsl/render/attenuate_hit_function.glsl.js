import {
		MATERIAL_ALPHA_TRANSFORM_TEXEL,
		MATERIAL_TRANSFORM_TEXEL,
		MATERIAL_WRAP_TEXEL_OFFSET,
} from '../shader/structs/materialStride.js';

export const attenuate_hit_function = /* glsl */`

		// Step through visibility hits. Alpha/null skips and castShadow:false may
		// continue; accepted physical surfaces terminate the straight connection.
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
		if ( isShadowRay ) {
			filterShadowMediumStack(
				state.mediumStack, materials, mediumStack, fogMaterial
			);
		}
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

			bool invalidRange = false;
			bool surfaceFound = ray.minimumDistanceExclusive >= 0.0
				? bvhIntersectExactRangeFirstHit(
					ray, surfaceHit, invalidRange
				)
				: bvhIntersectCanonicalInitialFirstHit(
					ray, surfaceHit, invalidRange
				);
			if ( invalidRange ) {
				result = true;
				break;
			}
                        float remainingDistance = finiteRayDistance
                                ? max( rayDist - traveledDistance, 0.0 )
                                : INFINITY;
                        float segmentDistance = surfaceFound
                                ? min( max( surfaceHit.dist, 0.0 ), remainingDistance )
                                : remainingDistance;
			float effectiveSegmentDistance = mediumEffectiveSegmentDistance(
				mediumStack, segmentDistance
			);
			if ( effectiveSegmentDistance < 0.0 ) {
				result = true;
				break;
			}
                        color *= opticalVisibilitySegmentTransmittance(
                                materials,
                                mediumStack,
                                fogMaterial,
				effectiveSegmentDistance,
                                state.wavelength
                        );
			if ( ! consumeMediumSegmentDistance(
				mediumStack, segmentDistance, materials, fogMaterial
			) ) {
				result = true;
				break;
			}
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

					// Physical transmission is not a null-opacity event. Collapsing a glass
					// interface into this straight segment would omit its refraction,
					// Fresnel/roughness terms, Jacobians, and BDPT vertex/PDF. This helper
					// therefore carries only alpha/null policy and medium-segment extinction.

				uint materialIndex = uTexelFetch1D( materialIndexAttribute, surfaceHit.faceIndices.w ).r;
				Material material;
				readMaterialInfo( materials, materialIndex, material );

						if ( finiteRayDistance ) {
						traveledDistance += max( surfaceHit.dist, 0.0 );
					}

					vec4 vertexColor = textureSampleBarycoord( attributesArray, ATTR_COLOR, surfaceHit.barycoord, surfaceHit.faceIndices.xyz );

				#define ATTENUATE_MAP_UV(mapIndex) textureSampleBarycoord( attributesArray, readMaterialMapUvLayer( materials, materialIndex, mapIndex ), surfaceHit.barycoord, surfaceHit.faceIndices.xyz ).xy
				#define ATTENUATE_MAP_SAMPLE(layer,transformOffset,policyOffset,uvCoord,outValue) sampleMappedMaterialTexture( materials, textures, materialIndex, layer, transformOffset, policyOffset, uvCoord, outValue )
				#define ATTENUATE_SRGB_MAP_SAMPLE(layer,transformOffset,policyOffset,uvCoord,outValue) sampleMappedSrgbMaterialTexture( materials, textures, materialIndex, layer, transformOffset, policyOffset, uvCoord, outValue )

				// albedo
				vec4 albedo = vec4( material.color, material.opacity );
				if ( material.map != - 1 ) {

					vec4 baseColorSample;
					if ( ATTENUATE_SRGB_MAP_SAMPLE(
						material.map, ${MATERIAL_TRANSFORM_TEXEL.baseColorMap}u,
						${MATERIAL_WRAP_TEXEL_OFFSET + 0}u,
						ATTENUATE_MAP_UV( 0u ), baseColorSample
					) ) albedo *= baseColorSample;

				}

				if ( material.vertexColors ) {

					albedo *= vertexColor;

				}

				// alphaMap
					if ( material.alphaMap != - 1 ) {

					vec4 alphaSample;
					if ( ATTENUATE_MAP_SAMPLE(
						material.alphaMap, ${MATERIAL_ALPHA_TRANSFORM_TEXEL}u,
						${MATERIAL_WRAP_TEXEL_OFFSET + 6}u,
						ATTENUATE_MAP_UV( 6u ), alphaSample
					) ) albedo.a *= alphaSample.x;

					}

					#undef ATTENUATE_MAP_SAMPLE
					#undef ATTENUATE_SRGB_MAP_SAMPLE
					#undef ATTENUATE_MAP_UV

						float alphaTest = material.alphaTest;
					bool useAlphaTest = alphaTest != 0.0;
					bool skipSurface =
						material.side != 0.0 && surfaceHit.side != material.side ||
						useAlphaTest && albedo.a < alphaTest ||
						material.transparent && ! useAlphaTest &&
							rand( 10 ) >= representedBernoulliProbabilityF32( albedo.a );

					if ( skipSurface ) {

						if ( transmissiveTraversals > 0 ) {
							remainingTraversals ++;
							transmissiveTraversals --;
						}
						if ( ray.minimumDistanceExclusive >= 0.0 ) {
							if ( ! setExactRayRangeFromSurfaceHit( ray, surfaceHit ) ) {
								result = true;
								break;
							}
						} else {
							ray.origin = stepRayOrigin(
								ray.origin, ray.direction,
								- surfaceHit.faceNormal, surfaceHit.dist
							);
							setOrdinaryRayRange( ray );
						}
						continue;

					}

					if ( ! material.castShadow && isShadowRay ) {

							// The filtered shadow stack omits this material's volume
							// extinction, so its boundary is a visibility null event.
						if ( ! setExactRayRangeFromSurfaceHit( ray, surfaceHit ) ) {
							result = true;
							break;
						}
						if ( transmissiveTraversals > 0 ) {
							remainingTraversals ++;
							transmissiveTraversals --;
						}
						continue;

					}

					bool exactBaseRoughness = material.roughness == 0.0;
					bool exactFrontRoughness =
						! material.hasFrontLayer ||
						material.frontLayerRoughness < 0.0 ||
						material.frontLayerRoughness == 0.0;
					bool exactBackRoughness =
						! material.hasBackLayer ||
						material.backLayerRoughness < 0.0 ||
						material.backLayerRoughness == 0.0;
					if (
						material.thinFilm && material.transmission > 0.0 &&
						exactBaseRoughness &&
						exactFrontRoughness && exactBackRoughness
					) {
						SurfaceRecord sheetSurface;
						int sheetStatus = getSurfaceRecord(
							materialIndex, surfaceHit, attributesArray,
							0.0, 0, state.wavelength, true, sheetSurface
						);
						vec3 sheetAttenuation;
						bool sheetConnectable =
							sheetStatus == HIT_SURFACE &&
							thinSheetExactVisibilityTransmission(
								- ray.direction, ray.direction,
								sheetSurface, state.wavelength,
								sheetAttenuation
							);
						if ( ! sheetConnectable ) {
							result = true;
							break;
						}
						color *= sheetAttenuation;
						if ( ! setExactRayRangeFromSurfaceHit( ray, surfaceHit ) ) {
							result = true;
							break;
						}
						if ( transmissiveTraversals > 0 ) {
							remainingTraversals ++;
							transmissiveTraversals --;
						}
						continue;
					}

					result = true;
					break;

							}

		}

		// reset the bounce index
		sobolBounceIndex = originalBounceIndex;
		return result;

	}

`;
