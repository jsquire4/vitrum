import {
	MATERIAL_ALPHA_TRANSFORM_TEXEL,
	MATERIAL_LAYER_NORMAL_TEXEL_OFFSET,
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
		out vec3 color
	) {

		// store the original bounce index so we can reset it after
		uint originalBounceIndex = sobolBounceIndex;

		int traversals = state.traversals;
		int transmissiveTraversals = state.transmissiveTraversals;
                bool isShadowRay = state.isShadowRay;
                FogMaterial fogMaterial = state.fogMaterial;
                MediumStack mediumStack = state.mediumStack;

		vec3 startPoint = ray.origin;

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
                        float traveled = distance( startPoint, ray.origin );
                        float remainingDistance = max( rayDist - traveled, 0.0 );
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
                        if ( ! surfaceFound || surfaceHit.dist > remainingDistance ) {
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

				// albedo
				vec4 albedo = vec4( material.color, material.opacity );
				if ( material.map != - 1 ) {

					albedo *= ATTENUATE_MAP_SAMPLE(
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

				// stainedglass fork — opt-in caustic-texture patch.
				// Default conservative: FEATURE_STAINED_GLASS_SHADOW_NORMAL_PERTURBATION
				// is intentionally undefined/false unless a host compiles this fork
				// path in. When enabled for shadow rays, the material normalMap
				// applies a small perturbation to the ray direction so NEE caustic
				// projections can pick up per-pixel surface relief.
				#if FEATURE_STAINED_GLASS_SHADOW_NORMAL_PERTURBATION

				bool frontFaceHitForNormal = surfaceHit.side == 1.0 || transmission == 0.0;
				bool hasFaceLayerForNormal = frontFaceHitForNormal ? material.hasFrontLayer : material.hasBackLayer;
				int activeShadowNormalMap = material.normalMap;
				uint activeShadowNormalMapTransformOffset = ${MATERIAL_TRANSFORM_TEXEL.normalMap}u;
				vec2 activeShadowNormalScale = material.normalScale;
				uint activeShadowNormalMapPolicyOffset = ${MATERIAL_WRAP_TEXEL_OFFSET + 5}u;
				int activeShadowNormalUvLayer = readMaterialMapUvLayer( materials, materialIndex, 5u );
				vec2 activeShadowNormalUv = textureSampleBarycoord( attributesArray, activeShadowNormalUvLayer, surfaceHit.barycoord, surfaceHit.faceIndices.xyz ).xy;
				if ( hasFaceLayerForNormal && frontFaceHitForNormal && material.frontLayerNormalMap != - 1 ) {
					activeShadowNormalMap = material.frontLayerNormalMap;
					activeShadowNormalMapTransformOffset = ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 1}u;
					activeShadowNormalScale = material.frontLayerNormalScale;
					activeShadowNormalMapPolicyOffset = ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 5}u;
					activeShadowNormalUvLayer = int( round( material.frontLayerNormalTexCoord ) );
					activeShadowNormalUv = textureSampleBarycoord( attributesArray, activeShadowNormalUvLayer, surfaceHit.barycoord, surfaceHit.faceIndices.xyz ).xy;
				} else if ( hasFaceLayerForNormal && ! frontFaceHitForNormal && material.backLayerNormalMap != - 1 ) {
					activeShadowNormalMap = material.backLayerNormalMap;
					activeShadowNormalMapTransformOffset = ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 3}u;
					activeShadowNormalScale = material.backLayerNormalScale;
					activeShadowNormalMapPolicyOffset = ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 6}u;
					activeShadowNormalUvLayer = int( round( material.backLayerNormalTexCoord ) );
					activeShadowNormalUv = textureSampleBarycoord( attributesArray, activeShadowNormalUvLayer, surfaceHit.barycoord, surfaceHit.faceIndices.xyz ).xy;
				}

				if ( isShadowRay && activeShadowNormalMap != - 1 ) {

					vec4 tangentSample = textureSampleBarycoord(
						attributesArray,
						ATTR_TANGENT,
						surfaceHit.barycoord,
						surfaceHit.faceIndices.xyz
					);

					vec3 faceN = surfaceHit.faceNormal * surfaceHit.side;
					mat3 shadowBasis = getBasisFromSelectedUv(
						bvh.position, attributesArray, activeShadowNormalUvLayer,
						surfaceHit.faceIndices.xyz, faceN, tangentSample
					);
					vec3 tangent = shadowBasis[ 0 ];
					vec3 bitangent = shadowBasis[ 1 ];
						vec3 texNormal = ATTENUATE_MAP_SAMPLE(
							activeShadowNormalMap,
							activeShadowNormalMapTransformOffset,
							activeShadowNormalMapPolicyOffset,
							activeShadowNormalUv
						).xyz * 2.0 - 1.0;
						texNormal.xy *= activeShadowNormalScale;
						// World-space perturbation vector in the tangent plane.
						vec3 dN = tangent * texNormal.x + bitangent * texNormal.y;
						float perturbStrength = ( material.ior - 1.0 ) * 0.1;
						ray.direction = normalize( ray.direction + dN * perturbStrength );

				}

				#endif

				#undef ATTENUATE_MAP_SAMPLE
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
