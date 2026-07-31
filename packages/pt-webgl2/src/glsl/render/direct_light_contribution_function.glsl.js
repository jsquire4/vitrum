export const direct_light_contribution_function = /*glsl*/`

        struct DirectLightSample {

                bool valid;
                vec3 direction;
                vec3 point;
                vec3 emission;
                float distance;
                float pdf;
                float delta;
                float castShadowDisabled;
                float contributionScale;
                float bdptLaunchPdf;
                float bdptInfiniteKind;
                bool hasTargetFace;
                uint targetFaceIndex;

        };

        vec2 neeOctEncodeDirection( vec3 direction ) {

                vec3 n = normalize( direction );
                n /= abs( n.x ) + abs( n.y ) + abs( n.z );
                vec2 encoded = n.xy;
                if ( n.z < 0.0 ) {

                        vec2 signNotZero = vec2(
                                encoded.x >= 0.0 ? 1.0 : -1.0,
                                encoded.y >= 0.0 ? 1.0 : -1.0
                        );
                        encoded = ( 1.0 - abs( encoded.yx ) ) * signNotZero;

                }
                return encoded;

        }

        float bdptDirectionalNeePower() {
                float total = 0.0;
                for ( uint i = 0u; i < lights.count; i ++ ) {
                        Light light = readLightInfo( lights.tex, i );
                        if ( light.type == DIR_LIGHT_TYPE ) {
                                total += finitePositiveLightPower( light.power );
                        }
                }
                return total;
        }

        float bdptDistantNeeDenom() {
                float directionalSlot = bdptDirectionalNeePower() > 0.0 ? 1.0 : 0.0;
                float environmentSlot =
                        envMapInfo.totalSum > 0.0 && environmentIntensity > 0.0 ? 1.0 : 0.0;
                return directionalSlot + environmentSlot;
        }

        float bdptCandidateEmitterLogPower( float power ) {
                return power > 0.0 && ! isnan( power ) && ! isinf( power )
                        ? log2( power )
                        : - INFINITY;
        }

        float bdptCandidateEnvironmentLogPower() {
                if (
                        ! ( environmentIntensity > 0.0 ) ||
                        ! ( envMapInfo.totalSum > 0.0 ) ||
                        isnan( environmentIntensity ) || isinf( environmentIntensity ) ||
                        isnan( envMapInfo.totalSum ) || isinf( envMapInfo.totalSum )
                ) return - INFINITY;
                return log2( environmentIntensity ) + log2( envMapInfo.totalSum );
        }

        float bdptCandidateMaxLogPower() {
                float maxLogPower = bdptCandidateEnvironmentLogPower();
                for ( uint i = 0u; i < lights.count; i ++ ) {
                        maxLogPower = max(
                                maxLogPower,
                                bdptCandidateEmitterLogPower(
                                        finitePositiveLightPower(
                                                readLightInfo( lights.tex, i ).power
                                        )
                                )
                        );
                }
                for ( uint i = 0u; i < uMeshLightCount; i ++ ) {
                        maxLogPower = max(
                                maxLogPower,
                                bdptCandidateEmitterLogPower(
                                        finitePositiveLightPower(
                                                readMeshTriLight( uMeshLights, i ).power
                                        )
                                )
                        );
                }
                return maxLogPower;
        }

        float bdptCandidateScaledWeight( float logPower, float maxLogPower ) {
                return logPower > - INFINITY && maxLogPower > - INFINITY
                        ? exp2( logPower - maxLogPower )
                        : 0.0;
        }

        float bdptCandidateTotalScaledWeight( float maxLogPower ) {
                if ( ! ( maxLogPower > - INFINITY ) ) return 0.0;
                float total = bdptCandidateScaledWeight(
                        bdptCandidateEnvironmentLogPower(), maxLogPower
                );
                for ( uint i = 0u; i < lights.count; i ++ ) {
                        total += bdptCandidateScaledWeight(
                                bdptCandidateEmitterLogPower(
                                        finitePositiveLightPower(
                                                readLightInfo( lights.tex, i ).power
                                        )
                                ),
                                maxLogPower
                        );
                }
                for ( uint i = 0u; i < uMeshLightCount; i ++ ) {
                        total += bdptCandidateScaledWeight(
                                bdptCandidateEmitterLogPower(
                                        finitePositiveLightPower(
                                                readMeshTriLight( uMeshLights, i ).power
                                        )
                                ),
                                maxLogPower
                        );
                }
                return total;
        }

        float bdptCandidateEmitterDiscretePdf( float logPower ) {
                float maxLogPower = bdptCandidateMaxLogPower();
                float total = bdptCandidateTotalScaledWeight( maxLogPower );
                return total > 0.0
                        ? bdptCandidateScaledWeight( logPower, maxLogPower ) / total
                        : 0.0;
        }

        bool bdptSampleDirectionalNee(
                vec3 rayOrigin,
                vec3 ruv,
                out LightRecord rec,
                out float discretePdf,
                out float selectedEmitterLogPower
        ) {
                float totalPower = bdptDirectionalNeePower();
                if ( totalPower <= 0.0 ) return false;
                float pick = ruv.x * totalPower;
                float cumulative = 0.0;
                uint selected = 0u;
                float selectedPower = 0.0;
                for ( uint i = 0u; i < lights.count; i ++ ) {
                        Light candidate = readLightInfo( lights.tex, i );
                        float power = candidate.type == DIR_LIGHT_TYPE
                                ? finitePositiveLightPower( candidate.power )
                                : 0.0;
                        if ( power > 0.0 ) {
                                selected = i;
                                selectedPower = power;
                                cumulative += power;
                                if ( pick <= cumulative ) break;
                        }
                }
                if ( selectedPower <= 0.0 ) return false;
                Light light = readLightInfo( lights.tex, selected );
                rec.dist = INFINITY;
                if ( light.angularDiameter > 0.0 ) {
                        float conePdf;
                        rec.direction = sampleDirectionalCone(
                                light.u, light.angularDiameter, ruv.yz, conePdf
                        );
                        rec.pdf = conePdf;
                        rec.delta = 0.0;
                } else {
                        rec.direction = normalize( light.u );
                        rec.pdf = 1.0;
                        rec.delta = 1.0;
                }
                rec.point = rayOrigin + rec.direction * rec.dist;
                rec.normal = - rec.direction;
                rec.emission = light.color * light.intensity;
                rec.type = light.type;
                rec.discretePdf = selectedPower / totalPower;
                rec.castShadowDisabled = light.castShadowDisabled;
                rec.hasTargetFace = false;
                rec.targetFaceIndex = 0u;
                discretePdf = rec.discretePdf;
                selectedEmitterLogPower = log2( selectedPower );
                return true;
        }

        DirectLightSample sampleDirectLight( const in SurfaceRecord surf, vec3 rayOrigin ) {

		DirectLightSample lightSample;
		lightSample.valid = false;
		lightSample.direction = vec3( 0.0 );
		lightSample.point = vec3( 0.0 );
		lightSample.emission = vec3( 0.0 );
		lightSample.distance = 0.0;
		lightSample.pdf = 0.0;
		lightSample.delta = 0.0;
                lightSample.castShadowDisabled = 0.0;
                lightSample.contributionScale = 1.0;
                lightSample.bdptLaunchPdf = 0.0;
                lightSample.bdptInfiniteKind = 0.0;
                lightSample.hasTargetFace = false;
                lightSample.targetFaceIndex = 0u;
                float neeStrategyU = rand( 5 );

                #if FEATURE_BDPT

                // BDPT owns every finite-emitter direct strategy through c=0.
                // Only the non-connectable distant roots (directional + environment)
                // remain in ordinary NEE, sampled as two explicit strategy slots.
                float distantDenom = bdptDistantNeeDenom();
                bool hasDirectional = bdptDirectionalNeePower() > 0.0;
                float directionalCutoff = hasDirectional && distantDenom > 0.0
                        ? 1.0 / distantDenom
                        : 0.0;
                if ( distantDenom > 0.0 && neeStrategyU < directionalCutoff ) {
                        LightRecord lightRec;
                        float discretePdf;
                        float selectedEmitterLogPower;
                        if (
                                bdptSampleDirectionalNee(
                                        rayOrigin,
                                        rand3( 6 ),
                                        lightRec,
                                        discretePdf,
                                        selectedEmitterLogPower
                                )
                        ) {
                                bool below = ! surf.volumeParticle &&
                                        dot( surf.faceNormal, lightRec.direction ) < 0.0;
                                if ( ! below && lightRec.pdf > 0.0 ) {
                                        lightSample.valid = true;
                                        lightSample.direction = lightRec.direction;
                                        lightSample.point = lightRec.point;
                                        lightSample.emission = lightRec.emission;
                                        lightSample.distance = lightRec.dist;
                                        lightSample.pdf = lightRec.pdf * discretePdf / distantDenom;
                                        lightSample.delta = lightRec.delta;
                                        lightSample.castShadowDisabled = lightRec.castShadowDisabled;
                                        lightSample.hasTargetFace = lightRec.hasTargetFace;
                                        lightSample.targetFaceIndex = lightRec.targetFaceIndex;
                                        float launchRadius =
                                                max( uBdptSceneRadius, 1.175494351e-38 );
                                        VitrumAreaVectorMeasure launchArea =
                                                vitrumMeasureAreaVector(
                                                        vec3( launchRadius, 0.0, 0.0 ),
                                                        vec3( 0.0, launchRadius, 0.0 ),
                                                        PI
                                        );
                                        float emitterDiscretePdf =
                                                bdptCandidateEmitterDiscretePdf(
                                                        selectedEmitterLogPower
                                                );
                                        lightSample.bdptLaunchPdf =
                                                emitterDiscretePdf > 0.0 && launchArea.valid
                                                ? emitterDiscretePdf *
                                                        lightRec.pdf / launchArea.area
                                                : 0.0;
                                        lightSample.bdptInfiniteKind = 2.0;
                                }
                        }
                } else if (
                        distantDenom > 0.0 &&
                        envMapInfo.totalSum > 0.0 && environmentIntensity > 0.0
                ) {
                        vec3 envColor;
                        vec3 envDirection;
                        float envPdf = sampleEquirectProbability(
                                rand2( 7 ), envColor, envDirection
                        );
                        envDirection = invEnvRotation3x3 * envDirection;
                        bool below = ! surf.volumeParticle &&
                                dot( surf.faceNormal, envDirection ) < 0.0;
                        bool needsTransmission = below && surf.transmission > 0.0;
                        if ( ( ! below || needsTransmission ) && envPdf > 0.0 ) {
                                lightSample.valid = true;
                                lightSample.direction = envDirection;
                                lightSample.point = vec3( 0.0 );
                                lightSample.emission = finiteEquirectRadiance(
                                        envColor, surf.envMapIntensity
                                );
                                lightSample.distance = INFINITY;
                                lightSample.pdf = envPdf / distantDenom;
                                lightSample.delta = 0.0;
                                float launchRadius =
                                        max( uBdptSceneRadius, 1.175494351e-38 );
                                VitrumAreaVectorMeasure launchArea =
                                        vitrumMeasureAreaVector(
                                                vec3( launchRadius, 0.0, 0.0 ),
                                                vec3( 0.0, launchRadius, 0.0 ),
                                        PI
                                );
                                float environmentDiscretePdf =
                                        bdptCandidateEmitterDiscretePdf(
                                                bdptCandidateEnvironmentLogPower()
                                        );
                                lightSample.bdptLaunchPdf =
                                        environmentDiscretePdf > 0.0 && launchArea.valid
                                        ? environmentDiscretePdf *
                                                envPdf / launchArea.area
                                        : 0.0;
                                lightSample.bdptInfiniteKind = 1.0;
                        }
                }

                #else

                float analyticCutoff = lightsDenom != 0.0
                        ? float( lights.count ) / lightsDenom
                        : 0.0;
                float meshCutoff = lightsDenom != 0.0
                        ? float( lights.count + 1u ) / lightsDenom
                        : 0.0;

                if ( lightsDenom != 0.0 && neeStrategyU < analyticCutoff ) {

                        // One analytic-light proposal is one Monte Carlo trial. Retrying
                        // rejected back-facing samples would condition the proposal without
                        // applying the corresponding acceptance-probability correction and
                        // therefore bias the estimator. A rejected draw is a zero-valued
                        // (null) sample under the original one-draw density.
                        LightRecord lightRec = randomLightSample(
                                lights.tex, lights.count, rayOrigin, rand3( 6 )
                        );
                        bool isSampleBelowSurface =
                                ! surf.volumeParticle &&
                                dot( surf.faceNormal, lightRec.direction ) < 0.0;
                        if ( ! isSampleBelowSurface && lightRec.pdf > 0.0 ) {

					lightSample.valid = true;
					lightSample.direction = lightRec.direction;
					lightSample.point = lightRec.point;
					lightSample.emission = lightRec.emission;
				lightSample.distance = lightRec.dist;
				lightSample.pdf =
                                        lightRec.pdf / lightsDenom *
                                        float( lights.count ) *
                                        lightRec.discretePdf;
					// Rect/disc emitters have a matching forward-hit BSDF strategy,
					// so preserve their continuous measure. Directional lights have
					// no forward miss evaluator in the non-BDPT path (only the
					// environment does), and therefore remain NEE-owned even when
					// angularDiameter gives their proposal finite width.
					lightSample.delta = lightRec.castShadowDisabled > 0.5
						? 1.0
						: lightRec.type == DIR_LIGHT_TYPE
							? 1.0
							: lightRec.delta;
				lightSample.castShadowDisabled = lightRec.castShadowDisabled;
				lightSample.hasTargetFace = lightRec.hasTargetFace;
				lightSample.targetFaceIndex = lightRec.targetFaceIndex;

                        }

                } else if (
                        lightsDenom != 0.0 &&
                        uMeshLightCount != 0u &&
                        neeStrategyU < meshCutoff
                ) {

                        LightRecord lightRec = sampleMeshAreaLight(
                                uMeshLights,
                                uMeshLightCount,
                                uTotalEmissivePower,
                                rayOrigin,
                                rand3( 6 )
                        );
                        bool isSampleBelowSurface =
                                ! surf.volumeParticle &&
                                dot( surf.faceNormal, lightRec.direction ) < 0.0;
                        if ( ! isSampleBelowSurface && lightRec.pdf > 0.0 ) {

					lightSample.valid = true;
					lightSample.direction = lightRec.direction;
					lightSample.point = lightRec.point;
					lightSample.emission = lightRec.emission;
					lightSample.distance = lightRec.dist;
				lightSample.pdf = lightRec.pdf / lightsDenom;
				// A shadow-disabled finite emitter deliberately bypasses scene
				// visibility. A continuation ray cannot reproduce that proposal,
				// and its ordinary forward hit is suppressed, so NEE owns the
				// contribution exactly like a singular strategy.
				lightSample.delta = lightRec.castShadowDisabled > 0.5
					? 1.0
					: lightRec.delta;
				lightSample.castShadowDisabled = lightRec.castShadowDisabled;
				lightSample.hasTargetFace = lightRec.hasTargetFace;
				lightSample.targetFaceIndex = lightRec.targetFaceIndex;

                        }

                } else if ( envMapInfo.totalSum != 0.0 && environmentIntensity != 0.0 ) {

                        vec3 envColor;
                        vec3 envDirection;
                        float envPdf = sampleEquirectProbability(
                                rand2( 7 ), envColor, envDirection
                        );
                        envDirection = invEnvRotation3x3 * envDirection;

                        bool isSampleBelowSurface =
                                ! surf.volumeParticle &&
                                dot( surf.faceNormal, envDirection ) < 0.0;
                        bool envSampleNeedsTransmission =
                                isSampleBelowSurface && surf.transmission > 0.0;
                        if (
                                ( ! isSampleBelowSurface || envSampleNeedsTransmission ) &&
                                envPdf > 0.0
                        ) {

					lightSample.valid = true;
					lightSample.direction = envDirection;
					lightSample.point = vec3( 0.0 );
					lightSample.emission = finiteEquirectRadiance(
						envColor, surf.envMapIntensity
					);
				lightSample.distance = INFINITY;
				lightSample.pdf = envPdf / lightsDenom;
					// Environment directions have a finite solid-angle density and
				// therefore compete with the continuation BSDF strategy.
					lightSample.delta = 0.0;
				lightSample.castShadowDisabled = 0.0;

                        }

                }

                #endif

                return lightSample;

        }

	DirectLightSample prepareDirectLightSample(
		const in SurfaceRecord surf,
		RenderState state,
		vec3 rayOrigin,
		DirectLightSample lightSample
	) {

		if (
			! lightSample.valid ||
			lightSample.pdf <= 0.0 ||
			! isDirectionValid( lightSample.direction, surf.normal, surf.faceNormal )
		) {

			lightSample.valid = false;
			return lightSample;

		}

		bool lightIsBelowSurface =
			! surf.volumeParticle &&
			dot( surf.faceNormal, lightSample.direction ) < 0.0;
		Ray lightRay;
			lightRay.origin = stepRayOrigin(
				rayOrigin,
				vec3( 0.0 ),
				lightIsBelowSurface ? - surf.faceNormal : surf.faceNormal,
				0.0
			);
			lightRay.direction = lightSample.direction;
			if ( ! vitrumIsInfiniteDistance( lightSample.distance ) ) {

				// The proposal is measured from the geometric shading point, while
				// visibility starts at a coordinate-aware offset. Rebuild the finite
				// endpoint ray from that actual origin so large translations cannot
				// turn a sampled emitter into a parallel, prematurely terminated ray.
				vec3 toLightEndpoint = lightSample.point - lightRay.origin;
				float endpointDistance = vitrumLengthVec3( toLightEndpoint );
				if ( ! ( endpointDistance > 0.0 ) ) {

					lightSample.valid = false;
					return lightSample;

				}
				lightRay.direction = toLightEndpoint / endpointDistance;
				lightSample.distance = endpointDistance;
				if (
					! isDirectionValid(
						lightRay.direction, surf.normal, surf.faceNormal
					)
				) {

					lightSample.valid = false;
					return lightSample;

				}

			}
		vec3 attenuatedColor = vec3( 1.0 );
		RenderState visibilityState = state;
		visibilityState.isShadowRay = true;
		bool visible =
			lightSample.castShadowDisabled > 0.5 ||
			! attenuateHit(
				visibilityState,
				lightRay,
				lightSample.distance,
				lightSample.hasTargetFace,
				lightSample.targetFaceIndex,
				attenuatedColor
			);
		if ( ! visible ) {

			lightSample.valid = false;

		} else {

			lightSample.emission *= attenuatedColor;

		}
		return lightSample;

	}

	vec3 evaluatePreparedDirectLightSample(
		RenderState state,
		DirectLightSample lightSample,
		vec3 sampleColor,
		float lightMaterialPdf,
		float lightFamilyProbability,
		float continuationFamilyProbability
	) {

                float sampledLightPdf =
                        lightFamilyProbability * lightSample.pdf;
                if (
                        ! lightSample.valid ||
                        sampledLightPdf <= 0.0 ||
                        lightMaterialPdf <= 0.0 ||
			! all( greaterThanEqual( sampleColor, vec3( 0.0 ) ) )
		) {

			return vec3( 0.0 );

		}

                float competingBsdfPdf =
			continuationFamilyProbability * lightMaterialPdf;
		float misWeight = lightSample.delta > 0.5
			? 1.0
			: misHeuristic( sampledLightPdf, competingBsdfPdf );
		vec3 spectralContribution =
			state.throughput *
			pathThroughputFromRgb( lightSample.emission, state.wavelength ) *
			pathThroughputFromRgb( sampleColor, state.wavelength );
		return wavelengthToRGB(
			state.wavelength,
			spectralContribution,
			state.wavelengthPdf
                ) * lightSample.contributionScale * misWeight /
                        sampledLightPdf;

	}

`;
