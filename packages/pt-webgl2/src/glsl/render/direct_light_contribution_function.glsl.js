export const direct_light_contribution_function = /*glsl*/`

	vec3 directLightContribution( vec3 worldWo, SurfaceRecord surf, RenderState state, vec3 rayOrigin ) {

		vec3 result = vec3( 0.0 );
		vec3 throughputRgb = wavelengthToRGB( state.wavelength, state.throughput, state.wavelengthPdf );

		// Uniformly pick one NEE strategy slot with a single shared variate.
		// PCG-backed rand(v) advances on every call, so using separate
		// rand(5) calls for the analytic and mesh thresholds biases the
		// slot probabilities away from the PDFs below.
		float neeStrategyU = rand( 5 );
		float analyticCutoff = lightsDenom != 0.0 ? float( lights.count ) / lightsDenom : 0.0;
		float meshCutoff = lightsDenom != 0.0 ? float( lights.count + 1u ) / lightsDenom : 0.0;
		if( lightsDenom != 0.0 && neeStrategyU < analyticCutoff ) {

			// sample a light or environment. Back-face candidates are resampled up to
			// 4 attempts before giving up to zero contribution.
			LightRecord lightRec;
			bool foundFrontFacingLightSample = false;
			for ( int attempt = 0; attempt < 4; attempt ++ ) {

				lightRec = randomLightSample( lights.tex, lights.count, rayOrigin, rand3( 6 + attempt ) );
				bool isSampleBelowSurface = ! surf.volumeParticle && dot( surf.faceNormal, lightRec.direction ) < 0.0;
				if ( ! isSampleBelowSurface ) {

					foundFrontFacingLightSample = true;
					break;

				}

			}

			if ( ! foundFrontFacingLightSample ) {

				lightRec.pdf = 0.0;

			}

			// check if a ray could even reach the light area
			Ray lightRay;
			lightRay.origin = rayOrigin;
			lightRay.direction = lightRec.direction;
			// SHADOW-01 — emitter castShadow:false skips the occlusion/attenuation
			// test entirely (light passes through blockers, no glass tint).
			// attenuatedColor stays 1.0 in that case; attenuateHit overwrites it on
			// the default path, so flag-less scenes are behaviorally identical.
			vec3 attenuatedColor = vec3( 1.0 );
			if (
				lightRec.pdf > 0.0 &&
				isDirectionValid( lightRec.direction, surf.normal, surf.faceNormal ) &&
				( lightRec.castShadowDisabled > 0.5 || ! attenuateHit( state, lightRay, lightRec.dist, attenuatedColor ) )
			) {

				// get the material pdf
				vec3 sampleColor;
				float lightMaterialPdf = bsdfResult( worldWo, lightRec.direction, surf, state.wavelength, sampleColor );
				bool isValidSampleColor = all( greaterThanEqual( sampleColor, vec3( 0.0 ) ) );
				if ( lightMaterialPdf > 0.0 && isValidSampleColor ) {

					// weight the direct light contribution
					float lightPdf =
						lightRec.pdf / lightsDenom * float( lights.count ) * lightRec.discretePdf;
					bool deltaLight = lightRec.delta > 0.5;
					float misWeight = deltaLight ? 1.0 : misHeuristic( lightPdf, lightMaterialPdf );
					result = attenuatedColor * lightRec.emission * throughputRgb * sampleColor * misWeight / lightPdf;

				}

			}

		} else if (
			lightsDenom != 0.0 &&
			uMeshLightCount != 0u &&
			neeStrategyU < meshCutoff
		) {

				// B4 — mesh-area triangle-light NEE. One strategy slot; the chosen point is
				// emitted-power-proportional over emissive triangles / texel cells, so the
				// forward-hit PDF can recover the same area density from surf.emission.
				LightRecord lightRec = sampleMeshAreaLight(
					uMeshLights, uMeshLightCount, uTotalEmissivePower, rayOrigin, rand3( 6 )
				);

			bool isSampleBelowSurface = ! surf.volumeParticle && dot( surf.faceNormal, lightRec.direction ) < 0.0;
			if ( isSampleBelowSurface ) lightRec.pdf = 0.0;

			Ray lightRay;
			lightRay.origin = rayOrigin;
			lightRay.direction = lightRec.direction;
			vec3 attenuatedColor = vec3( 1.0 );
			if (
				lightRec.pdf > 0.0 &&
				isDirectionValid( lightRec.direction, surf.normal, surf.faceNormal ) &&
				( lightRec.castShadowDisabled > 0.5 || ! attenuateHit( state, lightRay, lightRec.dist - 1e-3, attenuatedColor ) )
			) {

				vec3 sampleColor;
				float lightMaterialPdf = bsdfResult( worldWo, lightRec.direction, surf, state.wavelength, sampleColor );
				bool isValidSampleColor = all( greaterThanEqual( sampleColor, vec3( 0.0 ) ) );
				if ( lightMaterialPdf > 0.0 && isValidSampleColor ) {

					// The mesh-light strategy is ONE slot of lightsDenom, chosen with
					// probability 1/lightsDenom; the full NEE pdf of this sample is therefore
					// (1/lightsDenom)·lightRec.pdf. (Mirrors the analytic branch's
					// lightRec.pdf/lightsDenom·count·discretePdf — there count·discretePdf is
					// the in-branch light selection; here the single area-proportional pick
					// makes the in-branch factor 1.)
					float lightPdf = lightRec.pdf / lightsDenom;
					float misWeight = misHeuristic( lightPdf, lightMaterialPdf );
					result = attenuatedColor * lightRec.emission * throughputRgb * sampleColor * misWeight / lightPdf;

				}

			}

		} else if ( envMapInfo.totalSum != 0.0 && environmentIntensity != 0.0 ) {

			// find a sample in the environment map to include in the contribution
			vec3 envColor, envDirection;
			float envPdf = sampleEquirectProbability( rand2( 7 ), envColor, envDirection );
			envDirection = invEnvRotation3x3 * envDirection;

			// this env sampling is not set up for transmissive sampling and yields overly bright
			// results so we ignore the sample in this case.
			// TODO: this should be improved but how? The env samples could traverse a few layers?
			bool isSampleBelowSurface = ! surf.volumeParticle && dot( surf.faceNormal, envDirection ) < 0.0;
			if ( isSampleBelowSurface ) {

				envPdf = 0.0;

			}

			// check if a ray could even reach the surface
			Ray envRay;
			envRay.origin = rayOrigin;
			envRay.direction = envDirection;
			vec3 attenuatedColor;
			if (
				envPdf > 0.0 &&
				isDirectionValid( envDirection, surf.normal, surf.faceNormal ) &&
				! attenuateHit( state, envRay, INFINITY, attenuatedColor )
			) {

				// get the material pdf
				vec3 sampleColor;
				float envMaterialPdf = bsdfResult( worldWo, envDirection, surf, state.wavelength, sampleColor );
				bool isValidSampleColor = all( greaterThanEqual( sampleColor, vec3( 0.0 ) ) );
				if ( envMaterialPdf > 0.0 && isValidSampleColor ) {

					// weight the direct light contribution
					// D3 — surf.envMapIntensity: per-material env scale on the NEE half;
					// the forward (BSDF) half applies the same factor via
					// state.envMapIntensity, keeping the MIS estimator consistent.
					envPdf /= lightsDenom;
					float misWeight = misHeuristic( envPdf, envMaterialPdf );
					result = attenuatedColor * surf.envMapIntensity * environmentIntensity * envColor * throughputRgb * sampleColor * misWeight / envPdf;

				}

			}

		}

		// Function changed to have a single return statement to potentially help with crashes on Mac OS.
		// See issue #470
		return result;

	}

`;
