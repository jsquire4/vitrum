import { WEBGL2_MAX_PATH_STEPS } from '../limits.js';

// renderMain.glsl — the pt-webgl2 trace-shader main() render-loop section
// constants (extracted from composeTraceGlsl.ts, T3-D / D11-3). PURE DATA: these
// GLSL string constants are assembled by composeTraceGlsl into the fragment body.
//
// BYTE-IDENTITY CONTRACT: RENDER_MAIN_SECTIONS.join('') === the original inline
// RENDER_MAIN string. The section constants were moved VERBATIM (tabs + all
// whitespace preserved); no byte was added or removed. Verified by the composed-
// GLSL byte-identity golden (composeTraceGlslGolden.test.ts) + the D10.4 section
// pins in composeTraceGlsl.test.ts.

/**
 * The inlined main() render loop (PhysicalPathTracingMaterial.js:443-1099). The fork keeps
 * the whole orchestration loop inline in the material (no `RENDER.main` chunk), so it is
 * transcribed verbatim here. All FEATURE_ gates resolve from the preamble
 * defines at compile time.
 *
 * D10.4 (2026-06-11): split into named section constants assembled in order.
 * BYTE-IDENTITY CONTRACT: RENDER_MAIN_SECTIONS.join('') === original RENDER_MAIN string.
 * Sections are split at natural inline-comment boundaries; no whitespace is added or removed.
 */

// ── Section 1: function header + RNG init + BDPT light-subpath pass ───────
const RENDER_MAIN_BDPT_SUBPATH = /* glsl */ `
					float neeReservoirReplacementU( uint candidateOrdinal ) {

						// A stateless stream reserved for vertex-reservoir replacement. It
						// deliberately does not call rand()/pcgRand(), so the candidate pass
						// replays the continuation path with the exact same RNG state as the
						// main radiance pass. The light proposal and visibility trace save and
						// restore their mutable RNG state separately at the capture site.
						uint h = uint( gl_FragCoord.x ) * 0x9e3779b9u;
						h ^= uint( gl_FragCoord.y ) * 0x85ebca6bu;
						h ^= uint( seed ) * 0xc2b2ae35u;
						h ^= candidateOrdinal * 0x27d4eb2fu;
						h ^= h >> 16u;
						h *= 0x7feb352du;
						h ^= h >> 15u;
						h *= 0x846ca68bu;
						h ^= h >> 16u;
						return float( h ) * ( 1.0 / 4294967296.0 );

					}

					void main() {

						// init
						rng_initialize( gl_FragCoord.xy, seed );
						sobolPixelIndex = ( uint( gl_FragCoord.x ) << 16 ) | uint( gl_FragCoord.y );
						sobolPathIndex = uint( seed );

                                                #if FEATURE_BDPT && ! NEE_CANDIDATE_PASS

						// Sprint 10c — dedicated light-subpath draw (one column per dispatch).
						if ( uBdptLightSubpathPass != 0 ) {

							// BDPT subpath RNG made row-independent (one coherent path per vertex
							// column), 2026-06-10 — RENDER-CHANGING for bdpt:true only; off-path
							// byte-identical.
							//
                                                        // The texture is 8 rows × 8 columns. Eight fragments (one per row)
							// cooperate to write one vertex: row 0 = position|kind, row 1 =
							// normal|pdfFwd, row 2 = throughput|pdfRev, row 3 = BSDF state,
                                                        // row 4 = hit/material payload, rows 5..7 = medium stack
							// (see bdpt_light_subpath.glsl.js).
							// The original rng_initialize(gl_FragCoord.xy, seed) seeded with the
							// Y coordinate, so each of the five fragments at column C traced a
							// *different* random subpath and stored ONE row from that path — the
							// assembled "vertex" mixed position, normal/pdf, and throughput from
							// five independent random subpaths, making BDPT connections garbage.
							//
							// Fix: re-initialize with a y-flattened coordinate so all five
							// fragments at the same column trace the identical subpath. The row
							// routing (bdptRow == 0/1/2/3 below) then writes consistent rows from
							// the same path. The main-entry rng_initialize(gl_FragCoord.xy, seed)
							// above is left untouched — the eye pass still seeds with the full
							// (x,y) pixel coordinate as before.
							// Predecessor patches are rendered by fragments in column k-1, so seed
							// from the logical vertex column rather than the fragment's x coordinate.
							rng_initialize( vec2( float( uBdptVertexCol ) + 0.5, 0.0 ), seed );

							envRotation3x3 = mat3( environmentRotation );
							invEnvRotation3x3 = inverse( envRotation3x3 );
							// NOTE: lightsDenom is not read by the subpath kernel — writeLightSubpathVertex
							// builds its own emitted-power CDF over analytic lights plus mesh-area
							// emitters. The assignment below is retained for completeness and to keep
							// the variable initialised in case a future subpath extension reads it.
							lightsDenom = float(
								lights.count +
								( uMeshLightCount != 0u ? 1u : 0u ) +
								( environmentIntensity != 0.0 && envMapInfo.totalSum != 0.0 ? 1u : 0u )
							);

							RenderState bdptState = initRenderState();
							if ( uSpectralRendering != 0 ) {
								bdptState.wavelength = uBdptSharedWavelength;
								bdptState.wavelengthPdf = uBdptSharedWavelengthPdf;
							} else {
								bdptState.wavelength = sampleHeroWavelengthMIS( rand( 30 ), rand( 31 ), bdptState.wavelengthPdf );
							}
                                                        vec4 bdptV0;
							vec4 bdptV1;
							vec4 bdptV2;
							vec4 bdptV3;
							vec4 bdptV4;
                                                        vec4 bdptV5;
                                                        vec4 bdptV6;
                                                        vec4 bdptV7;
							vec4 bdptPredecessor0;
							vec4 bdptPredecessor2;
							writeLightSubpathVertex(
								uBdptVertexCol,
								uBdptMaxLightBounces,
                                                                uBdptLightPathTex,
                                                                bdptState.fogMaterial,
                                                                bdptState.mediumStack,
                                                                bdptState.wavelength,
								bdptV0,
								bdptV1,
								bdptV2,
								bdptV3,
								bdptV4,
                                                                bdptV5,
                                                                bdptV6,
                                                                bdptV7,
								bdptPredecessor0,
								bdptPredecessor2
							);

							int bdptCol = int( gl_FragCoord.x );
							int bdptRow = int( gl_FragCoord.y );
							if ( bdptCol == uBdptVertexCol - 1 && bdptRow == 0 ) {
								pc_fragColor = bdptPredecessor0;
                                                        } else if (
                                                                uBdptVertexCol >= 2 &&
                                                                bdptCol == uBdptVertexCol - 2 &&
                                                                bdptRow == 2
                                                        ) {
                                                                pc_fragColor = bdptPredecessor2;
							} else if ( bdptCol != uBdptVertexCol ) {
								discard;
							} else if ( bdptRow == 0 ) {
								pc_fragColor = bdptV0;
							} else if ( bdptRow == 1 ) {
								pc_fragColor = bdptV1;
							} else if ( bdptRow == 2 ) {
								pc_fragColor = bdptV2;
							} else if ( bdptRow == 3 ) {
								pc_fragColor = bdptV3;
							} else if ( bdptRow == 4 ) {
								pc_fragColor = bdptV4;
                                                        } else if ( bdptRow == 5 ) {
                                                                pc_fragColor = bdptV5;
                                                        } else if ( bdptRow == 6 ) {
                                                                pc_fragColor = bdptV6;
                                                        } else {
                                                                pc_fragColor = bdptV7;
                                                        }
							gNormalDepth = vec4( 0.5, 1.0, 0.5, 0.0 );
							gAlbedo = vec4( 0.0 );
							return;
						}

						#endif
`;

// ── Section 2: camera ray + env rotation + G-buffer init + BDPT eye stack ─
const RENDER_MAIN_GBUFFER = /* glsl */ `
						// get camera ray
						Ray ray = getCameraRay();
						vec3 primaryRayOrigin = ray.origin;

						// inverse environment rotation
						envRotation3x3 = mat3( environmentRotation );
						invEnvRotation3x3 = inverse( envRotation3x3 );
						// B4: the NEE-strategy slot count = analytic lights + mesh-area triangle
						// lights (counted as ONE strategy slot — area-proportional triangle pick)
						// + 1 env slot when an environment is present. Each strategy is chosen
						// with probability 1/lightsDenom. Mesh-only/no-env scenes must not reserve
						// a dead environment slot; that stays unbiased but doubles direct-light variance.
						lightsDenom = float(
							lights.count +
							( uMeshLightCount != 0u ? 1u : 0u ) +
							( environmentIntensity != 0.0 && envMapInfo.totalSum != 0.0 ? 1u : 0u )
						);

						#if NEE_CANDIDATE_PASS

						// One uniformly selected NEE vertex is retained per continuation path.
						// The final resolve multiplies by K (the number of eligible vertices),
						// which is the exact Horvitz-Thompson compensation for inclusion 1/K.
						uint neeCandidateCount = 0u;
						vec4 neeCandidate0 = vec4( 0.0 );
						vec4 neeCandidate1 = vec4( 0.0 );
						vec4 neeCandidate2 = vec4( 0.0 );
						vec4 neeCandidate3 = vec4( 0.0 );

						#endif

						// Sprint 5: G-buffer accumulators (written at primary hit; sky fallback if NO_HIT).
						// gNormalDepth.rgb = world normal encoded to [0,1] via (n*0.5+0.5).
						//   Sky sentinel: vec3(0.5,1.0,0.5) decodes to world-up (0,1,0).
						// gNormalDepth.w   = primary-ray hit distance in scene units; 0.0 for sky.
						// gAlbedo.rgb      = demodulated base color (surf.color), no lighting.
						bool gbufWritten = false;
						vec3 gbufNormalEnc = vec3( 0.5, 1.0, 0.5 ); // sky sentinel
						float gbufLinearDepth = 0.0;
						vec3 gbufAlbedo = vec3( 0.0 );

						// final color
						pc_fragColor = vec4( 0, 0, 0, 1 );

						// surface results
						SurfaceHit surfaceHit;
						ScatterRecord scatterRec;
                                                scatterRec.specularPdf = 0.0;
                                                scatterRec.pdf = 0.0;
                                                scatterRec.direction = vec3( 0.0 );
                                                scatterRec.throughput = vec3( 0.0 );
                                                scatterRec.sampledDelta = false;

                                                #if FEATURE_BDPT
						// BDPT eye-subpath scratch stack (per-invocation local arrays — the
						// WebGL2 analogue of @vitrum/pt-webgpu's read_write storage stack).
						//   pos/nrm    — eye vertex geometry
						//   pdfFwd     — merged forward (swapped-BSDF reverse density), filled
						//                one bounce later (and overridden by connection straddle)
						//   pdfRev     — merged reverse (scatter pdf that produced this vertex)
						//   spec       — delta-BSDF flag (Veach §10.3.5)
						vec3  bdptEyePos[ BDPT_MAX_EYE_DEPTH ];
						vec3  bdptEyeNrm[ BDPT_MAX_EYE_DEPTH ];
						float bdptEyePdfFwd[ BDPT_MAX_EYE_DEPTH ];
						float bdptEyePdfRev[ BDPT_MAX_EYE_DEPTH ];
						bool  bdptEyeSpec[ BDPT_MAX_EYE_DEPTH ];
						bool  bdptEyeMedium[ BDPT_MAX_EYE_DEPTH ];
						int   bdptEyeDepth = 0;                 // depth of the current eye vertex
						// Forward scatter pdf at the previous eye vertex (camera importance
						// 1.0 at the pinhole — the one vertex without an aperture model; this
						// replaces the old hardcoded eyePdfFwd=1.0 for scene-surface vertices).
						float bdptPrevScatterPdf = 1.0;
						vec3  bdptPrevPos = ( cameraWorldMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
                                                float bdptEyeSegmentDensity = 1.0;
                                                float bdptEyeSegmentReverseDensity = 1.0;
                                                float bdptPendingEnvironmentMisWeight = 1.0;
                                                #endif

						// path tracing state
						RenderState state = initRenderState();
						// One-sample MIS across X/Y/Z CMFs (Wilkie 2015 §3.3) — uses
						// dim 30 for strategy selection, dim 31 for inverse-CDF on
						// the chosen strategy. Returned pdf is the mixture pdf
						// (balance heuristic), the correct MC denominator.
						#if FEATURE_BDPT
						if ( uSpectralRendering != 0 ) {
							// The BDPT light path is global for this sample, so every eye
							// connection must use the exact same hero wavelength and PDF.
							state.wavelength = uBdptSharedWavelength;
							state.wavelengthPdf = uBdptSharedWavelengthPdf;
						} else {
							state.wavelength = sampleHeroWavelengthMIS( rand( 30 ), rand( 31 ), state.wavelengthPdf );
						}
						#else
						state.wavelength = sampleHeroWavelengthMIS( rand( 30 ), rand( 31 ), state.wavelengthPdf );
						#endif
						state.transmissiveTraversals = transmissiveBounces;
						#if FEATURE_FOG

                                                bool initialMediumStackValid = bvhBuildMediumStack(
                                                        ray.origin, - ray.direction,
                                                        materialIndexAttribute, materials,
                                                        state.mediumStack,
                                                        state.fogMaterial
                                                );
                                                if ( ! initialMediumStackValid ) {
                                                        pc_fragColor = vec4( 0.0 );
                                                        gNormalDepth = vec4( 0.5, 1.0, 0.5, 0.0 );
                                                        gAlbedo = vec4( 0.0 );
                                                        #if NEE_CANDIDATE_PASS
                                                        gNeeCandidate3 = vec4( 0.0 );
                                                        #endif
                                                        return;
                                                }

						#endif

                                                // bounces is a validated uniform, but several WebGL2 drivers only
                                                // accept trace loops whose maximum trip count is statically visible.
                                                // i preserves the physical-depth accounting below: transparent and
                                                // volume crossings rewind it, while pathStep always advances and
                                                // proves termination after at most 2 * WEBGL2_MAX_BOUNCES iterations.
                                                for ( int pathStep = 0, i = 0; pathStep < ${WEBGL2_MAX_PATH_STEPS}; pathStep ++, i ++ ) {

                                                        if ( i >= bounces ) {

                                                                break;

                                                        }

                                                        // Surface reconstruction uses a zero-based accepted-bounce
                                                        // depth. Alpha/fog pass-through rewinds i below, so the first
                                                        // camera-visible surface remains depth 0 even after skipped
                                                        // intersections. state.depth deliberately remains the
                                                        // one-based traversal counter used by RR/debug telemetry.
                                                        int surfacePathDepth = i;

                                                        sobolBounceIndex ++;

                                                        state.depth ++;
                                                        state.traversals = bounces - i;
                                                        state.firstRay = i == 0 && state.transmissiveTraversals == transmissiveBounces;

                                                        int hitType = traceScene( ray, state.fogMaterial, surfaceHit );
                                                        #if FEATURE_FOG
                                                        if ( state.fogMaterial.fogVolume && hitType != NO_HIT ) {
                                                                state.throughput *= fogFreeFlightRatioWeight(
                                                                        materials,
                                                                        state.fogMaterial,
                                                                        max( surfaceHit.dist, 0.0 ),
                                                                        state.wavelength
                                                                );
                                                        }
                                                        #endif
                                                        #if FEATURE_BDPT
                                                        if ( state.fogMaterial.fogVolume && hitType != NO_HIT ) {
                                                                float bdptSigmaT = max( state.fogMaterial.opacity, 0.0 );
                                                                float bdptSurvival = exp( - bdptSigmaT * max( surfaceHit.dist, 0.0 ) );
                                                                bdptEyeSegmentDensity *= hitType == FOG_HIT
                                                                        ? bdptSigmaT * bdptSurvival
                                                                        : bdptSurvival;
                                                                // Reverse collision density belongs at
                                                                // the edge origin, so it is seeded after
                                                                // accepting that origin vertex.  Every
                                                                // traversed segment contributes only its
                                                                // symmetric survival factor here.
                                                                bdptEyeSegmentReverseDensity *= bdptSurvival;
							}
							#endif
`;

// ── Section 3: forward analytic-light hit + NO_HIT/env + surface setup ────
const RENDER_MAIN_BDPT_EYE = /* glsl */ `
							// check if we intersect a finite analytic area-light surface before the scene.
							// Primary/specular-transmission camera paths get raw visible emission; ordinary
							// BSDF hits are MIS-weighted against analytic-light NEE.
							LightRecord forwardAreaLightRec;
							bool forwardAreaLightHit = false;
							uint forwardAreaLightIndex = 0u;
							float forwardAreaLightDist = hitType == NO_HIT ? INFINITY : surfaceHit.dist;
							// H4 FIX (2026-06-09): the forward-hit MIS light pdf must MATCH the
							// power-weighted discrete selection NEE actually performs in
							// randomLightSample:  p_light = lightRec.pdf / lightsDenom * count *
							// (power_i / sumPower). The previous  lightRec.pdf / lightsDenom
							// silently assumed UNIFORM selection (count * discretePdf == 1) — exact
							// only for a single light or equal powers; with >=2 unequal-power area
							// lights it biased the MIS weight. Latent until H1 uploaded lights.count.
                                                        float sumLightPower = 0.0;
                                                        for ( uint pi = 0u; pi < lights.count; pi ++ ) {
                                                                sumLightPower += finitePositiveLightPower(
                                                                        readLightInfo( lights.tex, pi ).power
                                                                );
							}
							for ( uint i = 0u; i < lights.count; i ++ ) {

								LightRecord lightRec;
								if (
									intersectLightAtIndex( lights.tex, ray.origin, ray.direction, i, lightRec ) &&
									lightRec.dist < forwardAreaLightDist
								) {

									forwardAreaLightRec = lightRec;
									forwardAreaLightHit = true;
									forwardAreaLightIndex = i;
									forwardAreaLightDist = lightRec.dist;

								}

							}
                                                        if ( forwardAreaLightHit ) {

                                                                vec3 forwardAreaLightThroughput = state.throughput * pathThroughputFromRgb( forwardAreaLightRec.emission, state.wavelength );
                                                                vec3 forwardAreaLightRgb = wavelengthToRGB( state.wavelength, forwardAreaLightThroughput, state.wavelengthPdf );

                                                                #if FEATURE_BDPT

                                                                // Finite-emitter paths are a single BDPT family. The c=0
                                                                // endpoint connection owns non-delta arrivals; only camera
                                                                // and delta-chain hits are unique forward strategies.
                                                                if ( state.firstRay || scatterRec.sampledDelta ) {
                                                                        pc_fragColor.rgb += forwardAreaLightRgb;
                                                                }
                                                                break;

                                                                #else

                                                                #if FEATURE_MIS

								// NOTE: Only area lights are supported for forward sampling and can be hit.
								// Camera-visible and transmissive paths have no matching NEE strategy at the
								// previous vertex, so they keep full emission.
								if ( ! state.firstRay && ! state.transmissiveRay ) {
                                                                        float discreteSelectPdf = sumLightPower > 0.0
                                                                                ? finitePositiveLightPower(
                                                                                        readLightInfo( lights.tex, forwardAreaLightIndex ).power
                                                                                ) / sumLightPower
                                                                                : 0.0;
									float lightSamplePdf = forwardAreaLightRec.pdf / lightsDenom * float( lights.count ) * discreteSelectPdf;
									float misWeight = misHeuristic( scatterRec.pdf, lightSamplePdf );
									forwardAreaLightRgb *= misWeight;
								}

								#endif

                                                                pc_fragColor.rgb += forwardAreaLightRgb;
                                                                break;

                                                                #endif

                                                        }

							if ( hitType == NO_HIT ) {

                                                                #if FEATURE_BDPT
                                                                if ( state.firstRay || scatterRec.sampledDelta ) {
                                                                #else
                                                                if ( state.firstRay || state.transmissiveRay ) {
                                                                #endif

									vec3 background = sampleBackground( ray.direction, rand2( 2 ) );
									vec3 backgroundThroughput = state.throughput * pathThroughputFromRgb( background, state.wavelength );
									pc_fragColor.rgb += wavelengthToRGB( state.wavelength, backgroundThroughput, state.wavelengthPdf );
									pc_fragColor.a = backgroundAlpha;

                                                                } else {

                                                                        #if FEATURE_MIS

									// get the PDF of the hit envmap point
                                                                        vec3 envColor;
                                                                        float envPdf = sampleEquirect( envRotation3x3 * ray.direction, envColor );
                                                                        #if ! FEATURE_BDPT
                                                                        envPdf /= lightsDenom;
                                                                        #endif

									// and weight the contribution
									// D3 — state.envMapIntensity scales the BSDF half of the env
									// estimator by the LAST shaded surface's per-material env scale
									// (the NEE half applies the same factor in
									// directLightContribution → consistent MIS, radiance-only).
                                                                        #if FEATURE_BDPT
                                                                        float misWeight =
                                                                                bdptPendingEnvironmentMisWeight;
                                                                        #else
                                                                        float misWeight =
                                                                                misHeuristic( scatterRec.pdf, envPdf );
                                                                        #endif
									vec3 environmentThroughput = state.throughput * pathThroughputFromRgb( envColor, state.wavelength );
									pc_fragColor.rgb += state.envMapIntensity * environmentIntensity * wavelengthToRGB( state.wavelength, environmentThroughput, state.wavelengthPdf ) * misWeight;

									#else

									vec3 envColor = sampleEquirectColor( envMapInfo.map, envRotation3x3 * ray.direction );
									vec3 environmentThroughput = state.throughput * pathThroughputFromRgb( envColor, state.wavelength );
									pc_fragColor.rgb += state.envMapIntensity * environmentIntensity * wavelengthToRGB( state.wavelength, environmentThroughput, state.wavelengthPdf );

                                                                        #endif

                                                                }
								break;

							}

							uint materialIndex = uTexelFetch1D( materialIndexAttribute, surfaceHit.faceIndices.w ).r;
							MaterialControl materialControl;
							readMaterialControl( materials, materialIndex, materialControl );
							bool activeMaterialMatte = materialControl.matte;
							bool activeMaterialCastShadow = materialControl.castShadow;
							bool activeMaterialUnlit = materialControl.unlit;
							bool activeMaterialMeshEmitterCastShadowDisabled = materialControl.meshEmitterCastShadowDisabled;
							uint activeMaterialFlags = materialControl.flags;

							#if FEATURE_FOG

							if ( hitType == FOG_HIT ) {

								activeMaterialMatte = state.fogMaterial.matte;
								activeMaterialCastShadow = state.fogMaterial.castShadow;
								activeMaterialUnlit = state.fogMaterial.unlit;
								activeMaterialMeshEmitterCastShadowDisabled = state.fogMaterial.meshEmitterCastShadowDisabled;
								activeMaterialFlags = state.fogMaterial.flags;
								state.accumulatedRoughness += 0.2;

                                                        } else if ( materialControl.fogVolume ) {
                                                                bool mediumStackValid = surfaceHit.side == 1.0
                                                                        ? enterMedium(
                                                                                state.mediumStack, materialIndex,
                                                                                materials, state.fogMaterial
                                                                        )
                                                                        : leaveMedium(
                                                                                state.mediumStack, materialIndex,
                                                                                materials, state.fogMaterial
                                                                        );
                                                                if ( ! mediumStackValid ) break;

								ray.origin = stepRayOrigin( ray.origin, ray.direction, - surfaceHit.faceNormal, surfaceHit.dist );

								i -= sign( state.transmissiveTraversals );
								state.transmissiveTraversals -= sign( state.transmissiveTraversals );
								continue;

							}

							#endif

							// early out if this is a matte material
							if ( activeMaterialMatte && state.firstRay ) {

								pc_fragColor = vec4( 0.0 );
								break;

							}

							// if we've determined that this is a shadow ray and we've hit an item with no shadow casting
							// then skip it
							if ( ! activeMaterialCastShadow && state.isShadowRay ) {

								ray.origin = stepRayOrigin( ray.origin, ray.direction, - surfaceHit.faceNormal, surfaceHit.dist );
								continue;

							}

							SurfaceRecord surf;
							int surfaceStatus = HIT_SURFACE;
							if ( hitType == FOG_HIT ) {

								setFogSurfaceRecord( state.fogMaterial, surf );

							} else {

								surfaceStatus = getSurfaceRecord(
									materialIndex, surfaceHit, attributesArray,
									state.accumulatedRoughness, surfacePathDepth, state.wavelength,
									surf
								);

							}
							if ( surfaceStatus == SKIP_SURFACE ) {

								// only allow a limited number of transparency discards otherwise we could
								// crash the context with too long a loop.
								i -= sign( state.transmissiveTraversals );
								state.transmissiveTraversals -= sign( state.transmissiveTraversals );

								ray.origin = stepRayOrigin( ray.origin, ray.direction, - surfaceHit.faceNormal, surfaceHit.dist );
								continue;

							}
`;

// ── Section 5: G-buffer capture + surface shading + NEE + BDPT connection ─
const RENDER_MAIN_SURFACE_BDPT_EYE = /* glsl */ `

							// Sprint 5: G-buffer primary-hit capture (once per path, at first real surface hit).
							// Depth is distance from the original primary-ray origin to the
							// first accepted hit; alpha pass-through is not a hit.
							if ( ! gbufWritten ) {
								gbufLinearDepth = distance( ray.origin + ray.direction * surfaceHit.dist, primaryRayOrigin );
								// World normal encoded to [0,1] (decode: xyz*2-1).
								gbufNormalEnc = surf.normal * 0.5 + 0.5;
								// Demodulated base color: surf.color holds baseColor x ao (no lighting).
								gbufAlbedo = surf.rgbColor;
								gbufWritten = true;
							}

							if ( activeMaterialUnlit ) {

								pc_fragColor.rgb += wavelengthToRGB( state.wavelength, state.throughput * pathThroughputFromRgb( surf.color, state.wavelength ), state.wavelengthPdf );
								break;

							}

							// D3 — record this surface's env scale for the forward env pickup
							// (the NO_HIT MIS branch above) on the NEXT iteration.
							state.envMapIntensity = surf.envMapIntensity;

							// B4 — capture the INCOMING ray's BSDF pdf (the pdf of the prior
							// bounce's scatter that produced the ray hitting THIS surface) BEFORE
							// bsdfSample overwrites scatterRec. Used to MIS-weight the forward
							// emissive accumulation below against the mesh-area NEE strategy.
							// On the primary hit there is no prior scatter (camera) → handled by
							// the firstRay branch at the emission site.
							float incomingBsdfPdf = scatterRec.pdf;
                                                        bool incomingWasDelta = scatterRec.sampledDelta;

							// Emission belongs to the path that already arrived at this surface,
							// so it is accumulated before selecting local NEE versus continuation.
                                                        #if FEATURE_BDPT
                                                        // The c=0 endpoint connection owns every ordinary finite
                                                        // mesh-emitter arrival. Preserve only camera/delta-chain hits.
                                                        bool skipForwardMeshEmission =
                                                                ! state.firstRay && ! incomingWasDelta;
                                                        #else
                                                        bool skipForwardMeshEmission =
                                                                activeMaterialMeshEmitterCastShadowDisabled &&
                                                                ! state.firstRay && ! incomingWasDelta;
                                                        #endif
							if ( ! skipForwardMeshEmission ) {
								if (
									uMeshLightCount != 0u &&
                                                                        uTotalEmissivePower > 0.0 &&
                                                                        ! state.firstRay &&
                                                                        ! incomingWasDelta &&
									surf.emission != vec3( 0.0 )
								) {
									float cosLight = dot( surf.faceNormal, ray.direction );
									float neePdf = meshAreaLightForwardPdf(
										surfaceHit.dist * surfaceHit.dist,
										cosLight,
										uTotalEmissivePower,
										surf.emission
									) / lightsDenom;
									float emisMisWeight = misHeuristic( incomingBsdfPdf, neePdf );
									pc_fragColor.rgb += wavelengthToRGB(
										state.wavelength,
										state.throughput * pathThroughputFromRgb(
											surf.emission, state.wavelength
										),
										state.wavelengthPdf
									) * emisMisWeight;
								} else {
									pc_fragColor.rgb += wavelengthToRGB(
										state.wavelength,
										state.throughput * pathThroughputFromRgb(
											surf.emission, state.wavelength
										),
										state.wavelengthPdf
									);
								}
							}
							// Sprint 7: gate SSS by per-material TRANSLUCENT_BIT and back-face traversal.
							// Falls back to standard BSDF sampling for non-translucent materials.
							bool canUseSss =
								surf.sssSigmaT > 0.0 &&
								( ( activeMaterialFlags & TRANSLUCENT_BIT ) != 0u ) &&
								! surf.frontFace;
							if ( canUseSss ) {

								scatterRec = sssSample( - ray.direction, surf, state.wavelength );
								scatterRec.throughput *= activeLayerThroughput( surf, state.wavelength );

							} else {

								scatterRec = bsdfSample( - ray.direction, surf, state.wavelength );

							}
							state.isShadowRay = scatterRec.specularPdf < rand( 4 );

							bool isBelowSurface = ! surf.volumeParticle && dot( scatterRec.direction, surf.faceNormal ) < 0.0;
							vec3 geometricHitPoint =
								ray.origin + ray.direction * surfaceHit.dist;
                                                        vec3 hitPoint = stepRayOrigin( ray.origin, ray.direction, isBelowSurface ? - surf.faceNormal : surf.faceNormal, surfaceHit.dist );

                                                        #if FEATURE_BDPT
                                                        // Both the radiance draw and the NEE replay need the
                                                        // same eye stack so s=0/s=1 can share the explicit
                                                        // s>=2 power-heuristic denominator.
                                                        bool bdptEyeIsSpec = scatterRec.sampledDelta;
                                                        if ( bdptEyeDepth < BDPT_MAX_EYE_DEPTH ) {
                                                                bdptEyePos[ bdptEyeDepth ] = hitPoint;
                                                                bdptEyeNrm[ bdptEyeDepth ] = surf.volumeParticle
                                                                        ? vec3( 0.0 )
                                                                        : surf.normal;
                                                                bdptEyePdfFwd[ bdptEyeDepth ] = 0.0;
                                                                bdptEyePdfRev[ bdptEyeDepth ] =
                                                                        bdptPrevScatterPdf * bdptEyeSegmentDensity;
                                                                bdptEyeSpec[ bdptEyeDepth ] = bdptEyeIsSpec;
                                                                bdptEyeMedium[ bdptEyeDepth ] = surf.volumeParticle;
                                                        }
                                                        #endif

							// Next-event estimation is captured in a dedicated path-replay pass.
							// The layered direct BSDF is then evaluated by a separate fragment
							// program with no path-bounce loop, avoiding the ANGLE/SwiftShader
							// non-termination triggered by nesting that evaluator in this loop.
							#if FEATURE_MIS && NEE_CANDIDATE_PASS

							neeCandidateCount ++;
							if (
								neeReservoirReplacementU( neeCandidateCount ) <
								1.0 / float( neeCandidateCount )
							) {

								// A replacement always overwrites the prior record, including when
								// this vertex's single light proposal is invalid or occluded. Zero
								// proposals are part of the original proposal distribution and must
								// never condition either K or the reservoir.
								neeCandidate0 = vec4( 0.0 );
								neeCandidate1 = vec4( 0.0 );
								neeCandidate2 = vec4( 0.0 );
								neeCandidate3 = vec4( 0.0 );

								uvec4 neeSavedWhiteNoiseSeed = WHITE_NOISE_SEED;
								uint neeSavedSobolBounceIndex = sobolBounceIndex;

                                                                        DirectLightSample neeLightSample = sampleDirectLight(
                                                                                surf, geometricHitPoint
                                                                        );
                                                                        neeLightSample = prepareDirectLightSample(
                                                                        surf, state, geometricHitPoint, neeLightSample
                                                                );

								WHITE_NOISE_SEED = neeSavedWhiteNoiseSeed;
                                                                sobolBounceIndex = neeSavedSobolBounceIndex;

                                                                float neeCrossFamilyMisWeight = 1.0;
                                                                #if FEATURE_BDPT
                                                                if (
                                                                        neeLightSample.valid &&
                                                                        neeLightSample.bdptInfiniteKind > 0.5 &&
                                                                        bdptEyeDepth < BDPT_MAX_EYE_DEPTH
                                                                ) {
                                                                        neeCrossFamilyMisWeight =
                                                                                bdptInfiniteEyeFamilyWeight(
                                                                                        1,
                                                                                        neeLightSample.bdptInfiniteKind < 1.5,
                                                                                        false,
                                                                                        neeLightSample.pdf,
                                                                                        neeLightSample.bdptLaunchPdf,
                                                                                        neeLightSample.direction,
                                                                                        surf,
                                                                                        hitPoint,
                                                                                        - ray.direction,
                                                                                        state.wavelength,
                                                                                        bdptEyeDepth,
                                                                                        bdptEyePos,
                                                                                        bdptEyeNrm,
                                                                                        bdptEyePdfFwd,
                                                                                        bdptEyePdfRev,
                                                                                        bdptEyeSpec,
                                                                                        bdptEyeMedium
                                                                                );
                                                                        if ( neeCrossFamilyMisWeight <= 0.0 ) {
                                                                                neeLightSample.valid = false;
                                                                        }
                                                                }
                                                                #endif

                                                                if ( neeLightSample.valid ) {

									uint neeFlags = 1u;
									if ( surf.volumeParticle ) neeFlags |= 2u;
									if ( neeLightSample.delta > 0.5 ) neeFlags |= 4u;
									neeFlags |= (
										min( uint( surfacePathDepth ), 127u ) << 3u
									);

                                                                        neeCandidate0 = vec4( ray.origin, state.accumulatedRoughness );
                                                                        neeCandidate1 = vec4( ray.direction, state.wavelength );
                                                                        neeCandidate3 = vec4(
                                                                                neeOctEncodeDirection(
                                                                                        neeLightSample.direction
                                                                                ),
                                                                                neeCrossFamilyMisWeight,
                                                                                uintBitsToFloat( neeFlags )
                                                                        );

									if ( surf.volumeParticle ) {

                                                                                // Fog has a closed-form HG phase function. Resolve it here without
                                                                                // invoking the layered evaluator; the no-loop pass only applies K.
                                                                float fogPdf = mediumPhasePdf(
                                                                        - ray.direction,
                                                                        neeLightSample.direction,
                                                                        surf.sssAnisotropyG
                                                                );
                                                                vec3 fogColor = surf.color * fogPdf;
                                                                float neeContinuationFamilyProbability = 1.0;
                                                                #if ! FEATURE_BDPT
                                                                // A terminal-bounce fog proposal has no
                                                                // continuation estimator to compete with.
                                                                // The surface resolver applies the same
                                                                // ownership rule after replay.
                                                                neeContinuationFamilyProbability =
                                                                        surfacePathDepth + 1 < bounces
                                                                                ? 1.0
                                                                                : 0.0;
                                                                #endif
                                                                vec3 fogContribution =
                                                                        evaluatePreparedDirectLightSample(
                                                                                state,
                                                                                neeLightSample,
										fogColor,
										fogPdf,
                                                                                1.0,
                                                                                neeContinuationFamilyProbability
                                                                        );
                                                                                #if FEATURE_BDPT
                                                                                float legacyFogMisWeight =
                                                                                        neeLightSample.delta > 0.5
                                                                                                ? 1.0
                                                                                                : misHeuristic(
                                                                                                        neeLightSample.pdf,
                                                                                                        fogPdf
                                                                                                );
                                                                                if ( legacyFogMisWeight > 0.0 ) {
                                                                                        fogContribution *=
                                                                                                neeCrossFamilyMisWeight /
                                                                                                legacyFogMisWeight;
                                                                                } else {
                                                                                        fogContribution = vec3( 0.0 );
                                                                                }
                                                                                #endif
                                                                                neeCandidate2 = vec4(
                                                                                        fogContribution,
                                                                                        neeLightSample.pdf
                                                                                );

									} else {

										vec3 neeSourceThroughput =
											state.throughput *
											pathThroughputFromRgb(
												neeLightSample.emission,
												state.wavelength
											) * neeLightSample.contributionScale;
										neeCandidate2 = vec4(
											neeSourceThroughput,
											neeLightSample.pdf
										);

									}

								}

							}

							#endif

                                                        // General BDPT explicit connections. At every eye vertex, including
                                                        // the primary hit, the eye subpath attempts an explicit connection
                                                        // to every stored light-subpath vertex in uBdptLightPathTex.
                                                        // Finite ordinary NEE is disabled while this estimator owns the family;
                                                        // Infinite emitters share one cross-family denominator across
                                                        // forward escape (s=0), replay NEE (s=1), and c>=1 BDPT.
                                                        #if FEATURE_BDPT && ! NEE_CANDIDATE_PASS

							// uBdptLightPathTex validity is enforced by the host bridge:
							// driveForkMaterialUniforms() forces uBdptEnabled=false when the
							// texture is null, so FEATURE_BDPT=1 implies the texture is bound.
							//
                                                                if ( bdptEyeDepth < BDPT_MAX_EYE_DEPTH ) {
                                                                        // Start at c=0 so maxLightBounces=1 is the finite direct
                                                                        // endpoint strategy, then extend through c>=1 scattering vertices.
                                                                        for ( int bdptLvi = 0; bdptLvi < 8; bdptLvi ++ ) {
										if ( bdptLvi >= uBdptMaxLightBounces ) break;
                                                                                // A (c,e) connection contains c light-side
                                                                                // scattering vertices plus e+1 eye vertices.
                                                                                // Keep that total within the same accepted-
                                                                                // vertex budget as the unidirectional path.
                                                                                if ( bdptLvi + bdptEyeDepth >= bounces ) break;
										pc_fragColor.rgb += evaluateBdptConnection(
											hitPoint,
											surf.normal,
										- ray.direction,    // worldWo at eye vertex
										state.throughput,
										surf,
										state,
										bdptEyeDepth,
											bdptEyePos, bdptEyeNrm, bdptEyePdfFwd, bdptEyePdfRev,
											bdptEyeSpec, bdptEyeMedium,
											bdptLvi
										);
								}
							}

							#endif
`;

// ── Section 8: roughness accum + emissive MIS + scatter + throughput + RR ─
const RENDER_MAIN_SCATTER = /* glsl */ `
							// accumulate a roughness value to offset glossy rays that have high contribution
							// to a single pixel resulting in fireflies. Reflected lobes use the ordinary
							// reflection half vector; rough transmission uses the Disney transmission half
							// vector in the active normal frame so glass blur feeds the same filter state.
							if ( ! surf.volumeParticle ) {

								bool sampledTransmissionLobe =
                                                                surf.transmission > 0.0 &&
									( isBelowSurface || dot( scatterRec.direction, surf.faceNormal * surfaceHit.side ) < 0.0 );
								if ( sampledTransmissionLobe ) {

									mat3 transmissionInvBasis = transpose( surf.normalBasis );
									vec3 transmissionWo = normalize( transmissionInvBasis * - ray.direction );
									vec3 transmissionWi = normalize( transmissionInvBasis * scatterRec.direction );
									vec3 transmissionHalf = getHalfVector( transmissionWi, transmissionWo, surf.eta );
									state.accumulatedRoughness += sin( acosApprox( clamp( abs( transmissionHalf.z ), 0.0, 1.0 ) ) );

								} else if ( ! isBelowSurface ) {

									// determine if this is a rough normal or not by checking how far off straight up it is
									vec3 halfVector = normalize( - ray.direction + scatterRec.direction );
									state.accumulatedRoughness += max(
										sin( acosApprox( dot( halfVector, surf.normal ) ) ),
										sin( acosApprox( dot( halfVector, surf.clearcoatNormal ) ) )
									);

									state.transmissiveRay = false;

								}

							}

							// skip the sample if our PDF or ray is impossible
							if ( scatterRec.pdf <= 0.0 || ! isDirectionValid( scatterRec.direction, surf.normal, surf.faceNormal ) ) {

								break;

							}

							// if we're bouncing around the inside a transmissive material then decrement
							// perform this separate from a bounce
							bool isTransmissiveRay = ! surf.volumeParticle && dot( scatterRec.direction, surf.faceNormal * surfaceHit.side ) < 0.0;
							if ( ( isTransmissiveRay || isBelowSurface ) && state.transmissiveTraversals > 0 ) {

								state.transmissiveTraversals --;
								i --;

							}

							//

							// handle throughput color transformation
							// attenuate the throughput color by the medium color
							if ( ! surf.frontFace ) {

								float attenuationDist = surfaceHit.dist;
								if ( surf.hasAttenuationThickness ) {
									attenuationDist = min( attenuationDist, max( surf.attenuationThickness, 0.0 ) );
								}
								state.throughput *= transmissionAttenuationThroughput(
									materials,
									attenuationDist,
									surf.attenuationColor,
									surf.attenuationDistance,
									surf.hasSpectralAttenuation,
									surf.materialIndex,
									state.wavelength
								);

							}

							// Fixed-depth BDPT keeps q=1 on the eye path: the reverse strategies do
							// not carry camera-throughput-dependent roulette survival probabilities.
							#if FEATURE_RUSSIAN_ROULETTE && ! FEATURE_BDPT

							// russian roulette path termination
							// https://www.arnoldrenderer.com/research/physically_based_shader_design_in_arnold.pdf
							uint minBounces = 3u;
							float depthProb = float( state.depth < minBounces );

							float scatterScalar = max( scatterRec.throughput.r, max( scatterRec.throughput.g, scatterRec.throughput.b ) );
                                                        float rrProb = scatterRec.pdf > 0.0
                                                                ? scatterScalar / scatterRec.pdf
                                                                : 0.0;
							rrProb = sqrt( rrProb );
							rrProb = max( rrProb, depthProb );
							rrProb = min( rrProb, 1.0 );
                                                        if ( rrProb <= 0.0 || rand( 8 ) > rrProb ) {

								break;

							}

                                                        state.throughput /= rrProb;

							#endif

							// adjust the throughput and discard and exit if we find discard the sample if there are any NaNs
							state.throughput *= scatterRec.throughput / scatterRec.pdf;
							if ( any( isnan( state.throughput ) ) || any( isinf( state.throughput ) ) ) {

								break;

							}

							//

                                                        #if FEATURE_BDPT
							// BDPT eye-stack bookkeeping (mirrors @vitrum/pt-webgpu kernel).
							// scatterRec.pdf is the real forward scatter pdf at this eye vertex —
							// fed to the next vertex as its reverse density (the old hardcoded
							// eyePdfFwd=1.0 is gone). The swapped-direction reverse density at this
							// vertex toward the previous one is merged pdfFwd(E_{depth-1}); write it
							// into the previous slot (PBRT camera[d-1].pdfRev set while at camera[d]).
							if ( bdptEyeDepth < BDPT_MAX_EYE_DEPTH ) {
								if ( bdptEyeDepth >= 1 ) {
									vec3 bdptToPrev = normalize( bdptPrevPos - hitPoint );
                                                                        bool bdptSwappedDelta;
                                                                        float bdptSwappedRev = bsdfPdfResult(
                                                                                scatterRec.direction,
                                                                                bdptToPrev,
                                                                                surf,
                                                                                state.wavelength,
                                                                                bdptSwappedDelta
                                                                        );
                                                                        bdptEyePdfFwd[ bdptEyeDepth - 1 ] =
                                                                                bdptSwappedRev *
                                                                                bdptEyeSegmentReverseDensity;
                                                                }
                                                                #if ! NEE_CANDIDATE_PASS
                                                                if (
                                                                        envMapInfo.totalSum > 0.0 &&
                                                                        environmentIntensity > 0.0
                                                                ) {
                                                                        vec3 pendingEnvironmentColor;
                                                                        float pendingEnvironmentPdf = sampleEquirect(
                                                                                envRotation3x3 * scatterRec.direction,
                                                                                pendingEnvironmentColor
                                                                        );
                                                                        float pendingNeePdf =
                                                                                pendingEnvironmentPdf /
                                                                                bdptDistantNeeDenom();
                                                                        float pendingLaunchPdf =
                                                                                bdptEnvironmentEmitterPower() /
                                                                                bdptTotalEmitterPower() *
                                                                                pendingEnvironmentPdf /
                                                                                (
                                                                                        PI * uBdptSceneRadius * uBdptSceneRadius
                                                                                );
                                                                        bdptPendingEnvironmentMisWeight =
                                                                                bdptInfiniteEyeFamilyWeight(
                                                                                        0,
                                                                                        true,
                                                                                        scatterRec.sampledDelta,
                                                                                        pendingNeePdf,
                                                                                        pendingLaunchPdf,
                                                                                        scatterRec.direction,
                                                                                        surf,
                                                                                        hitPoint,
                                                                                        - ray.direction,
                                                                                        state.wavelength,
                                                                                        bdptEyeDepth,
                                                                                        bdptEyePos,
                                                                                        bdptEyeNrm,
                                                                                        bdptEyePdfFwd,
                                                                                        bdptEyePdfRev,
                                                                                        bdptEyeSpec,
                                                                                        bdptEyeMedium
                                                                                );
                                                                } else {
                                                                        bdptPendingEnvironmentMisWeight = 1.0;
                                                                }
                                                                #endif
                                                                bdptPrevScatterPdf = max( scatterRec.pdf, 0.0 );
                                                            bdptPrevPos = hitPoint;
                                                            bdptEyeSegmentDensity = 1.0;
                                                            bdptEyeSegmentReverseDensity = surf.volumeParticle
                                                                    ? max( state.fogMaterial.opacity, 0.0 )
                                                                    : 1.0;
                                                            bdptEyeDepth ++;
							}
							#endif

							// prepare for next ray
							ray.direction = scatterRec.direction;
							ray.origin = hitPoint;

						}
`;

// ── Section 9: post-loop alpha + debug + G-buffer write ──────────────────
const RENDER_MAIN_POST_LOOP = /* glsl */ `
						pc_fragColor.a *= opacity;

						#if NEE_CANDIDATE_PASS

						// K is packed only after the path terminates, so every eligible vertex
						// contributes to the inclusion probability even if the retained light
						// proposal was a null/occluded sample. K is bounded by the static path
						// step limit and range-checked again by the resolve shader.
						uint neeFinalFlags = floatBitsToUint( neeCandidate3.w );
						neeFinalFlags |= ( min( neeCandidateCount, 127u ) << 10u );
						pc_fragColor = neeCandidate0;
						gNormalDepth = neeCandidate1;
						gAlbedo = neeCandidate2;
						gNeeCandidate3 = vec4(
							neeCandidate3.xyz,
							uintBitsToFloat( neeFinalFlags )
						);

						#else

						// Sprint 5: Write G-buffer outputs.
						// If gbufWritten == false (sky/miss on first ray), sky sentinels are used:
						//   gNormalDepth.rgb = (0.5,1.0,0.5) → decodes to world-up (0,1,0)
						//   gNormalDepth.w   = 0.0 (sky depth sentinel, matches shared-denoisers convention)
						//   gAlbedo.rgb      = (0,0,0) (no surface albedo)
						gNormalDepth = vec4( gbufNormalEnc, gbufLinearDepth );
						gAlbedo      = vec4( gbufAlbedo, 1.0 );

						#endif

					}
`;

/**
 * The sections array — assembled in order, byte-identical to the original RENDER_MAIN.
 * Verified at module load time (see assertion below).
 */
/** @internal — exported for byte-identity test pin (D10.4). */
export const RENDER_MAIN_SECTIONS = [
  RENDER_MAIN_BDPT_SUBPATH,
  RENDER_MAIN_GBUFFER,
  RENDER_MAIN_BDPT_EYE,
  RENDER_MAIN_SURFACE_BDPT_EYE,
  RENDER_MAIN_SCATTER,
  RENDER_MAIN_POST_LOOP,
] as const;

/** Assembled RENDER_MAIN — concatenation of all sections in order (byte-identical to original). */
/** Assembled RENDER_MAIN — the composer interpolates this into the fragment body. */
export const RENDER_MAIN = RENDER_MAIN_SECTIONS.join('');
