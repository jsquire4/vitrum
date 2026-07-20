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
 * transcribed verbatim here. All FEATURE_ and DEBUG_MODE gates resolve from the preamble
 * defines at compile time.
 *
 * D10.4 (2026-06-11): split into named section constants assembled in order.
 * BYTE-IDENTITY CONTRACT: RENDER_MAIN_SECTIONS.join('') === original RENDER_MAIN string.
 * Sections are split at natural inline-comment boundaries; no whitespace is added or removed.
 */

// ── Section 1: function header + RNG init + BDPT light-subpath pass ───────
const RENDER_MAIN_BDPT_SUBPATH = /* glsl */ `
					void main() {

						// init
						rng_initialize( gl_FragCoord.xy, seed );
						sobolPixelIndex = ( uint( gl_FragCoord.x ) << 16 ) | uint( gl_FragCoord.y );
						sobolPathIndex = uint( seed );

						#if FEATURE_BDPT

						// Sprint 10c — dedicated light-subpath draw (one column per dispatch).
						if ( uBdptLightSubpathPass != 0 ) {

							// BDPT subpath RNG made row-independent (one coherent path per vertex
							// column), 2026-06-10 — RENDER-CHANGING for bdpt:true only; off-path
							// byte-identical.
							//
							// The texture is 5 rows × N columns. Five fragments (one per row)
							// cooperate to write one vertex: row 0 = position|kind, row 1 =
							// normal|pdfFwd, row 2 = throughput|pdfRev, row 3 = BSDF state,
							// row 4 = hit/material payload
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
							rng_initialize( vec2( gl_FragCoord.x, 0.0 ), seed );

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
							bdptState.wavelength = sampleHeroWavelengthMIS( rand( 30 ), rand( 31 ), bdptState.wavelengthPdf );
							#if FEATURE_FOG

							Ray fogRay;
							fogRay.origin = vec3( 0.0 );
							fogRay.direction = vec3( 0.0, 1.0, 0.0 );
							bdptState.fogMaterial.fogVolume = bvhIntersectFogVolumeHit(
								fogRay.origin, - fogRay.direction,
								materialIndexAttribute, materials,
								bdptState.fogMaterial
							);

							#endif

							vec4 bdptV0;
							vec4 bdptV1;
							vec4 bdptV2;
							vec4 bdptV3;
							vec4 bdptV4;
							writeLightSubpathVertex(
								uBdptVertexCol,
								uBdptMaxLightBounces,
								uBdptLightPathTex,
								bdptState.fogMaterial,
								bdptState.wavelength,
								bdptV0,
								bdptV1,
								bdptV2,
								bdptV3,
								bdptV4
							);

							if ( int( gl_FragCoord.x ) != uBdptVertexCol ) {
								discard;
							}

							int bdptRow = int( gl_FragCoord.y );
							if ( bdptRow == 0 ) {
								pc_fragColor = bdptV0;
							} else if ( bdptRow == 1 ) {
								pc_fragColor = bdptV1;
							} else if ( bdptRow == 2 ) {
								pc_fragColor = bdptV2;
							} else if ( bdptRow == 3 ) {
								pc_fragColor = bdptV3;
							} else {
								pc_fragColor = bdptV4;
							}
							gNormalDepth = vec4( 0.0 );
							gAlbedo = vec4( 0.0 );
							return;
						}

						#endif
`;

// ── Section 2: camera ray + env rotation + G-buffer init + BDPT eye stack ─
const RENDER_MAIN_GBUFFER = /* glsl */ `
						// get camera ray
						Ray ray = getCameraRay();

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

						// Sprint 5: G-buffer accumulators (written at primary hit; sky fallback if NO_HIT).
						// gNormalDepth.rgb = world normal encoded to [0,1] via (n*0.5+0.5).
						//   Sky sentinel: vec3(0.5,1.0,0.5) decodes to world-up (0,1,0).
						// gNormalDepth.w   = linear depth (camera-space, always positive); 0.0 for sky.
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
						int   bdptEyeDepth = 0;                 // depth of the current eye vertex
						// Forward scatter pdf at the previous eye vertex (camera importance
						// 1.0 at the pinhole — the one vertex without an aperture model; this
						// replaces the old hardcoded eyePdfFwd=1.0 for scene-surface vertices).
						float bdptPrevScatterPdf = 1.0;
						vec3  bdptPrevPos = ( cameraWorldMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
						#endif

						// path tracing state
						RenderState state = initRenderState();
						// One-sample MIS across X/Y/Z CMFs (Wilkie 2015 §3.3) — uses
						// dim 30 for strategy selection, dim 31 for inverse-CDF on
						// the chosen strategy. Returned pdf is the mixture pdf
						// (balance heuristic), the correct MC denominator.
						state.wavelength = sampleHeroWavelengthMIS( rand( 30 ), rand( 31 ), state.wavelengthPdf );
						state.transmissiveTraversals = transmissiveBounces;
						#if FEATURE_FOG

						state.fogMaterial.fogVolume = bvhIntersectFogVolumeHit(
							ray.origin, - ray.direction,
							materialIndexAttribute, materials,
							state.fogMaterial
						);

						#endif

						for ( int i = 0; i < bounces; i ++ ) {

							sobolBounceIndex ++;

							state.depth ++;
							state.traversals = bounces - i;
							state.firstRay = i == 0 && state.transmissiveTraversals == transmissiveBounces;

							int hitType = traceScene( ray, state.fogMaterial, surfaceHit );
							vec3 throughputRgb = wavelengthToRGB( state.wavelength, state.throughput, state.wavelengthPdf );
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
								sumLightPower += max( readLightInfo( lights.tex, pi ).power, 1e-20 );
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

								vec3 forwardAreaLightRgb = forwardAreaLightRec.emission * throughputRgb;

								#if FEATURE_MIS

								// NOTE: Only area lights are supported for forward sampling and can be hit.
								// Camera-visible and transmissive paths have no matching NEE strategy at the
								// previous vertex, so they keep full emission.
								if ( ! state.firstRay && ! state.transmissiveRay ) {
									float discreteSelectPdf = sumLightPower > 1e-30
										? max( readLightInfo( lights.tex, forwardAreaLightIndex ).power, 1e-20 ) / sumLightPower
										: 1.0 / max( float( lights.count ), 1.0 );
									float lightSamplePdf = forwardAreaLightRec.pdf / lightsDenom * float( lights.count ) * discreteSelectPdf;
									float misWeight = misHeuristic( scatterRec.pdf, lightSamplePdf );
									forwardAreaLightRgb *= misWeight;
								}

								#endif

								pc_fragColor.rgb += forwardAreaLightRgb;
								break;

							}

							if ( hitType == NO_HIT ) {

								if ( state.firstRay || state.transmissiveRay ) {

									pc_fragColor.rgb += sampleBackground( ray.direction, rand2( 2 ) ) * throughputRgb;
									pc_fragColor.a = backgroundAlpha;

								} else {

									#if FEATURE_MIS

									// get the PDF of the hit envmap point
									vec3 envColor;
									float envPdf = sampleEquirect( envRotation3x3 * ray.direction, envColor );
									envPdf /= lightsDenom;

									// and weight the contribution
									// D3 — state.envMapIntensity scales the BSDF half of the env
									// estimator by the LAST shaded surface's per-material env scale
									// (the NEE half applies the same factor in
									// directLightContribution → consistent MIS, radiance-only).
									float misWeight = misHeuristic( scatterRec.pdf, envPdf );
									pc_fragColor.rgb += state.envMapIntensity * environmentIntensity * envColor * throughputRgb * misWeight;

									#else

									pc_fragColor.rgb +=
										state.envMapIntensity *
										environmentIntensity *
										sampleEquirectColor( envMapInfo.map, envRotation3x3 * ray.direction ) *
										throughputRgb;

									#endif

								}
								break;

							}

							uint materialIndex = uTexelFetch1D( materialIndexAttribute, surfaceHit.faceIndices.w ).r;
							Material material = readMaterialInfo( materials, materialIndex );

							#if FEATURE_FOG

							if ( hitType == FOG_HIT ) {

								material = state.fogMaterial;
								state.accumulatedRoughness += 0.2;

							} else if ( material.fogVolume ) {

								state.fogMaterial = material;
								state.fogMaterial.fogVolume = surfaceHit.side == 1.0;

								ray.origin = stepRayOrigin( ray.origin, ray.direction, - surfaceHit.faceNormal, surfaceHit.dist );

								i -= sign( state.transmissiveTraversals );
								state.transmissiveTraversals -= sign( state.transmissiveTraversals );
								continue;

							}

							#endif

							// early out if this is a matte material
							if ( material.matte && state.firstRay ) {

								pc_fragColor = vec4( 0.0 );
								break;

							}

							// if we've determined that this is a shadow ray and we've hit an item with no shadow casting
							// then skip it
							if ( ! material.castShadow && state.isShadowRay ) {

								ray.origin = stepRayOrigin( ray.origin, ray.direction, - surfaceHit.faceNormal, surfaceHit.dist );
								continue;

							}

							SurfaceRecord surf;
							if (
								getSurfaceRecord(
									material, materialIndex, surfaceHit, attributesArray,
									state.accumulatedRoughness, int( state.depth ), state.wavelength,
									surf
								) == SKIP_SURFACE
							) {

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
							// Linear depth: project world-space hit point onto camera -Z axis.
							//   camForward = camera's -Z world-space direction (Three.js convention).
							//   camPos     = camera world-space position.
							//   linearDepth = dot( hitPoint - camPos, camForward ) — always positive in front of camera.
							if ( state.firstRay && ! gbufWritten ) {
								vec3 hitPos = ray.origin + ray.direction * surfaceHit.dist;
								// Camera forward direction in world space: cameraWorldMatrix * (0,0,-1,0)
								vec3 camForward = normalize( ( cameraWorldMatrix * vec4( 0.0, 0.0, - 1.0, 0.0 ) ).xyz );
								// Camera world position: cameraWorldMatrix * (0,0,0,1)
								vec3 camPos = ( cameraWorldMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
								// Linear (positive) depth along camera -Z axis.
								gbufLinearDepth = dot( hitPos - camPos, camForward );
								// World normal encoded to [0,1] (decode: xyz*2-1).
								gbufNormalEnc = surf.normal * 0.5 + 0.5;
								// Demodulated base color: surf.color holds baseColor x ao (no lighting).
								gbufAlbedo = surf.color;
								gbufWritten = true;
							}

							if ( material.unlit ) {

								pc_fragColor.rgb += surf.color * throughputRgb;
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
							bool incomingWasSpecular = scatterRec.specularPdf > 0.999;

							// Sprint 7: gate SSS by per-material TRANSLUCENT_BIT and back-face traversal.
							// Falls back to standard BSDF sampling for non-translucent materials.
							bool canUseSss =
								surf.sssSigmaT > 0.0 &&
								( ( material.flags & TRANSLUCENT_BIT ) != 0u ) &&
								! surf.frontFace;
							if ( canUseSss ) {
								scatterRec = sssSample( - ray.direction, surf, state.wavelength );
								scatterRec.throughput *= activeLayerThroughput( surf, state.wavelength );
							} else {
								scatterRec = bsdfSample( - ray.direction, surf, state.wavelength );
							}
							state.isShadowRay = scatterRec.specularPdf < rand( 4 );

							bool isBelowSurface = ! surf.volumeParticle && dot( scatterRec.direction, surf.faceNormal ) < 0.0;
							vec3 hitPoint = stepRayOrigin( ray.origin, ray.direction, isBelowSurface ? - surf.faceNormal : surf.faceNormal, surfaceHit.dist );

							// next event estimation
							#if FEATURE_MIS

							pc_fragColor.rgb += directLightContribution( - ray.direction, surf, state, hitPoint );

							#endif

							// Sprint 10c — BDPT explicit connections (depth > 0 only; skip primary hit).
							// At each indirect bounce the eye subpath attempts an explicit connection
							// to every stored light-subpath vertex in uBdptLightPathTex.
							// Primary hit (state.firstRay) is skipped to avoid double-counting with
							// the unidirectional NEE path above (direct_light_contribution_function).
							#if FEATURE_BDPT

							// uBdptLightPathTex validity is enforced by the host bridge:
							// driveForkMaterialUniforms() forces uBdptEnabled=false when the
							// texture is null, so FEATURE_BDPT=1 implies the texture is bound.
							//
							// Push this eye vertex (E_bdptEyeDepth) onto the local scratch stack
							// BEFORE connecting: pdfRev = forward scatter pdf at the previous
							// vertex that produced it (camera importance 1.0 at the primary hit).
							// pdfFwd is filled one bounce later by the swapped reverse density
							// (and overridden by the connection straddle when this is E_e/E_{e-1}).
							bool bdptEyeIsSpec = ( surf.transmission > 0.5 && surf.filteredRoughness < 0.05 );
							if ( bdptEyeDepth < BDPT_MAX_EYE_DEPTH ) {
								bdptEyePos[ bdptEyeDepth ] = hitPoint;
								bdptEyeNrm[ bdptEyeDepth ] = surf.normal;
								bdptEyePdfFwd[ bdptEyeDepth ] = 0.0; // filled next bounce / overridden
								bdptEyePdfRev[ bdptEyeDepth ] = bdptPrevScatterPdf;
								bdptEyeSpec[ bdptEyeDepth ] = bdptEyeIsSpec;
							}
								// Skip the primary hit: an explicit connection there double-counts
								// with the unidirectional NEE above (fork !state.firstRay).
								if ( ! state.firstRay && bdptEyeDepth < BDPT_MAX_EYE_DEPTH ) {
									vec3 throughputRgbBdpt = wavelengthToRGB( state.wavelength, state.throughput, state.wavelengthPdf );
									// bdptLvi=0 is the emitter endpoint, which matches the same
									// per-bounce direct-light strategy already estimated by NEE.
									// Start at the first scattered light vertex so the safe
									// maxLightBounces=1 default remains radiometrically neutral.
									for ( int bdptLvi = 1; bdptLvi < uBdptMaxLightBounces; bdptLvi ++ ) {
										pc_fragColor.rgb += evaluateBdptConnection(
											hitPoint,
											surf.normal,
										- ray.direction,    // worldWo at eye vertex
										throughputRgbBdpt,
										surf,
										state,
										bdptEyeDepth,
										bdptEyePos, bdptEyeNrm, bdptEyePdfFwd, bdptEyePdfRev, bdptEyeSpec,
										bdptLvi
									);
								}
							}

							#endif
`;

// ── Section 6: caustic manifold-NEE heuristic (strategy 1) ─────────────── 
const RENDER_MAIN_CAUSTIC_MANIFOLD = /* glsl */ `
							// RFE-05 strategy behavior hook:
							// strategy 1 ('manifold-nee') => deterministic refraction-walk heuristic.
							//   NOT the Newton-solve MNEE of pt-webgpu. Walks the refracted chain,
							//   treats escape-to-environment as reachedLight=true, adds
							//   throughput * color * pow(dot(walkDir,-rayDir), 10) as a focus weight.
							//   No constraint manifold, no Newton solver — a heuristic approximation.
							//   Port of the real pt-webgpu MNEE is a road-to-100 fidelity item.
							// strategy 2 ('photon-map') => deterministic cone-traced density estimate.
							//   Casts 8 cone sample rays, uses an inverse-distance kernel for hits and
							//   adds 1.0 for escaped rays (no-hit). Known approximation: the escaped-ray
							//   energy-add (~21% energy bias at typical cone sizes) is a deliberate
							//   trade-off for visual clarity over physical accuracy, not a full
							//   bidirectional photon map.
							if ( uCausticStrategy > 0 && surf.transmission > 0.0 ) {
								if ( uCausticStrategy == 1 ) {
									// Skip manifold mode on rough refractive surfaces: the fixed-step
									// walk is intended for near-specular interfaces.
									if ( surf.filteredRoughness < 0.12 ) {
										float etaM = surf.frontFace ? ( 1.0 / max( surf.ior, 1.0 ) ) : max( surf.ior, 1.0 );
										vec3 walkDir = refract( ray.direction, surf.normal, etaM );
										if ( length( walkDir ) > 0.0 ) {
											walkDir = normalize( walkDir );
											vec3 walkOrigin = hitPoint;
											int maxWalkIter = int( clamp( floor( uMneeMaxIterations + 0.5 ), 1.0, 16.0 ) );
											int maxChain = int( clamp( floor( uMneeMaxChainLength + 0.5 ), 1.0, 8.0 ) );
											int traversedChain = 0;
											bool reachedLight = false;
											float chainAttenuation = 1.0;
											for ( int walkIter = 0; walkIter < 16; walkIter ++ ) {
												if ( walkIter >= maxWalkIter || traversedChain >= maxChain ) break;
												Ray walkRay;
												walkRay.origin = walkOrigin;
												walkRay.direction = walkDir;
												SurfaceHit walkHit;
												int walkHitType = traceScene( walkRay, state.fogMaterial, walkHit );
												if ( walkHitType == NO_HIT ) {
													reachedLight = true;
													break;
												}
												uint walkMaterialIndex = uTexelFetch1D( materialIndexAttribute, walkHit.faceIndices.w ).r;
												Material walkMaterial = readMaterialInfo( materials, walkMaterialIndex );
												if ( walkMaterial.transmission <= 0.0 ) {
													break;
												}
												vec3 walkHitPoint = stepRayOrigin( walkOrigin, walkDir, walkHit.faceNormal, walkHit.dist );
												float etaWalk = walkHit.side > 0.0
													? ( 1.0 / max( walkMaterial.ior, 1.0 ) )
													: max( walkMaterial.ior, 1.0 );
												vec3 nextDir = refract( walkDir, walkHit.faceNormal, etaWalk );
												if ( length( nextDir ) <= 1e-5 ) {
													break;
												}
												walkOrigin = walkHitPoint;
												walkDir = normalize( nextDir );
												chainAttenuation *= clamp( walkMaterial.transmission, 0.0, 1.0 );
												traversedChain ++;
											}
											if ( reachedLight ) {
												float focus = pow( max( dot( walkDir, - ray.direction ), 0.0 ), 10.0 );
												float chainNorm = 1.0 / max( float( traversedChain + 1 ), 1.0 );
												float manifoldWeight = focus * chainNorm * chainAttenuation;
												pc_fragColor.rgb += throughputRgb * surf.color * manifoldWeight;
											}
										}
									}
								} else if ( uCausticStrategy == 2 ) {`;

// ── Section 7: caustic photon-density estimate (strategy 2) ─────────────── 
const RENDER_MAIN_CAUSTIC_PHOTON = /* glsl */ `
									// Photon-density style estimate: cast a deterministic refracted cone
									// and estimate visible light density with an inverse-distance kernel.
									float etaP = surf.frontFace ? ( 1.0 / max( surf.ior, 1.0 ) ) : max( surf.ior, 1.0 );
									vec3 refrDir = refract( ray.direction, surf.normal, etaP );
									if ( length( refrDir ) > 0.0 ) {
										refrDir = normalize( refrDir );
										vec3 tangentA = normalize( abs( refrDir.x ) > 0.5 ? cross( refrDir, vec3( 0.0, 1.0, 0.0 ) ) : cross( refrDir, vec3( 1.0, 0.0, 0.0 ) ) );
										vec3 tangentB = normalize( cross( refrDir, tangentA ) );
										float coneRadius = mix( 0.01, 0.12, clamp( surf.filteredRoughness, 0.0, 1.0 ) );
										float photonAccum = 0.0;
										const int PHOTON_SAMPLES = 8;
										for ( int p = 0; p < PHOTON_SAMPLES; p ++ ) {
											float u = ( float( p ) + 0.5 ) / float( PHOTON_SAMPLES );
											float v = rand( 42 + p );
											float r = coneRadius * sqrt( u );
											float phi = 6.28318530718 * v;
											vec3 coneDir = normalize( refrDir + ( cos( phi ) * r ) * tangentA + ( sin( phi ) * r ) * tangentB );
											Ray photonRay;
											photonRay.origin = hitPoint;
											photonRay.direction = coneDir;
											SurfaceHit photonHit;
											int photonHitType = traceScene( photonRay, state.fogMaterial, photonHit );
											if ( photonHitType == NO_HIT ) {
												photonAccum += 1.0;
											} else {
												float d = max( photonHit.dist, 1e-3 );
												photonAccum += 1.0 / ( 1.0 + d * d );
											}
										}
										float density = photonAccum / float( PHOTON_SAMPLES );
										pc_fragColor.rgb += throughputRgb * surf.color * density * surf.transmission;
									}
								}
							}
`;

// ── Section 8: roughness accum + emissive MIS + scatter + throughput + RR ─
const RENDER_MAIN_SCATTER = /* glsl */ `
							// accumulate a roughness value to offset glossy rays that have high contribution
							// to a single pixel resulting in fireflies. Reflected lobes use the ordinary
							// reflection half vector; rough transmission uses the Disney transmission half
							// vector in the active normal frame so glass blur feeds the same filter state.
							if ( ! surf.volumeParticle ) {

								bool sampledTransmissionLobe =
									surf.transmission > 0.001 &&
									( isBelowSurface || dot( scatterRec.direction, surf.faceNormal * surfaceHit.side ) < 0.0 );
								if ( sampledTransmissionLobe ) {

									vec3 transmissionWo = normalize( surf.normalInvBasis * - ray.direction );
									vec3 transmissionWi = normalize( surf.normalInvBasis * scatterRec.direction );
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

							// accumulate emissive color
							// B4 — MIS the forward emissive hit against the mesh-area NEE strategy.
							// The fold (foldEmissiveEmitters) puts every mesh-area emitter's
							// radiance on its material, so any emissive (surf.emission > 0) surface is ALSO
							// a mesh-NEE triangle light. The forward hit (BSDF sampling) and the
							// NEE sample (light sampling) are the two MIS strategies; the forward
							// hit's weight is misHeuristic(bsdfPdf_incoming, neePdf). neePdf is the
								// emitted-power pdf (meshAreaLightForwardPdf), scaled by the
								// 1/lightsDenom strategy-selection probability.
							//   • primary hit / specular incoming: NEE could not have made this
							//     sample → weight 1 (full emission), no double-count.
							//   • else: balance/power-heuristic split with the NEE estimate.
							// When uMeshLightCount==0 this reduces to the raw add (byte-identical).
							bool skipForwardMeshEmission = material.meshEmitterCastShadowDisabled &&
								! state.firstRay && ! incomingWasSpecular;
							if ( ! skipForwardMeshEmission ) {
									if ( uMeshLightCount != 0u && uTotalEmissivePower > 0.0 &&
										! state.firstRay && ! incomingWasSpecular &&
										surf.emission != vec3( 0.0 ) && hitType != NO_HIT ) {
										float cosLight = dot( surf.faceNormal, ray.direction );
										float neePdf = meshAreaLightForwardPdf(
											surfaceHit.dist * surfaceHit.dist, cosLight, uTotalEmissivePower, surf.emission
										) / lightsDenom;
									float emisMisWeight = misHeuristic( incomingBsdfPdf, neePdf );
									pc_fragColor.rgb += ( surf.emission * throughputRgb * emisMisWeight );
								} else {
									pc_fragColor.rgb += ( surf.emission * throughputRgb );
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

							#if FEATURE_RUSSIAN_ROULETTE

							// russian roulette path termination
							// https://www.arnoldrenderer.com/research/physically_based_shader_design_in_arnold.pdf
							uint minBounces = 3u;
							float depthProb = float( state.depth < minBounces );

							float scatterScalar = max( scatterRec.throughput.r, max( scatterRec.throughput.g, scatterRec.throughput.b ) );
							float rrProb = scatterScalar / max( scatterRec.pdf, 1e-6 );
							rrProb = sqrt( rrProb );
							rrProb = max( rrProb, depthProb );
							rrProb = min( rrProb, 1.0 );
							if ( rand( 8 ) > rrProb ) {

								break;

							}

							// perform sample clamping here to avoid bright pixels
							state.throughput *= min( 1.0 / rrProb, 20.0 );

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
									vec3 bdptSwapColor;
									float bdptSwappedRev = bsdfResult( scatterRec.direction, bdptToPrev, surf, state.wavelength, bdptSwapColor );
									bdptEyePdfFwd[ bdptEyeDepth - 1 ] = bdptSwappedRev;
								}
								bdptPrevScatterPdf = max( scatterRec.pdf, 0.0 );
								bdptPrevPos = hitPoint;
								bdptEyeDepth ++;
							}
							#endif

							// prepare for next ray
							ray.direction = scatterRec.direction;
							ray.origin = hitPoint;

						}
`;

// ── Section 9: post-loop radiance clamp + alpha + debug + G-buffer write ──
const RENDER_MAIN_POST_LOOP = /* glsl */ `
						if ( uRadianceClamp > 0.0 ) {
							float sampleLuminance = dot( pc_fragColor.rgb, vec3( 0.2126, 0.7152, 0.0722 ) );
							if ( sampleLuminance > uRadianceClamp ) {
								pc_fragColor.rgb *= uRadianceClamp / sampleLuminance;
							}
						}

						pc_fragColor.a *= opacity;

						#if DEBUG_MODE == 1

						// output the number of rays checked in the path and number of
						// transmissive rays encountered.
						pc_fragColor.rgb = vec3(
							float( state.depth ),
							transmissiveBounces - state.transmissiveTraversals,
							0.0
						);
						pc_fragColor.a = 1.0;

						#endif

						// Sprint 5: Write G-buffer outputs.
						// If gbufWritten == false (sky/miss on first ray), sky sentinels are used:
						//   gNormalDepth.rgb = (0.5,1.0,0.5) → decodes to world-up (0,1,0)
						//   gNormalDepth.w   = 0.0 (sky depth sentinel, matches shared-denoisers convention)
						//   gAlbedo.rgb      = (0,0,0) (no surface albedo)
						gNormalDepth = vec4( gbufNormalEnc, gbufLinearDepth );
						gAlbedo      = vec4( gbufAlbedo, 1.0 );

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
  RENDER_MAIN_CAUSTIC_MANIFOLD,
  RENDER_MAIN_CAUSTIC_PHOTON,
  RENDER_MAIN_SCATTER,
  RENDER_MAIN_POST_LOOP,
] as const;

/** Assembled RENDER_MAIN — concatenation of all sections in order (byte-identical to original). */
/** Assembled RENDER_MAIN — the composer interpolates this into the fragment body. */
export const RENDER_MAIN = RENDER_MAIN_SECTIONS.join('');
