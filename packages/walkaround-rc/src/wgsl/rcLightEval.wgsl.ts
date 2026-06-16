/**
 * rcLightEval.wgsl.ts — RC probe-ray light evaluation functions.
 *
 * Factored out of probeRayCast.wgsl.ts (D13.3) following the
 * OCTAHEDRAL_CORE_WGSL / PCG_HASH_TO_F32_WGSL composition pattern.
 * probeRayCast.wgsl.ts composes two sub-exports at their respective
 * positions in the shader:
 *
 *   RC_SUN_VISIBILITY_WGSL — inserted where the inline traceSunVisibility
 *     was, BEFORE the RCLight struct / bind-group declarations (the function
 *     only references rcTraceFirstHit, rc_triMatId, rc_materials — all
 *     declared before the bind-group block in probeRayCast.wgsl.ts).
 *
 *   RC_NEE_POINTSPOT_WGSL — inserted where the inline rcEmitterNEE and
 *     evalRCPointSpotLights were, AFTER the RCLight struct / bind-group
 *     declarations (those functions reference rc_emitters, rc_lights,
 *     RC_LIGHT_POINT, RC_LIGHT_SPOT, RC_MAX_LIGHTS — all declared in
 *     the bind-group block just above).
 *
 * RC_LIGHT_EVAL_WGSL is the combined export (= RC_SUN_VISIBILITY_WGSL +
 * separator + RC_NEE_POINTSPOT_WGSL) kept for external consumers that want
 * the full light-eval block in one string.
 *
 * Exported functions:
 *   traceSunVisibility    — glass-aware sun shadow test (with Beer-Lambert glass)
 *   rcEmitterNEE          — one-sample-per-emitter rect-area NEE (2026-06-07)
 *   evalRCPointSpotLights — point/spot analytic light evaluation (A7, 2026-06-10)
 */

/**
 * Glass-aware sun shadow test (traceSunVisibility).
 * Inserted BEFORE the RCLight struct / bind-group declarations in
 * probeRayCast.wgsl.ts — only references rcTraceFirstHit, rc_triMatId,
 * and rc_materials, which are all declared earlier in that file.
 */
export const RC_SUN_VISIBILITY_WGSL = /* wgsl */`// ─── Sun visibility helper ────────────────────────────────────────────────────
// Glass-aware sun shadow test.  Verbatim from sunVisibilityHelper wgslFn.

// M14 audit remediation: slabStepSize replaces the Cornell-specific 0.5-unit
// glass-slab step. Callers compute it from the scene extent
// (min(roomSize) * 0.001) so the step is proportional to the actual scene.
fn traceSunVisibility(
  origin:        vec3f,
  sunDir:        vec3f,
  slabStepSize:  f32,
  triEps:        f32,
) -> vec3f {
  var visibility = vec3f(1.0);
  var rayOrigin  = origin;
  for (var iter: u32 = 0u; iter < 3u; iter = iter + 1u) {
    var sRay = Ray();
    sRay.origin    = rayOrigin;
    sRay.direction = sunDir;
    let sHit = rcTraceFirstHit(sRay, triEps);
    if (!sHit.didHit) {
      return visibility;
    }
    let hitPos = rayOrigin + sunDir * sHit.dist;
    let sMatId = rc_triMatId[sHit.indices.w];
    let sMat   = rc_materials[sMatId];
    if ((sMat.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) != 0u) {
      rayOrigin = hitPos + sunDir * slabStepSize;
      continue;
    }
    if (sMat.transmission <= 0.5) {
      return vec3f(0.0);
    }
    let gThick    = max(0.001, sMat.thickness);
    let gAttenCol = sMat.attenuationColor;
    let gColor    = sMat.baseColor;
    let beerAtten = exp(-gAttenCol * (gThick / max(0.001, sMat.attenuationDistance)));
    visibility = visibility * gColor * beerAtten;
    rayOrigin  = hitPos + sunDir * slabStepSize;
  }
  return vec3f(0.0);
}`;

/**
 * Rect-area emitter NEE (rcEmitterNEE) and point/spot analytic lights
 * (evalRCPointSpotLights). Inserted AFTER the RCLight struct / bind-group
 * declarations in probeRayCast.wgsl.ts — these functions reference
 * rc_emitters, rc_lights, RC_LIGHT_POINT, RC_LIGHT_SPOT, RC_MAX_LIGHTS,
 * RCLight, and RCLightBuffer, which are all declared in the bind-group
 * block just above the insertion point.
 */
export const RC_NEE_POINTSPOT_WGSL = /* wgsl */`// ─── Rect-area emitter NEE ────────────────────────────────────────────────────
// RC's prior light model (radiance = directSun + emissive + envTransmission)
// could see emissive GEOMETRY a probe ray directly hit, but NOT the abstract
// rect-area emitter list — so a rect-area-only scene produced all-zero cascades
// (the 2026-06-07 "RC cascade-zero" regime gap). This adds one-sample-per-
// emitter next-event estimation at the probe-ray hit: for each emitter triangle
// sample a point, shadow-test through RC's own BVH unless the source emitter
// set castShadow:false, and add the Lambertian
// diffuse-reflected contribution. Summing one sample per emitter (rather than
// CDF-importance-sampling a single emitter) is unbiased and lower-variance for
// the handful of emitters a walkaround scene carries, and needs no CDF buffer.
//
// Estimator (area-form, pdf = 1/area ⇒ 1/pdf = area):
//   Lo += (albedo/π) · Le · (cosSurf · cosLight / dist²) · area · vis
// cosLight uses the emitter's front face only (one-sided), matching the
// shade/ReSTIR-DI convention. The shadow ray uses RC's glass-skip any-hit path:
// an OPAQUE hit before the light occludes; transmissive hits are ignored here
// rather than tinting visibility (kept simple — RC is a coarse GI cache).
// A7 (2026-06-10): scene-scale-proportional normal bias for shadow rays.
// Uses the smallest room-size axis * 0.001 (mirroring DDGI's gridParams.spacing
// * 0.001 — M13 precedent); replaces the hardcoded world-unit values 0.01/0.02
// that were Cornell-specific and silently wrong for other scene scales.
// Passed by the entry point as 'normalBias' so it is computed once per thread.

fn rcEmitterNEE(hitPos: vec3f, n: vec3f, albedo: vec3f, count: u32, seed0: u32, triEps: f32, normalBias: f32) -> vec3f {
  var Lo = vec3f(0.0);
  for (var ei: u32 = 0u; ei < count; ei = ei + 1u) {
    let e = rc_emitters[ei];
    // Per-emitter jittered area sample.
    let s0 = pcgHashToF32(seed0 ^ (ei * 0x9E3779B9u + 0x1u));
    let s1 = pcgHashToF32(seed0 * 7919u ^ (ei * 0x85EBCA6Bu + 0x2u));
    let su = sqrt(s0);
    let localBary = vec3f(1.0 - su, su * (1.0 - s1), su * s1);
    let pos = localBary.x * e.vA + localBary.y * e.vB + localBary.z * e.vC;

    let toL    = pos - hitPos;
    let dist2  = max(dot(toL, toL), 1e-8);
    let dist   = sqrt(dist2);
    let wi     = toL / dist;
    let cosSurf  = dot(n, wi);
    let cosLight = dot(e.normal, -wi);   // emitter front face only
    if (cosSurf <= 0.0 || cosLight <= 0.0) { continue; }

    // Opaque shadow test toward the light sample (stop just short of it).
    // H37: skip transmissive geometry so emitters behind stained glass still
    // contribute to the coarse RC cache. Glass tint is intentionally ignored.
    // Emitter castShadow:false rides the shared EmitterTri fifth-vec4 .w lane.
    let shadowTMax = max(0.0, dist - normalBias);
    if (e.castShadowDisabled < 0.5 && shadowTMax > 0.0 && rcTraceAnyCastShadow(hitPos + n * normalBias, wi, shadowTMax, triEps, true)) {
      continue;
    }

    let G = (cosSurf * cosLight) / dist2;
    let Le = rcSampleEmitterLeAtBary(e, localBary, e.Le);
    Lo = Lo + albedo * 0.31831 * Le * G * e.area;   // 0.31831 = 1/π
  }
  return Lo;
}

// ─── Point/spot analytic lights (A7, 2026-06-10) ─────────────────────────────
// Mirrors DDGI's evalPointLight (probeUpdateRays.wgsl.ts) with the same
// conventions: distance falloff 1/(r²+1), spot-cone smoothstep, shadow ray
// using the same scene-scale-proportional normalBias and glass-skip visibility.
// Called only when lightCount > 0, so sun-only and emitter-only scenes are
// byte-identical.
fn evalRCPointSpotLights(hitPos: vec3f, n: vec3f, albedo: vec3f, normalBias: f32, triEps: f32) -> vec3f {
  let count = min(rc_lights.count, RC_MAX_LIGHTS);
  if (count == 0u) { return vec3f(0.0); }
  var Lo = vec3f(0.0);
  for (var li: u32 = 0u; li < count; li = li + 1u) {
    let light = rc_lights.items[li];
    let kind = light.kind & RC_LIGHT_KIND_MASK;
    let castShadowDisabled = (light.kind & RC_LIGHT_CAST_SHADOW_DISABLED) != 0u;
    if (kind != RC_LIGHT_POINT && kind != RC_LIGHT_SPOT) { continue; }

    let toLight = light.position - hitPos;
    let dist    = length(toLight);
    if (dist < 1e-6) { continue; }
    let lightDir = toLight / dist;
    let nDotL = max(0.0, dot(n, lightDir));
    if (nDotL < 1e-3) { continue; }

    // Spot cone falloff (KHR_lights_punctual convention; point → no cone).
    let axisLen2 = dot(light.direction, light.direction);
    var coneFalloff = 1.0;
    if (axisLen2 > 0.25) {
      let cosToP = dot(lightDir, light.direction * inverseSqrt(axisLen2));
      coneFalloff = smoothstep(light.outerCone, light.innerCone, cosToP);
      if (coneFalloff <= 0.0) { continue; }
    }

    // Shadow test — stop just short of the light position. H37 mirrors emitter
    // NEE: transmissive geometry does not fully occlude coarse RC direct light.
    let shadowTMax = max(0.0, dist - normalBias);
    if (!castShadowDisabled && shadowTMax > 0.0 && rcTraceAnyCastShadow(hitPos + n * normalBias, lightDir, shadowTMax, triEps, true)) {
      continue;
    }

    // 1/(r²+1) falloff (matches DDGI's evalPointLight — softens singularity at r→0).
    let atten = light.intensity / (dist * dist + 1.0);
    // Lambertian receiver: Lo += (albedo/π) · light_irradiance
    Lo = Lo + albedo * 0.31831 * light.color * atten * nDotL * coneFalloff;
  }
  return Lo;
}`;

/**
 * Combined WGSL block: RC_SUN_VISIBILITY_WGSL + '\n\n' + RC_NEE_POINTSPOT_WGSL.
 * Retained as a convenience export for external consumers that want the full
 * light-eval block in one string. probeRayCast.wgsl.ts uses the two
 * sub-exports (RC_SUN_VISIBILITY_WGSL / RC_NEE_POINTSPOT_WGSL) for their
 * respective insertion points (byte-identity verified in F6 completion, 2026-06-11).
 * Export this constant if/when external callers need the combined block.
 */
const _RC_LIGHT_EVAL_WGSL = `${RC_SUN_VISIBILITY_WGSL}\n\n${RC_NEE_POINTSPOT_WGSL}`;
