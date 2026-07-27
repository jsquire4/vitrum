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
 *     only references rcTraceFirstHit and the packed scene-arena loaders — all
 *     declared before the bind-group block in probeRayCast.wgsl.ts).
 *
 *   RC_NEE_POINTSPOT_WGSL — inserted where the inline rcEmitterNEE and
 *     evalRCPointSpotLights were, AFTER the RCLight struct / bind-group
 *     declarations (those functions reference rc_emitters, rc_lights,
 *     RC light kinds and runtime alias decoders — all declared in
 *     the bind-group block just above).
 *
 * RC_LIGHT_EVAL_WGSL is the combined export (= RC_SUN_VISIBILITY_WGSL +
 * separator + RC_NEE_POINTSPOT_WGSL) kept for external consumers that want
 * the full light-eval block in one string.
 *
 * Exported functions:
 *   traceSunVisibility    — alpha + glass-aware sun shadow test
 *   rcEmitterNEE          — one power-aliased rect-area NEE sample
 *   evalRCPointSpotLights — one aliased punctual/directional light sample
 */

/**
 * Alpha + glass-aware sun shadow test (traceSunVisibility).
 * Inserted before the RCLight struct in probeRayCast.wgsl.ts. It calls the
 * RC alpha-coverage helpers that are declared later in the composed module;
 * WGSL module-scope functions are order-independent.
 */
export const RC_SUN_VISIBILITY_WGSL = /* wgsl */`// ─── Sun visibility helper ────────────────────────────────────────────────────
// Alpha + glass-aware sun shadow test. Alpha coverage mirrors the DDGI/shade
// material-atlas transmittance model; transmissive slabs keep the existing
// Beer-Lambert continuation path.

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
  // A straight visibility ray cannot encounter more distinct surfaces than
  // the scene contains triangles. Use that geometry-derived bound rather than
  // truncating valid layered/alpha/glass paths after three interfaces.
  let surfaceBudget = arrayLength(&rc_geom_index) + 1u;
  for (var iter: u32 = 0u; iter < surfaceBudget; iter = iter + 1u) {
    var sRay = Ray();
    sRay.origin    = rayOrigin;
    sRay.direction = sunDir;
    let sHit = rcTraceFirstHit(sRay, triEps);
    if (!sHit.didHit) {
      return visibility;
    }
    let hitPos = rayOrigin + sunDir * sHit.dist;
    let sMatId = rcLoadTriMaterialId(sHit.indices.w);
    let sMat   = rcLoadMaterial(sMatId);
    if ((sMat.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) != 0u) {
      rayOrigin = hitPos + sunDir * slabStepSize;
      continue;
    }
    let alphaT = rcAlphaShadowTransmittanceForHit(sHit);
    if (alphaT >= 1.0) {
      rayOrigin = hitPos + sunDir * slabStepSize;
      continue;
    }
    if (alphaT > 0.0) {
      visibility = visibility * alphaT;
      if ((sMat.flags & MATERIAL_FLAG_IS_GLASS) == 0u) {
        rayOrigin = hitPos + sunDir * slabStepSize;
        continue;
      }
    } else if ((sMat.flags & MATERIAL_FLAG_IS_GLASS) == 0u) {
      return vec3f(0.0);
    }
    let gThick    = max(0.0, sMat.thickness);
    let gAttenCol = sMat.attenuationColor;
    let gColor    = sMat.baseColor;
    var beerAtten = vec3f(1.0);
    if (gThick > 0.0 && sMat.attenuationDistance > 0.0) {
      beerAtten = exp(-max(gAttenCol, vec3f(0.0)) *
        (gThick / sMat.attenuationDistance));
    }
    visibility = visibility * gColor * beerAtten;
    rayOrigin  = hitPos + sunDir * slabStepSize;
  }
  return visibility;
}`;

/**
 * Rect-area emitter NEE and punctual/directional analytic lights
 * (evalRCPointSpotLights). Inserted AFTER the RCLight struct / bind-group
 * declarations in probeRayCast.wgsl.ts — these functions reference
 * rc_emitters, rc_lights, RC light kinds, and RCLight, which are declared in the bind-group
 * block just above the insertion point.
 */
export const RC_NEE_POINTSPOT_WGSL = /* wgsl */`// ─── Rect-area emitter NEE ────────────────────────────────────────────────────
// RC's prior light model (radiance = directSun + emissive + envTransmission)
// could see emissive GEOMETRY a probe ray directly hit, but NOT the abstract
// rect-area emitter list — so a rect-area-only scene produced all-zero cascades
// (the 2026-06-07 "RC cascade-zero" regime gap). This adds one-sample-per-
// emitter next-event estimation at the probe-ray hit: for each emitter triangle
// sample a point, shadow-test through RC's own BVH unless the source emitter
// set castShadow:false, and add a compact material BRDF response using the
// probe-hit albedo/roughness/metalness/specular payload. The host-built power
// alias table selects exactly one emitter; draw.pmf is the represented
// discrete probability and the estimator divides by it.
//
// Estimator (area-form, conditional surface pdf = 1/area):
//   Lo += f_r(albedo, rough, metal, specular) · cosSurf · Le
//         · (cosLight / dist²) · area · vis / p(emitter)
// cosLight uses the emitter's front face only (one-sided), matching the
// shade/ReSTIR-DI convention. The shadow ray uses RC's alpha transmittance walk:
// alpha-mask/blend surfaces attenuate through the material atlas while scalar
// transmissive glass preserves the coarse RC "skip glass" policy.
// A7 (2026-06-10): scene-scale-proportional normal bias for shadow rays.
// Uses the smallest room-size axis * 0.001 (mirroring DDGI's gridParams.spacing
// * 0.001 — M13 precedent); replaces the hardcoded world-unit values 0.01/0.02
// that were Cornell-specific and silently wrong for other scene scales.
// Passed by the entry point as 'normalBias' so it is computed once per thread.

fn rcEmitterNEE(hitPos: vec3f, n: vec3f, wo: vec3f, material: RCProbeHitMaterial, count: u32, seed0: u32, triEps: f32, normalBias: f32) -> vec3f {
  if (count == 0u) { return vec3f(0.0); }
  let draw = rcEmitterAliasDraw(count, seed0 ^ 0x61c88647u);
  if (draw.pmf <= 0.0) { return vec3f(0.0); }
  let e = rcLoadEmitter(draw.index);
    let s0 = pcgHashToF32(seed0 ^ 0x243f6a88u);
    let s1 = pcgHashToF32(seed0 ^ 0xb7e15162u);
    let su = sqrt(s0);
    let localBary = vec3f(1.0 - su, su * (1.0 - s1), su * s1);
    let pos = localBary.x * e.vA + localBary.y * e.vB + localBary.z * e.vC;

    let toL    = pos - hitPos;
    let dist2  = dot(toL, toL);
    if (dist2 <= 0.0) { return vec3f(0.0); }
    let dist   = sqrt(dist2);
    let wi     = toL / dist;
    let response = rcEvaluateProbeDirectResponse(material, n, wo, wi);
    let responsePower = max(response.r, max(response.g, response.b));
    let cosLight = dot(e.normal, -wi);   // emitter front face only
    if (responsePower <= 0.0 || cosLight <= 0.0) { return vec3f(0.0); }

    // Alpha-aware shadow transmittance toward the light sample (stop just short
    // of it). H37's scalar-glass skip is preserved inside the helper so emitters
    // behind stained glass still contribute to the coarse RC cache.
    // Emitter castShadow:false rides the shared EmitterTri fifth-vec4 .w lane.
    let shadowTMax = max(0.0, dist - normalBias);
    var shadowT = 1.0;
    if (e.castShadowDisabled < 0.5 && shadowTMax > 0.0) {
      shadowT = rcTraceShadowTransmittance(hitPos + n * normalBias, wi, shadowTMax, triEps, true);
      if (shadowT <= 0.0) { return vec3f(0.0); }
    }

    let G = cosLight / dist2;
    let Le = rcSampleEmitterLeAtBary(e, localBary, e.Le);
    return response * Le * G * e.area * shadowT / draw.pmf;
}

// ─── Point/spot analytic lights (A7, 2026-06-10) ─────────────────────────────
// Mirrors DDGI's evalPointLight (probeUpdateRays.wgsl.ts) with the same
// authored decay law, spot-cone smoothstep, shadow ray
// using the same scene-scale-proportional normalBias and alpha/glass visibility.
// Called only when lightCount > 0, so sun-only and emitter-only scenes are
// byte-identical.
fn rcPointSpotDistanceAttenuation(
  dist: f32,
  cutoffDistance: f32,
  decay: f32,
  dist2Floor: f32,
) -> f32 {
  var attenuation = 1.0;
  if (decay > 0.0) {
    let regularizedDist2 = max(dist * dist, dist2Floor);
    if (!(regularizedDist2 > 0.0)) { return 0.0; }
    attenuation = 1.0 / pow(sqrt(regularizedDist2), decay);
  }
  if (cutoffDistance > 0.0) {
    let x = clamp(1.0 - pow(dist / cutoffDistance, 4.0), 0.0, 1.0);
    attenuation = attenuation * x * x;
  }
  return attenuation;
}

fn evalRCPointSpotLights(hitPos: vec3f, n: vec3f, wo: vec3f, material: RCProbeHitMaterial, normalBias: f32, triEps: f32, seed0: u32) -> vec3f {
  let count = min(rc_lights[0u], rc_u.lightCount);
  if (count == 0u || rc_lights[3u] != RC_LIGHTS_ABI_MAGIC) { return vec3f(0.0); }
    let draw = rcLightAliasDraw(count, seed0 ^ 0x3c6ef372u);
    if (draw.pmf <= 0.0) { return vec3f(0.0); }
    let light = rcLoadLight(draw.index);
    let kind = light.kind & RC_LIGHT_KIND_MASK;
    let castShadowDisabled = (light.kind & RC_LIGHT_CAST_SHADOW_DISABLED) != 0u;
    if (kind == RC_LIGHT_DIRECTIONAL) {
      let axisLen2 = dot(light.direction, light.direction);
      if (axisLen2 <= 0.0) { return vec3f(0.0); }
      let lightDir = rcSoftSunDirection(
        -light.direction * inverseSqrt(axisLen2),
        light.outerCone,
        hitPos,
        rc_u.roomSize,
        rc_u.cascadeIndex,
      );
      let response = rcEvaluateProbeDirectResponse(material, n, wo, lightDir);
      var visibility = vec3f(1.0);
      if (!castShadowDisabled) {
        let slabStep = min(rc_u.roomSize.x, min(rc_u.roomSize.y, rc_u.roomSize.z)) * 0.001;
        visibility = traceSunVisibility(hitPos + n * normalBias, lightDir, slabStep, triEps);
      }
      return response * light.color * light.intensity * visibility / draw.pmf;
    }
    if (kind != RC_LIGHT_POINT && kind != RC_LIGHT_SPOT) { return vec3f(0.0); }

    let toLight = light.position - hitPos;
    let dist    = length(toLight);
    if (dist <= 0.0) { return vec3f(0.0); }
    let lightDir = toLight / dist;
    let response = rcEvaluateProbeDirectResponse(material, n, wo, lightDir);
    if (max(response.r, max(response.g, response.b)) <= 0.0) { return vec3f(0.0); }

    // Spot cone falloff (KHR_lights_punctual convention; point → no cone).
    // light.direction is the spot beam/travel axis, so a receiver-to-light
    // vector is inside the cone when it aligns with -axis.
    let axisLen2 = dot(light.direction, light.direction);
    var coneFalloff = 1.0;
    if (kind == RC_LIGHT_SPOT) {
      if (!(axisLen2 > 0.0)) { return vec3f(0.0); }
      let cosToP = dot(-light.direction * inverseSqrt(axisLen2), lightDir);
      if (cosToP < light.outerCone) { return vec3f(0.0); }
      if (light.innerCone == light.outerCone) {
        coneFalloff = 1.0;
      } else {
        coneFalloff = smoothstep(light.outerCone, light.innerCone, cosToP);
      }
      if (coneFalloff <= 0.0) { return vec3f(0.0); }
    }

    // Shadow transmittance — stop just short of the light position. H37 mirrors
    // emitter NEE: scalar transmissive glass does not fully occlude coarse RC
    // direct light, while alpha-map/blend surfaces attenuate.
    let shadowTMax = max(0.0, dist - normalBias);
    var shadowT = 1.0;
    if (!castShadowDisabled && shadowTMax > 0.0) {
      shadowT = rcTraceShadowTransmittance(hitPos + n * normalBias, lightDir, shadowTMax, triEps, true);
      if (shadowT <= 0.0) { return vec3f(0.0); }
    }

    let dist2Floor = max(normalBias * normalBias, triEps * triEps);
    let distanceAttenuation = rcPointSpotDistanceAttenuation(
      dist, light.distance, light.decay, dist2Floor,
    );
    let atten = light.intensity * distanceAttenuation;
    return response * light.color * atten * coneFalloff * shadowT / draw.pmf;
}`;
