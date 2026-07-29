/**
 * analyticLightFalloff.wgsl.ts — shared spot-cone-falloff + point/spot distance
 * attenuation WGSL math.
 *
 * The spot-cone falloff and point/spot distance attenuation are centralized
 * here so opaque, transparent, NRC-teacher, manifold-caustic, and DDGI routes
 * cannot drift on the KHR_lights_punctual range window. The functions are
 * BINDING-FREE PURE MATH (function-scope args only; no `ubo`/`scene`/texture
 * references), so per the composeWgsl ordering rule they COULD be a WgslModule.
 *
 * They are shared here as a RAW-STRING BUILDER rather than a WgslModule for one
 * reason: byte-identity. The two consumers use DIFFERENT function-name prefixes
 * (`analytic*` vs `oit*`) at DIFFERENT interior positions of their bodies. A
 * WgslModule would emit a single canonical name once, in composer dep-order —
 * changing both the definition-site bytes and the name at every call site. The
 * raw-string builder keeps each consumer's local naming convention. The single
 * parameterized slot is the name prefix.
 */

/**
 * Emit the spot-cone-falloff + point/spot-attenuation pair with a name prefix.
 * `prefix('analytic')` → `analyticSpotConeFalloff` / `analyticPointSpotAttenuation`;
 * `prefix('oit')`      → `oitSpotConeFalloff`      / `oitPointSpotAttenuation`.
 * The body bytes are identical for every prefix.
 */
export function analyticLightFalloffWgsl(prefix: string): string {
  return /* wgsl */ `fn ${prefix}SpotConeFalloff(lightDir: vec3f, wi: vec3f, cosInner: f32, cosOuter: f32) -> f32 {
  let axisLen2 = dot(lightDir, lightDir);
  if (axisLen2 <= 0.0) { return 1.0; }
  let axis = lightDir * inverseSqrt(axisLen2);
  let cosTheta = dot(-axis, wi);
  if (cosTheta < cosOuter) { return 0.0; }
  if (cosInner == cosOuter) {
    return select(0.0, 1.0, cosTheta >= cosOuter);
  }
  return smoothstep(cosOuter, cosInner, cosTheta);
}

fn ${prefix}PointSpotAttenuation(dist: f32, cutoffDistance: f32, decay: f32, dist2Floor: f32) -> f32 {
  var attenuation = 1.0;
  if (decay > 0.0) {
    let regularizedDist2 = max(dist * dist, dist2Floor);
    if (!(regularizedDist2 > 0.0)) { return 0.0; }
    attenuation = 1.0 / pow(sqrt(regularizedDist2), decay);
  }
  if (cutoffDistance > 0.0) {
    let x = clamp(1.0 - pow(dist / cutoffDistance, 4.0), 0.0, 1.0);
    attenuation = attenuation * x;
  }
  return attenuation;
}`;
}
