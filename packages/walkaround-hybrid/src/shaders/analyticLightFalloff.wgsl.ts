/**
 * analyticLightFalloff.wgsl.ts — shared spot-cone-falloff + point/spot distance
 * attenuation WGSL math.
 *
 * D8-2 (complexity-sweep 2026-07-20, T4-3): the spot-cone falloff and the
 * point/spot distance-attenuation functions are byte-identical between
 * `shadingTerms.wgsl.ts` (the `analytic*`-named copies interpolated into
 * SHADE_WGSL) and `transparentOit.wgsl.ts` (the `oit*`-named copies). Both are
 * BINDING-FREE PURE MATH (function-scope args only; no `ubo`/`scene`/texture
 * references), so per the composeWgsl ordering rule they COULD be a WgslModule.
 *
 * They are shared here as a RAW-STRING BUILDER rather than a WgslModule for one
 * reason: byte-identity. The two consumers use DIFFERENT function-name prefixes
 * (`analytic*` vs `oit*`) at DIFFERENT interior positions of their bodies. A
 * WgslModule would emit a single canonical name once, in composer dep-order —
 * changing both the definition-site bytes and the name at every call site. The
 * raw-string builder keeps each consumer's function name and position exactly,
 * so the composed WGSL for `shade` and `transparent-oit` stays byte-for-byte
 * unchanged. The single parameterized slot is the name prefix.
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
  if (axisLen2 <= 0.01) { return 1.0; }
  let axis = lightDir * inverseSqrt(axisLen2);
  let cosTheta = dot(-axis, wi);
  if (cosTheta < cosOuter) { return 0.0; }
  if (abs(cosInner - cosOuter) < 1e-5) {
    return select(0.0, 1.0, cosTheta >= cosOuter);
  }
  return smoothstep(cosOuter, cosInner, cosTheta);
}

fn ${prefix}PointSpotAttenuation(dist: f32, cutoffDistance: f32, decay: f32, dist2Floor: f32) -> f32 {
  var attenuation = 1.0;
  if (decay > 0.01) {
    if (abs(decay - 2.0) < 1e-5) {
      attenuation = 1.0 / (dist * dist + dist2Floor);
    } else {
      attenuation = 1.0 / max(pow(max(dist, 1.0), decay), max(dist2Floor, 1e-6));
    }
  }
  if (cutoffDistance > 0.0) {
    let x = clamp(1.0 - pow(dist / cutoffDistance, 4.0), 0.0, 1.0);
    attenuation = attenuation * x * x;
  }
  return attenuation;
}`;
}
