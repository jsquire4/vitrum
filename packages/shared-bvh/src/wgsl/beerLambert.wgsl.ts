/**
 * Canonical RGB Beer-Lambert transmittance for realtime WGSL consumers.
 *
 * `attenuationColor` is transmittance at `attenuationDistance`, matching
 * KHR_materials_volume and the core MaterialSpec contract:
 *
 *   T(path) = attenuationColor ^ (path / attenuationDistance)
 *
 * A zero path or positive-infinite/default distance is the identity. A zero
 * colour channel is exactly opaque for every positive finite path; no
 * artificial colour floor is introduced.
 */
export const BEER_LAMBERT_WGSL = /* wgsl */ `
fn beerLambertTransmittanceRgb(
  attenuationColor: vec3f,
  attenuationDistance: f32,
  pathLength: f32,
) -> vec3f {
  let distance = max(pathLength, 0.0);
  if (distance <= 0.0 || attenuationDistance >= 3.402823e+38) {
    return vec3f(1.0);
  }
  if (!(attenuationDistance > 0.0)) {
    return vec3f(0.0);
  }
  let color = clamp(attenuationColor, vec3f(0.0), vec3f(1.0));
  let exponent = distance / attenuationDistance;
  let positive = pow(max(color, vec3f(1e-30)), vec3f(exponent));
  return select(positive, vec3f(0.0), color <= vec3f(0.0));
}
`;
