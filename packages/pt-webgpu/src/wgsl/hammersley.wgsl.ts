/**
 * Ported from stainedGlass walkaround WGSL helpers.
 * Low-discrepancy sampling utilities are backend-agnostic and reusable in
 * path tracing kernels (camera/lens sampling, light sampling, etc.).
 */
export const HAMMERSLEY_WGSL = /* wgsl */ `
fn radicalInverse_VdC(n: u32) -> f32 {
  var bits = (n << 16u) | (n >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return f32(bits) * 2.3283064365386963e-10;
}

fn hammersleyUniform(i: u32, numSamples: u32) -> vec2f {
  return vec2f(f32(i) / f32(numSamples), radicalInverse_VdC(i));
}

fn uniformSphere(u: vec2f) -> vec3f {
  let phi = u.x * 6.283185307;
  let cosTheta = 1.0 - 2.0 * u.y;
  let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  return vec3f(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

fn rotateAngleAxis(v: vec3f, angleAxis: vec3f) -> vec3f {
  let angle = length(angleAxis);
  if (angle < 1e-6) {
    return v;
  }
  let axis = angleAxis / angle;
  let cosA = cos(angle);
  let sinA = sin(angle);
  return v * cosA + cross(axis, v) * sinA + axis * dot(axis, v) * (1.0 - cosA);
}
`;
