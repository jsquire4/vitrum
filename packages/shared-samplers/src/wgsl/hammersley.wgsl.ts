/**
 * Hammersley sequence + sphere sampling for DDGI probe rays.
 * Low-discrepancy stratified sampling with a per-frame random rotation
 * to decorrelate frames (golden-ratio rotation on the sphere).
 */

export const HAMMERSLEY_WGSL = /* wgsl */`

fn radicalInverse_VdC(n: u32) -> f32 {
  var bits = (n << 16u) | (n >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  // Converting all 32 bits to f32 first can round values near 0xffffffffu to
  // 2^32 and return the forbidden endpoint 1.0.  Keep the 24 high-quality
  // radical-inverse bits that binary32 represents exactly, yielding [0, 1).
  return f32(bits >> 8u) * 5.960464477539063e-8;
}

fn hammersleyUniform(i: u32, numSamples: u32) -> vec2f {
  return vec2f(f32(i) / f32(max(numSamples, 1u)), radicalInverse_VdC(i));
}

// Uniform sphere sampling from 2D [0,1]^2 input.
// Outputs a unit-length direction vector on the sphere.
fn uniformSphere(u: vec2f) -> vec3f {
  let phi   = u.x * 6.283185307;   // 2π
  let cosT  = 1.0 - 2.0 * u.y;
  let sinT  = sqrt(max(0.0, 1.0 - cosT * cosT));
  return vec3f(sinT * cos(phi), sinT * sin(phi), cosT);
}

// Rodrigues rotation: angleAxis.xyz = axis * angle(radians).
fn rotateAngleAxis(v: vec3f, angleAxis: vec3f) -> vec3f {
  let angle = length(angleAxis);
  if (angle < 1e-6) { return v; }
  let axis  = angleAxis / angle;
  let cosA  = cos(angle);
  let sinA  = sin(angle);
  return v * cosA + cross(axis, v) * sinA + axis * dot(axis, v) * (1.0 - cosA);
}

// Stratified sphere direction for ray index i out of numSamples, with
// an extra per-frame golden-ratio rotation applied.
fn ddgiRayDirection(i: u32, numSamples: u32, randomRotation: vec3f) -> vec3f {
  let uv  = hammersleyUniform(i, numSamples);
  let dir = uniformSphere(uv);
  return rotateAngleAxis(dir, randomRotation);
}

`;
