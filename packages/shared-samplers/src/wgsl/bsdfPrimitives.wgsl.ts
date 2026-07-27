/** Canonical WGSL BRDF primitives shared across renderers. */
export const BSDF_PRIMITIVES_MODULE_NAME = 'bsdfPrimitives';

export const BSDF_PRIMITIVES_WGSL = /* wgsl */ `
fn samplerSafeNormal(n: vec3f) -> vec3f {
  let scale = max(max(abs(n.x), abs(n.y)), abs(n.z));
  if (!(scale > 1e-12) || !(scale <= 3.402823e38)) { return vec3f(0.0, 1.0, 0.0); }
  let scaled = n / scale;
  return scaled * inverseSqrt(dot(scaled, scaled));
}

// Build an orthonormal basis around a normal.
fn buildONB(n: vec3f, T: ptr<function, vec3f>, B: ptr<function, vec3f>) {
  let nn = samplerSafeNormal(n);
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(nn.y) > 0.999) { up = vec3f(1.0, 0.0, 0.0); }
  *T = normalize(cross(up, nn));
  *B = cross(nn, *T);
}

// Cosine-hemisphere sample in local space, returns world-space direction.
fn sampleCosineHemisphere(n: vec3f, rng: ptr<function, u32>) -> vec3f {
  let xi = rand2(rng);
  let r = sqrt(xi.x);
  let phi = 2.0 * PI * xi.y;
  let localDir = vec3f(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - xi.x)));
  var T: vec3f; var B: vec3f;
  let nn = samplerSafeNormal(n);
  buildONB(nn, &T, &B);
  return localDir.x * T + localDir.y * B + localDir.z * nn;
}

fn cosineHemispherePdf(n: vec3f, wi: vec3f) -> f32 {
  let wiScale = max(max(abs(wi.x), abs(wi.y)), abs(wi.z));
  if (!(wiScale > 1e-12) || !(wiScale <= 3.402823e38)) { return 0.0; }
  let wiScaled = wi / wiScale;
  let wiUnit = wiScaled * inverseSqrt(dot(wiScaled, wiScaled));
  return max(0.0, dot(samplerSafeNormal(n), wiUnit)) * INV_PI;
}

// Schlick Fresnel.
fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
  let c = 1.0 - cosTheta;
  return F0 + (1.0 - F0) * (c * c * c * c * c);
}
`;

// NOTE (complexity-sweep 2026-06-02): GGX microfacet terms (NDF/geometry) are
// deliberately NOT in this shared module. The two GPU backends floor roughness
// differently — walkaround floors `rough` at 0.01; pt-webgpu floors alpha=rough²
// at 1e-3 plus a 1e-6 denominator floor — so a shared GGX would change one
// backend's low-roughness specular. They keep local copies BY DESIGN
// (walkaround: shaders/ggxBrdf.wgsl.ts; pt-webgpu: wgsl/pathTrace/material.wgsl.ts).
// `fresnelSchlick` above is the unclamped reference; pt-webgpu's local copy adds a
// defensive clamp(1-cosθ,0,1) — also intentional, not a divergence to "fix".
