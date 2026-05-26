/**
 * Lite caustic stubs — MNEE / photon-map paths disabled on the compatibility tier.
 */
export const PT_WEBGPU_PATH_TRACE_CAUSTIC_LITE_WGSL = /* wgsl */ `
fn manifoldNeeContribution(
  rng: ptr<function, u32>,
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  throughput: vec3f,
) -> vec3f {
  _ = rng;
  _ = hitPos;
  _ = normal;
  _ = wo;
  _ = baseColor;
  _ = roughness;
  _ = metallic;
  _ = transmission;
  _ = throughput;
  return vec3f(0.0);
}

fn photonMapContribution(
  rng: ptr<function, u32>,
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  throughput: vec3f,
) -> vec3f {
  _ = rng;
  _ = hitPos;
  _ = normal;
  _ = wo;
  _ = baseColor;
  _ = roughness;
  _ = metallic;
  _ = transmission;
  _ = throughput;
  return vec3f(0.0);
}
`;
