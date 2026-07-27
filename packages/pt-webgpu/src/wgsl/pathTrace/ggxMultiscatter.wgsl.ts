/**
 * Shared Kulla-Conty GGX multiple-scattering compensation.
 *
 * The forward path tracer and the direct-light adjoint must evaluate the same
 * BRDF. Keeping the directional-albedo table and compensation formula in one
 * WGSL fragment prevents inverse gradients from silently drifting when the
 * production forward model changes.
 */
export const PT_WEBGPU_GGX_MULTISCATTER_WGSL = /* wgsl */ `
// Precomputed 8x8 single-scatter GGX directional-albedo LUT, E_ss(roughness, mu).
// Rows = roughness 0..1 (8 steps), columns = N.V 0..1 (8 steps), row-major.
const GGX_E_LUT_DIM = 8u;
const GGX_E_LUT = array<f32, 64>(
  0.1375, 0.5617, 0.7546, 0.8522, 0.9111, 0.9505, 0.9788, 1.0,
  0.2955, 0.515,  0.7091, 0.8192, 0.889,  0.937,  0.9721, 0.9988,
  0.5794, 0.5541, 0.6677, 0.7691, 0.8451, 0.9021, 0.9457, 0.98,
  0.7011, 0.6486, 0.6669, 0.7199, 0.7776, 0.8305, 0.8764, 0.9155,
  0.7335, 0.6901, 0.6696, 0.6756, 0.6972, 0.7262, 0.7578, 0.7893,
  0.7153, 0.6712, 0.6355, 0.6145, 0.6052, 0.6045, 0.6101, 0.6199,
  0.6669, 0.6137, 0.5657, 0.5286, 0.5,    0.478,  0.4611, 0.4483,
  0.6017, 0.537,  0.4773, 0.4296, 0.3905, 0.358,  0.3305, 0.3069,
);

// Hemispherical average E_avg(roughness) = 2 integral E_ss(mu,r) mu dmu.
const GGX_EAVG_LUT = array<f32, 8>(
  0.9106, 0.8931, 0.8629, 0.8094, 0.725, 0.6147, 0.4931, 0.3766,
);

fn ggxDirectionalAlbedo(cosTheta: f32, roughness: f32) -> f32 {
  let mu = clamp(cosTheta, 0.0, 1.0);
  let r = clamp(roughness, 0.0, 1.0);
  let fr = r * f32(GGX_E_LUT_DIM - 1u);
  let fm = mu * f32(GGX_E_LUT_DIM - 1u);
  let r0 = u32(floor(fr));
  let m0 = u32(floor(fm));
  let r1 = min(r0 + 1u, GGX_E_LUT_DIM - 1u);
  let m1 = min(m0 + 1u, GGX_E_LUT_DIM - 1u);
  let tr = fr - f32(r0);
  let tm = fm - f32(m0);
  let e00 = GGX_E_LUT[r0 * GGX_E_LUT_DIM + m0];
  let e01 = GGX_E_LUT[r0 * GGX_E_LUT_DIM + m1];
  let e10 = GGX_E_LUT[r1 * GGX_E_LUT_DIM + m0];
  let e11 = GGX_E_LUT[r1 * GGX_E_LUT_DIM + m1];
  return clamp(mix(mix(e00, e01, tm), mix(e10, e11, tm), tr), 0.02, 1.0);
}

fn ggxAverageAlbedo(roughness: f32) -> f32 {
  let r = clamp(roughness, 0.0, 1.0);
  let fr = r * f32(GGX_E_LUT_DIM - 1u);
  let r0 = u32(floor(fr));
  let r1 = min(r0 + 1u, GGX_E_LUT_DIM - 1u);
  let tr = fr - f32(r0);
  return clamp(mix(GGX_EAVG_LUT[r0], GGX_EAVG_LUT[r1], tr), 0.3, 1.0);
}

// Kulla-Conty compensation BRDF kernel (without the caller's N.L factor).
fn ggxMultiscatterLobeRoughness(
  f0: vec3f,
  roughnessV: f32,
  roughnessL: f32,
  roughnessAvg: f32,
  nDotV: f32,
  nDotL: f32,
) -> vec3f {
  let eAvg = ggxAverageAlbedo(roughnessAvg);
  let oneMinusEavg = 1.0 - eAvg;
  if (oneMinusEavg <= 0.0) { return vec3f(0.0); }
  let eo = ggxDirectionalAlbedo(nDotV, roughnessV);
  let ei = ggxDirectionalAlbedo(nDotL, roughnessL);
  let fAvg = f0 + (vec3f(1.0) - f0) * (1.0 / 21.0);
  let seriesDenom = vec3f(1.0) - fAvg * oneMinusEavg;
  if (any(seriesDenom <= vec3f(0.0))) { return vec3f(0.0); }
  let fMs = fAvg * fAvg * eAvg / seriesDenom;
  let shape = (1.0 - eo) * (1.0 - ei) / (PI * oneMinusEavg);
  let result = fMs * shape;
  let finite = all(result == result) &&
    all(abs(result) <= vec3f(3.402823e38));
  return select(vec3f(0.0), result, finite);
}

fn ggxMultiscatterLobe(
  f0: vec3f,
  roughness: f32,
  nDotV: f32,
  nDotL: f32,
) -> vec3f {
  return ggxMultiscatterLobeRoughness(
    f0, roughness, roughness, roughness, nDotV, nDotL,
  );
}
`;
