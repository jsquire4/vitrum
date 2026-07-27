import {
  PT_WEBGPU_MICROFACET_ALPHA_FLOOR,
  ptWebgpuMicrofacetAlpha,
  roughDielectricSmithG1 as smithG1,
  roughDielectricSmithG1RoughnessDerivative,
} from '../math/roughDielectric.js';

/**
 * brdfAdjoint.ts — CPU reference for the path-replay BSDF adjoint (WS5 Phase 1).
 *
 * This is the LOAD-BEARING oracle for the differentiable-RT Phase-1 gradient,
 * the same role `cpuTracer.ts` plays for the forward MC integrator and
 * `nrcEncoding.ts` plays for the NRC encoding. The emitted WGSL adjoint
 * (`../wgsl/pathTrace/pathTraceAdjoint.wgsl.ts`) is hand-verified line-for-line
 * against the partials computed here; the codegen-shape tests pin that the WGSL
 * keeps emitting the SAME arithmetic. The V24 shader-compile A/B confirmed it
 * runs on real hardware and matches this oracle to f32 precision (GPU-validated
 * on lavapipe, 2026-06-03).
 *
 * What it differentiates:
 *  - the Cook-Torrance unified BRDF `evaluateBrdf` (mirror of
 *    `bsdf.wgsl.ts:evaluateBrdf`) w.r.t. `baseColor` (rgb) and `roughness`
 *    (scalar) — the original Phase-1 set. The sampled directions `wo`, `wi`
 *    are HELD CONSTANT: path-replay freezes the random choices from the
 *    forward pass, so the adjoint differentiates only the continuous shading
 *    and never differentiates through sampling. That sidesteps the
 *    visibility / lobe-choice discontinuities path tracing is full of.
 *  - the additive emission term `Le` w.r.t. `emissive` (rgb) — emission is NOT
 *    a BRDF term, so its partial is a CONTRIBUTION-level identity, not a
 *    `dBrdf_*`. See `dContribution_dEmissive` for the derivation.
 *  - KHR_materials_specular dielectric F0 controls (`specularColor` and
 *    `specularIntensity`) plus metallic through the same frozen direct-light
 *    BRDF partial.
 *  - scalar KHR_materials_clearcoat controls (`clearcoat` and
 *    `clearcoatRoughness`) for the additive, map-free direct-light clearcoat
 *    lobe. Clearcoat normal maps remain outside this oracle.
 *  - KHR_materials_sheen controls (`sheen`, `sheenColor`, and
 *    `sheenRoughness`) for the additive direct-light sheen lobe. Texture-map
 *    factors are handled as local chain-rule multipliers in the replay pass.
 *  - scalar KHR_materials_iridescence (`iridescence`) through the
 *    thin-film-modified base F0 used by the opaque direct-light specular and
 *    diffuse partition. Map-free `iridescenceIor` is differentiated through a
 *    local symmetric derivative of that thin-film F0 term. Authored
 *    `iridescenceThicknessRange` gradients are differentiated by chaining the
 *    sampled thickness (`V·H` or readable thickness-map texel) to min/max
 *    endpoints. Texture-pixel gradients remain outside this oracle.
 *  - KHR_materials_anisotropy scalar controls (`anisotropy` and
 *    `anisotropyRotation`) through a local symmetric derivative of the
 *    anisotropic GGX specular lobe. Anisotropy map factors are handled as local
 *    strength/rotation chain-rule terms in the replay pass.
 *  - the dielectric Fresnel reflectance `frDielectric` w.r.t. `ior` (scalar).
 *    NOTE: `ior` does NOT enter the opaque `evaluateBrdf` F0 term — dielectric
 *    F0 is controlled by KHR_materials_specular and metallic F0 by baseColor —
 *    so `∂evaluateBrdf/∂ior ≡ 0`. The only differentiable `ior` dependence in
 *    the forward kernel is `frDielectric` (the transmissive reflect/refract
 *    Fresnel partition). `dFrDielectric_dIor` provides that partial; see its doc
 *    for the (large) caveat on end-to-end consumption.
 *
 * Ref: Vicini, Speierer, Jakob, "Path Replay Backpropagation of Light
 *      Transport," ACM TOG 40(4), SIGGRAPH 2021.
 *      Nimier-David, Vicini, Zeltner, Jakob, "Radiative Backpropagation: An
 *      Adjoint Method for Lightweight Differentiable Rendering," ACM TOG 39(4),
 *      SIGGRAPH 2020.
 *      BRDF primitives: Pharr, Jakob, Humphreys, PBR 4th ed. §9.6–9.8;
 *      Heitz 2018 VNDF (sampling, not differentiated here).
 */

export type Vec3 = readonly [number, number, number];

const PI = 3.14159265358979;
const INV_PI = 0.31830988618;
const IRIDESCENCE_IOR_DERIV_STEP = 1e-3;
const IRIDESCENCE_THICKNESS_DERIV_STEP = 1e-2;
const ANISOTROPY_DERIV_STEP = 1e-3;
const ANISOTROPY_ROTATION_DERIV_STEP = 1e-3;
const ANISOTROPIC_BASE_PARAM_DERIV_STEP = 1e-4;
const MULTISCATTER_ROUGHNESS_DERIV_STEP = 1e-4;
const GGX_E_LUT_DIM = 8;
const GGX_E_LUT: readonly number[] = [
  0.1375, 0.5617, 0.7546, 0.8522, 0.9111, 0.9505, 0.9788, 1.0,
  0.2955, 0.515, 0.7091, 0.8192, 0.889, 0.937, 0.9721, 0.9988,
  0.5794, 0.5541, 0.6677, 0.7691, 0.8451, 0.9021, 0.9457, 0.98,
  0.7011, 0.6486, 0.6669, 0.7199, 0.7776, 0.8305, 0.8764, 0.9155,
  0.7335, 0.6901, 0.6696, 0.6756, 0.6972, 0.7262, 0.7578, 0.7893,
  0.7153, 0.6712, 0.6355, 0.6145, 0.6052, 0.6045, 0.6101, 0.6199,
  0.6669, 0.6137, 0.5657, 0.5286, 0.5, 0.478, 0.4611, 0.4483,
  0.6017, 0.537, 0.4773, 0.4296, 0.3905, 0.358, 0.3305, 0.3069,
];
const GGX_EAVG_LUT: readonly number[] = [
  0.9106, 0.8931, 0.8629, 0.8094, 0.725, 0.6147, 0.4931, 0.3766,
];

// ── primitive mirrors (match material.wgsl.ts exactly) ──────────────────────

function safeNormalize(v: Vec3): Vec3 {
  const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (len < 1e-8) return [0, 0, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function buildOnb(n: Vec3): [Vec3, Vec3] {
  const up: Vec3 = Math.abs(n[1]) > 0.95 ? [1, 0, 0] : [0, 1, 0];
  const t = safeNormalize(cross(up, n));
  return [t, cross(n, t)];
}

/** fresnelSchlick (material.wgsl.ts:313) — per-channel. */
function fresnelSchlick(cosTheta: number, f0: Vec3): Vec3 {
  const m = Math.min(Math.max(1.0 - cosTheta, 0.0), 1.0);
  const m2 = m * m;
  const m5 = m2 * m2 * m;
  return [
    f0[0] + (1.0 - f0[0]) * m5,
    f0[1] + (1.0 - f0[1]) * m5,
    f0[2] + (1.0 - f0[2]) * m5,
  ];
}

function iridXyzToRec709(xyz: Vec3): Vec3 {
  return [
    3.2404542 * xyz[0] - 1.5371385 * xyz[1] - 0.4985314 * xyz[2],
    -0.9692660 * xyz[0] + 1.8760108 * xyz[1] + 0.0415560 * xyz[2],
    0.0556434 * xyz[0] - 0.2040259 * xyz[1] + 1.0572252 * xyz[2],
  ];
}

function iridFresnel0ToIor(f0: Vec3): Vec3 {
  return f0.map((v) => {
    const sqrtF0 = Math.sqrt(Math.min(Math.max(v, 0.0), 0.9999));
    return (1.0 + sqrtF0) / (1.0 - sqrtF0);
  }) as unknown as Vec3;
}

function iridIorToFresnel0Scalar(transmittedIor: number, incidentIor: number): number {
  const r = (transmittedIor - incidentIor) / (transmittedIor + incidentIor);
  return r * r;
}

function iridIorToFresnel0Vec(transmittedIor: Vec3, incidentIor: number): Vec3 {
  return [
    ((transmittedIor[0] - incidentIor) / (transmittedIor[0] + incidentIor)) ** 2,
    ((transmittedIor[1] - incidentIor) / (transmittedIor[1] + incidentIor)) ** 2,
    ((transmittedIor[2] - incidentIor) / (transmittedIor[2] + incidentIor)) ** 2,
  ];
}

function iridSchlickScalar(cosTheta: number, f0: number): number {
  const m = Math.min(Math.max(1.0 - cosTheta, 0.0), 1.0);
  const m2 = m * m;
  return f0 + (1.0 - f0) * m2 * m2 * m;
}

function iridSchlickVec(cosTheta: number, f0: Vec3): Vec3 {
  return fresnelSchlick(cosTheta, f0);
}

function iridEvalSensitivity(OPD: number, shift: Vec3): Vec3 {
  const phase = 2.0 * PI * OPD * 1.0e-9;
  const val: Vec3 = [5.4856e-13, 4.4201e-13, 5.2481e-13];
  const pos: Vec3 = [1.6810e+06, 1.7953e+06, 2.2084e+06];
  const vari: Vec3 = [4.3278e+09, 9.3046e+09, 6.6121e+09];
  const xyz: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    xyz[c] = val[c]! * Math.sqrt(2.0 * PI * vari[c]!) *
      Math.cos(pos[c]! * phase + shift[c]!) *
      Math.exp(-phase * phase * vari[c]!);
  }
  xyz[0] += 9.7470e-14 * Math.sqrt(2.0 * PI * 4.5282e+09) *
    Math.cos(2.2399e+06 * phase + shift[0]) *
    Math.exp(-4.5282e+09 * phase * phase);
  return iridXyzToRec709([xyz[0] / 1.0685e-7, xyz[1] / 1.0685e-7, xyz[2] / 1.0685e-7]);
}

function evalIridescence(
  outsideIOR: number,
  eta2: number,
  cosTheta1: number,
  thicknessNm: number,
  baseF0: Vec3,
): Vec3 {
  const t = Math.min(Math.max(thicknessNm / 0.03, 0.0), 1.0);
  const smooth = t * t * (3.0 - 2.0 * t);
  const iridescenceIor = outsideIOR + (eta2 - outsideIOR) * smooth;
  const sinTheta2Sq = ((outsideIOR / iridescenceIor) ** 2) *
    Math.max(0.0, 1.0 - cosTheta1 * cosTheta1);
  const cosTheta2Sq = 1.0 - sinTheta2Sq;
  if (cosTheta2Sq < 0.0) return [1, 1, 1];
  const cosTheta2 = Math.sqrt(cosTheta2Sq);

  const R0Scalar = iridIorToFresnel0Scalar(iridescenceIor, outsideIOR);
  const R12 = iridSchlickScalar(cosTheta1, R0Scalar);
  const T121 = 1.0 - R12;
  const phi12 = iridescenceIor < outsideIOR ? PI : 0.0;
  const phi21 = PI - phi12;

  const baseIOR = iridFresnel0ToIor(baseF0);
  const R1Vec = iridIorToFresnel0Vec(baseIOR, iridescenceIor);
  const R23 = iridSchlickVec(cosTheta2, R1Vec);
  const phi23: Vec3 = [
    baseIOR[0] < iridescenceIor ? PI : 0.0,
    baseIOR[1] < iridescenceIor ? PI : 0.0,
    baseIOR[2] < iridescenceIor ? PI : 0.0,
  ];
  const OPD = 2.0 * iridescenceIor * thicknessNm * cosTheta2;
  const phi: Vec3 = [phi21 + phi23[0], phi21 + phi23[1], phi21 + phi23[2]];
  const R123: Vec3 = [
    Math.min(Math.max(R12 * R23[0], 1e-5), 0.9999),
    Math.min(Math.max(R12 * R23[1], 1e-5), 0.9999),
    Math.min(Math.max(R12 * R23[2], 1e-5), 0.9999),
  ];
  const r123: Vec3 = [Math.sqrt(R123[0]), Math.sqrt(R123[1]), Math.sqrt(R123[2])];
  const Rs: Vec3 = [
    (T121 * T121) * R23[0] / (1.0 - R123[0]),
    (T121 * T121) * R23[1] / (1.0 - R123[1]),
    (T121 * T121) * R23[2] / (1.0 - R123[2]),
  ];
  const out: [number, number, number] = [R12 + Rs[0], R12 + Rs[1], R12 + Rs[2]];
  let Cm: [number, number, number] = [Rs[0] - T121, Rs[1] - T121, Rs[2] - T121];
  for (let m = 1; m <= 2; m++) {
    Cm = [Cm[0] * r123[0], Cm[1] * r123[1], Cm[2] * r123[2]];
    const Sm = iridEvalSensitivity(m * OPD, [m * phi[0], m * phi[1], m * phi[2]]);
    out[0] += Cm[0] * 2.0 * Sm[0];
    out[1] += Cm[1] * 2.0 * Sm[1];
    out[2] += Cm[2] * 2.0 * Sm[2];
  }
  return [Math.max(out[0], 0.0), Math.max(out[1], 0.0), Math.max(out[2], 0.0)];
}

function iridescenceModifiedF0(
  baseF0: Vec3,
  iridescence: number,
  iridescenceIor: number,
  thicknessMin: number,
  thicknessMax: number,
  cosTheta: number,
): Vec3 {
  if (iridescence <= 0) return baseF0;
  const thicknessNm = thicknessMin + (thicknessMax - thicknessMin) * Math.min(Math.max(cosTheta, 0.0), 1.0);
  const iridF = evalIridescence(1.0, iridescenceIor, cosTheta, thicknessNm, baseF0);
  return [
    baseF0[0] + (iridF[0] - baseF0[0]) * iridescence,
    baseF0[1] + (iridF[1] - baseF0[1]) * iridescence,
    baseF0[2] + (iridF[2] - baseF0[2]) * iridescence,
  ];
}

function clamp01(x: number): number {
  return Math.min(Math.max(x, 0), 1);
}

function materialSpecularF0(
  baseColor: Vec3,
  metallic: number,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  const si = clamp01(specularIntensity);
  const dielectricF0: Vec3 = [
    clamp01(0.04 * clamp01(specularColor[0]) * si),
    clamp01(0.04 * clamp01(specularColor[1]) * si),
    clamp01(0.04 * clamp01(specularColor[2]) * si),
  ];
  return [
    dielectricF0[0] + (baseColor[0] - dielectricF0[0]) * metallic,
    dielectricF0[1] + (baseColor[1] - dielectricF0[1]) * metallic,
    dielectricF0[2] + (baseColor[2] - dielectricF0[2]) * metallic,
  ];
}

/** ggxD (material.wgsl.ts:344). */
function ggxD(nDotH: number, alpha: number): number {
  const a2 = alpha * alpha;
  const n2 = Math.min(1, Math.max(0, nDotH * nDotH));
  const d = (1 - n2) + n2 * a2;
  return a2 / (PI * d * d);
}

function mixNumber(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Exact CPU mirror of `ggxMultiscatter.wgsl.ts:ggxDirectionalAlbedo`. */
function ggxDirectionalAlbedo(cosTheta: number, roughness: number): number {
  const mu = clamp01(cosTheta);
  const r = clamp01(roughness);
  const fr = r * (GGX_E_LUT_DIM - 1);
  const fm = mu * (GGX_E_LUT_DIM - 1);
  const r0 = Math.floor(fr);
  const m0 = Math.floor(fm);
  const r1 = Math.min(r0 + 1, GGX_E_LUT_DIM - 1);
  const m1 = Math.min(m0 + 1, GGX_E_LUT_DIM - 1);
  const tr = fr - r0;
  const tm = fm - m0;
  const e00 = GGX_E_LUT[r0 * GGX_E_LUT_DIM + m0]!;
  const e01 = GGX_E_LUT[r0 * GGX_E_LUT_DIM + m1]!;
  const e10 = GGX_E_LUT[r1 * GGX_E_LUT_DIM + m0]!;
  const e11 = GGX_E_LUT[r1 * GGX_E_LUT_DIM + m1]!;
  return Math.min(
    Math.max(mixNumber(mixNumber(e00, e01, tm), mixNumber(e10, e11, tm), tr), 0.02),
    1.0,
  );
}

/** Exact CPU mirror of `ggxMultiscatter.wgsl.ts:ggxAverageAlbedo`. */
function ggxAverageAlbedo(roughness: number): number {
  const r = clamp01(roughness);
  const fr = r * (GGX_E_LUT_DIM - 1);
  const r0 = Math.floor(fr);
  const r1 = Math.min(r0 + 1, GGX_E_LUT_DIM - 1);
  const tr = fr - r0;
  return Math.min(Math.max(mixNumber(GGX_EAVG_LUT[r0]!, GGX_EAVG_LUT[r1]!, tr), 0.3), 1.0);
}

function ggxMultiscatterLobeRoughness(
  f0: Vec3,
  roughnessV: number,
  roughnessL: number,
  roughnessAvg: number,
  nDotV: number,
  nDotL: number,
): Vec3 {
  const eAvg = ggxAverageAlbedo(roughnessAvg);
  const oneMinusEavg = 1.0 - eAvg;
  if (oneMinusEavg <= 0.0) return [0, 0, 0];
  const eo = ggxDirectionalAlbedo(nDotV, roughnessV);
  const ei = ggxDirectionalAlbedo(nDotL, roughnessL);
  const shape = ((1.0 - eo) * (1.0 - ei)) / (PI * oneMinusEavg);
  const out: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const fAvg = f0[c]! + (1.0 - f0[c]!) * (1.0 / 21.0);
    const seriesDenom = 1.0 - fAvg * oneMinusEavg;
    if (seriesDenom <= 0.0) return [0, 0, 0];
    const value = ((fAvg * fAvg * eAvg) / seriesDenom) * shape;
    out[c] = Number.isFinite(value) ? value : 0.0;
  }
  return out;
}

function ggxMultiscatterLobe(
  f0: Vec3,
  roughness: number,
  nDotV: number,
  nDotL: number,
): Vec3 {
  return ggxMultiscatterLobeRoughness(
    f0, roughness, roughness, roughness, nDotV, nDotL,
  );
}

/**
 * Per-channel derivative of the Kulla-Conty lobe with respect to F0. The
 * directional-albedo shape is achromatic; the colour-series derivative is
 * evaluated analytically so all base/specular/iridescence partials include
 * the production forward model's multiscatter term.
 */
function dGgxMultiscatter_dF0(
  f0: Vec3,
  roughness: number,
  nDotV: number,
  nDotL: number,
): Vec3 {
  const eAvg = ggxAverageAlbedo(roughness);
  const oneMinusEavg = 1.0 - eAvg;
  if (oneMinusEavg <= 0.0) return [0, 0, 0];
  const eo = ggxDirectionalAlbedo(nDotV, roughness);
  const ei = ggxDirectionalAlbedo(nDotL, roughness);
  const shape = ((1.0 - eo) * (1.0 - ei)) / (PI * oneMinusEavg);
  const dFavg_dF0 = 20.0 / 21.0;
  const out: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const fAvg = f0[c]! + (1.0 - f0[c]!) * (1.0 / 21.0);
    const seriesDenom = 1.0 - fAvg * oneMinusEavg;
    if (seriesDenom <= 0.0) return [0, 0, 0];
    const numerator = eAvg * (2.0 * fAvg - fAvg * fAvg * oneMinusEavg);
    const value = shape * dFavg_dF0 * numerator / (seriesDenom * seriesDenom);
    out[c] = Number.isFinite(value) ? value : 0.0;
  }
  return out;
}

function dGgxMultiscatter_dRoughness(
  f0: Vec3,
  roughness: number,
  nDotV: number,
  nDotL: number,
): Vec3 {
  const rp = clamp01(roughness + MULTISCATTER_ROUGHNESS_DERIV_STEP);
  const rm = clamp01(roughness - MULTISCATTER_ROUGHNESS_DERIV_STEP);
  const denom = rp - rm;
  if (denom <= 1e-8) return [0, 0, 0];
  const fp = ggxMultiscatterLobe(f0, rp, nDotV, nDotL);
  const fm = ggxMultiscatterLobe(f0, rm, nDotV, nDotL);
  return [
    (fp[0] - fm[0]) / denom,
    (fp[1] - fm[1]) / denom,
    (fp[2] - fm[2]) / denom,
  ];
}

/**
 * frDielectric (material.wgsl.ts:502 — PBR4e §9.3 FrDielectric). Unpolarised
 * Fresnel reflectance of a smooth dielectric interface. Mirrored here as the
 * forward the `ior` adjoint differentiates. Returns 1.0 on TIR (a hard
 * discontinuity where the derivative is 0 / undefined — `dFrDielectric_dIor`
 * returns 0 there, consistent with path-replay's frozen-event convention).
 *
 * `cosThetaI` is the cosine of the incident angle (caller passes the absolute
 * value at the surface); `eta` is the relative IOR (transmitted / incident).
 */
export function frDielectric(cosThetaIIn: number, etaIn: number): number {
  let cosThetaI = Math.min(Math.max(cosThetaIIn, -1.0), 1.0);
  let eta = etaIn;
  if (cosThetaI < 0.0) {
    eta = 1.0 / eta;
    cosThetaI = -cosThetaI;
  }
  const sin2ThetaI = Math.max(0.0, 1.0 - cosThetaI * cosThetaI);
  const sin2ThetaT = sin2ThetaI / (eta * eta);
  if (sin2ThetaT >= 1.0) return 1.0; // Total Internal Reflection.
  const cosThetaT = Math.sqrt(Math.max(0.0, 1.0 - sin2ThetaT));
  const rPar = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
  const rPerp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
  return 0.5 * (rPar * rPar + rPerp * rPerp);
}

// ── forward BRDF (mirror of bsdf.wgsl.ts:evaluateBrdf) ───────────────────────

/**
 * Cook-Torrance unified diffuse + specular BRDF. Exact mirror of
 * `bsdf.wgsl.ts:evaluateBrdf`. Returns the per-channel BRDF value (no cosine,
 * no Li — those are applied by the caller at the throughput level).
 */
export function evaluateBrdf(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const f0 = materialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  const f = fresnelSchlick(vDotH, f0);
  const alpha = ptWebgpuMicrofacetAlpha(roughness);
  const d = ggxD(nDotH, alpha);
  const g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  const specScale = (d * g) / Math.max(4.0 * nDotV * nDotL, 1e-6);
  const kd0 = 1.0 - metallic;
  const ms = ggxMultiscatterLobe(f0, roughness, nDotV, nDotL);
  const out: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const spec = specScale * f[c]!;
    const kd = (1.0 - f[c]!) * kd0;
    const diff = kd * baseColor[c]! * INV_PI;
    out[c] = diff + spec + ms[c]!;
  }
  return out;
}

function evaluateBrdfWithF0(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  f0: Vec3,
): Vec3 {
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const f = fresnelSchlick(vDotH, f0);
  const alpha = ptWebgpuMicrofacetAlpha(roughness);
  const d = ggxD(nDotH, alpha);
  const g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  const specScale = (d * g) / Math.max(4.0 * nDotV * nDotL, 1e-6);
  const kd0 = 1.0 - metallic;
  const ms = ggxMultiscatterLobe(f0, roughness, nDotV, nDotL);
  return [
    ((1.0 - f[0]) * kd0 * baseColor[0] * INV_PI) + specScale * f[0] + ms[0],
    ((1.0 - f[1]) * kd0 * baseColor[1] * INV_PI) + specScale * f[1] + ms[1],
    ((1.0 - f[2]) * kd0 * baseColor[2] * INV_PI) + specScale * f[2] + ms[2],
  ];
}

function evalClearcoatLobe(
  clearcoat: number,
  clearcoatRoughness: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
): Vec3 {
  if (clearcoat <= 0) return [0, 0, 0];
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const f = fresnelSchlick(vDotH, [0.04, 0.04, 0.04]);
  const alpha = ptWebgpuMicrofacetAlpha(clearcoatRoughness);
  const d = ggxD(nDotH, alpha);
  const g = smithG1(nDotV, clearcoatRoughness) * smithG1(nDotL, clearcoatRoughness);
  const specScale = (d * g) / Math.max(4.0 * nDotV * nDotL, 1e-6);
  return [clearcoat * f[0] * specScale, clearcoat * f[1] * specScale, clearcoat * f[2] * specScale];
}

function charlieD(nDotH: number, alpha: number): number {
  const invAlpha = 1.0 / Math.max(alpha, 1e-4);
  const sinThetaH = Math.sqrt(Math.max(0.0, 1.0 - nDotH * nDotH));
  return ((2.0 + invAlpha) * Math.pow(sinThetaH, invAlpha)) / (2.0 * PI);
}

function sheenVisibility(nDotL: number, nDotV: number): number {
  return 1.0 / Math.max(4.0 * (nDotL + nDotV - nDotL * nDotV), 1e-6);
}

function evalSheenLobe(
  sheen: number,
  sheenRoughness: number,
  sheenColor: Vec3,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
): Vec3 {
  if (sheen <= 0) return [0, 0, 0];
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const alpha = ptWebgpuMicrofacetAlpha(sheenRoughness);
  const d = charlieD(nDotH, alpha);
  const vis = sheenVisibility(nDotL, nDotV);
  return [
    sheen * sheenColor[0] * d * vis,
    sheen * sheenColor[1] * d * vis,
    sheen * sheenColor[2] * d * vis,
  ];
}

function ggxDAnis(hT: number, hB: number, hN: number, ax: number, ay: number): number {
  const d = (hT / ax) * (hT / ax) + (hB / ay) * (hB / ay) + hN * hN;
  return 1.0 / Math.max(PI * ax * ay * d * d, 1e-10);
}

function smithG1Anis(vT: number, vB: number, vN: number, ax: number, ay: number): number {
  const vN2 = Math.max(vN * vN, 1e-10);
  const numer = 2.0 * vN;
  const denom = vN + Math.sqrt(vN2 + (vT * ax) * (vT * ax) + (vB * ay) * (vB * ay));
  return numer / Math.max(denom, 1e-6);
}

function evalBrdfSpecAnisotropic(
  f0: Vec3,
  roughness: number,
  anisotropy: number,
  normal: Vec3,
  tangent: Vec3,
  bitangent: Vec3,
  wo: Vec3,
  wi: Vec3,
): Vec3 {
  const nDotV = Math.max(dot(normal, wo), 1e-6);
  const nDotL = Math.max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const vDotH = Math.max(dot(wo, h), 1e-6);
  const f = fresnelSchlick(vDotH, f0);
  const alpha = ptWebgpuMicrofacetAlpha(roughness);
  const aspect = Math.sqrt(Math.max(1.0 - 0.9 * anisotropy, 1e-4));
  const ax = Math.max(alpha / aspect, 1e-4);
  const ay = Math.max(alpha * aspect, 1e-4);
  const hT = dot(h, tangent);
  const hB = dot(h, bitangent);
  const hN = Math.max(dot(h, normal), 0.0);
  const woT = dot(wo, tangent);
  const woB = dot(wo, bitangent);
  const woN = Math.max(dot(wo, normal), 1e-6);
  const wiT = dot(wi, tangent);
  const wiB = dot(wi, bitangent);
  const wiN = Math.max(dot(wi, normal), 1e-6);
  const d = ggxDAnis(hT, hB, hN, ax, ay);
  const g = smithG1Anis(woT, woB, woN, ax, ay) * smithG1Anis(wiT, wiB, wiN, ax, ay);
  const specScale = (d * g) / Math.max(4.0 * nDotV * nDotL, 1e-6);
  return [specScale * f[0], specScale * f[1], specScale * f[2]];
}

function anisotropicAxes(alpha: number, anisotropy: number): readonly [number, number] {
  const aspect = Math.sqrt(Math.max(1.0 - 0.9 * anisotropy, 1e-4));
  return [Math.max(alpha / aspect, 1e-4), Math.max(alpha * aspect, 1e-4)];
}

function anisotropicProjectedRoughness(
  dir: Vec3,
  tangent: Vec3,
  bitangent: Vec3,
  roughness: number,
  anisotropy: number,
): number {
  const alpha = ptWebgpuMicrofacetAlpha(roughness);
  const [ax, ay] = anisotropicAxes(alpha, anisotropy);
  const dT = dot(dir, tangent);
  const dB = dot(dir, bitangent);
  const tangentLen2 = dT * dT + dB * dB;
  const projectionBlend = Math.min(Math.max(0.15 * anisotropy, 0.0), 0.15);
  if (tangentLen2 <= 1e-6) {
    const projectedNormal = Math.sqrt(Math.min(Math.max(0.5 * (ax + ay), 1e-4), 1.0));
    return mixNumber(roughness, projectedNormal, projectionBlend);
  }
  const alphaEff = Math.sqrt(
    ((dT * ax) * (dT * ax) + (dB * ay) * (dB * ay)) / tangentLen2,
  );
  const projected = Math.sqrt(Math.min(Math.max(alphaEff, 1e-4), 1.0));
  return mixNumber(roughness, projected, projectionBlend);
}

function anisotropicAverageRoughness(roughness: number, anisotropy: number): number {
  const alpha = ptWebgpuMicrofacetAlpha(roughness);
  const [ax, ay] = anisotropicAxes(alpha, anisotropy);
  const alphaRms = Math.sqrt(0.5 * (ax * ax + ay * ay));
  const projected = Math.sqrt(Math.min(Math.max(alphaRms, 1e-4), 1.0));
  return mixNumber(
    roughness,
    projected,
    Math.min(Math.max(0.15 * anisotropy, 0.0), 0.15),
  );
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(Math.max((x - edge0) / (edge1 - edge0), 0.0), 1.0);
  return t * t * (3.0 - 2.0 * t);
}

function anisotropicMultiscatterScale(anisotropy: number, roughnessForScale: number): number {
  const anisoReduction = smoothstep(0.0, 0.35, clamp01(anisotropy));
  return mixNumber(1.0, 0.6, anisoReduction) *
    smoothstep(0.35, 0.9, roughnessForScale);
}

/**
 * Map-free forward mirror of `evaluateBrdfFull(... clearcoat, clearcoatRoughness,
 * sheen=0, iridescence=0, anisotropy=0)`. Used by the clearcoat adjoint FD gate.
 */
export function evaluateBrdfWithClearcoat(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  clearcoat: number,
  clearcoatRoughness: number,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  const base = evaluateBrdf(
    baseColor,
    roughness,
    metallic,
    normal,
    wo,
    wi,
    specularColor,
    specularIntensity,
  );
  const cc = evalClearcoatLobe(clearcoat, clearcoatRoughness, normal, wo, wi);
  return [base[0] + cc[0], base[1] + cc[1], base[2] + cc[2]];
}

/**
 * Map-free forward mirror of the additive KHR_materials_sheen lobe. Used by the
 * scalar sheen adjoint FD gate; the production adjoint pass chains readable
 * sheenColor/sheenRoughness texture factors outside this CPU mirror.
 */
export function evaluateBrdfWithSheen(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  sheen: number,
  sheenRoughness: number,
  sheenColor: Vec3,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  const base = evaluateBrdf(
    baseColor,
    roughness,
    metallic,
    normal,
    wo,
    wi,
    specularColor,
    specularIntensity,
  );
  const sh = evalSheenLobe(sheen, sheenRoughness, sheenColor, normal, wo, wi);
  return [base[0] + sh[0], base[1] + sh[1], base[2] + sh[2]];
}

/**
 * Map-free forward mirror of the KHR_materials_iridescence scalar path in the
 * opaque isotropic direct-light domain. Iridescence modifies the base F0 before
 * the Cook-Torrance diffuse/specular partition; clearcoat/sheen/maps are omitted.
 */
export function evaluateBrdfWithIridescence(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  iridescence: number,
  iridescenceIor: number,
  iridescenceThicknessMin: number,
  iridescenceThicknessMax: number,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const baseF0 = materialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  const f0 = iridescenceModifiedF0(
    baseF0,
    iridescence,
    iridescenceIor,
    iridescenceThicknessMin,
    iridescenceThicknessMax,
    vDotH,
  );
  return evaluateBrdfWithF0(baseColor, roughness, metallic, normal, wo, wi, f0);
}

/**
 * Map-free forward mirror of the KHR_materials_anisotropy scalar path in the
 * opaque direct-light domain. It mirrors the anisotropic replacement of the
 * base specular lobe; additive clearcoat/sheen and texture maps are outside
 * this helper.
 */
export function evaluateBrdfWithAnisotropy(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  anisotropy: number,
  anisotropyRotation: number,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const f0 = materialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  const f = fresnelSchlick(vDotH, f0);
  const kd0 = 1.0 - metallic;
  const diff: Vec3 = [
    (1.0 - f[0]) * kd0 * baseColor[0] * INV_PI,
    (1.0 - f[1]) * kd0 * baseColor[1] * INV_PI,
    (1.0 - f[2]) * kd0 * baseColor[2] * INV_PI,
  ];
  if (anisotropy <= 0) {
    const iso = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi, specularColor, specularIntensity);
    return iso;
  }
  const [tanT, tanB] = buildOnb(normal);
  const c = Math.cos(anisotropyRotation);
  const s = Math.sin(anisotropyRotation);
  const anisoT: Vec3 = [
    c * tanT[0] + s * tanB[0],
    c * tanT[1] + s * tanB[1],
    c * tanT[2] + s * tanB[2],
  ];
  const anisoB: Vec3 = [
    -s * tanT[0] + c * tanB[0],
    -s * tanT[1] + c * tanB[1],
    -s * tanT[2] + c * tanB[2],
  ];
  const spec = evalBrdfSpecAnisotropic(f0, roughness, anisotropy, normal, anisoT, anisoB, wo, wi);
  const roughnessAvg = anisotropicAverageRoughness(roughness, anisotropy);
  const msBase = ggxMultiscatterLobeRoughness(
    f0,
    anisotropicProjectedRoughness(wo, anisoT, anisoB, roughness, anisotropy),
    anisotropicProjectedRoughness(wi, anisoT, anisoB, roughness, anisotropy),
    roughnessAvg,
    nDotV,
    nDotL,
  );
  const msScale = anisotropicMultiscatterScale(anisotropy, roughnessAvg);
  return [
    diff[0] + spec[0] + msScale * msBase[0],
    diff[1] + spec[1] + msScale * msBase[1],
    diff[2] + spec[2] + msScale * msBase[2],
  ];
}

// ── analytic partials ────────────────────────────────────────────────────────

/**
 * Analytic ∂(evaluateBrdf)_c / ∂(baseColor_j) — a 3×3 Jacobian (channel × input
 * component). Because Fresnel mixes baseColor into f0 (metallic > 0), changing
 * baseColor_j perturbs BOTH the diffuse term of channel j AND the specular term
 * of channel j (through f0_j → f_j). There is no cross-channel coupling, so the
 * Jacobian is diagonal: ∂out_c/∂baseColor_j = 0 for c ≠ j. We return the diagonal.
 */
export function dBrdf_dBaseColor(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const alpha = ptWebgpuMicrofacetAlpha(roughness);
  const d = ggxD(nDotH, alpha);
  const g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  const specScale = (d * g) / Math.max(4.0 * nDotV * nDotL, 1e-6);
  const kd0 = 1.0 - metallic;
  const currentF0 = materialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  const dMs_dF0 = dGgxMultiscatter_dF0(currentF0, roughness, nDotV, nDotL);

  // Schlick m5 factor (same for every channel).
  const m = Math.min(Math.max(1.0 - vDotH, 0.0), 1.0);
  const m2 = m * m;
  const m5 = m2 * m2 * m;

  // For channel c: f0_c = 0.04 + (baseColor_c - 0.04)·metallic.
  //   df0_c/dbaseColor_c = metallic.
  // f_c = f0_c + (1 - f0_c)·m5  ⇒  df_c/df0_c = 1 - m5.
  //   ⇒ df_c/dbaseColor_c = (1 - m5)·metallic.
  // out_c = (1 - f_c)·kd0·baseColor_c·INV_PI + specScale·f_c.
  //   d(diff_c)/dbaseColor_c = kd0·INV_PI·[ (1 - f_c) + baseColor_c·(-df_c/dbaseColor_c) ]
  //   d(spec_c)/dbaseColor_c = specScale·df_c/dbaseColor_c.
  const out: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const dielectricF0 = clamp01(0.04 * clamp01(specularColor[c]!) * clamp01(specularIntensity));
    const f0c = dielectricF0 + (baseColor[c]! - dielectricF0) * metallic;
    const fc = f0c + (1.0 - f0c) * m5;
    const dfc = (1.0 - m5) * metallic;
    const dDiff = kd0 * INV_PI * ((1.0 - fc) + baseColor[c]! * -dfc);
    const dSpec = specScale * dfc;
    out[c] = dDiff + dSpec + dMs_dF0[c]! * metallic;
  }
  return out;
}

/**
 * Analytic ∂(evaluateBrdf)_c / ∂roughness — per channel (the specular term is
 * channel-coupled only through Fresnel, but roughness affects D and both G1s,
 * which are achromatic; the per-channel result differs only via the Fresnel
 * weight f_c). The diffuse term is roughness-independent; the specular D/G
 * term and Kulla-Conty multiscatter compensation both contribute.
 *
 * alpha uses `PT_WEBGPU_MICROFACET_ALPHA_FLOOR`. Below that numerical floor,
 * dalpha/droughness = 0; above it = 2·roughness. Event classification remains
 * exact-zero delta versus positive finite and is deliberately separate.
 */
export function dBrdf_dRoughness(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const f0 = materialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  const f = fresnelSchlick(vDotH, f0);

  const alpha = ptWebgpuMicrofacetAlpha(roughness);
  const alphaClamped = roughness * roughness < PT_WEBGPU_MICROFACET_ALPHA_FLOOR;
  const dAlpha_dRough = alphaClamped ? 0.0 : 2.0 * roughness;

  // ── dD/dalpha ──
  //   D = a² / (PI·den²),  den = nDotH²·(a²-1)+1,  (ignoring the 1e-6 max floor)
  //   dDen/da² = nDotH²
  //   dD/da² = [PI·den² − a²·PI·2·den·nDotH²] / (PI·den²)²
  //          = [den − 2·a²·nDotH²] / (PI·den³)
  //   da²/droughness = 2·alpha·dAlpha_dRough
  const a2 = alpha * alpha;
  const n2 = Math.min(1, Math.max(0, nDotH * nDotH));
  const den = (1 - n2) + n2 * a2;
  const dD_da2 = (den - 2.0 * a2 * n2) / (PI * den * den * den);
  const da2_dRough = 2.0 * alpha * dAlpha_dRough;
  const dD_dRough = dD_da2 * da2_dRough;
  const d = ggxD(nDotH, alpha);

  // The exact Smith derivative shares the forward oracle's alpha=roughness² map.
  const g1V = smithG1(nDotV, roughness);
  const g1L = smithG1(nDotL, roughness);
  const g = g1V * g1L;
  const dG_dRough =
    roughDielectricSmithG1RoughnessDerivative(nDotV, roughness) * g1L +
    g1V * roughDielectricSmithG1RoughnessDerivative(nDotL, roughness);

  // spec_c = [(D·G)/(4·nDotV·nDotL)] · f_c.
  //   d(spec_c)/droughness = [(dD·G + D·dG)/(4·nDotV·nDotL)] · f_c.
  const invDenom = 1.0 / Math.max(4.0 * nDotV * nDotL, 1e-6);
  const dSpecScale = (dD_dRough * g + d * dG_dRough) * invDenom;
  const dMs = dGgxMultiscatter_dRoughness(f0, roughness, nDotV, nDotL);
  return [
    dSpecScale * f[0] + dMs[0],
    dSpecScale * f[1] + dMs[1],
    dSpecScale * f[2] + dMs[2],
  ];
}

/**
 * Analytic ∂(evaluateBrdf)_c / ∂metallic. Metallic affects both halves of the
 * opaque Disney base BRDF:
 *   - diffuse weight `kd0 = 1 - metallic` fades the Lambertian lobe out,
 *   - F0 blends from dielectric specular controls to baseColor, changing the
 *     Schlick Fresnel term that drives both specular energy and diffuse
 *     partitioning.
 *
 * The derivative is evaluated in the smooth unclamped interior; the optimizer
 * clamps metallic to [0,1] after each step, matching the forward material range.
 */
export function dBrdf_dMetallic(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const alpha = ptWebgpuMicrofacetAlpha(roughness);
  const d = ggxD(nDotH, alpha);
  const g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  const specScale = (d * g) / Math.max(4.0 * nDotV * nDotL, 1e-6);
  const kd0 = 1.0 - metallic;
  const currentF0 = materialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  const dMs_dF0 = dGgxMultiscatter_dF0(currentF0, roughness, nDotV, nDotL);
  const m = Math.min(Math.max(1.0 - vDotH, 0.0), 1.0);
  const m2 = m * m;
  const m5 = m2 * m2 * m;
  const out: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const dielectricF0 = clamp01(0.04 * clamp01(specularColor[c]!) * clamp01(specularIntensity));
    const f0c = dielectricF0 + (baseColor[c]! - dielectricF0) * metallic;
    const fc = f0c + (1.0 - f0c) * m5;
    const dfc = (1.0 - m5) * (baseColor[c]! - dielectricF0);
    const dDiff = baseColor[c]! * INV_PI * (-kd0 * dfc - (1.0 - fc));
    const dSpec = specScale * dfc;
    const dF0_dMetallic = baseColor[c]! - dielectricF0;
    out[c] = dDiff + dSpec + dMs_dF0[c]! * dF0_dMetallic;
  }
  return out;
}

/**
 * Analytic ∂(evaluateBrdf)_c / ∂specularColor_c for KHR_materials_specular.
 * The optimizer clamps this field to [0,1], and the derivative below is for the
 * smooth unclamped interior. The dielectric specular controls fade out as
 * metallic approaches 1 because metallic F0 is authored by baseColor.
 */
export function dBrdf_dSpecularColor(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  const dF0 = 0.04 * clamp01(specularIntensity) * (1.0 - metallic);
  const currentF0 = materialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  return dBrdf_dSpecularF0(
    baseColor, roughness, metallic, normal, wo, wi,
    currentF0, [dF0, dF0, dF0],
  );
}

/**
 * Analytic ∂(evaluateBrdf)_c / ∂specularIntensity for KHR_materials_specular.
 */
export function dBrdf_dSpecularIntensity(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  const currentF0 = materialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  return dBrdf_dSpecularF0(
    baseColor, roughness, metallic, normal, wo, wi,
    currentF0,
    [
      0.04 * clamp01(specularColor[0]) * (1.0 - metallic),
      0.04 * clamp01(specularColor[1]) * (1.0 - metallic),
      0.04 * clamp01(specularColor[2]) * (1.0 - metallic),
    ],
  );
}

/**
 * Analytic ∂(evaluateBrdfFull)_c / ∂clearcoat for the additive map-free
 * KHR_materials_clearcoat lobe. The returned value is the per-unit lobe kernel;
 * callers still multiply by NdotL and incident radiance.
 */
export function dBrdf_dClearcoat(
  clearcoatRoughness: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
): Vec3 {
  return evalClearcoatLobe(1.0, clearcoatRoughness, normal, wo, wi);
}

/**
 * Analytic ∂(evaluateBrdfFull)_c / ∂clearcoatRoughness for the additive
 * KHR_materials_clearcoat lobe. This mirrors `evalClearcoatLobe` with frozen
 * directions; callers pass the effective clearcoat normal when a clearcoat
 * normal map has already been replayed.
 */
export function dBrdf_dClearcoatRoughness(
  clearcoat: number,
  clearcoatRoughness: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
): Vec3 {
  if (clearcoat <= 0) return [0, 0, 0];
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const f = fresnelSchlick(vDotH, [0.04, 0.04, 0.04]);

  const alpha = ptWebgpuMicrofacetAlpha(clearcoatRoughness);
  const alphaClamped = clearcoatRoughness * clearcoatRoughness < PT_WEBGPU_MICROFACET_ALPHA_FLOOR;
  const dAlpha_dRough = alphaClamped ? 0.0 : 2.0 * clearcoatRoughness;

  const a2 = alpha * alpha;
  const n2 = Math.min(1, Math.max(0, nDotH * nDotH));
  const den = (1 - n2) + n2 * a2;
  const dD_da2 = (den - 2.0 * a2 * n2) / (PI * den * den * den);
  const da2_dRough = 2.0 * alpha * dAlpha_dRough;
  const dD_dRough = dD_da2 * da2_dRough;
  const d = ggxD(nDotH, alpha);

  const g1V = smithG1(nDotV, clearcoatRoughness);
  const g1L = smithG1(nDotL, clearcoatRoughness);
  const g = g1V * g1L;
  const dG_dRough =
    roughDielectricSmithG1RoughnessDerivative(nDotV, clearcoatRoughness) * g1L +
    g1V * roughDielectricSmithG1RoughnessDerivative(nDotL, clearcoatRoughness);

  const invDenom = 1.0 / Math.max(4.0 * nDotV * nDotL, 1e-6);
  const dSpecScale = (dD_dRough * g + d * dG_dRough) * invDenom;
  return [clearcoat * f[0] * dSpecScale, clearcoat * f[1] * dSpecScale, clearcoat * f[2] * dSpecScale];
}

/**
 * Analytic ∂(evaluateBrdfFull)_c / ∂sheen for the additive map-free
 * KHR_materials_sheen lobe. The returned value is the per-unit sheen lobe;
 * callers still multiply by NdotL and incident radiance.
 */
export function dBrdf_dSheen(
  sheenRoughness: number,
  sheenColor: Vec3,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
): Vec3 {
  return evalSheenLobe(1.0, sheenRoughness, sheenColor, normal, wo, wi);
}

/**
 * Analytic ∂(evaluateBrdfFull)_c / ∂sheenColor_c for the additive map-free
 * KHR_materials_sheen lobe. There is no cross-channel coupling, so the returned
 * Vec3 is the diagonal of the RGB Jacobian.
 */
export function dBrdf_dSheenColor(
  sheen: number,
  sheenRoughness: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
): Vec3 {
  return evalSheenLobe(sheen, sheenRoughness, [1, 1, 1], normal, wo, wi);
}

/**
 * Analytic ∂(evaluateBrdfFull)_c / ∂sheenRoughness for the additive map-free
 * KHR_materials_sheen Charlie lobe. Directions and sheenColor are frozen.
 */
export function dBrdf_dSheenRoughness(
  sheen: number,
  sheenRoughness: number,
  sheenColor: Vec3,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
): Vec3 {
  if (sheen <= 0) return [0, 0, 0];
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const sinThetaH = Math.sqrt(Math.max(0.0, 1.0 - nDotH * nDotH));
  const alphaRaw = sheenRoughness * sheenRoughness;
  const dAlpha_dRough = alphaRaw < PT_WEBGPU_MICROFACET_ALPHA_FLOOR ? 0.0 : 2.0 * sheenRoughness;
  if (sinThetaH <= 1e-6 || dAlpha_dRough === 0.0) return [0, 0, 0];

  const alpha = Math.max(alphaRaw, PT_WEBGPU_MICROFACET_ALPHA_FLOOR);
  const q = 1.0 / Math.max(alpha, 1e-4);
  const powTerm = Math.pow(sinThetaH, q);
  const logSin = Math.log(sinThetaH);
  const dD_dQ = (powTerm * (1.0 + (2.0 + q) * logSin)) / (2.0 * PI);
  const dQ_dAlpha = -1.0 / (alpha * alpha);
  const dD_dRough = dD_dQ * dQ_dAlpha * dAlpha_dRough;
  const vis = sheenVisibility(nDotL, nDotV);
  return [
    sheen * sheenColor[0] * dD_dRough * vis,
    sheen * sheenColor[1] * dD_dRough * vis,
    sheen * sheenColor[2] * dD_dRough * vis,
  ];
}

/**
 * Analytic ∂(evaluateBrdfFull)_c / ∂iridescence for the map-free
 * KHR_materials_iridescence scalar. Directions, IOR, thickness range, and
 * texture-free material state are frozen; the derivative is the existing F0
 * partial with dF0/dIridescence = F_iridescent - F0_base.
 */
export function dBrdf_dIridescence(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  iridescence: number,
  iridescenceIor: number,
  iridescenceThicknessMin: number,
  iridescenceThicknessMax: number,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const baseF0 = materialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  const thicknessNm = iridescenceThicknessMin +
    (iridescenceThicknessMax - iridescenceThicknessMin) * Math.min(Math.max(vDotH, 0.0), 1.0);
  const iridF = evalIridescence(1.0, iridescenceIor, vDotH, thicknessNm, baseF0);
  const amount = clamp01(iridescence);
  const currentF0: Vec3 = [
    baseF0[0] + (iridF[0] - baseF0[0]) * amount,
    baseF0[1] + (iridF[1] - baseF0[1]) * amount,
    baseF0[2] + (iridF[2] - baseF0[2]) * amount,
  ];
  return dBrdf_dSpecularF0(
    baseColor, roughness, metallic, normal, wo, wi,
    currentF0,
    [
      iridF[0] - baseF0[0],
      iridF[1] - baseF0[1],
      iridF[2] - baseF0[2],
    ],
  );
}

/**
 * Path-replay partial for map-free KHR_materials_iridescence `iridescenceIor`.
 *
 * The Belcour thin-film helper is piecewise and branchy (TIR, phase flips,
 * clamps), so this mirrors the WGSL replay path with a local symmetric
 * derivative of the thin-film F0 colour, then chains that through the existing
 * ∂BRDF/∂F0 helper. This is still a single replay pass over the frozen path; it
 * is not a full-render finite-difference probe.
 */
export function dBrdf_dIridescenceIor(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  iridescence: number,
  iridescenceIor: number,
  iridescenceThicknessMin: number,
  iridescenceThicknessMax: number,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  if (iridescence <= 0) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const baseF0 = materialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  const thicknessNm = iridescenceThicknessMin +
    (iridescenceThicknessMax - iridescenceThicknessMin) * Math.min(Math.max(vDotH, 0.0), 1.0);
  const step = IRIDESCENCE_IOR_DERIV_STEP;
  const iorP = Math.max(1.0, iridescenceIor + step);
  const iorM = Math.max(1.0, iridescenceIor - step);
  const denom = iorP - iorM;
  if (denom <= 1e-6) return [0, 0, 0];
  const currentIridF = evalIridescence(
    1.0, iridescenceIor, vDotH, thicknessNm, baseF0,
  );
  const currentF0: Vec3 = [
    baseF0[0] + (currentIridF[0] - baseF0[0]) * iridescence,
    baseF0[1] + (currentIridF[1] - baseF0[1]) * iridescence,
    baseF0[2] + (currentIridF[2] - baseF0[2]) * iridescence,
  ];
  const fp = evalIridescence(1.0, iorP, vDotH, thicknessNm, baseF0);
  const fm = evalIridescence(1.0, iorM, vDotH, thicknessNm, baseF0);
  return dBrdf_dSpecularF0(
    baseColor, roughness, metallic, normal, wo, wi,
    currentF0,
    [
      iridescence * (fp[0] - fm[0]) / denom,
      iridescence * (fp[1] - fm[1]) / denom,
      iridescence * (fp[2] - fm[2]) / denom,
    ],
  );
}

/**
 * Path-replay partials for KHR_materials_iridescence
 * `iridescenceThicknessRange`.
 *
 * The forward single-layer helper evaluates a sampled thickness. In the
 * map-free direct-light domain that sample is `V·H`; with
 * `iridescenceThicknessMap` it is the readable G-channel texel. We take a local
 * symmetric derivative of that thin-film F0 colour with respect to sampled
 * thickness, then chain it to the authored range endpoints by
 * ∂thickness/∂min = 1 − t and ∂thickness/∂max = t. This mirrors the WGSL replay
 * pass and is still a frozen-path local partial, not a full-render FD probe.
 */
export function dBrdf_dIridescenceThicknessRange(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  iridescence: number,
  iridescenceIor: number,
  iridescenceThicknessMin: number,
  iridescenceThicknessMax: number,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
  iridescenceThicknessTexel: number | null = null,
): { min: Vec3; max: Vec3 } {
  if (iridescence <= 0) return { min: [0, 0, 0], max: [0, 0, 0] };
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const vDotH = Math.min(Math.max(dot(wo, h), 0.0), 1.0);
  const rangeT = iridescenceThicknessTexel == null || !Number.isFinite(iridescenceThicknessTexel)
    ? vDotH
    : Math.min(Math.max(iridescenceThicknessTexel, 0.0), 1.0);
  const baseF0 = materialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  const thicknessNm = iridescenceThicknessMin +
    (iridescenceThicknessMax - iridescenceThicknessMin) * rangeT;
  const step = IRIDESCENCE_THICKNESS_DERIV_STEP;
  const tp = Math.max(0.0, thicknessNm + step);
  const tm = Math.max(0.0, thicknessNm - step);
  const denom = tp - tm;
  if (denom <= 1e-6) return { min: [0, 0, 0], max: [0, 0, 0] };
  const currentIridF = evalIridescence(
    1.0, iridescenceIor, vDotH, thicknessNm, baseF0,
  );
  const currentF0: Vec3 = [
    baseF0[0] + (currentIridF[0] - baseF0[0]) * iridescence,
    baseF0[1] + (currentIridF[1] - baseF0[1]) * iridescence,
    baseF0[2] + (currentIridF[2] - baseF0[2]) * iridescence,
  ];
  const fp = evalIridescence(1.0, iridescenceIor, vDotH, tp, baseF0);
  const fm = evalIridescence(1.0, iridescenceIor, vDotH, tm, baseF0);
  const dBrdf_dThickness = dBrdf_dSpecularF0(
    baseColor, roughness, metallic, normal, wo, wi,
    currentF0,
    [
      iridescence * (fp[0] - fm[0]) / denom,
      iridescence * (fp[1] - fm[1]) / denom,
      iridescence * (fp[2] - fm[2]) / denom,
    ],
  );
  const minWeight = 1.0 - rangeT;
  const maxWeight = rangeT;
  return {
    min: [
      dBrdf_dThickness[0] * minWeight,
      dBrdf_dThickness[1] * minWeight,
      dBrdf_dThickness[2] * minWeight,
    ],
    max: [
      dBrdf_dThickness[0] * maxWeight,
      dBrdf_dThickness[1] * maxWeight,
      dBrdf_dThickness[2] * maxWeight,
    ],
  };
}

/**
 * Path-replay partial for map-free KHR_materials_anisotropy `anisotropy`.
 *
 * The anisotropic GGX branch changes the roughness axes and activates only
 * above the zero-anisotropy guard, so this uses a local symmetric derivative of
 * the map-free anisotropic BRDF mirror with frozen directions. This is still a
 * replay-local partial, not a full-render finite-difference optimization step.
 */
export function dBrdf_dAnisotropy(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  anisotropy: number,
  anisotropyRotation: number,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  const step = ANISOTROPY_DERIV_STEP;
  const ap = Math.min(1.0, Math.max(0.0, anisotropy + step));
  const am = Math.min(1.0, Math.max(0.0, anisotropy - step));
  const denom = ap - am;
  if (denom <= 1e-6) return [0, 0, 0];
  const fp = evaluateBrdfWithAnisotropy(
    baseColor, roughness, metallic, normal, wo, wi,
    ap, anisotropyRotation, specularColor, specularIntensity,
  );
  const fm = evaluateBrdfWithAnisotropy(
    baseColor, roughness, metallic, normal, wo, wi,
    am, anisotropyRotation, specularColor, specularIntensity,
  );
  return [(fp[0] - fm[0]) / denom, (fp[1] - fm[1]) / denom, (fp[2] - fm[2]) / denom];
}

/**
 * Path-replay partial for map-free KHR_materials_anisotropy
 * `anisotropyRotation`. Rotation is inert when anisotropy is zero.
 */
export function dBrdf_dAnisotropyRotation(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  anisotropy: number,
  anisotropyRotation: number,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  if (anisotropy <= 0) return [0, 0, 0];
  const step = ANISOTROPY_ROTATION_DERIV_STEP;
  const fp = evaluateBrdfWithAnisotropy(
    baseColor, roughness, metallic, normal, wo, wi,
    anisotropy, anisotropyRotation + step, specularColor, specularIntensity,
  );
  const fm = evaluateBrdfWithAnisotropy(
    baseColor, roughness, metallic, normal, wo, wi,
    anisotropy, anisotropyRotation - step, specularColor, specularIntensity,
  );
  return [
    (fp[0] - fm[0]) / (2 * step),
    (fp[1] - fm[1]) / (2 * step),
    (fp[2] - fm[2]) / (2 * step),
  ];
}

/**
 * Replay-local partials for base-BRDF controls when KHR_materials_anisotropy is
 * active. The closed-form isotropic partials above use an isotropic GGX
 * D/G scale; when anisotropy is nonzero, these helpers differentiate the same
 * frozen-direction anisotropic BRDF mirror used by the forward path. This is
 * still a single path-replay pass, not a full-render finite-difference probe.
 */
export function dBrdf_dBaseColorWithAnisotropy(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  anisotropy: number,
  anisotropyRotation: number,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  if (anisotropy <= 0) {
    return dBrdf_dBaseColor(
      baseColor, roughness, metallic, normal, wo, wi, specularColor, specularIntensity,
    );
  }
  const out: [number, number, number] = [0, 0, 0];
  const step = ANISOTROPIC_BASE_PARAM_DERIV_STEP;
  for (let c = 0; c < 3; c++) {
    const plus = perturbChannelClamped(baseColor, c, step);
    const minus = perturbChannelClamped(baseColor, c, -step);
    const denom = plus[c]! - minus[c]!;
    if (denom <= 1e-8) continue;
    const fp = evaluateBrdfWithAnisotropy(
      plus, roughness, metallic, normal, wo, wi,
      anisotropy, anisotropyRotation, specularColor, specularIntensity,
    );
    const fm = evaluateBrdfWithAnisotropy(
      minus, roughness, metallic, normal, wo, wi,
      anisotropy, anisotropyRotation, specularColor, specularIntensity,
    );
    out[c] = (fp[c]! - fm[c]!) / denom;
  }
  return out;
}

export function dBrdf_dRoughnessWithAnisotropy(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  anisotropy: number,
  anisotropyRotation: number,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  if (anisotropy <= 0) {
    return dBrdf_dRoughness(
      baseColor, roughness, metallic, normal, wo, wi, specularColor, specularIntensity,
    );
  }
  const step = ANISOTROPIC_BASE_PARAM_DERIV_STEP;
  const rp = Math.min(1.0, Math.max(0.0, roughness + step));
  const rm = Math.min(1.0, Math.max(0.0, roughness - step));
  const denom = rp - rm;
  if (denom <= 1e-8) return [0, 0, 0];
  const fp = evaluateBrdfWithAnisotropy(
    baseColor, rp, metallic, normal, wo, wi,
    anisotropy, anisotropyRotation, specularColor, specularIntensity,
  );
  const fm = evaluateBrdfWithAnisotropy(
    baseColor, rm, metallic, normal, wo, wi,
    anisotropy, anisotropyRotation, specularColor, specularIntensity,
  );
  return [(fp[0] - fm[0]) / denom, (fp[1] - fm[1]) / denom, (fp[2] - fm[2]) / denom];
}

export function dBrdf_dMetallicWithAnisotropy(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  anisotropy: number,
  anisotropyRotation: number,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  if (anisotropy <= 0) {
    return dBrdf_dMetallic(
      baseColor, roughness, metallic, normal, wo, wi, specularColor, specularIntensity,
    );
  }
  const step = ANISOTROPIC_BASE_PARAM_DERIV_STEP;
  const mp = Math.min(1.0, Math.max(0.0, metallic + step));
  const mm = Math.min(1.0, Math.max(0.0, metallic - step));
  const denom = mp - mm;
  if (denom <= 1e-8) return [0, 0, 0];
  const fp = evaluateBrdfWithAnisotropy(
    baseColor, roughness, mp, normal, wo, wi,
    anisotropy, anisotropyRotation, specularColor, specularIntensity,
  );
  const fm = evaluateBrdfWithAnisotropy(
    baseColor, roughness, mm, normal, wo, wi,
    anisotropy, anisotropyRotation, specularColor, specularIntensity,
  );
  return [(fp[0] - fm[0]) / denom, (fp[1] - fm[1]) / denom, (fp[2] - fm[2]) / denom];
}

export function dBrdf_dSpecularColorWithAnisotropy(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  anisotropy: number,
  anisotropyRotation: number,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  if (anisotropy <= 0) {
    return dBrdf_dSpecularColor(
      baseColor, roughness, metallic, normal, wo, wi, specularColor, specularIntensity,
    );
  }
  const out: [number, number, number] = [0, 0, 0];
  const step = ANISOTROPIC_BASE_PARAM_DERIV_STEP;
  for (let c = 0; c < 3; c++) {
    const plus = perturbChannelClamped(specularColor, c, step);
    const minus = perturbChannelClamped(specularColor, c, -step);
    const denom = plus[c]! - minus[c]!;
    if (denom <= 1e-8) continue;
    const fp = evaluateBrdfWithAnisotropy(
      baseColor, roughness, metallic, normal, wo, wi,
      anisotropy, anisotropyRotation, plus, specularIntensity,
    );
    const fm = evaluateBrdfWithAnisotropy(
      baseColor, roughness, metallic, normal, wo, wi,
      anisotropy, anisotropyRotation, minus, specularIntensity,
    );
    out[c] = (fp[c]! - fm[c]!) / denom;
  }
  return out;
}

export function dBrdf_dSpecularIntensityWithAnisotropy(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  anisotropy: number,
  anisotropyRotation: number,
  specularColor: Vec3 = [1, 1, 1],
  specularIntensity = 1,
): Vec3 {
  if (anisotropy <= 0) {
    return dBrdf_dSpecularIntensity(
      baseColor, roughness, metallic, normal, wo, wi, specularColor, specularIntensity,
    );
  }
  const step = ANISOTROPIC_BASE_PARAM_DERIV_STEP;
  const ip = Math.min(1.0, Math.max(0.0, specularIntensity + step));
  const im = Math.min(1.0, Math.max(0.0, specularIntensity - step));
  const denom = ip - im;
  if (denom <= 1e-8) return [0, 0, 0];
  const fp = evaluateBrdfWithAnisotropy(
    baseColor, roughness, metallic, normal, wo, wi,
    anisotropy, anisotropyRotation, specularColor, ip,
  );
  const fm = evaluateBrdfWithAnisotropy(
    baseColor, roughness, metallic, normal, wo, wi,
    anisotropy, anisotropyRotation, specularColor, im,
  );
  return [(fp[0] - fm[0]) / denom, (fp[1] - fm[1]) / denom, (fp[2] - fm[2]) / denom];
}

function perturbChannelClamped(v: Vec3, channel: number, delta: number): Vec3 {
  const out: [number, number, number] = [v[0], v[1], v[2]];
  out[channel] = Math.min(1.0, Math.max(0.0, out[channel]! + delta));
  return out;
}

function dBrdf_dSpecularF0(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  currentF0: Vec3,
  dF0: Vec3,
): Vec3 {
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const alpha = ptWebgpuMicrofacetAlpha(roughness);
  const d = ggxD(nDotH, alpha);
  const g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  const specScale = (d * g) / Math.max(4.0 * nDotV * nDotL, 1e-6);
  const kd0 = 1.0 - metallic;
  const m = Math.min(Math.max(1.0 - vDotH, 0.0), 1.0);
  const m2 = m * m;
  const m5 = m2 * m2 * m;
  const dMs_dF0 = dGgxMultiscatter_dF0(currentF0, roughness, nDotV, nDotL);
  return [
    dF0[0] * ((1.0 - m5) * (specScale - kd0 * baseColor[0] * INV_PI) + dMs_dF0[0]),
    dF0[1] * ((1.0 - m5) * (specScale - kd0 * baseColor[1] * INV_PI) + dMs_dF0[1]),
    dF0[2] * ((1.0 - m5) * (specScale - kd0 * baseColor[2] * INV_PI) + dMs_dF0[2]),
  ];
}

/**
 * Analytic ∂(contribution)_c / ∂(emissive_c) — the emissive-on-hit partial.
 *
 * Emission is NOT a BSDF term: in the rendering equation it is an ADDITIVE
 * source `Le` along the outgoing direction, gated by the path throughput up to
 * the hit. The forward kernel adds it on a camera-/refraction-visible hit as
 *   `radiance += throughput · emissive`            (shadePrologue.wgsl.ts:63)
 * where the SHADER's `emissive` is the PACKED value `emissive_param ·
 * emissiveIntensity` (materialPacking.ts:122-124 folds the intensity in). So the
 * per-channel contribution partial w.r.t. the optimizable `emissive` PARAMETER
 * is diagonal and CONSTANT (independent of geometry / BRDF):
 *   ∂(contribution)_c / ∂(emissive_param_c) = throughput_c · emissiveIntensity,
 *   ∂(contribution)_c / ∂(emissive_param_j) = 0   for c ≠ j.
 * For a PRIMARY (camera) hit `throughput = 1`, so the partial is the identity
 * scaled by `emissiveIntensity` (= 1 when the host left intensity at its
 * default). `throughput` is passed so an indirect-bounce extension can scale it;
 * the single-bounce adjoint pass differentiates the primary hit (throughput 1).
 *
 * This is a CONTRIBUTION-level partial, not a `dBrdf_*` — it bypasses the BRDF
 * entirely. The chain rule is `∂loss/∂emissive_c = dLoss/dRendered_c · (this)_c`.
 */
export function dContribution_dEmissive(
  throughput: Vec3,
  emissiveIntensity: number,
): Vec3 {
  return [
    throughput[0] * emissiveIntensity,
    throughput[1] * emissiveIntensity,
    throughput[2] * emissiveIntensity,
  ];
}

/**
 * Analytic ∂(contribution)/∂emissiveIntensity — scalar partial of the same
 * additive primary-hit emission term as {@link dContribution_dEmissive}.
 *
 * Forward:
 *   contribution_c = throughput_c * emissive_c * emissiveIntensity
 *
 * Therefore:
 *   ∂loss/∂emissiveIntensity = Σ_c dLoss/dRendered_c * throughput_c * emissive_c
 *
 * The end-to-end adjoint pass carries the UNFACTORED material emissive RGB in
 * its parameter descriptor so intensity=0 remains differentiable; it never
 * divides the packed `emissive * emissiveIntensity` value back out.
 */
export function dContribution_dEmissiveIntensity(
  throughput: Vec3,
  emissive: Vec3,
): Vec3 {
  return [
    throughput[0] * emissive[0],
    throughput[1] * emissive[1],
    throughput[2] * emissive[2],
  ];
}

/**
 * Analytic ∂(frDielectric)/∂ior — the scalar partial of the dielectric Fresnel
 * reflectance w.r.t. the index of refraction, evaluated at a frozen incident
 * cosine. This is the ONLY differentiable `ior` dependence in the current
 * forward kernel (opaque dielectric F0 is controlled by KHR_materials_specular,
 * and metallic F0 by baseColor — see the file header), so it is the honest
 * `ior` partial.
 *
 * Forward (mirror of `frDielectric` above / material.wgsl.ts:502):
 *   eta = ior (front face) or 1/ior (back face);  cosI = |cosThetaI|;
 *   s = 1 - cosI² ;  sin2T = s/eta² ;  cosT = sqrt(1 - sin2T);
 *   r∥ = (eta·cosI - cosT)/(eta·cosI + cosT);
 *   r⊥ = (cosI - eta·cosT)/(cosI + eta·cosT);
 *   Fr = ½(r∥² + r⊥²).
 * Derivatives (all w.r.t. eta first, then chain to ior):
 *   dcosT/deta = s / (cosT·eta³)            (since d(sin2T)/deta = -2s·eta⁻³)
 *   r∥ = a/b,  a = eta·cosI - cosT,  b = eta·cosI + cosT:
 *     da/deta = cosI - dcosT/deta ;  db/deta = cosI + dcosT/deta
 *     dr∥/deta = (a'b - a·b') / b²
 *   r⊥ = c/d,  c = cosI - eta·cosT,  d = cosI + eta·cosT:
 *     dc/deta = -(cosT + eta·dcosT/deta) ;  dd/deta = -dc/deta
 *     dr⊥/deta = (c'd - c·d') / d²
 *   dFr/deta = r∥·dr∥/deta + r⊥·dr⊥/deta
 *   dFr/dior = dFr/deta · (front ? 1 : -1/ior²)     (eta = ior vs 1/ior)
 * TIR (sin2T ≥ 1) and grazing (cosT → 0) are hard events: Fr is pinned to 1 and
 * the derivative is returned as 0 (path-replay's frozen-discontinuity convention).
 *
 * ⚠ CONSUMPTION CAVEAT: the Phase-1 engine adjoint pass (adjointPass.wgsl.ts)
 * differentiates only single-bounce diffuse/glossy DIRECT lighting — it does NOT
 * trace the transmissive reflect/refract partition where `frDielectric` lives.
 * So this partial is GPU-validatable in isolation (analytic == FD on `frDielectric`)
 * but is NOT yet wired into an end-to-end `∂loss/∂ior` gradient. Wiring it
 * requires a transmissive-NEE adjoint (a deliberate follow-up).
 */
export function dFrDielectric_dIor(cosThetaIIn: number, ior: number): number {
  let cosThetaI = Math.min(Math.max(cosThetaIIn, -1.0), 1.0);
  // eta = ior (front) or 1/ior (back); track the chain factor deta/dior.
  let eta: number;
  let dEta_dIor: number;
  if (cosThetaI < 0.0) {
    eta = 1.0 / ior;
    dEta_dIor = -1.0 / (ior * ior);
    cosThetaI = -cosThetaI;
  } else {
    eta = ior;
    dEta_dIor = 1.0;
  }
  const s = Math.max(0.0, 1.0 - cosThetaI * cosThetaI);
  const sin2ThetaT = s / (eta * eta);
  if (sin2ThetaT >= 1.0) return 0.0; // TIR — Fr pinned to 1, derivative 0.
  const cosThetaT = Math.sqrt(Math.max(0.0, 1.0 - sin2ThetaT));
  if (cosThetaT <= 1e-6) return 0.0; // grazing — guard the 1/cosT below.

  const dCosT_dEta = s / (cosThetaT * eta * eta * eta);

  // r∥ = a/b.
  const a = eta * cosThetaI - cosThetaT;
  const b = eta * cosThetaI + cosThetaT;
  const da = cosThetaI - dCosT_dEta;
  const db = cosThetaI + dCosT_dEta;
  const rPar = a / b;
  const dRPar_dEta = (da * b - a * db) / (b * b);

  // r⊥ = c/d.
  const c = cosThetaI - eta * cosThetaT;
  const dd = cosThetaT + eta * dCosT_dEta; // = d(eta·cosT)/deta
  const cc = cosThetaI + eta * cosThetaT;
  const dcNum = -dd; // dc/deta = -(cosT + eta·dcosT/deta)
  const rPerp = c / cc;
  const dRPerp_dEta = (dcNum * cc - c * dd) / (cc * cc);

  const dFr_dEta = rPar * dRPar_dEta + rPerp * dRPerp_dEta;
  return dFr_dEta * dEta_dIor;
}
