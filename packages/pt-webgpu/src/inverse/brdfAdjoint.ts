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
 *  - map-free KHR_materials_sheen controls (`sheen`, `sheenColor`, and
 *    `sheenRoughness`) for the additive, map-free direct-light sheen lobe.
 *    Sheen colour/roughness maps remain outside this oracle.
 *  - map-free scalar KHR_materials_iridescence (`iridescence`) through the
 *    thin-film-modified base F0 used by the opaque direct-light specular and
 *    diffuse partition. Iridescence maps, thickness maps, and IOR/thickness
 *    parameter gradients remain outside this oracle.
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

// ── primitive mirrors (match material.wgsl.ts exactly) ──────────────────────

function safeNormalize(v: Vec3): Vec3 {
  const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (len < 1e-8) return [0, 0, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
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
  if (iridescence < 1e-4) return baseF0;
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
  const d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / Math.max(PI * d * d, 1e-6);
}

/** smithG1 (material.wgsl.ts:350). */
function smithG1(nDotV: number, roughness: number): number {
  const r = roughness + 1.0;
  const k = r * r * 0.125;
  return nDotV / Math.max(nDotV * (1.0 - k) + k, 1e-6);
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
  const alpha = Math.max(roughness * roughness, 1e-3);
  const d = ggxD(nDotH, alpha);
  const g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  const specScale = (d * g) / Math.max(4.0 * nDotV * nDotL, 1e-6);
  const kd0 = 1.0 - metallic;
  const out: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const spec = specScale * f[c]!;
    const kd = (1.0 - f[c]!) * kd0;
    const diff = kd * baseColor[c]! * INV_PI;
    out[c] = diff + spec;
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
  const alpha = Math.max(roughness * roughness, 1e-3);
  const d = ggxD(nDotH, alpha);
  const g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  const specScale = (d * g) / Math.max(4.0 * nDotV * nDotL, 1e-6);
  const kd0 = 1.0 - metallic;
  return [
    ((1.0 - f[0]) * kd0 * baseColor[0] * INV_PI) + specScale * f[0],
    ((1.0 - f[1]) * kd0 * baseColor[1] * INV_PI) + specScale * f[1],
    ((1.0 - f[2]) * kd0 * baseColor[2] * INV_PI) + specScale * f[2],
  ];
}

function evalClearcoatLobe(
  clearcoat: number,
  clearcoatRoughness: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
): Vec3 {
  if (clearcoat < 1e-4) return [0, 0, 0];
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const f = fresnelSchlick(vDotH, [0.04, 0.04, 0.04]);
  const alpha = Math.max(clearcoatRoughness * clearcoatRoughness, 1e-3);
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
  if (sheen < 1e-4) return [0, 0, 0];
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const alpha = Math.max(sheenRoughness * sheenRoughness, 1e-3);
  const d = charlieD(nDotH, alpha);
  const vis = sheenVisibility(nDotL, nDotV);
  return [
    sheen * sheenColor[0] * d * vis,
    sheen * sheenColor[1] * d * vis,
    sheen * sheenColor[2] * d * vis,
  ];
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
 * scalar sheen adjoint FD gate; sheenColor gradients/maps remain finite
 * difference because the pass does not yet replay texture sampling for this lobe.
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
  const alpha = Math.max(roughness * roughness, 1e-3);
  const d = ggxD(nDotH, alpha);
  const g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  const specScale = (d * g) / Math.max(4.0 * nDotV * nDotL, 1e-6);
  const kd0 = 1.0 - metallic;

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
    out[c] = dDiff + dSpec;
  }
  return out;
}

/**
 * Analytic ∂(evaluateBrdf)_c / ∂roughness — per channel (the specular term is
 * channel-coupled only through Fresnel, but roughness affects D and both G1s,
 * which are achromatic; the per-channel result differs only via the Fresnel
 * weight f_c). The diffuse term is roughness-independent, so only the specular
 * term contributes.
 *
 * alpha = max(roughness², 1e-3). Below the clamp boundary (roughness² < 1e-3,
 * i.e. roughness < ~0.0316) dalpha/droughness = 0; above it = 2·roughness.
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

  const alpha = Math.max(roughness * roughness, 1e-3);
  const alphaClamped = roughness * roughness < 1e-3;
  const dAlpha_dRough = alphaClamped ? 0.0 : 2.0 * roughness;

  // ── dD/dalpha ──
  //   D = a² / (PI·den²),  den = nDotH²·(a²-1)+1,  (ignoring the 1e-6 max floor)
  //   dDen/da² = nDotH²
  //   dD/da² = [PI·den² − a²·PI·2·den·nDotH²] / (PI·den²)²
  //          = [den − 2·a²·nDotH²] / (PI·den³)
  //   da²/droughness = 2·alpha·dAlpha_dRough
  const a2 = alpha * alpha;
  const den = nDotH * nDotH * (a2 - 1.0) + 1.0;
  const dD_da2 = (den - 2.0 * a2 * (nDotH * nDotH)) / Math.max(PI * den * den * den, 1e-12);
  const da2_dRough = 2.0 * alpha * dAlpha_dRough;
  const dD_dRough = dD_da2 * da2_dRough;
  const d = ggxD(nDotH, alpha);

  // ── dG1/droughness ── G1(x) = x / (x·(1-k) + k),  k = (roughness+1)²/8.
  //   dk/droughness = (roughness+1)/4.
  //   dG1/dk = -x·(1 - x) / (x·(1-k)+k)²
  const k = (roughness + 1.0) * (roughness + 1.0) * 0.125;
  const dk_dRough = (roughness + 1.0) * 0.25;
  const g1 = (x: number): number => x / Math.max(x * (1.0 - k) + k, 1e-6);
  const dG1_dRough = (x: number): number => {
    const denom = x * (1.0 - k) + k;
    if (denom <= 1e-6) return 0.0; // clamp region: derivative of the floor is 0
    return (-x * (1.0 - x) / (denom * denom)) * dk_dRough;
  };
  const g1V = g1(nDotV);
  const g1L = g1(nDotL);
  const g = g1V * g1L;
  const dG_dRough = dG1_dRough(nDotV) * g1L + g1V * dG1_dRough(nDotL);

  // spec_c = [(D·G)/(4·nDotV·nDotL)] · f_c.
  //   d(spec_c)/droughness = [(dD·G + D·dG)/(4·nDotV·nDotL)] · f_c.
  const invDenom = 1.0 / Math.max(4.0 * nDotV * nDotL, 1e-6);
  const dSpecScale = (dD_dRough * g + d * dG_dRough) * invDenom;
  return [dSpecScale * f[0], dSpecScale * f[1], dSpecScale * f[2]];
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
  const alpha = Math.max(roughness * roughness, 1e-3);
  const d = ggxD(nDotH, alpha);
  const g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  const specScale = (d * g) / Math.max(4.0 * nDotV * nDotL, 1e-6);
  const kd0 = 1.0 - metallic;
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
    out[c] = dDiff + dSpec;
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
  void specularColor;
  const dF0 = 0.04 * clamp01(specularIntensity) * (1.0 - metallic);
  return dBrdf_dSpecularF0(baseColor, roughness, metallic, normal, wo, wi, [dF0, dF0, dF0]);
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
): Vec3 {
  return dBrdf_dSpecularF0(baseColor, roughness, metallic, normal, wo, wi, [
    0.04 * clamp01(specularColor[0]) * (1.0 - metallic),
    0.04 * clamp01(specularColor[1]) * (1.0 - metallic),
    0.04 * clamp01(specularColor[2]) * (1.0 - metallic),
  ]);
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
 * map-free KHR_materials_clearcoat lobe. This mirrors `evalClearcoatLobe` with
 * frozen directions and no clearcoat normal map.
 */
export function dBrdf_dClearcoatRoughness(
  clearcoat: number,
  clearcoatRoughness: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
): Vec3 {
  if (clearcoat < 1e-4) return [0, 0, 0];
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const f = fresnelSchlick(vDotH, [0.04, 0.04, 0.04]);

  const alpha = Math.max(clearcoatRoughness * clearcoatRoughness, 1e-3);
  const alphaClamped = clearcoatRoughness * clearcoatRoughness < 1e-3;
  const dAlpha_dRough = alphaClamped ? 0.0 : 2.0 * clearcoatRoughness;

  const a2 = alpha * alpha;
  const den = nDotH * nDotH * (a2 - 1.0) + 1.0;
  const dD_da2 = (den - 2.0 * a2 * (nDotH * nDotH)) / Math.max(PI * den * den * den, 1e-12);
  const da2_dRough = 2.0 * alpha * dAlpha_dRough;
  const dD_dRough = dD_da2 * da2_dRough;
  const d = ggxD(nDotH, alpha);

  const k = (clearcoatRoughness + 1.0) * (clearcoatRoughness + 1.0) * 0.125;
  const dk_dRough = (clearcoatRoughness + 1.0) * 0.25;
  const g1 = (x: number): number => x / Math.max(x * (1.0 - k) + k, 1e-6);
  const dG1_dRough = (x: number): number => {
    const denom = x * (1.0 - k) + k;
    if (denom <= 1e-6) return 0.0;
    return (-x * (1.0 - x) / (denom * denom)) * dk_dRough;
  };
  const g1V = g1(nDotV);
  const g1L = g1(nDotL);
  const g = g1V * g1L;
  const dG_dRough = dG1_dRough(nDotV) * g1L + g1V * dG1_dRough(nDotL);

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
  if (sheen < 1e-4) return [0, 0, 0];
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const sinThetaH = Math.sqrt(Math.max(0.0, 1.0 - nDotH * nDotH));
  const alphaRaw = sheenRoughness * sheenRoughness;
  const dAlpha_dRough = alphaRaw < 1e-3 ? 0.0 : 2.0 * sheenRoughness;
  if (sinThetaH <= 1e-6 || dAlpha_dRough === 0.0) return [0, 0, 0];

  const alpha = Math.max(alphaRaw, 1e-3);
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
  return dBrdf_dSpecularF0(baseColor, roughness, metallic, normal, wo, wi, [
    iridF[0] - baseF0[0],
    iridF[1] - baseF0[1],
    iridF[2] - baseF0[2],
  ]);
}

function dBrdf_dSpecularF0(
  baseColor: Vec3,
  roughness: number,
  metallic: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
  dF0: Vec3,
): Vec3 {
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const alpha = Math.max(roughness * roughness, 1e-3);
  const d = ggxD(nDotH, alpha);
  const g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  const specScale = (d * g) / Math.max(4.0 * nDotV * nDotL, 1e-6);
  const kd0 = 1.0 - metallic;
  const m = Math.min(Math.max(1.0 - vDotH, 0.0), 1.0);
  const m2 = m * m;
  const m5 = m2 * m2 * m;
  return [
    dF0[0] * (1.0 - m5) * (specScale - kd0 * baseColor[0] * INV_PI),
    dF0[1] * (1.0 - m5) * (specScale - kd0 * baseColor[1] * INV_PI),
    dF0[2] * (1.0 - m5) * (specScale - kd0 * baseColor[2] * INV_PI),
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
