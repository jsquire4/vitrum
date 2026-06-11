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
 *  - the dielectric Fresnel reflectance `frDielectric` w.r.t. `ior` (scalar).
 *    NOTE: `ior` does NOT enter `evaluateBrdf` — the opaque-reflective F0 is a
 *    fixed `mix(0.04, baseColor, metallic)` (mirror of `bsdf.wgsl.ts:29` /
 *    `kernel.wgsl.ts:334`), so `∂evaluateBrdf/∂ior ≡ 0`. The only differentiable
 *    `ior` dependence in the forward kernel is `frDielectric` (the transmissive
 *    reflect/refract Fresnel partition). `dFrDielectric_dIor` provides that
 *    partial; see its doc for the (large) caveat on end-to-end consumption.
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
): Vec3 {
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const f0: Vec3 = [
    0.04 + (baseColor[0] - 0.04) * metallic,
    0.04 + (baseColor[1] - 0.04) * metallic,
    0.04 + (baseColor[2] - 0.04) * metallic,
  ];
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
    const f0c = 0.04 + (baseColor[c]! - 0.04) * metallic;
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
): Vec3 {
  const nDotL = Math.max(dot(normal, wi), 0.0);
  const nDotV = Math.max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = safeNormalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
  const nDotH = Math.max(dot(normal, h), 0.0);
  const vDotH = Math.max(dot(wo, h), 0.0);
  const f0: Vec3 = [
    0.04 + (baseColor[0] - 0.04) * metallic,
    0.04 + (baseColor[1] - 0.04) * metallic,
    0.04 + (baseColor[2] - 0.04) * metallic,
  ];
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
 * Analytic ∂(frDielectric)/∂ior — the scalar partial of the dielectric Fresnel
 * reflectance w.r.t. the index of refraction, evaluated at a frozen incident
 * cosine. This is the ONLY differentiable `ior` dependence in the current
 * forward kernel (the opaque-reflective F0 is a fixed 0.04 — see the file
 * header), so it is the honest `ior` partial.
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
