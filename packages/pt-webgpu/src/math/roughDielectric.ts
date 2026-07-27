/**
 * CPU double-precision oracle for the WebGPU Walter/Heitz rough dielectric.
 *
 * This module deliberately mirrors the production WGSL's solid-angle measures:
 * `etaTOverI` is eta_t / eta_i for the side containing `wo`; all directions
 * point away from the interface; and VNDF sampling returns a microfacet normal
 * on the `wo` side. It is used by numerical conservation and BDPT-measure gates.
 */

export type RoughDielectricVec3 = readonly [number, number, number];

export interface RoughDielectricConfig {
  readonly roughness: number;
  readonly etaTOverI: number;
  readonly anisotropy?: number;
  readonly anisotropyRotation?: number;
}

export interface RoughDielectricEventProbabilities {
  readonly reflection: number;
  readonly diffuse: number;
  readonly transmission: number;
}

export interface RoughDielectricSample {
  readonly wm: RoughDielectricVec3;
  readonly wi: RoughDielectricVec3 | null;
  readonly fresnel: number;
  readonly probabilities: RoughDielectricEventProbabilities;
}

/**
 * Shared finite-density alpha floor for every pt-webgpu microfacet eval, PDF,
 * sampler, and adjoint route. This is a numerical representation contract, not
 * an event classifier: roughness === 0 is delta, while every roughness > 0 is a
 * finite event evaluated with `max(roughness², floor)`.
 */
export const PT_WEBGPU_MICROFACET_ALPHA_FLOOR = 1e-3;

export function ptWebgpuMicrofacetAlpha(roughness: number): number {
  return Math.max(
    roughness * roughness,
    PT_WEBGPU_MICROFACET_ALPHA_FLOOR,
  );
}

/**
 * Emits the exact isotropic GGX Smith masking function used by production WGSL.
 * Keeping the expression here makes the CPU numerical oracle and every shader
 * consumer share the same roughness parameterization (`alpha = roughness^2`).
 */
export function roughDielectricSmithG1Wgsl(functionName: string): string {
  return /* wgsl */ `fn ${functionName}(nDotV: f32, roughness: f32) -> f32 {
  let cosTheta = max(nDotV, 0.0);
  if (cosTheta <= 1e-6) { return 0.0; }
  let alpha = max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  let tanTheta2 = max(0.0, 1.0 - cosTheta * cosTheta) /
    max(cosTheta * cosTheta, 1e-10);
  return 2.0 / (1.0 + sqrt(1.0 + alpha * alpha * tanTheta2));
}`;
}

export function roughDielectricSmithG1RoughnessDerivative(
  nDotV: number,
  roughness: number,
): number {
  const cosTheta = Math.max(nDotV, 0);
  if (cosTheta <= 1e-6) return 0;
  const alphaUnclamped = roughness * roughness;
  if (alphaUnclamped < PT_WEBGPU_MICROFACET_ALPHA_FLOOR) return 0;
  const alpha = Math.max(
    alphaUnclamped,
    PT_WEBGPU_MICROFACET_ALPHA_FLOOR,
  );
  const tanTheta2 = Math.max(0, 1 - cosTheta * cosTheta) /
    Math.max(cosTheta * cosTheta, 1e-20);
  const root = Math.sqrt(1 + alpha * alpha * tanTheta2);
  const dRootDRoughness = alpha * (2 * roughness) * tanTheta2 / root;
  return -2 * dRootDRoughness / ((1 + root) * (1 + root));
}

/** Emits the analytic roughness derivative of `roughDielectricSmithG1Wgsl`. */
export function roughDielectricSmithG1DerivativeWgsl(functionName: string): string {
  return /* wgsl */ `fn ${functionName}(nDotV: f32, roughness: f32) -> f32 {
  let cosTheta = max(nDotV, 0.0);
  if (cosTheta <= 1e-6) { return 0.0; }
  let alphaUnclamped = roughness * roughness;
  if (alphaUnclamped < ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR}) { return 0.0; }
  let alpha = max(alphaUnclamped, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  let tanTheta2 = max(0.0, 1.0 - cosTheta * cosTheta) /
    max(cosTheta * cosTheta, 1e-10);
  let root = sqrt(1.0 + alpha * alpha * tanTheta2);
  let dRootDRoughness =
    alpha * (2.0 * roughness) * tanTheta2 / root;
  return -2.0 * dRootDRoughness /
    ((1.0 + root) * (1.0 + root));
}`;
}

/** Exactly zero roughness is a Dirac interface; every positive value is finite. */
export const ROUGH_DIELECTRIC_SMOOTH_THRESHOLD = 0;

const PI = Math.PI;
const EPS = 1e-12;

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}

export function roughDielectricDot(a: RoughDielectricVec3, b: RoughDielectricVec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function roughDielectricScale(
  v: RoughDielectricVec3,
  scale: number,
): RoughDielectricVec3 {
  return [v[0] * scale, v[1] * scale, v[2] * scale];
}

export function roughDielectricAdd(
  a: RoughDielectricVec3,
  b: RoughDielectricVec3,
): RoughDielectricVec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function roughDielectricCross(
  a: RoughDielectricVec3,
  b: RoughDielectricVec3,
): RoughDielectricVec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function roughDielectricNormalize(v: RoughDielectricVec3): RoughDielectricVec3 {
  const length = Math.hypot(v[0], v[1], v[2]);
  if (!(length > EPS) || !Number.isFinite(length)) return [0, 0, 0];
  return [v[0] / length, v[1] / length, v[2] / length];
}

function roughDielectricBuildOnb(
  normal: RoughDielectricVec3,
): readonly [RoughDielectricVec3, RoughDielectricVec3] {
  const up: RoughDielectricVec3 = Math.abs(normal[1]) > 0.999 ? [1, 0, 0] : [0, 1, 0];
  const tangent = roughDielectricNormalize(roughDielectricCross(up, normal));
  return [tangent, roughDielectricCross(normal, tangent)];
}

export function roughDielectricFresnel(cosThetaIInput: number, etaInput: number): number {
  let cosThetaI = clamp(cosThetaIInput, -1, 1);
  let eta = Math.max(etaInput, 1e-8);
  if (cosThetaI < 0) {
    eta = 1 / eta;
    cosThetaI = -cosThetaI;
  }
  const sin2ThetaI = Math.max(0, 1 - cosThetaI * cosThetaI);
  const sin2ThetaT = sin2ThetaI / (eta * eta);
  if (sin2ThetaT >= 1) return 1;
  const cosThetaT = Math.sqrt(Math.max(0, 1 - sin2ThetaT));
  const rParallel = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
  const rPerpendicular = (cosThetaI - eta * cosThetaT) /
    (cosThetaI + eta * cosThetaT);
  return 0.5 * (rParallel * rParallel + rPerpendicular * rPerpendicular);
}

/**
 * CPU oracle for `materialDielectricFresnel` in the production WGSL.
 *
 * `authoredF0` is the already-composed KHR_materials_specular /
 * KHR_materials_iridescence F0 at this angle. The ratio keeps the historical
 * exact-IOR Fresnel curve unchanged for the no-op 4% F0 while allowing coloured
 * material controls to affect reflection and complementary transmission.
 */
export function roughDielectricMaterialFresnel(
  cosThetaInput: number,
  etaTOverI: number,
  authoredF0: RoughDielectricVec3,
): RoughDielectricVec3 {
  const cosTheta = Math.abs(cosThetaInput);
  const exact = roughDielectricFresnel(cosTheta, Math.max(etaTOverI, 1e-8));
  if (exact >= 1) return [1, 1, 1];
  const schlick = (f0: number) => {
    const m = clamp(1 - cosTheta, 0, 1);
    return f0 + (1 - f0) * m ** 5;
  };
  const baseline = Math.max(schlick(0.04), 1e-12);
  const channel = (f0: number) =>
    clamp(exact * schlick(clamp(f0, 0, 1)) / baseline, 0, 1);
  return [channel(authoredF0[0]), channel(authoredF0[1]), channel(authoredF0[2])];
}

export function roughDielectricMaterialEventProbabilities(
  transmission: number,
  macroFresnel: RoughDielectricVec3,
  microfacetFresnel: RoughDielectricVec3,
): RoughDielectricEventProbabilities {
  const luminance = (rgb: RoughDielectricVec3) =>
    clamp(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2], 0, 1);
  const t = clamp(transmission, 0, 1);
  const macroProbability = luminance(macroFresnel);
  const microfacetProbability = luminance(microfacetFresnel);
  const diffuse = (1 - macroProbability) * (1 - t);
  const dielectric = Math.max(1 - diffuse, 0);
  const norm = Math.max(
    microfacetProbability + t * (1 - microfacetProbability),
    1e-12,
  );
  return {
    reflection: dielectric * microfacetProbability / norm,
    diffuse,
    transmission: dielectric * t * (1 - microfacetProbability) / norm,
  };
}

export function roughDielectricGgxD(nDotM: number, alpha: number): number {
  const a2 = alpha * alpha;
  const n2 = Math.min(1, Math.max(0, nDotM * nDotM));
  // (1 - n²) + n²α² is algebraically identical to n²(α² - 1) + 1,
  // but it does not cancel to zero at the narrow-lobe peak.
  const d = (1 - n2) + n2 * a2;
  return a2 / (PI * d * d);
}

export function roughDielectricSmithG1(nDotV: number, roughness: number): number {
  const cosTheta = Math.max(nDotV, 0);
  if (cosTheta <= 1e-12) return 0;
  const alpha = ptWebgpuMicrofacetAlpha(roughness);
  const tanTheta2 = Math.max(0, 1 - cosTheta * cosTheta) /
    Math.max(cosTheta * cosTheta, 1e-20);
  return 2 / (1 + Math.sqrt(1 + alpha * alpha * tanTheta2));
}

function roughDielectricGgxDAnis(
  hT: number,
  hB: number,
  hN: number,
  ax: number,
  ay: number,
): number {
  const d = (hT / ax) ** 2 + (hB / ay) ** 2 + hN * hN;
  return 1 / Math.max(PI * ax * ay * d * d, 1e-12);
}

function roughDielectricSmithG1Anis(
  vT: number,
  vB: number,
  vN: number,
  ax: number,
  ay: number,
): number {
  const vN2 = Math.max(vN * vN, 1e-12);
  return (2 * vN) /
    Math.max(vN + Math.sqrt(vN2 + (vT * ax) ** 2 + (vB * ay) ** 2), 1e-12);
}

function roughDielectricAxes(roughness: number, anisotropy: number): readonly [number, number] {
  const alpha = ptWebgpuMicrofacetAlpha(roughness);
  const aspect = Math.sqrt(Math.max(1 - 0.9 * anisotropy, 1e-4));
  return [Math.max(alpha / aspect, 1e-4), Math.max(alpha * aspect, 1e-4)];
}

function roughDielectricFrame(
  normal: RoughDielectricVec3,
  rotation: number,
): readonly [RoughDielectricVec3, RoughDielectricVec3] {
  const [tangent, bitangent] = roughDielectricBuildOnb(normal);
  const c = Math.cos(rotation);
  const s = Math.sin(rotation);
  return [
    roughDielectricAdd(roughDielectricScale(tangent, c), roughDielectricScale(bitangent, s)),
    roughDielectricAdd(roughDielectricScale(tangent, -s), roughDielectricScale(bitangent, c)),
  ];
}

function orientRoughDielectricInterface(
  normal: RoughDielectricVec3,
  wo: RoughDielectricVec3,
  etaTOverI: number,
): { readonly normal: RoughDielectricVec3; readonly etaTOverI: number } {
  if (roughDielectricDot(normal, wo) >= 0) {
    return { normal, etaTOverI: Math.max(etaTOverI, 1e-8) };
  }
  return {
    normal: roughDielectricScale(normal, -1),
    etaTOverI: 1 / Math.max(etaTOverI, 1e-8),
  };
}

export function roughDielectricHalfVector(
  normal: RoughDielectricVec3,
  wo: RoughDielectricVec3,
  wi: RoughDielectricVec3,
  etaTOverI: number,
): RoughDielectricVec3 {
  const oriented = orientRoughDielectricInterface(normal, wo, etaTOverI);
  let wm = roughDielectricNormalize(
    roughDielectricAdd(wo, roughDielectricScale(wi, oriented.etaTOverI)),
  );
  if (roughDielectricDot(wm, oriented.normal) < 0) {
    wm = roughDielectricScale(wm, -1);
  }
  return wm;
}

export function roughDielectricVisibleNormalPdf(
  config: RoughDielectricConfig,
  normal: RoughDielectricVec3,
  wo: RoughDielectricVec3,
  wm: RoughDielectricVec3,
): number {
  const nDotO = roughDielectricDot(normal, wo);
  const oDotM = roughDielectricDot(wo, wm);
  if (nDotO <= 1e-8 || oDotM <= 1e-12 || roughDielectricDot(normal, wm) <= 0) return 0;
  const anisotropy = config.anisotropy ?? 0;
  if (anisotropy > 0) {
    const [tangent, bitangent] = roughDielectricFrame(normal, config.anisotropyRotation ?? 0);
    const [ax, ay] = roughDielectricAxes(config.roughness, anisotropy);
    const d = roughDielectricGgxDAnis(
      roughDielectricDot(wm, tangent),
      roughDielectricDot(wm, bitangent),
      Math.max(roughDielectricDot(wm, normal), 0),
      ax,
      ay,
    );
    const g1 = roughDielectricSmithG1Anis(
      roughDielectricDot(wo, tangent),
      roughDielectricDot(wo, bitangent),
      nDotO,
      ax,
      ay,
    );
    return d * g1 * oDotM / nDotO;
  }
  const alpha = ptWebgpuMicrofacetAlpha(config.roughness);
  return roughDielectricGgxD(Math.max(roughDielectricDot(normal, wm), 0), alpha) *
    roughDielectricSmithG1(nDotO, config.roughness) * oDotM / nDotO;
}

export function roughDielectricJacobian(
  normal: RoughDielectricVec3,
  wo: RoughDielectricVec3,
  wi: RoughDielectricVec3,
  etaTOverI: number,
): number {
  const oriented = orientRoughDielectricInterface(normal, wo, etaTOverI);
  const wm = roughDielectricHalfVector(normal, wo, wi, etaTOverI);
  const denominator = roughDielectricDot(wi, wm) +
    roughDielectricDot(wo, wm) / oriented.etaTOverI;
  return Math.abs(roughDielectricDot(wi, wm)) /
    Math.max(denominator * denominator, 1e-20);
}

export function roughDielectricTransmissionPdf(
  config: RoughDielectricConfig,
  normal: RoughDielectricVec3,
  wo: RoughDielectricVec3,
  wi: RoughDielectricVec3,
): number {
  if (config.roughness <= ROUGH_DIELECTRIC_SMOOTH_THRESHOLD) return 0;
  const oriented = orientRoughDielectricInterface(normal, wo, config.etaTOverI);
  const cosO = roughDielectricDot(oriented.normal, wo);
  const cosI = roughDielectricDot(oriented.normal, wi);
  if (cosO <= 1e-8 || cosI >= -1e-8) return 0;
  const wm = roughDielectricHalfVector(normal, wo, wi, config.etaTOverI);
  if (
    roughDielectricDot(wm, wi) * cosI < 0 ||
    roughDielectricDot(wm, wo) * cosO < 0
  ) return 0;
  return roughDielectricVisibleNormalPdf(config, oriented.normal, wo, wm) *
    roughDielectricJacobian(normal, wo, wi, config.etaTOverI);
}

export function roughDielectricTransmissionEval(
  config: RoughDielectricConfig,
  normal: RoughDielectricVec3,
  wo: RoughDielectricVec3,
  wi: RoughDielectricVec3,
  transportModeImportance: boolean,
): number {
  if (config.roughness <= ROUGH_DIELECTRIC_SMOOTH_THRESHOLD) return 0;
  const oriented = orientRoughDielectricInterface(normal, wo, config.etaTOverI);
  const cosO = roughDielectricDot(oriented.normal, wo);
  const cosI = roughDielectricDot(oriented.normal, wi);
  if (cosO <= 1e-8 || cosI >= -1e-8) return 0;
  const wm = roughDielectricHalfVector(normal, wo, wi, config.etaTOverI);
  if (
    roughDielectricDot(wm, wi) * cosI < 0 ||
    roughDielectricDot(wm, wo) * cosO < 0
  ) return 0;
  const anisotropy = config.anisotropy ?? 0;
  let d: number;
  let g: number;
  if (anisotropy > 0) {
    const [tangent, bitangent] = roughDielectricFrame(oriented.normal, config.anisotropyRotation ?? 0);
    const [ax, ay] = roughDielectricAxes(config.roughness, anisotropy);
    d = roughDielectricGgxDAnis(
      roughDielectricDot(wm, tangent), roughDielectricDot(wm, bitangent),
      Math.max(roughDielectricDot(wm, oriented.normal), 0), ax, ay,
    );
    g = roughDielectricSmithG1Anis(
      roughDielectricDot(wo, tangent), roughDielectricDot(wo, bitangent), cosO, ax, ay,
    ) * roughDielectricSmithG1Anis(
      roughDielectricDot(wi, tangent), roughDielectricDot(wi, bitangent), Math.abs(cosI), ax, ay,
    );
  } else {
    d = roughDielectricGgxD(
      Math.max(roughDielectricDot(oriented.normal, wm), 0),
      ptWebgpuMicrofacetAlpha(config.roughness),
    );
    g = roughDielectricSmithG1(cosO, config.roughness) *
      roughDielectricSmithG1(Math.abs(cosI), config.roughness);
  }
  const fresnel = roughDielectricFresnel(
    Math.abs(roughDielectricDot(wo, wm)), oriented.etaTOverI,
  );
  const denominator = roughDielectricDot(wi, wm) +
    roughDielectricDot(wo, wm) / oriented.etaTOverI;
  let value = d * (1 - fresnel) * g * Math.abs(
    roughDielectricDot(wi, wm) * roughDielectricDot(wo, wm) /
      Math.max(Math.abs(denominator * denominator * cosI * cosO), 1e-20),
  );
  if (!transportModeImportance) value /= oriented.etaTOverI ** 2;
  return value;
}

export function roughDielectricReflectionEval(
  config: RoughDielectricConfig,
  normal: RoughDielectricVec3,
  wo: RoughDielectricVec3,
  wi: RoughDielectricVec3,
): number {
  if (config.roughness <= ROUGH_DIELECTRIC_SMOOTH_THRESHOLD) return 0;
  const oriented = orientRoughDielectricInterface(normal, wo, config.etaTOverI);
  const cosO = roughDielectricDot(oriented.normal, wo);
  const cosI = roughDielectricDot(oriented.normal, wi);
  if (cosO <= 1e-8 || cosI <= 1e-8) return 0;
  let wm = roughDielectricNormalize(roughDielectricAdd(wo, wi));
  if (roughDielectricDot(wm, oriented.normal) < 0) {
    wm = roughDielectricScale(wm, -1);
  }
  const anisotropy = config.anisotropy ?? 0;
  let d: number;
  let g: number;
  if (anisotropy > 0) {
    const [tangent, bitangent] = roughDielectricFrame(
      oriented.normal,
      config.anisotropyRotation ?? 0,
    );
    const [ax, ay] = roughDielectricAxes(config.roughness, anisotropy);
    d = roughDielectricGgxDAnis(
      roughDielectricDot(wm, tangent),
      roughDielectricDot(wm, bitangent),
      Math.max(roughDielectricDot(wm, oriented.normal), 0),
      ax,
      ay,
    );
    g = roughDielectricSmithG1Anis(
      roughDielectricDot(wo, tangent), roughDielectricDot(wo, bitangent), cosO, ax, ay,
    ) * roughDielectricSmithG1Anis(
      roughDielectricDot(wi, tangent), roughDielectricDot(wi, bitangent), cosI, ax, ay,
    );
  } else {
    d = roughDielectricGgxD(
      Math.max(roughDielectricDot(oriented.normal, wm), 0),
      ptWebgpuMicrofacetAlpha(config.roughness),
    );
    g = roughDielectricSmithG1(cosO, config.roughness) *
      roughDielectricSmithG1(cosI, config.roughness);
  }
  const fresnel = roughDielectricFresnel(
    Math.abs(roughDielectricDot(wo, wm)),
    oriented.etaTOverI,
  );
  return d * g * fresnel / Math.max(4 * cosO * cosI, 1e-20);
}

export function roughDielectricEventProbabilities(
  config: RoughDielectricConfig,
  normal: RoughDielectricVec3,
  wo: RoughDielectricVec3,
  wm: RoughDielectricVec3,
  transmission: number,
): RoughDielectricEventProbabilities {
  const oriented = orientRoughDielectricInterface(normal, wo, config.etaTOverI);
  const t = clamp(transmission, 0, 1);
  const macroF = roughDielectricFresnel(
    Math.abs(roughDielectricDot(oriented.normal, wo)), oriented.etaTOverI,
  );
  const diffuse = (1 - macroF) * (1 - t);
  const dielectric = Math.max(1 - diffuse, 0);
  const microfacetF = roughDielectricFresnel(
    Math.abs(roughDielectricDot(wo, wm)), oriented.etaTOverI,
  );
  const norm = Math.max(microfacetF + t * (1 - microfacetF), 1e-12);
  return {
    reflection: dielectric * microfacetF / norm,
    diffuse,
    transmission: dielectric * t * (1 - microfacetF) / norm,
  };
}

export function roughDielectricRefract(
  incoming: RoughDielectricVec3,
  wm: RoughDielectricVec3,
  etaIOverT: number,
): RoughDielectricVec3 | null {
  const dotNI = roughDielectricDot(wm, incoming);
  const k = 1 - etaIOverT * etaIOverT * (1 - dotNI * dotNI);
  if (k < 0) return null;
  return roughDielectricNormalize(roughDielectricAdd(
    roughDielectricScale(incoming, etaIOverT),
    roughDielectricScale(wm, -(etaIOverT * dotNI + Math.sqrt(k))),
  ));
}

export function sampleRoughDielectricVisibleNormal(
  config: RoughDielectricConfig,
  normal: RoughDielectricVec3,
  wo: RoughDielectricVec3,
  u1: number,
  u2: number,
): RoughDielectricVec3 {
  const anisotropy = config.anisotropy ?? 0;
  const [tangent, bitangent] = roughDielectricFrame(
    normal,
    config.anisotropyRotation ?? 0,
  );
  const alpha = ptWebgpuMicrofacetAlpha(config.roughness);
  const [ax, ay] = anisotropy > 0
    ? roughDielectricAxes(config.roughness, anisotropy)
    : [alpha, alpha];
  const woLocal: RoughDielectricVec3 = [
    roughDielectricDot(wo, tangent),
    roughDielectricDot(wo, bitangent),
    roughDielectricDot(wo, normal),
  ];
  const vh = roughDielectricNormalize([ax * woLocal[0], ay * woLocal[1], woLocal[2]]);
  const lensq = vh[0] * vh[0] + vh[1] * vh[1];
  const tangent1: RoughDielectricVec3 = lensq > 1e-10
    ? [-vh[1] / Math.sqrt(lensq), vh[0] / Math.sqrt(lensq), 0]
    : [1, 0, 0];
  const tangent2 = roughDielectricCross(vh, tangent1);
  const radius = Math.sqrt(clamp(u1, 0, 1));
  const phi = 2 * PI * clamp(u2, 0, 1);
  const diskX = radius * Math.cos(phi);
  let diskY = radius * Math.sin(phi);
  const projection = 0.5 * (1 + vh[2]);
  diskY = (1 - projection) * Math.sqrt(Math.max(0, 1 - diskX * diskX)) +
    projection * diskY;
  const nh = roughDielectricAdd(
    roughDielectricAdd(
      roughDielectricScale(tangent1, diskX),
      roughDielectricScale(tangent2, diskY),
    ),
    roughDielectricScale(
      vh,
      Math.sqrt(Math.max(0, 1 - diskX * diskX - diskY * diskY)),
    ),
  );
  const wmLocal = roughDielectricNormalize([
    ax * nh[0],
    ay * nh[1],
    Math.max(1e-6, nh[2]),
  ]);
  return roughDielectricNormalize(roughDielectricAdd(
    roughDielectricAdd(
      roughDielectricScale(tangent, wmLocal[0]),
      roughDielectricScale(bitangent, wmLocal[1]),
    ),
    roughDielectricScale(normal, wmLocal[2]),
  ));
}

export function sampleRoughDielectric(
  config: RoughDielectricConfig,
  normal: RoughDielectricVec3,
  wo: RoughDielectricVec3,
  transmission: number,
  u1: number,
  u2: number,
): RoughDielectricSample {
  const oriented = orientRoughDielectricInterface(normal, wo, config.etaTOverI);
  const wm = sampleRoughDielectricVisibleNormal(config, oriented.normal, wo, u1, u2);
  const wi = roughDielectricRefract(
    roughDielectricScale(wo, -1),
    wm,
    1 / oriented.etaTOverI,
  );
  return {
    wm,
    wi,
    fresnel: roughDielectricFresnel(
      Math.abs(roughDielectricDot(wo, wm)),
      oriented.etaTOverI,
    ),
    probabilities: roughDielectricEventProbabilities(
      config, normal, wo, wm, transmission,
    ),
  };
}
