/**
 * HYB-GI-01 / HYB-GI-02 — independent CPU oracle for the walkaround-hybrid
 * ReSTIR-DI estimator (RIS seeding → W finalize → shade consumption).
 *
 * CURRENT PRODUCTION PATH UNDER AUDIT:
 *   1. RIS candidate generation evaluates the selected sample point:
 *      - finite emitters store the triangle sample `xi` that generated p̂,
 *      - BRDF-to-emitter candidates invert hit barycentrics back to that same
 *        `xi`,
 *      - HDRI candidates store ENV_SAMPLE_SENTINEL plus an encoded direction.
 *   2. Finite-emitter and HDRI source weights are scaled by
 *      `Mtotal / nDomain`; attempted `areaM` / `envM` (including null draws)
 *      are persisted, and W finalization divides the resulting tagged-union
 *      `w_sum` by `Mtotal * p̂(selected)`.
 *   3. Shade consumption uses the stored `r.xi` for both finite emitters and
 *      env sentinels, so candidate p̂, finalization p̂, visibility, and the
 *      shaded contribution all refer to the same sample.
 *
 * The tests below keep historical pre-fix variants as characterization controls:
 * centroid p̂ + fresh shade xi under-estimated large close emitters, and a
 * single mixed-measure M under-weighted HDRI samples by the finite-emitter
 * candidate count. The production/regression variants must remain ≈ unbiased.
 *
 * GROUND TRUTH (fresh, first-principles):
 *   I_lights = Σ_tri ∫_A Le·evalGGX(wo,wi)·cosθ_L/d² dA
 *   I_env    = ∫_Ω L_env(ω)·evalGGX(wo,ω) dω
 * The GGX evaluator is shared between transcription and ground truth so the
 * comparison isolates the RIS/W/shade estimator, not the BRDF model.
 */
import { describe, expect, it } from 'vitest';

type V3 = [number, number, number];
type V2 = [number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const mulv = (a: V3, b: V3): V3 => [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const norm = (a: V3): V3 => {
  const l = Math.sqrt(dot(a, a));
  return [a[0] / l, a[1] / l, a[2] / l];
};

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const PI = Math.PI;
const INV_PI = 1 / PI;

// ── walkaround GGX transcription (shared by both sides) ──────────────────────
// shared-samplers luminance.wgsl.ts:28-30
const luminance = (c: V3) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
// shared-samplers bsdfPrimitives.wgsl.ts:29-32 (walkaround composes the
// UNCLAMPED reference form)
function fresnelSchlick(cosTheta: number, F0: V3): V3 {
  const c = 1 - cosTheta;
  const c5 = c * c * c * c * c;
  return [F0[0] + (1 - F0[0]) * c5, F0[1] + (1 - F0[1]) * c5, F0[2] + (1 - F0[2]) * c5];
}
// ggxBrdf.wgsl.ts:28-33
function distributionGGX(NdotH: number, rough: number): number {
  const a = rough * rough;
  const a2 = a * a;
  const d = NdotH * NdotH * (a2 - 1) + 1;
  return a2 / (PI * d * d);
}
// ggxBrdf.wgsl.ts:36-40
function geometrySchlickGGX(NdotV: number, rough: number): number {
  const r = rough + 1;
  const k = (r * r) / 8;
  return NdotV / (NdotV * (1 - k) + k);
}
// ggxBrdf.wgsl.ts:42-44
const geometrySmith = (NdotV: number, NdotL: number, rough: number) =>
  geometrySchlickGGX(NdotV, rough) * geometrySchlickGGX(NdotL, rough);
// ggxBrdf.wgsl.ts:47-65 — evalGGX (returns BRDF·NdotL)
function evalGGX(albedo: V3, rough: number, metal: number, n: V3, wo: V3, wi: V3): V3 {
  const h = norm(add(wo, wi));
  const NdotL = Math.max(0, dot(n, wi));
  const NdotV = Math.max(1e-4, dot(n, wo));
  const NdotH = Math.max(0, dot(n, h));
  const VdotH = Math.max(0, dot(wo, h));
  if (NdotL < 1e-6 || NdotV < 1e-6) return [0, 0, 0];
  const F0: V3 = [
    0.04 * (1 - metal) + albedo[0] * metal,
    0.04 * (1 - metal) + albedo[1] * metal,
    0.04 * (1 - metal) + albedo[2] * metal,
  ];
  const F = fresnelSchlick(VdotH, F0);
  const D = distributionGGX(NdotH, Math.max(0.01, rough));
  const G = geometrySmith(NdotV, NdotL, Math.max(0.01, rough));
  const spec = scale(F, (D * G) / (4 * NdotV * NdotL));
  const diff: V3 = [
    (1 - F[0]) * (1 - metal) * albedo[0] * INV_PI,
    (1 - F[1]) * (1 - metal) * albedo[1] * INV_PI,
    (1 - F[2]) * (1 - metal) * albedo[2] * INV_PI,
  ];
  return scale(add(diff, spec), NdotL);
}
// walkaroundUbo.wgsl.ts:205-208
const emitterGeometry = (nlDotL: number, dist2: number, dist2Floor: number) =>
  nlDotL / Math.max(dist2, dist2Floor);

// ── emitter model (EmitterTri, reservoirDi.wgsl.ts:21-32) ─────────────────────
interface EmitterTri {
  vA: V3;
  vB: V3;
  vC: V3;
  normal: V3;
  area: number;
  Le: V3;
}
// emitterSampling.wgsl.ts:25-40
function sampleEmitterPoint(e: EmitterTri, xi: V2): { pos: V3; pdfArea: number } {
  const s = Math.sqrt(xi[0]);
  const u = 1 - s;
  const v = s * xi[1];
  const w = s * (1 - xi[1]);
  return {
    pos: add(add(scale(e.vA, u), scale(e.vB, v)), scale(e.vC, w)),
    pdfArea: 1 / e.area,
  };
}
// reservoirDi.wgsl.ts:55-71 — env xi codec
const ENV_SENTINEL = 0xffffffff;
function envDirToXi(d: V3): V2 {
  const phi = Math.atan2(d[2], d[0]);
  const theta = Math.acos(Math.min(Math.max(d[1], -1), 1));
  const fract = (x: number) => x - Math.floor(x);
  return [theta * INV_PI, fract((phi * INV_PI) / 2 + 0.5)];
}
function envDirFromXi(xi: V2): V3 {
  const theta = xi[0] * PI;
  const phi = (xi[1] - 0.5) * 2 * PI;
  const st = Math.sin(theta);
  return norm([Math.cos(phi) * st, Math.cos(theta), Math.sin(phi) * st]);
}

// reservoirDi.wgsl.ts:73-92 — ReservoirDI + WRS update
interface ReservoirDI {
  lightId: number;
  M: number;
  w_sum: number;
  W: number;
  xi: V2;
  areaM: number;
  envM: number;
}
function updateReservoirDI(r: ReservoirDI, lid: number, xi: V2, w: number, rng: () => number): void {
  r.w_sum += w;
  if (rng() * r.w_sum < w) {
    r.lightId = lid;
    r.xi = xi;
  }
}
function updateReservoirDIHistorical(
  r: ReservoirDI,
  lid: number,
  xi: V2,
  w: number,
  rng: () => number,
): void {
  r.M += 1;
  updateReservoirDI(r, lid, xi, w, rng);
}

// ── scene / receiver ──────────────────────────────────────────────────────────
const pos: V3 = [0, 0, 0];
const normal: V3 = [0, 1, 0];
const wo: V3 = norm([0.3, 1, 0.2]);
const albedo: V3 = [0.65, 0.65, 0.65];
const roughness = 0.85;
const metalness = 0;
const dist2Floor = 1e-6; // ubo.emitterDist2Floor — tiny so it never binds and
// the ground truth (no floor) is comparable.

// Large CLOSE emitter quad (split into 2 tris, y = 1.2, normal −Y) so the
// centroid-vs-sampled-point p̂ mismatch is significant.
function makeTri(vA: V3, vB: V3, vC: V3, Le: V3): EmitterTri {
  const n = norm(cross(sub(vB, vA), sub(vC, vA)));
  const area = 0.5 * Math.sqrt(dot(cross(sub(vB, vA), sub(vC, vA)), cross(sub(vB, vA), sub(vC, vA))));
  return { vA, vB, vC, normal: n, area, Le };
}
const TRI1 = makeTri([-1, 1.2, -1], [1, 1.2, -1], [1, 1.2, 1], [4, 3, 2]); // normal (0,-1,0)
const TRI2 = makeTri([-1, 1.2, -1], [1, 1.2, 1], [-1, 1.2, 1], [1, 2, 5]); // normal (0,-1,0)

// Constant env model: walkaround's envRadiance returns the HDRI texel; the
// oracle uses a constant L_env (the 1/M-underweight law is content-independent)
// and models envImportanceSample as a uniform-sphere draw with its exact pdf
// 1/(4π) — a valid importance distribution; the transcription divides by the
// same pdf it samples from, exactly as the shader does with its CDF pdf.
const L_ENV: V3 = [0.8, 0.9, 1.0];
const envRadiance = (_dir: V3): V3 => L_ENV;
function envImportanceSample(rng: () => number): { dir: V3; pdf: number; color: V3 } {
  const u1 = rng();
  const u2 = rng();
  const z = 1 - 2 * u1;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  const phi = 2 * PI * u2;
  const dir: V3 = [r * Math.cos(phi), z, r * Math.sin(phi)];
  return { dir, pdf: 1 / (4 * PI), color: L_ENV };
}

const M_LIGHT = 64; // ris.wgsl.ts:82
const M_ENV = 1; // ris.wgsl.ts:93

interface SceneCfg {
  emitters: EmitterTri[];
  envHasMap: boolean;
  envMapIntensity?: number;
}

const envMapIntensityForCfg = (cfg: SceneCfg): number => Math.max(0, cfg.envMapIntensity ?? 1);

// ── restir_di_compute_phat_xi transcription (restirPHat.wgsl.ts:46-73) ───────
function computePhatCentroid(cfg: SceneCfg, lid: number): number {
  const e = cfg.emitters[lid]!;
  const centroid = scale(add(add(e.vA, e.vB), e.vC), 1 / 3);
  const toL = sub(centroid, pos);
  const dist2 = dot(toL, toL);
  if (dist2 < 1e-8) return 0;
  const wi = scale(toL, 1 / Math.sqrt(dist2));
  const nDotL = Math.max(0, dot(normal, wi));
  const nlDotL = Math.max(0, dot(scale(e.normal, -1), wi));
  if (nDotL < 1e-6 || nlDotL < 1e-6) return 0;
  const G = emitterGeometry(nlDotL, dist2, dist2Floor);
  const brdf = evalGGX(albedo, roughness, metalness, normal, wo, wi);
  return luminance(scale(mulv(e.Le, brdf), G));
}

function computePhatXi(cfg: SceneCfg, lid: number, xi: V2): number {
  if (lid === ENV_SENTINEL) {
    if (!cfg.envHasMap) return 0; // L50
    const wi = envDirFromXi(xi); // L51
    const nDotL = Math.max(0, dot(normal, wi)); // L52
    if (nDotL < 1e-6) return 0;
    const color = scale(envRadiance(wi), envMapIntensityForCfg(cfg)); // L54
    const brdf = evalGGX(albedo, roughness, metalness, normal, wo, wi); // L55
    return luminance(mulv(color, brdf)); // L56
  }
  const e = cfg.emitters[lid]!;
  const ls = sampleEmitterPoint(e, xi);
  const toL = sub(ls.pos, pos);
  const dist2 = dot(toL, toL);
  if (dist2 < 1e-8) return 0;
  const wi = scale(toL, 1 / Math.sqrt(dist2));
  const nDotL = Math.max(0, dot(normal, wi));
  const nlDotL = Math.max(0, dot(scale(e.normal, -1), wi));
  if (nDotL < 1e-6 || nlDotL < 1e-6) return 0;
  const G = emitterGeometry(nlDotL, dist2, dist2Floor); // L70
  const brdf = evalGGX(albedo, roughness, metalness, normal, wo, wi); // L71
  return luminance(scale(mulv(e.Le, brdf), G)); // L72
}

// ── RIS pass transcription (ris.wgsl.ts risMain, flat-CDF mode) ──────────────
type Variant = 'shader' | 'historical';

function runRis(cfg: SceneCfg, rng: () => number, variant: Variant = 'shader'): ReservoirDI {
  const r: ReservoirDI = {
    lightId: 0,
    M: 0,
    w_sum: 0,
    W: 0,
    xi: [0, 0],
    areaM: 0,
    envM: 0,
  };
  let areaSupportM = 0;
  let envSupportM = 0;
  const scheduledAreaM = M_LIGHT;
  const scheduledEnvM = cfg.envHasMap ? M_ENV : 0;
  const scheduledTotalM = scheduledAreaM + scheduledEnvM;
  const areaRisScale = scheduledTotalM / Math.max(1, scheduledAreaM);
  const envRisScale = scheduledTotalM / Math.max(1, scheduledEnvM);
  const totalPower = Math.max(
    cfg.emitters.reduce((s, e) => s + luminance(e.Le) * e.area, 0),
    1e-8,
  ); // L188 (host totalEmPower = Σ lum·area)
  // M_LIGHT loop [L208-263], flat power CDF branch [L228-233]
  for (let i = 0; i < M_LIGHT; i++) {
    areaSupportM += 1;
    const xiEm = rng();
    // sampleEmitterIdx over the power CDF [emitterSampling.wgsl.ts:42-58]
    let cum = 0;
    let lid = cfg.emitters.length - 1;
    for (let k = 0; k < cfg.emitters.length; k++) {
      cum += (luminance(cfg.emitters[k]!.Le) * cfg.emitters[k]!.area) / totalPower;
      if (xiEm < cum) {
        lid = k;
        break;
      }
    }
    const emitterSelPmf = (luminance(cfg.emitters[lid]!.Le) * cfg.emitters[lid]!.area) / totalPower; // L232
    const e = cfg.emitters[lid]!;
    const xiTri: V2 = [rng(), rng()]; // L235
    const ls = sampleEmitterPoint(e, xiTri); // L236
    const toL = sub(ls.pos, pos);
    const dist2 = dot(toL, toL);
    if (dist2 < 1e-8) continue; // zero-weight attempt remains in support M
    const wi = scale(toL, 1 / Math.sqrt(dist2));
    const nDotL = Math.max(0, dot(normal, wi));
    const nlDotL = Math.max(0, dot(scale(e.normal, -1), wi));
    if (nDotL < 1e-6 || nlDotL < 1e-6) continue;
    const G = emitterGeometry(nlDotL, dist2, dist2Floor); // L251
    const brdf = evalGGX(albedo, roughness, metalness, normal, wo, wi); // L252
    const pHat = luminance(scale(mulv(e.Le, brdf), G)); // L253 — SAMPLED-POINT p̂
    const pX = Math.max(1e-15, emitterSelPmf * ls.pdfArea); // L260
    const w = pHat > 0 ? pHat / pX : 0; // L261
    if (variant === 'historical') {
      updateReservoirDIHistorical(r, lid, xiTri, w, rng);
    } else {
      updateReservoirDI(r, lid, xiTri, w * areaRisScale, rng);
    }
  }
  // M_ENV loop [L375-389]
  for (let ei = 0; ei < M_ENV; ei++) {
    const envS = envImportanceSample(rng); // L376
    if (!cfg.envHasMap) continue;
    envSupportM += 1;
    if (envS.pdf <= 1e-8) continue;
    const nDotL = Math.max(0, dot(normal, envS.dir));
    if (nDotL < 1e-6) continue; // L379
    const brdfE = evalGGX(albedo, roughness, metalness, normal, wo, envS.dir); // L380
    const pHatE = luminance(scale(mulv(envS.color, brdfE), envMapIntensityForCfg(cfg))); // L382 — SA measure, no G
    const pXe = Math.max(1e-15, envS.pdf); // L384
    const wE = pHatE > 0 ? pHatE / pXe : 0; // L385
    if (variant === 'historical') {
      updateReservoirDIHistorical(r, ENV_SENTINEL, envDirToXi(envS.dir), wE, rng);
    } else {
      updateReservoirDI(
        r,
        ENV_SENTINEL,
        envDirToXi(envS.dir),
        wE * envRisScale,
        rng,
      );
    }
  }
  if (variant === 'shader') {
    r.areaM = areaSupportM;
    r.envM = envSupportM;
    r.M = areaSupportM + envSupportM;
  }
  // W finalize [L392-461] — unoccluded scene, so the shadow tests pass.
  if (r.M > 0 && r.w_sum > 0) {
    const pHatZ =
      variant === 'historical' && r.lightId !== ENV_SENTINEL
        ? computePhatCentroid(cfg, r.lightId)
        : computePhatXi(cfg, r.lightId, r.xi);
    r.W = pHatZ > 0 ? r.w_sum / (r.M * pHatZ) : 0;
  }
  return r;
}

// ── shade consumption transcription (shadingTerms.wgsl.ts lo_direct) ─────────
function loDirect(cfg: SceneCfg, r: ReservoirDI, rng: () => number, variant: Variant = 'shader'): V3 {
  if (r.W <= 0 || r.M === 0) return [0, 0, 0]; // L238
  if (r.lightId === ENV_SENTINEL) {
    if (!cfg.envHasMap) return [0, 0, 0]; // L248
    const envDir = envDirFromXi(r.xi); // L249
    const nDotL = Math.max(0, dot(normal, envDir));
    if (nDotL < 1e-6) return [0, 0, 0];
    const envColor = scale(envRadiance(envDir), envMapIntensityForCfg(cfg)); // L252
    const brdfE = evalGGX(albedo, roughness, metalness, normal, wo, envDir); // L253
    return scale(mulv(envColor, brdfE), r.W); // L254
  }
  const e = cfg.emitters[r.lightId]!;
  const lsXi: V2 = variant === 'historical' ? [rng(), rng()] : r.xi;
  const ls = sampleEmitterPoint(e, lsXi); // L266
  const toL = sub(ls.pos, pos);
  const dist = Math.sqrt(dot(toL, toL));
  if (dist <= 1e-4) return [0, 0, 0];
  const wi = scale(toL, 1 / dist);
  const nDotL = Math.max(0, dot(normal, wi));
  const nlDotL = Math.max(0, dot(scale(e.normal, -1), wi));
  if (nDotL <= 1e-6 || nlDotL <= 1e-6) return [0, 0, 0]; // L273
  // shadow ray L278-284: unoccluded
  const G = emitterGeometry(nlDotL, dist * dist, dist2Floor); // L285
  const brdf = evalGGX(albedo, roughness, metalness, normal, wo, wi); // L286
  return scale(scale(mulv(e.Le, brdf), G), r.W); // L287
}

function measurePipeline(cfg: SceneCfg, trials: number, seed: number, variant: Variant = 'shader'): V3 {
  const rng = mulberry32(seed);
  const acc: V3 = [0, 0, 0];
  for (let t = 0; t < trials; t++) {
    const r = runRis(cfg, rng, variant);
    const v = loDirect(cfg, r, rng, variant);
    acc[0] += v[0];
    acc[1] += v[1];
    acc[2] += v[2];
  }
  return scale(acc, 1 / trials);
}

// ── ground truths (fresh first-principles MC; see header) ─────────────────────
function groundTruthLights(emitters: EmitterTri[], nPerTri: number, seed: number): V3 {
  const rng = mulberry32(seed);
  const acc: V3 = [0, 0, 0];
  for (const e of emitters) {
    const triAcc: V3 = [0, 0, 0];
    for (let i = 0; i < nPerTri; i++) {
      // uniform on the triangle, written fresh (NOT via sampleEmitterPoint):
      let u = rng();
      let v = rng();
      if (u + v > 1) {
        u = 1 - u;
        v = 1 - v;
      }
      const p = add(e.vA, add(scale(sub(e.vB, e.vA), u), scale(sub(e.vC, e.vA), v)));
      const toL = sub(p, pos);
      const d2 = dot(toL, toL);
      const wi = scale(toL, 1 / Math.sqrt(d2));
      const cosL = Math.max(0, dot(scale(e.normal, -1), wi));
      if (cosL <= 0) continue;
      const f = evalGGX(albedo, roughness, metalness, normal, wo, wi); // contains cosE
      const integrand = scale(mulv(e.Le, f), cosL / d2);
      triAcc[0] += integrand[0];
      triAcc[1] += integrand[1];
      triAcc[2] += integrand[2];
    }
    acc[0] += (triAcc[0] / nPerTri) * e.area;
    acc[1] += (triAcc[1] / nPerTri) * e.area;
    acc[2] += (triAcc[2] / nPerTri) * e.area;
  }
  return acc;
}
function groundTruthEnv(nSamples: number, seed: number, envMapIntensity = 1): V3 {
  const rng = mulberry32(seed);
  const acc: V3 = [0, 0, 0];
  for (let i = 0; i < nSamples; i++) {
    const u1 = rng();
    const u2 = rng();
    const z = 1 - 2 * u1;
    const rr = Math.sqrt(Math.max(0, 1 - z * z));
    const phi = 2 * PI * u2;
    const wi: V3 = [rr * Math.cos(phi), z, rr * Math.sin(phi)];
    const f = evalGGX(albedo, roughness, metalness, normal, wo, wi); // 0 below horizon
    const integrand = scale(mulv(L_ENV, f), Math.max(0, envMapIntensity));
    acc[0] += integrand[0] * 4 * PI;
    acc[1] += integrand[1] * 4 * PI;
    acc[2] += integrand[2] * 4 * PI;
  }
  return scale(acc, 1 / nSamples);
}

const lum3 = (c: V3) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

describe('HYB-GI-01/02 oracle — walkaround ReSTIR-DI estimator vs brute force', () => {
  it('transcription sanity: BRDF-candidate barycentric→xi inversion round-trips sampleEmitterPoint', () => {
    // ris.wgsl.ts:319-328 inverts emitterSampling.wgsl.ts:25-31. If this drifts,
    // the reservoir's (lid, xi) representation silently stops matching the
    // visibility/shade reconstruction.
    const rng = mulberry32(7);
    for (let i = 0; i < 200; i++) {
      const xi: V2 = [rng(), rng()];
      const ls = sampleEmitterPoint(TRI1, xi);
      // recover barycentrics of ls.pos wrt TRI1, then invert
      const s = Math.sqrt(xi[0]);
      const bary: V3 = [1 - s, s * xi[1], s * (1 - xi[1])]; // (A,B,C) weights
      const sInv = 1 - bary[0];
      const xiX = sInv * sInv;
      const xiY = bary[1] / Math.max(bary[1] + bary[2], 1e-8);
      expect(Math.abs(xiX - xi[0])).toBeLessThan(1e-9);
      expect(Math.abs(xiY - xi[1])).toBeLessThan(1e-9);
      const ls2 = sampleEmitterPoint(TRI1, [xiX, xiY]);
      expect(Math.abs(ls2.pos[0] - ls.pos[0])).toBeLessThan(1e-9);
    }
  });

  it('HYB-GI-01 mechanism: finalize p̂ (centroid) ≠ candidate p̂ (sampled point) for a large close emitter', () => {
    // Direct demonstration that the two p̂s the estimator mixes (w_sum built
    // from sampled-point p̂, W divided by centroid p̂) disagree by large factors
    // across the triangle — the inconsistency HYB-GI-01 flags.
    const phatCentroid = computePhatCentroid({ emitters: [TRI1, TRI2], envHasMap: false }, 0);
    const rng = mulberry32(11);
    let minR = Infinity;
    let maxR = 0;
    for (let i = 0; i < 2000; i++) {
      const xi: V2 = [rng(), rng()];
      const ls = sampleEmitterPoint(TRI1, xi);
      const toL = sub(ls.pos, pos);
      const d2 = dot(toL, toL);
      const wi = scale(toL, 1 / Math.sqrt(d2));
      const nDotL = Math.max(0, dot(normal, wi));
      const nlDotL = Math.max(0, dot(scale(TRI1.normal, -1), wi));
      if (nDotL < 1e-6 || nlDotL < 1e-6) continue;
      const G = emitterGeometry(nlDotL, d2, dist2Floor);
      const brdf = evalGGX(albedo, roughness, metalness, normal, wo, wi);
      const pHatSample = luminance(scale(mulv(TRI1.Le, brdf), G));
      const ratio = pHatSample / phatCentroid;
      minR = Math.min(minR, ratio);
      maxR = Math.max(maxR, ratio);
    }
    // For this geometry the sampled-point p̂ swings far around the centroid p̂.
    expect(maxR / minR, `p̂(sample)/p̂(centroid) spans [${minR.toFixed(3)}, ${maxR.toFixed(3)}]`).toBeGreaterThan(3);
  });

  it('historical HYB-GI-01 characterization: centroid p̂ + fresh shade xi under-estimates ~30%', () => {
    const cfg: SceneCfg = { emitters: [TRI1, TRI2], envHasMap: false };
    const measured = measurePipeline(cfg, 24_000, 314159, 'historical');
    const truth = groundTruthLights(cfg.emitters, 400_000, 271828);
    const ratio = lum3(measured) / lum3(truth);
    const msg =
      `lights-only E[lo_direct]/I = ${ratio.toFixed(4)} (≈30% under-estimate). w_sum is ` +
      `built from SAMPLED-POINT p̂ (ris.wgsl.ts:253) but W divides by CENTROID p̂ ` +
      `(historical restirPHat path), and shade re-samples a FRESH point ` +
      `(shadingTerms.wgsl.ts:265). The expectation collapses to ≈ I·mean_A(p̂)/p̂(centroid), ` +
      `< 1 for any large emitter close to the receiver (the centroid sits near the p̂ peak). ` +
      `The current selected-xi regression sibling restores ≈1.00 with the same candidate stream.`;
    expect(ratio, msg).toBeGreaterThan(0.62);
    expect(ratio, msg).toBeLessThan(0.78);
    expect(Math.abs(ratio - 1), msg).toBeGreaterThan(0.2); // decisively biased
  });

  it('REGRESSION HYB-GI-01: selected-xi finalize/shade path is unbiased (≈1.000)', () => {
    // Same transcribed candidate generation, with every scheduled proposal
    // counted even when a `continue` turns it into a zero-weight/null draw.
    // W is finalized with p̂ at the SELECTED r.xi and shade consumes r.xi.
    // This restores E ≈ I because:
    //   (a) pins the lights-only bias on the centroid/fresh-xi consumption
    //       seam (HYB-GI-01), not on candidate generation; and
    //   (b) the tagged generalized-RIS scaling averages each support family
    //       over its scheduled attempts and then sums the two domain estimates.
    //       Null proposals therefore contribute zero weight and a real count;
    //       conditioning M on successful candidates would be biased.
    const cfg: SceneCfg = { emitters: [TRI1, TRI2], envHasMap: false };
    const measured = measurePipeline(cfg, 24_000, 314159);
    const truth = groundTruthLights(cfg.emitters, 400_000, 271828);
    const ratio = lum3(measured) / lum3(truth);
    expect(ratio).toBeGreaterThan(0.97);
    expect(ratio).toBeLessThan(1.03);
  });

  it('REGRESSION envMapIntensity: scales only the HDRI p-hat branch', () => {
    const cfg: SceneCfg = { emitters: [TRI1, TRI2], envHasMap: true, envMapIntensity: 0.25 };
    const xiEnv: V2 = [0.5, 0.5];
    const envScaled = computePhatXi(cfg, ENV_SENTINEL, xiEnv);
    const envDefault = computePhatXi({ ...cfg, envMapIntensity: 1 }, ENV_SENTINEL, xiEnv);
    expect(envScaled).toBeCloseTo(envDefault * 0.25, 10);

    const xiLight: V2 = [0.25, 0.75];
    const lightScaled = computePhatXi(cfg, 0, xiLight);
    const lightDefault = computePhatXi({ ...cfg, envMapIntensity: 1 }, 0, xiLight);
    expect(lightScaled).toBeCloseTo(lightDefault, 10);
  });

  it('historical HYB-GI-02 characterization: env-only direct light is under-estimated ≈ M× (≈66×)', () => {
    // Scene: env map + the same two emitters DARK (Le=0). The 64 emitter
    // candidates and the BRDF candidate still enter the pool (w=0 but
    // M incremented — reservoirDi.wgsl.ts:86 runs for every updateReservoirDI
    // call), so the single env candidate's disjoint-support integral is
    // averaged down by the full M ≈ 66 in W = w_sum/(M·p̂)  [ris.wgsl.ts:420].
    const darkTri1: EmitterTri = { ...TRI1, Le: [0, 0, 0] };
    const darkTri2: EmitterTri = { ...TRI2, Le: [0, 0, 0] };
    const cfg: SceneCfg = { emitters: [darkTri1, darkTri2], envHasMap: true };
    const measured = measurePipeline(cfg, 60_000, 161803, 'historical');
    const truth = groundTruthEnv(2_000_000, 141421);
    const ratio = lum3(measured) / lum3(truth);
    const msg =
      `env-only E[lo_direct]/I_env = ${ratio.toFixed(5)} ≈ 1/${(1 / ratio).toFixed(1)} ` +
      `(expected ≈ 1/66 = ${(1 / 66).toFixed(5)}). The env strategy has disjoint support ` +
      `from the 65 area-measure candidates, but the 1/M factor in W (ris.wgsl.ts:420) ` +
      `weights it as if all 66 candidates could have produced it — HDRI direct light is ` +
      `crushed ~66× whenever emitters exist in the scene pool.`;
    expect(ratio, msg).toBeGreaterThan(1 / 90);
    expect(ratio, msg).toBeLessThan(1 / 45);
  });

  it('historical mixed lights+env: both defects compose — measured matches the two-bias model', () => {
    const cfg: SceneCfg = { emitters: [TRI1, TRI2], envHasMap: true };
    const measured = measurePipeline(cfg, 24_000, 999331, 'historical');
    const truthL = groundTruthLights(cfg.emitters, 400_000, 271828);
    const truthE = groundTruthEnv(2_000_000, 141421);
    const truthTotal = add(truthL, truthE);
    const ratio = lum3(measured) / lum3(truthTotal);
    // Two-bias composition model:
    //   lights share carries the centroid-p̂ factor (measured in the
    //   lights-only pin) and a 65/66 dilution from the env candidate's M slot;
    //   env share is crushed to 1/66 by the disjoint-support balance.
    const lightsOnly = measurePipeline({ emitters: [TRI1, TRI2], envHasMap: false }, 24_000, 314159, 'historical');
    const centroidFactor = lum3(lightsOnly) / lum3(truthL);
    const predicted =
      ((65 / 66) * centroidFactor * lum3(truthL) + (1 / 66) * lum3(truthE)) / lum3(truthTotal);
    const msg =
      `mixed E[lo_direct]/I_total = ${ratio.toFixed(4)} (two-bias model prediction ` +
      `${predicted.toFixed(4)}; unbiased would be 1.0). I_env/I_total = ` +
      `${(lum3(truthE) / lum3(truthTotal)).toFixed(3)} of the true energy is reduced to ` +
      `~1/66 of itself, and the emitter share is further scaled by the centroid-p̂ ` +
      `factor ${centroidFactor.toFixed(3)}.`;
    expect(Math.abs(ratio - predicted), msg).toBeLessThan(0.05);
    expect(ratio, msg).toBeLessThan(0.6); // decisively short of unbiased (1.0)
  });

  it('REGRESSION HYB-GI-02: env-only and mixed scenes match brute force', () => {
    const darkTri1: EmitterTri = { ...TRI1, Le: [0, 0, 0] };
    const darkTri2: EmitterTri = { ...TRI2, Le: [0, 0, 0] };
    const envOnly = measurePipeline({ emitters: [darkTri1, darkTri2], envHasMap: true }, 60_000, 161803);
    const truthE = groundTruthEnv(2_000_000, 141421);
    expect(lum3(envOnly) / lum3(truthE)).toBeGreaterThan(0.95);
    expect(lum3(envOnly) / lum3(truthE)).toBeLessThan(1.05);

    const mixedCfg: SceneCfg = { emitters: [TRI1, TRI2], envHasMap: true };
    const mixed = measurePipeline(mixedCfg, 24_000, 999331);
    const truthL = groundTruthLights(mixedCfg.emitters, 400_000, 271828);
    const truthTotal = add(truthL, truthE);
    expect(lum3(mixed) / lum3(truthTotal)).toBeGreaterThan(0.95);
    expect(lum3(mixed) / lum3(truthTotal)).toBeLessThan(1.05);
  });
});
