/**
 * PTWG-BDPT-01 — independent CPU oracle for the BDPT eye↔light connection
 * radiometry (plan/road-to-100-gap-ledger-2026-06-11.md §PTWG-BDPT-01).
 *
 * WHAT THIS ORACLE IS
 * -------------------
 * It does NOT mirror the shader's own assembly as ground truth (that was the
 * documented failure mode of the previous CPU oracle — a shared radiometric
 * bias passes a mirror test). Instead it compares
 *
 *   (a) a line-by-line TS transcription of the WGSL connection contribution
 *       (every term cites the WGSL file:line it mirrors), against
 *   (b) a ground-truth direct-illumination estimator derived FRESH from the
 *       rendering equation (area-measure NEE over a rect emitter), with the
 *       SAME BSDF evaluator shared between the two sides so that the comparison
 *       isolates the CONNECTION assembly, not the BRDF model.
 *
 * SCOPE: the s=1 strategy (light subpath = emitter vertex L_0 only) connected
 * to an eye surface vertex E_e. The MIS weight `misW` is deliberately factored
 * OUT of the comparison: in Veach §10.2 each (s,t) strategy's UNWEIGHTED
 * contribution C*_{s,t} = β_L · f_l · G · f_e · β_E must by itself be an
 * unbiased estimator of the transported radiance for paths of that length
 * (the MIS weights then sum to 1 across strategies). A wrong C* cannot be
 * repaired by any valid MIS weighting, so auditing C* pre-MIS is sufficient
 * and is the only audit that is independent of the (separately tested)
 * bdptMISWeightFull recurrence.
 *
 * GROUND TRUTH (derived from first principles, no shader code):
 *   Direct radiance leaving the eye vertex toward the camera due to a diffuse
 *   rect emitter with radiance Le and area A:
 *     L = ∫_A Le · f_e(wo, wi(x)) · cosE(x) · cosL(x) / d(x)² dA(x)
 *   MC estimator with x ~ Uniform(A) (pdf 1/A):
 *     C_correct(x) = A · Le · f_e · cosE · cosL / d²
 *   (β_E = 1: the eye prefix throughput is shared verbatim by both sides —
 *   `eyeThroughput` multiplies the shader contribution at
 *   bdptConnection.wgsl.ts:407, so it cancels in the ratio.)
 *
 * SHADER TRANSCRIPTION (s=1, single rect light, no occlusion):
 *   Light vertex throughput (bdptLightSubpath.wgsl.ts:141-159 bdptFinishBounce0):
 *     pdfHemi    = cosEmit/π                       (cosineHemisphereSample)
 *     pdfJoint   = max(discretePdf · pdfHemi, 1e-8)  [L152]
 *     β_L0       = Le · cosEmit / pdfJoint = π · Le / discretePdf  [L153]
 *     NOTE: the emitter POSITION pdf (1/A for the uniform rect sample drawn at
 *     bdptWriteBounce0, bdptLightSubpath.wgsl.ts:266) is NEVER divided out —
 *     pdfLight passed to bdptFinishBounce0 is the discrete pick pmf only
 *     (bdptLightSubpath.wgsl.ts:269: `bdptFinishBounce0(col, emitPos,
 *     emitNormal, rr, discretePdf, rng)`). Veach's β_L0 = Le / (pdfA·pdfPick)
 *     = Le·A/pdfPick. The shader instead carries π·Le/pdfPick.
 *   Connection (bdptConnection.wgsl.ts evaluateBdptConnection):
 *     gTerm            = cosX·cosY/d²                       [L99-109]
 *     eyeBsdfCosTheta  = evaluateBrdf(...) · cosEye          [L318-323]
 *     lightBsdfCosTheta= cosLight/π   (emitter, lvMatId < 0) [L324, L335]
 *     contribution     = β_L0 · lightBsdfCosTheta · gTerm ·
 *                        eyeBsdfCosTheta · misW · eyeThroughput  [L406-407]
 *
 * PREDICTED BIAS LAW (algebra, then verified numerically below):
 *   C_shader(x) / C_correct(x)
 *     = [πLe · (cosL/π) · (cosE·cosL/d²) · f_e·cosE] / [A·Le·f_e·cosE·cosL/d²]
 *     = cosE · cosL / A
 *   i.e. ONE extra receiver cosine, ONE extra emitter cosine (cos² per
 *   connection-edge endpoint: once inside gTerm, once in each side's
 *   bsdfCosTheta term), and a missing emitter-area factor A (the uniform
 *   position pdf 1/A is never divided out of β_L0).
 *
 * VERDICT ENCODING: the confirmed bias is PINNED as a characterization test
 * (so any change to the connection math flips it loudly), and the correct
 * value lives in a sibling `it.skip` that the eventual fix should un-skip.
 */
import { describe, expect, it } from 'vitest';

// ───────────────────────────── vec3 helpers (plain math, no deps) ────────────
type V3 = [number, number, number];
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

// Deterministic RNG (the oracle audits expectations, not the WGSL PCG stream).
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

// ─────────────── pt-webgpu BSDF transcription (shared by BOTH sides) ─────────
// The BRDF model itself is NOT under audit here; sharing it between the shader
// transcription and the ground-truth integrand isolates the connection assembly.
const PI = Math.PI;
const INV_PI = 1 / PI;

// material.wgsl.ts:741-745
function ggxD(nDotH: number, alpha: number): number {
  const a2 = alpha * alpha;
  const d = nDotH * nDotH * (a2 - 1) + 1;
  return a2 / Math.max(PI * d * d, 1e-6);
}
// material.wgsl.ts:747-751
function smithG1(nDotV: number, roughness: number): number {
  const r = roughness + 1;
  const k = r * r * 0.125;
  return nDotV / Math.max(nDotV * (1 - k) + k, 1e-6);
}
// shared-samplers bsdfPrimitives.wgsl.ts:29-32 (pt-webgpu local copy clamps c)
function fresnelSchlick(cosTheta: number, F0: V3): V3 {
  const c = Math.min(Math.max(1 - cosTheta, 0), 1);
  const c5 = c * c * c * c * c;
  return [F0[0] + (1 - F0[0]) * c5, F0[1] + (1 - F0[1]) * c5, F0[2] + (1 - F0[2]) * c5];
}
// material.wgsl.ts:786-812 — Kulla-Conty E LUT (verbatim) + lookups
const GGX_E_LUT_DIM = 8;
// material.wgsl.ts:789-798
const GGX_E_LUT = [
  0.1375, 0.5617, 0.7546, 0.8522, 0.9111, 0.9505, 0.9788, 1.0,
  0.2955, 0.515, 0.7091, 0.8192, 0.889, 0.937, 0.9721, 0.9988,
  0.5794, 0.5541, 0.6677, 0.7691, 0.8451, 0.9021, 0.9457, 0.98,
  0.7011, 0.6486, 0.6669, 0.7199, 0.7776, 0.8305, 0.8764, 0.9155,
  0.7335, 0.6901, 0.6696, 0.6756, 0.6972, 0.7262, 0.7578, 0.7893,
  0.7153, 0.6712, 0.6355, 0.6145, 0.6052, 0.6045, 0.6101, 0.6199,
  0.6669, 0.6137, 0.5657, 0.5286, 0.5, 0.478, 0.4611, 0.4483,
  0.6017, 0.537, 0.4773, 0.4296, 0.3905, 0.358, 0.3305, 0.3069,
];
// material.wgsl.ts:801-803
const GGX_EAVG_LUT = [0.9106, 0.8931, 0.8629, 0.8094, 0.725, 0.6147, 0.4931, 0.3766];
// material.wgsl.ts:806-826
function ggxDirectionalAlbedo(cosTheta: number, roughness: number): number {
  const mu = Math.min(Math.max(cosTheta, 0), 1);
  const r = Math.min(Math.max(roughness, 0), 1);
  const fr = r * (GGX_E_LUT_DIM - 1);
  const fm = mu * (GGX_E_LUT_DIM - 1);
  const r0 = Math.floor(fr);
  const m0 = Math.floor(fm);
  const r1 = Math.min(r0 + 1, GGX_E_LUT_DIM - 1);
  const m1 = Math.min(m0 + 1, GGX_E_LUT_DIM - 1);
  const tr = fr - r0;
  const tm = fm - m0;
  const e0 = GGX_E_LUT[r0 * GGX_E_LUT_DIM + m0]! * (1 - tm) + GGX_E_LUT[r0 * GGX_E_LUT_DIM + m1]! * tm;
  const e1 = GGX_E_LUT[r1 * GGX_E_LUT_DIM + m0]! * (1 - tm) + GGX_E_LUT[r1 * GGX_E_LUT_DIM + m1]! * tm;
  return Math.min(Math.max(e0 * (1 - tr) + e1 * tr, 0.02), 1);
}
// material.wgsl.ts:829-835
function ggxAverageAlbedo(roughness: number): number {
  const r = Math.min(Math.max(roughness, 0), 1);
  const fr = r * (GGX_E_LUT_DIM - 1);
  const r0 = Math.floor(fr);
  const r1 = Math.min(r0 + 1, GGX_E_LUT_DIM - 1);
  const tr = fr - r0;
  return Math.min(Math.max(GGX_EAVG_LUT[r0]! * (1 - tr) + GGX_EAVG_LUT[r1]! * tr, 0.3), 1);
}
// material.wgsl.ts:838-853
function ggxMultiscatterLobe(f0: V3, roughness: number, nDotV: number, nDotL: number): V3 {
  const eAvg = ggxAverageAlbedo(roughness);
  const oneMinusEavg = 1 - eAvg;
  if (oneMinusEavg < 1e-4) return [0, 0, 0];
  const eo = ggxDirectionalAlbedo(nDotV, roughness);
  const ei = ggxDirectionalAlbedo(nDotL, roughness);
  const fAvg: V3 = [f0[0] + (1 - f0[0]) / 21, f0[1] + (1 - f0[1]) / 21, f0[2] + (1 - f0[2]) / 21];
  const fMs: V3 = [
    (fAvg[0] * fAvg[0] * eAvg) / Math.max(1 - fAvg[0] * oneMinusEavg, 1e-4),
    (fAvg[1] * fAvg[1] * eAvg) / Math.max(1 - fAvg[1] * oneMinusEavg, 1e-4),
    (fAvg[2] * fAvg[2] * eAvg) / Math.max(1 - fAvg[2] * oneMinusEavg, 1e-4),
  ];
  const shape = ((1 - eo) * (1 - ei)) / Math.max(PI * oneMinusEavg, 1e-6);
  return scale(fMs, shape);
}
// bsdf.wgsl.ts:634-654 — evaluateBrdf (Cook-Torrance + Lambert + multiscatter)
function evaluateBrdf(baseColor: V3, roughness: number, metallic: number, normal: V3, wo: V3, wi: V3): V3 {
  const nDotL = Math.max(dot(normal, wi), 0);
  const nDotV = Math.max(dot(normal, wo), 0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = norm(add(wi, wo));
  const nDotH = Math.max(dot(normal, h), 0);
  const vDotH = Math.max(dot(wo, h), 0);
  const f0: V3 = [
    0.04 * (1 - metallic) + baseColor[0] * metallic,
    0.04 * (1 - metallic) + baseColor[1] * metallic,
    0.04 * (1 - metallic) + baseColor[2] * metallic,
  ];
  const f = fresnelSchlick(vDotH, f0);
  const alpha = Math.max(roughness * roughness, 1e-3);
  const d = ggxD(nDotH, alpha);
  const g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  const specS = (d * g) / Math.max(4 * nDotV * nDotL, 1e-6);
  const spec: V3 = scale(f, specS);
  const kd: V3 = [(1 - f[0]) * (1 - metallic), (1 - f[1]) * (1 - metallic), (1 - f[2]) * (1 - metallic)];
  const diff: V3 = scale(mulv(kd, baseColor), INV_PI);
  const ms = ggxMultiscatterLobe(f0, roughness, nDotV, nDotL);
  return add(add(diff, spec), ms);
}

// ───────────────────────────── test scene ────────────────────────────────────
// Eye vertex (the connectable surface vertex E_e). β_E (eyeThroughput) = 1 on
// both sides — it multiplies both formulas identically (bdptConnection:407).
const eyePos: V3 = [0, 0, 0];
const eyeNormal: V3 = [0, 1, 0];
const eyeWo: V3 = norm([0.2, 1, 0.1]); // toward the previous eye vertex/camera
const baseColor: V3 = [0.6, 0.5, 0.4];
const roughness = 0.7;
const metallic = 0;

// Single rect area light (so the power-weighted discrete pick pmf == 1 and the
// discretePdf factor drops out on both sides):
//   center (0,1.5,0), u=(1,0,0), v=(0,0,1)
//   emitNormal = normalize(cross(u,v)) = (0,-1,0)   (bdptLightSubpath:268)
//   area = 4·|u×v| = 4                              (bdptLightSubpath:81-85)
//   sampled point = center + u·(ξ1·2−1) + v·(ξ2·2−1) (bdptLightSubpath:266)
const rectCenter: V3 = [0, 1.5, 0];
const rectU: V3 = [1, 0, 0];
const rectV: V3 = [0, 0, 1];
const rectNormal: V3 = norm(cross(rectU, rectV)); // (0,-1,0)
const rectArea = 4 * Math.sqrt(dot(cross(rectU, rectV), cross(rectU, rectV))); // = 4
const Le: V3 = [5, 4, 3];

function samplePointOnRect(x1: number, x2: number): V3 {
  return add(rectCenter, add(scale(rectU, x1 * 2 - 1), scale(rectV, x2 * 2 - 1)));
}

// ─────────── shader transcription: β_L0 (bdptFinishBounce0) ─────────────────
// bdptLightSubpath.wgsl.ts:149-153 with discretePdf = 1 (single light):
//   hemi.pdf = cosEmit/π; pdfJoint = max(1·cosEmit/π, 1e-8);
//   emitThroughput = Le·cosEmit/pdfJoint
// The hemisphere draw cosEmit cancels exactly → β_L0 = π·Le, independent of
// the random emission direction. Verified numerically in the first test.
function shaderLightVertexThroughput(cosEmit: number): V3 {
  const pdfHemi = cosEmit / PI; // cosineHemisphereSample pdf
  const pdfJoint = Math.max(1 * pdfHemi, 1e-8); // L152, discretePdf = 1
  return scale(Le, cosEmit / pdfJoint); // L153
}

// ─────────── shader transcription: connection contribution (pre-MIS) ─────────
// evaluateBdptConnection, bdptConnection.wgsl.ts — per-line transcription for
// the s=1 (emitter light vertex, lvMatId < 0) unoccluded case, misW factored
// out (see file header for why that is the right thing to audit).
function shaderConnectionContribution(lightPos: V3): V3 {
  const toLight = sub(lightPos, eyePos); // L304
  const dist = Math.sqrt(dot(toLight, toLight)); // L305
  if (dist < 1e-4) return [0, 0, 0]; // L306
  const connDir = scale(toLight, 1 / dist); // L309
  // bdptGeometricTerm, L99-109: cosX·cosY/d² (BOTH cosines inside G)
  const dist2 = dist * dist;
  const cosX = Math.abs(dot(eyeNormal, connDir));
  const cosY = Math.abs(dot(rectNormal, scale(connDir, -1)));
  const gTerm = (cosX * cosY) / dist2; // L108
  if (gTerm <= 0) return [0, 0, 0]; // L311
  // (shadow ray L314-317 omitted: scene has no occluder)
  const eyeBrdf = evaluateBrdf(baseColor, roughness, metallic, eyeNormal, eyeWo, connDir); // L318
  const cosEye = Math.max(dot(eyeNormal, connDir), 0); // L319
  if (cosEye <= 0) return [0, 0, 0]; // L320
  const eyeBsdfCosTheta = scale(eyeBrdf, cosEye); // L323  ← cosEye AGAIN (also in gTerm)
  const cosLight = Math.max(dot(rectNormal, scale(connDir, -1)), 0); // L324
  // Emitter vertex (lvMatId < 0): Lambertian emission profile, L335
  const lightBsdfCosTheta: V3 = [cosLight / PI, cosLight / PI, cosLight / PI]; // ← cosLight AGAIN
  const beta_L0 = shaderLightVertexThroughput(0.42 /* arbitrary hemi draw; cancels */);
  // L406: contribution = lightThroughput·lightBsdfCosTheta·gTerm·eyeBsdfCosTheta·misW
  // L407: contribution *= eyeThroughput (=1 here)
  // misW EXCLUDED (audits the unweighted strategy contribution C*; see header).
  return mulv(mulv(beta_L0, lightBsdfCosTheta), scale(eyeBsdfCosTheta, gTerm));
}

// ─────────── ground truth: fresh area-measure NEE estimator ──────────────────
// C_correct(x) = A · Le · f_e(wo, wi) · cosE · cosL / d²   (x ~ Uniform(A))
// Derived directly from L = ∫_A Le·f_e·cosE·cosL/d² dA — no shader code.
function correctConnectionContribution(lightPos: V3): V3 {
  const toLight = sub(lightPos, eyePos);
  const dist2 = dot(toLight, toLight);
  const dist = Math.sqrt(dist2);
  const wi = scale(toLight, 1 / dist);
  const cosE = Math.max(dot(eyeNormal, wi), 0);
  const cosL = Math.max(dot(rectNormal, scale(wi, -1)), 0);
  if (cosE <= 0 || cosL <= 0) return [0, 0, 0];
  const f = evaluateBrdf(baseColor, roughness, metallic, eyeNormal, eyeWo, wi);
  return scale(mulv(Le, f), (rectArea * cosE * cosL) / dist2);
}

describe('PTWG-BDPT-01 oracle — BDPT connection cosine/area audit', () => {
  it('transcription sanity: β_L0 = π·Le for every hemisphere draw (bdptFinishBounce0:149-153)', () => {
    for (const cosEmit of [0.05, 0.3, 0.7, 0.99]) {
      const beta = shaderLightVertexThroughput(cosEmit);
      expect(beta[0]).toBeCloseTo(PI * Le[0], 9);
      expect(beta[1]).toBeCloseTo(PI * Le[1], 9);
      expect(beta[2]).toBeCloseTo(PI * Le[2], 9);
    }
    // Veach's β_L0 would be Le·A/discretePdf = Le·4 here; the shader carries
    // Le·π. The missing-area half of the bias law below comes from exactly this.
  });

  it('per-sample bias law: C_shader(x) = C_correct(x) · cosE·cosL / A (exact, no MC noise)', () => {
    // This pins the bias ANALYTICALLY: the transcribed shader formula equals the
    // first-principles estimator times one extra cosine per connection-edge
    // endpoint (cos² per edge total) divided by the emitter area.
    const rng = mulberry32(1234);
    for (let i = 0; i < 64; i++) {
      const x = samplePointOnRect(rng(), rng());
      const cs = shaderConnectionContribution(x);
      const cc = correctConnectionContribution(x);
      const toLight = sub(x, eyePos);
      const d = Math.sqrt(dot(toLight, toLight));
      const wi = scale(toLight, 1 / d);
      const cosE = Math.max(dot(eyeNormal, wi), 0);
      const cosL = Math.max(dot(rectNormal, scale(wi, -1)), 0);
      const predicted = scale(cc, (cosE * cosL) / rectArea);
      for (const ch of [0, 1, 2] as const) {
        expect(Math.abs(cs[ch] - predicted[ch])).toBeLessThan(1e-9 * Math.max(1, Math.abs(predicted[ch])));
      }
    }
  });

  it('CONFIRMED BIAS (characterization pin): E[C_shader] / L ≈ 0.20 for this geometry, not 1', () => {
    // 2·10⁶-sample MC of both estimators over the same uniform-rect draws.
    // Ground truth L is the fresh rendering-equation estimator; the shader side
    // is the faithful transcription. If a fix lands in bdptConnection.wgsl.ts /
    // bdptLightSubpath.wgsl.ts the transcription MUST be updated and this pin
    // flipped with the it.skip sibling below.
    const N = 2_000_000;
    const rng = mulberry32(987654321);
    let sumShaderLum = 0;
    let sumCorrectLum = 0;
    let sumBiasFactor = 0;
    const lum = (c: V3) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    for (let i = 0; i < N; i++) {
      const x = samplePointOnRect(rng(), rng());
      sumShaderLum += lum(shaderConnectionContribution(x));
      sumCorrectLum += lum(correctConnectionContribution(x));
      const toLight = sub(x, eyePos);
      const d2 = dot(toLight, toLight);
      const d = Math.sqrt(d2);
      const wi = scale(toLight, 1 / d);
      sumBiasFactor +=
        (Math.max(dot(eyeNormal, wi), 0) * Math.max(dot(rectNormal, scale(wi, -1)), 0)) / rectArea;
    }
    const ratio = sumShaderLum / sumCorrectLum;
    const predictedBias = sumBiasFactor / N; // E[cosE·cosL]/A under the correct measure-weighting? No —
    // NOTE: predictedBias is E_uniform[cosE·cosL/A], an unweighted average; the
    // realized ratio is the C_correct-WEIGHTED average of the same factor, so the
    // two differ slightly. Both are reported; the assertion uses the realized ratio.
    const msg =
      `BDPT s=1 connection: measured E[C_shader]/L = ${ratio.toFixed(4)} ` +
      `(uniform-avg bias factor E[cosE·cosL]/A = ${predictedBias.toFixed(4)}). ` +
      `The shader applies cos² per connection edge (gTerm carries both cosines AND ` +
      `each side's bsdfCosTheta re-multiplies its cosine) and never divides the ` +
      `emitter-position pdf 1/A out of β_L0 — bdptConnection.wgsl.ts:108,323,335 + ` +
      `bdptLightSubpath.wgsl.ts:153,269.`;
    // The bias is geometry-dependent (cosE·cosL/A field over the rect); for THIS
    // geometry (2×2 rect, 1.5 above a horizontal receiver) the realized ratio is
    // ≈ 0.205 — a ~5× energy deficit. Pin with a band wide enough for MC noise
    // (deterministic seed → tight in practice).
    // eslint-disable-next-line no-console
    console.log(`[oracle.bdptConnectionCosine] ${msg}`);
    expect(ratio, msg).toBeGreaterThan(0.1);
    expect(ratio, msg).toBeLessThan(0.25);
    // And it is decisively NOT unbiased:
    expect(Math.abs(ratio - 1), msg).toBeGreaterThan(0.5);
  });

  // The fix should make the unweighted s=1 strategy contribution an unbiased
  // estimator of the direct-illumination integral: un-skip when the connection
  // assembly is corrected (single G with both cosines, plain BSDF values f_l/f_e
  // without re-multiplied cosines, β_L0 = Le/(pdfPick·pdfArea)).
  it.skip('CORRECT VALUE (un-skip with the fix): E[C_shader] / L = 1 ± MC noise', () => {
    const N = 2_000_000;
    const rng = mulberry32(987654321);
    let sumShaderLum = 0;
    let sumCorrectLum = 0;
    const lum = (c: V3) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    for (let i = 0; i < N; i++) {
      const x = samplePointOnRect(rng(), rng());
      sumShaderLum += lum(shaderConnectionContribution(x));
      sumCorrectLum += lum(correctConnectionContribution(x));
    }
    expect(sumShaderLum / sumCorrectLum).toBeCloseTo(1, 2);
  });
});
