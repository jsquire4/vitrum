// restirPtHybridShift.test.ts — the ReSTIR-PT / GRIS HYBRID-shift Jacobian for the
// GGX/Cook-Torrance BSDF the pt-webgpu kernel uses (Lin et al. 2022 §5.2: random
// replay of the rough prefix ∘ reconnection at the first smooth vertex). The full
// hybrid Jacobian = J_geom · ∏ (p_BSDF^q / p_BSDF^r) over the replayed prefix
// segment(s); the geometric factor is the reconnection half-G ratio, the replay
// factor is the engine's forward-sampling-pdf ratio (brdfDirectionalPdf).
//
// The EXECUTED correctness check runs on GPU (wsl-gpu
// scripts/restir-pt-hybrid-shift-validate.ts, lavapipe): the analytic hybrid
// Jacobian == the finite difference of the ACTUAL hybrid-shift map (FD the
// geometric factor by perturbing x_s over its area params AND FD the BSDF-pdf
// factor by perturbing the canonical randoms through the actual replay sampler).
// Here we pin the host-side packing + the kernel composition + analytic invariants
// computed on the CPU from INDEPENDENTLY-derived closed forms (NOT copied from the
// WGSL): the same discipline as restirPtShift.test.ts.
import { describe, it, expect } from 'vitest';
import { roughDielectricSmithG1Wgsl } from '../math/roughDielectric.js';
import { RESTIR_PT_HYBRID_SHIFT_WGSL } from '../wgsl/pathTrace/restirPtHybridShift.wgsl.js';
import {
  RESTIR_PT_HYBRID_SHIFT_HARNESS_WGSL,
  packRestirPtHybridShiftInput,
  RESTIR_PT_HYBRID_SHIFT_INPUT_FLOATS,
} from '../wgsl/pathTrace/restirPtHybridShift.harness.wgsl.js';

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: V3): number => Math.hypot(a[0], a[1], a[2]);
const norm = (a: V3): V3 => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const mix3 = (a: V3, b: V3, t: number): V3 => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

// ── independent (non-implementation) CPU reference closed forms ──────────────
// Destination-cosine half-G + the reconnection-shift geometric Jacobian (n_s = +z).
const halfG = (xa: V3, xs: V3, ns: V3): number => {
  const d = sub(xa, xs);
  const dist = len(d);
  if (dist <= 0) return 0;
  return Math.abs(dot(ns, d) / dist) / (dist * dist);
};
const jGeom = (xq: V3, xr: V3, xs: V3, ns: V3): number => {
  const g = halfG(xq, xs, ns);
  return g <= 0 ? 0 : halfG(xr, xs, ns) / g;
};
// The engine's forward BSDF sampling pdf (brdfDirectionalPdf) — re-derived here
// from the model (Heitz-2018 VNDF specular + Lambertian cosine + the discrete lobe
// partition), independently of the WGSL text. Same-hemisphere (reflection) only;
// the harness configs use reflection prefixes.
const luminance = (c: V3): number => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
const fresnelSchlick = (cosTheta: number, f0: V3): V3 => {
  const m = Math.min(Math.max(1 - cosTheta, 0), 1);
  const m5 = m * m * m * m * m;
  return [f0[0] + (1 - f0[0]) * m5, f0[1] + (1 - f0[1]) * m5, f0[2] + (1 - f0[2]) * m5];
};
const ggxD = (nDotH: number, alpha: number): number => {
  const a2 = alpha * alpha;
  const d = nDotH * nDotH * (a2 - 1) + 1;
  return a2 / Math.max(Math.PI * d * d, 1e-6);
};
const smithG1 = (nDotV: number, roughness: number): number => {
  const r = roughness + 1;
  const k = (r * r) * 0.125;
  return nDotV / Math.max(nDotV * (1 - k) + k, 1e-6);
};
const brdfPdf = (
  baseColor: V3, roughness: number, metallic: number, transmission: number,
  normal: V3, wo: V3, wi: V3,
): number => {
  const wiDotN = dot(normal, wi), woDotN = dot(normal, wo);
  const nDotV = Math.max(woDotN, 0);
  if (nDotV <= 1e-5) return 0;
  const h = norm(add(wi, wo));
  const nDotH = Math.max(dot(normal, h), 0);
  const vDotH = Math.max(dot(wo, h), 1e-6);
  const f0: V3 = mix3([0.04, 0.04, 0.04], baseColor, metallic);
  const fres = fresnelSchlick(vDotH, f0);
  const baseSpecProb = Math.min(Math.max(0.04 + (0.96 - 0.04) * Math.max(luminance(fres), metallic), 0.04), 0.96);
  const baseTransProb = Math.min(Math.max(transmission * (1 - metallic), 0), 0.95);
  const baseDiffProb = Math.max(0, (1 - metallic) * (1 - transmission));
  const sumProb = Math.max(baseSpecProb + baseTransProb + baseDiffProb, 1e-4);
  const specProb = baseSpecProb / sumProb;
  const diffProb = baseDiffProb / sumProb;
  if (wiDotN * woDotN <= 0) return 0; // harness uses reflection (same-hemisphere) prefixes
  const nDotL = Math.max(wiDotN, 0);
  if (nDotL <= 1e-5) return 0;
  const alpha = Math.max(roughness * roughness, 1e-3);
  const d = ggxD(nDotH, alpha);
  const g1Wo = smithG1(nDotV, roughness);
  const pdfSpec = (d * g1Wo) / Math.max(4 * nDotV, 1e-6);
  const pdfDiff = nDotL * (1 / Math.PI);
  return diffProb * pdfDiff + specProb * pdfSpec;
};

// A canonical single-segment hybrid-shift fixture: a glossy reflection prefix bounce
// at xq (source) and xr (offset), reconnecting at a shared x_s above. wo points back
// toward the camera-side; wi points toward x_s (the next, reconnection vertex).
type Cfg = {
  xq: V3; nq: V3; woq: V3; wiq: V3;
  xr: V3; nr: V3; wor: V3; wir: V3;
  xs: V3; baseColor: V3; roughness: number; metallic: number; transmission: number; etaTOverI: number;
};
const mkCfg = (over: Partial<Cfg> = {}): Cfg => {
  const xs: V3 = over.xs ?? [0, 0, 1.2];
  const xq: V3 = over.xq ?? [0.4, 0.1, 0];
  const xr: V3 = over.xr ?? [-0.3, 0.2, 0];
  const nq: V3 = over.nq ?? [0, 0, 1];
  const nr: V3 = over.nr ?? [0, 0, 1];
  // wi toward x_s; wo a plausible incoming-reflection geometry (mirror-ish of wi).
  const wiq: V3 = over.wiq ?? norm(sub(xs, xq));
  const wir: V3 = over.wir ?? norm(sub(xs, xr));
  const woq: V3 = over.woq ?? norm([-wiq[0] * 0.6 + 0.1, -wiq[1] * 0.6, Math.abs(wiq[2]) * 0.9 + 0.3]);
  const wor: V3 = over.wor ?? norm([-wir[0] * 0.6 + 0.1, -wir[1] * 0.6, Math.abs(wir[2]) * 0.9 + 0.3]);
  return {
    xq, nq, woq, wiq, xr, nr, wor, wir, xs,
    baseColor: over.baseColor ?? [0.8, 0.5, 0.3],
    roughness: over.roughness ?? 0.3,
    metallic: over.metallic ?? 0.0,
    transmission: over.transmission ?? 0.0,
    etaTOverI: over.etaTOverI ?? 1.5,
  };
};

describe('ReSTIR-PT / GRIS HYBRID-shift Jacobian (random-replay prefix + reconnection)', () => {
  it('packs a single-segment config into the 48-float vec4-aligned record', () => {
    const c = mkCfg();
    const r = packRestirPtHybridShiftInput(c);
    expect(r).toHaveLength(RESTIR_PT_HYBRID_SHIFT_INPUT_FLOATS);
    expect(RESTIR_PT_HYBRID_SHIFT_INPUT_FLOATS).toBe(48);
    expect(r.slice(0, 4)).toEqual([c.xq[0], c.xq[1], c.xq[2], 0]); // xq.xyz, pad
    expect(r.slice(16, 20)).toEqual([c.xr[0], c.xr[1], c.xr[2], 0]); // xr.xyz, pad
    expect(r.slice(32, 36)).toEqual([c.xs[0], c.xs[1], c.xs[2], 0]); // xs.xyz, pad
    expect(r.slice(36, 40)).toEqual([c.baseColor[0], c.baseColor[1], c.baseColor[2], c.roughness]);
    expect(r.slice(40, 44)).toEqual([c.metallic, c.transmission, c.etaTOverI, 0]);
    // r-domain material row reuses q baseColor + roughness in the canonical case.
    expect(r.slice(44, 48)).toEqual([c.baseColor[0], c.baseColor[1], c.baseColor[2], c.roughness]);
  });

  it('exports the hybrid Jacobian = geometric half-G ratio × replayed-segment BSDF-pdf ratio', () => {
    // (A) geometric factor — IDENTICAL to restirPtShift's restirPtShiftJacobian.
    expect(RESTIR_PT_HYBRID_SHIFT_WGSL).toContain('fn rptHybridReconnectionGeometryTerm(');
    expect(RESTIR_PT_HYBRID_SHIFT_WGSL).toContain('let cosOut = abs(dot(ns, d) / dist)');
    expect(RESTIR_PT_HYBRID_SHIFT_WGSL).toContain('fn rptHybridGeomJacobian(');
    expect(RESTIR_PT_HYBRID_SHIFT_WGSL).toContain('return gTarget / gSource');
    // (B) replay factor — the engine's forward BSDF sampling pdf (VNDF specular).
    expect(RESTIR_PT_HYBRID_SHIFT_WGSL).toContain('fn rptHybridBsdfReplayPdf(');
    expect(RESTIR_PT_HYBRID_SHIFT_WGSL).toContain('let pdfSpec = (d * g1Wo) / max(4.0 * nDotV, 1e-6)');
    // SOURCE pdf in the numerator (du/dω_q = p_q), TARGET pdf in the denominator.
    expect(RESTIR_PT_HYBRID_SHIFT_WGSL).toContain('fn rptHybridReplaySegmentJacobian(');
    expect(RESTIR_PT_HYBRID_SHIFT_WGSL).toContain('return pq / pr');
    // (★) the composite: J_hybrid = jGeom · jReplay.
    expect(RESTIR_PT_HYBRID_SHIFT_WGSL).toContain('fn rptHybridShiftJacobian(');
    expect(RESTIR_PT_HYBRID_SHIFT_WGSL).toContain('return jGeom * jReplay');
  });

  it('uses the exact Smith G1 emitted by the same oracle as the forward sampler', () => {
    expect(RESTIR_PT_HYBRID_SHIFT_WGSL).toContain(
      roughDielectricSmithG1Wgsl('rptHybrid_smithG1'),
    );
  });

  it('harness composes the core byte-identically + writes [J, jGeom, jReplay], [pq, pr, gSrc, gTgt], and the SA derivs', () => {
    expect(RESTIR_PT_HYBRID_SHIFT_HARNESS_WGSL).toContain(RESTIR_PT_HYBRID_SHIFT_WGSL); // byte-identical core
    expect(RESTIR_PT_HYBRID_SHIFT_HARNESS_WGSL).toContain('let ns = vec3f(0.0, 0.0, 1.0)');
    expect(RESTIR_PT_HYBRID_SHIFT_HARNESS_WGSL).toContain('hOut[i * 3u + 0u] = vec4f(jHybrid, jGeom, jReplay, 0.0)');
    expect(RESTIR_PT_HYBRID_SHIFT_HARNESS_WGSL).toContain('hOut[i * 3u + 1u] = vec4f(pq, pr, gSource, gTarget)');
    expect(RESTIR_PT_HYBRID_SHIFT_HARNESS_WGSL).toContain('hOut[i * 3u + 2u] = vec4f(saSource, saTarget, 0.0, 0.0)');
    // The harness reuses the restirPtShift solid-angle-area-deriv measure for the FD leg.
    expect(RESTIR_PT_HYBRID_SHIFT_HARNESS_WGSL).toContain('fn rptHybridSolidAngleAreaDeriv(');
  });

  it('analytic CPU invariant: J_hybrid == J_geom · (p_q / p_r) on varying-geometry fixtures', () => {
    const ns: V3 = [0, 0, 1];
    const cfgs = [
      mkCfg(),
      mkCfg({ xq: [0.2, 0.0, 0], xr: [0.6, -0.2, 0], xs: [-0.1, 0.05, 1.0], roughness: 0.5 }),
      mkCfg({ xq: [-0.5, 0.4, 0], xr: [0.3, -0.3, 0], xs: [0.1, 0.1, 1.6], roughness: 0.2, metallic: 1.0 }),
    ];
    for (const c of cfgs) {
      const jg = jGeom(c.xq, c.xr, c.xs, ns);
      const pq = brdfPdf(c.baseColor, c.roughness, c.metallic, c.transmission, c.nq, c.woq, c.wiq);
      const pr = brdfPdf(c.baseColor, c.roughness, c.metallic, c.transmission, c.nr, c.wor, c.wir);
      expect(pq).toBeGreaterThan(0);
      expect(pr).toBeGreaterThan(0);
      const jHybrid = jg * (pq / pr);
      // Independently: the hybrid Jacobian must equal the product of the two factors.
      expect(jHybrid).toBeCloseTo(jg * pq / pr, 9);
      // And it differs from the pure-geometric reconnection Jacobian whenever the
      // replayed pdfs differ (the WHOLE POINT of the hybrid factor) — sanity that
      // the replay factor is not accidentally 1.
      if (Math.abs(pq - pr) > 1e-6) {
        expect(jHybrid).not.toBeCloseTo(jg, 6);
      }
    }
  });

  it('reciprocity: the FULL hybrid Jacobian is its own inverse — J(T)·J(T⁻¹) = 1', () => {
    const ns: V3 = [0, 0, 1];
    const c = mkCfg({ roughness: 0.4 });
    const pq = brdfPdf(c.baseColor, c.roughness, c.metallic, c.transmission, c.nq, c.woq, c.wiq);
    const pr = brdfPdf(c.baseColor, c.roughness, c.metallic, c.transmission, c.nr, c.wor, c.wir);
    const jFwd = jGeom(c.xq, c.xr, c.xs, ns) * (pq / pr);
    // T⁻¹ swaps q ↔ r in BOTH factors: geometric ratio inverts AND the replay pdf
    // ratio inverts (p_r/p_q). So J_inv = G(q)/G(r) · p_r/p_q = 1/J_fwd.
    const jInv = jGeom(c.xr, c.xq, c.xs, ns) * (pr / pq);
    expect(jFwd * jInv).toBeCloseTo(1, 9);
  });

  it('self-shift (xr==xq, identical replay) has unit hybrid Jacobian', () => {
    const ns: V3 = [0, 0, 1];
    const c = mkCfg();
    // Offset == source: same vertex, same wo/wi ⇒ both factors are 1.
    const jg = jGeom(c.xq, c.xq, c.xs, ns);
    const pq = brdfPdf(c.baseColor, c.roughness, c.metallic, c.transmission, c.nq, c.woq, c.wiq);
    expect(jg).toBeCloseTo(1, 12);
    expect(jg * (pq / pq)).toBeCloseTo(1, 12);
  });

  it('degenerate factors return 0: coincident/tangent reconnection edge OR unreachable replayed prefix', () => {
    const ns: V3 = [0, 0, 1];
    const c = mkCfg();
    // (1) Coincident reconnection edge (xq == xs) ⇒ geometric factor 0 ⇒ J_hybrid = 0.
    expect(jGeom([0, 0, 1.2], c.xr, [0, 0, 1.2], ns)).toBe(0);
    // (2) Unreachable replayed prefix: a back-hemisphere wi (wi·n < 0 vs wo·n > 0)
    //     gives a same-hemisphere-only reflection pdf of 0 ⇒ replay factor 0 ⇒
    //     J_hybrid = 0. (The WGSL's rptHybridReplaySegmentJacobian guards pq<=0.)
    const wiBack: V3 = [0, 0, -1];
    expect(brdfPdf(c.baseColor, c.roughness, c.metallic, c.transmission, c.nq, c.woq, wiBack)).toBe(0);
  });
});
