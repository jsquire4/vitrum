// brdfAdjoint.test.ts — the CRITICAL Phase-1 gate (WS5).
//
// The analytic partials dBrdf/dBaseColor and dBrdf/dRoughness MUST match a
// finite-difference of `evaluateBrdf` to ≤1e-4. This mirrors the
// nrcEncoding.test.ts FD-cross-check pattern. Phase 1 does NOT ship without
// this passing (per the plan §2 WS5 DoD).
//
// The CPU oracle (`../inverse/brdfAdjoint.ts`) is the load-bearing reference;
// the emitted WGSL (`../wgsl/pathTrace/pathTraceAdjoint.wgsl.ts`) is pinned to
// the SAME arithmetic by the codegen-shape tests at the bottom.

import { describe, it, expect } from 'vitest';
import {
  evaluateBrdf,
  dBrdf_dBaseColor,
  dBrdf_dRoughness,
  dBrdf_dSpecularColor,
  dBrdf_dSpecularIntensity,
} from '../inverse/brdfAdjoint.js';
import {
  PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL,
  ADJOINT_GRAD_FP,
} from '../wgsl/pathTrace/pathTraceAdjoint.wgsl.js';

type V3 = [number, number, number];

function normalize(v: V3): V3 {
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
}

// A spread of non-degenerate shading configurations: varied normal, view, light
// directions and material params. Every config keeps NdotL, NdotV > 0 so the
// BRDF is in its smooth (differentiable) regime (the early-out is a hard
// discontinuity we deliberately stay away from, exactly as path-replay does).
interface Config {
  baseColor: V3;
  roughness: number;
  metallic: number;
  normal: V3;
  wo: V3;
  wi: V3;
}

const CONFIGS: Config[] = [
  {
    baseColor: [0.8, 0.2, 0.1], roughness: 0.4, metallic: 0.1,
    normal: [0, 0, 1], wo: normalize([0.3, 0.1, 1]), wi: normalize([-0.2, 0.4, 1]),
  },
  {
    baseColor: [0.5, 0.5, 0.5], roughness: 0.7, metallic: 0.0,
    normal: [0, 0, 1], wo: normalize([0.5, 0.5, 0.9]), wi: normalize([0.1, -0.3, 1]),
  },
  {
    baseColor: [0.9, 0.85, 0.05], roughness: 0.25, metallic: 0.9,
    normal: normalize([0.1, 0.2, 1]), wo: normalize([0.2, 0.0, 1]), wi: normalize([-0.4, 0.1, 1]),
  },
  {
    baseColor: [0.2, 0.6, 0.9], roughness: 0.55, metallic: 0.4,
    normal: [0, 0, 1], wo: normalize([-0.3, 0.2, 1]), wi: normalize([0.3, 0.3, 1]),
  },
  {
    baseColor: [0.05, 0.05, 0.05], roughness: 0.9, metallic: 0.0,
    normal: normalize([0, 0.3, 1]), wo: normalize([0.1, 0.5, 1]), wi: normalize([0.0, 0.1, 1]),
  },
];

function perturb(base: V3, j: number, h: number): V3 {
  const out: V3 = [base[0], base[1], base[2]];
  out[j] = base[j]! + h;
  return out;
}

describe('BRDF adjoint — analytic dBrdf/dBaseColor == finite difference', () => {
  it('matches FD to <= 1e-4 over a spread of shading configs (the Phase-1 gate)', () => {
    const h = 1e-4;
    for (const cfg of CONFIGS) {
      const analytic = dBrdf_dBaseColor(
        cfg.baseColor, cfg.roughness, cfg.metallic, cfg.normal, cfg.wo, cfg.wi,
      );
      for (let j = 0; j < 3; j++) {
        const fp = evaluateBrdf(perturb(cfg.baseColor, j, h), cfg.roughness, cfg.metallic, cfg.normal, cfg.wo, cfg.wi);
        const fm = evaluateBrdf(perturb(cfg.baseColor, j, -h), cfg.roughness, cfg.metallic, cfg.normal, cfg.wo, cfg.wi);
        const fd = (fp[j]! - fm[j]!) / (2 * h); // diagonal: ∂out_j/∂baseColor_j
        expect(Math.abs(analytic[j]! - fd)).toBeLessThan(1e-4);
      }
    }
  });

  it('off-diagonal channel coupling is zero (∂out_c/∂baseColor_j = 0 for c≠j)', () => {
    const h = 1e-4;
    const cfg = CONFIGS[0]!;
    for (let j = 0; j < 3; j++) {
      const fp = evaluateBrdf(perturb(cfg.baseColor, j, h), cfg.roughness, cfg.metallic, cfg.normal, cfg.wo, cfg.wi);
      const fm = evaluateBrdf(perturb(cfg.baseColor, j, -h), cfg.roughness, cfg.metallic, cfg.normal, cfg.wo, cfg.wi);
      for (let c = 0; c < 3; c++) {
        if (c === j) continue;
        const fd = (fp[c]! - fm[c]!) / (2 * h);
        expect(Math.abs(fd)).toBeLessThan(1e-6);
      }
    }
  });
});

describe('BRDF adjoint — analytic dBrdf/dRoughness == finite difference', () => {
  it('matches FD to <= 1e-4 over a spread of shading configs (the Phase-1 gate)', () => {
    const h = 1e-4;
    for (const cfg of CONFIGS) {
      // All CONFIGS have roughness >= 0.25, so roughness² >= 0.0625 >> 1e-3 —
      // safely above the alpha clamp boundary where the derivative is non-smooth.
      const analytic = dBrdf_dRoughness(
        cfg.baseColor, cfg.roughness, cfg.metallic, cfg.normal, cfg.wo, cfg.wi,
      );
      const fp = evaluateBrdf(cfg.baseColor, cfg.roughness + h, cfg.metallic, cfg.normal, cfg.wo, cfg.wi);
      const fm = evaluateBrdf(cfg.baseColor, cfg.roughness - h, cfg.metallic, cfg.normal, cfg.wo, cfg.wi);
      for (let c = 0; c < 3; c++) {
        const fd = (fp[c]! - fm[c]!) / (2 * h);
        expect(Math.abs(analytic[c]! - fd)).toBeLessThan(1e-4);
      }
    }
  });
});

describe('BRDF adjoint — analytic KHR_materials_specular partials == finite difference', () => {
  const cfg = {
    baseColor: [0.55, 0.32, 0.18] as V3,
    roughness: 0.46,
    metallic: 0.25,
    normal: normalize([0.1, 0.2, 1]),
    wo: normalize([0.25, -0.1, 1]),
    wi: normalize([-0.35, 0.25, 1]),
    specularColor: [0.7, 0.45, 0.9] as V3,
    specularIntensity: 0.62,
  };

  it('matches FD for specularColor over the unclamped interior', () => {
    const h = 1e-4;
    const analytic = dBrdf_dSpecularColor(
      cfg.baseColor, cfg.roughness, cfg.metallic, cfg.normal, cfg.wo, cfg.wi,
      cfg.specularColor, cfg.specularIntensity,
    );
    for (let j = 0; j < 3; j++) {
      const sp = perturb(cfg.specularColor, j, h);
      const sm = perturb(cfg.specularColor, j, -h);
      const fp = evaluateBrdf(
        cfg.baseColor, cfg.roughness, cfg.metallic, cfg.normal, cfg.wo, cfg.wi,
        sp, cfg.specularIntensity,
      );
      const fm = evaluateBrdf(
        cfg.baseColor, cfg.roughness, cfg.metallic, cfg.normal, cfg.wo, cfg.wi,
        sm, cfg.specularIntensity,
      );
      const fd = (fp[j]! - fm[j]!) / (2 * h);
      expect(Math.abs(analytic[j]! - fd)).toBeLessThan(1e-4);
    }
  });

  it('matches FD for specularIntensity over the unclamped interior', () => {
    const h = 1e-4;
    const analytic = dBrdf_dSpecularIntensity(
      cfg.baseColor, cfg.roughness, cfg.metallic, cfg.normal, cfg.wo, cfg.wi,
      cfg.specularColor,
    );
    const fp = evaluateBrdf(
      cfg.baseColor, cfg.roughness, cfg.metallic, cfg.normal, cfg.wo, cfg.wi,
      cfg.specularColor, cfg.specularIntensity + h,
    );
    const fm = evaluateBrdf(
      cfg.baseColor, cfg.roughness, cfg.metallic, cfg.normal, cfg.wo, cfg.wi,
      cfg.specularColor, cfg.specularIntensity - h,
    );
    for (let c = 0; c < 3; c++) {
      const fd = (fp[c]! - fm[c]!) / (2 * h);
      expect(Math.abs(analytic[c]! - fd)).toBeLessThan(1e-4);
    }
  });
});

describe('BRDF adjoint — analytic Lambert cross-check (pure diffuse)', () => {
  it('dBrdf/dBaseColor of a pure-diffuse (metallic=0) surface ≈ (1-F)/π on the diagonal', () => {
    // For metallic=0: f0 = 0.04 everywhere, F is achromatic, diffuse term
    // = (1-F)·baseColor/π. The specular term is baseColor-independent (f0 fixed),
    // so ∂out_c/∂baseColor_c = (1-F)/π exactly. Compare against that closed form.
    const baseColor: V3 = [0.6, 0.3, 0.2];
    const roughness = 0.5, metallic = 0;
    const normal: V3 = [0, 0, 1];
    const wo = normalize([0.2, 0.1, 1]);
    const wi = normalize([-0.1, 0.2, 1]);
    const analytic = dBrdf_dBaseColor(baseColor, roughness, metallic, normal, wo, wi);
    const h: V3 = normalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
    const vDotH = Math.max(wo[0] * h[0] + wo[1] * h[1] + wo[2] * h[2], 0);
    const m = Math.min(Math.max(1 - vDotH, 0), 1);
    const m5 = m * m * m * m * m;
    const f = 0.04 + (1 - 0.04) * m5;
    const expected = (1 - f) / Math.PI;
    for (let c = 0; c < 3; c++) expect(analytic[c]!).toBeCloseTo(expected, 6);
  });
});

describe('BRDF adjoint — WGSL codegen shape pins (oracle equivalence)', () => {
  const wgsl = PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL;

  it('emits both partial functions with the frozen-wi signature', () => {
    expect(wgsl).toContain('fn dBrdf_dBaseColor(');
    expect(wgsl).toContain('fn dBrdf_dRoughness(');
    expect(wgsl).toContain('fn dBrdf_dSpecularColor(');
    expect(wgsl).toContain('fn dBrdf_dSpecularIntensity(');
    // wi is an INPUT, never sampled inside — path-replay freezes it.
    expect(wgsl).not.toContain('rand_f32');
    expect(wgsl).not.toContain('cosineHemisphereSample');
  });

  it('baseColor partial uses the same df_c/dbaseColor_c = (1-m5)·metallic term', () => {
    expect(wgsl).toContain('let dfc = (1.0 - m5) * metallic;');
    expect(wgsl).toContain('let dDiff = kd0 * INV_PI * ((1.0 - fc) + bc * (-dfc));');
  });

  it('roughness partial uses the same da²/droughness and dk/droughness terms', () => {
    expect(wgsl).toContain('let da2_dRough = 2.0 * alpha * dAlpha_dRough;');
    expect(wgsl).toContain('let dk_dRough = (roughness + 1.0) * 0.25;');
    // alpha clamp boundary handled (derivative 0 below roughness²<1e-3).
    expect(wgsl).toContain('let dAlpha_dRough = select(2.0 * roughness, 0.0, alphaClamped);');
  });

  it('specular partials use the KHR_materials_specular dielectric F0 derivative', () => {
    expect(wgsl).toContain('fn adjointMaterialSpecularF0(');
    expect(wgsl).toContain('let dF0 = vec3f(0.04 * clamp(specularIntensity, 0.0, 1.0) * (1.0 - metallic));');
    expect(wgsl).toContain('let dF0 = 0.04 * clamp(specularColor, vec3f(0.0), vec3f(1.0)) * (1.0 - metallic);');
    expect(wgsl).toContain('return dF0 * (1.0 - m5) * (vec3f(specScale) - kd0 * baseColor * INV_PI);');
  });

  it('gradient atomics use the i32 fixed-point scale shared with NRC (2^20)', () => {
    expect(ADJOINT_GRAD_FP).toBe(1048576);
    expect(wgsl).toContain('atomicAdd(&gradAccum[slot]');
    expect(wgsl).toContain('i32(round(g * ADJOINT_GRAD_FP))');
  });
});
