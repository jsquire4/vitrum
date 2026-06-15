// brdfAdjointEmissiveIor.test.ts — Phase II.1 gate: widen the path-replay
// adjoint's differentiable set to `emissive` (rgb) + `ior` (scalar).
//
// Same discipline as brdfAdjoint.test.ts: the analytic partials MUST match a
// finite-difference of their FORWARD to ≤1e-4 (CPU oracle is the load-bearing
// reference). The WGSL twins are pinned to the same arithmetic by the codegen-
// shape tests, and the executed GPU == FD A/B runs in wsl-gpu
// scripts/adjoint-emissive-ior-validate.ts (build-only here).
//
// Subtlety this file documents (and the math respects):
//  - `emissive` is NOT a BSDF term: its partial is a CONTRIBUTION-level identity
//    (× emissiveIntensity), so it is `dContribution_dEmissive`, not `dBrdf_*`.
//  - `ior` does NOT enter `evaluateBrdf` (opaque-reflective F0 is a fixed 0.04),
//    so `∂evaluateBrdf/∂ior ≡ 0`. The only differentiable ior dependence in the
//    forward is `frDielectric`; `dFrDielectric_dIor` differentiates THAT.

import { describe, it, expect } from 'vitest';
import {
  evaluateBrdf,
  frDielectric,
  dContribution_dEmissive,
  dFrDielectric_dIor,
} from '../inverse/brdfAdjoint.js';
import { PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL } from '../wgsl/pathTrace/pathTraceAdjoint.wgsl.js';
import {
  ADJOINT_EMISSIVE_IOR_FD_WGSL,
  packEmissiveIorAdjointInput,
  ADJOINT_EMISSIVE_IOR_INPUT_FLOATS,
} from '../inverse/adjointHarness.wgsl.js';
import type { Scene, MaterialSpec, SceneEmitter } from '@vitrum/core';
import {
  PtWebgpuInverseSession,
  type InverseEngineHooks,
} from '../inverse/inverseSession.js';

type V3 = [number, number, number];

const norm = (v: V3): V3 => {
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
};

describe('emissive adjoint — dContribution_dEmissive == finite difference', () => {
  // The emissive contribution forward is `rendered_c = throughput_c · intensity ·
  // emissive_c`. The partial w.r.t. emissive_c is throughput_c · intensity,
  // diagonal and VALUE-independent — so an arbitrary emissive base FD-checks it.
  const CASES: { throughput: V3; intensity: number }[] = [
    { throughput: [1, 1, 1], intensity: 1 }, // primary hit, default intensity
    { throughput: [0.6, 0.3, 0.9], intensity: 1 }, // attenuated throughput
    { throughput: [1, 1, 1], intensity: 4 }, // intensity folds into the partial
    { throughput: [0.2, 0.8, 0.5], intensity: 2.5 },
  ];
  const emissiveBase: V3 = [0.4, 0.7, 0.25];

  it('matches FD of the additive forward to <= 1e-4 (diagonal identity × intensity)', () => {
    const h = 1e-4;
    for (const { throughput, intensity } of CASES) {
      const analytic = dContribution_dEmissive(throughput, intensity);
      for (let c = 0; c < 3; c++) {
        // FD of `rendered_c = throughput_c · intensity · emissive_c` w.r.t.
        // emissive_c — the other channels are constant so only channel c moves.
        const eC = emissiveBase[c]!;
        const fwdP = throughput[c]! * intensity * (eC + h);
        const fwdM = throughput[c]! * intensity * (eC - h);
        const fd = (fwdP - fwdM) / (2 * h);
        expect(Math.abs(analytic[c]! - fd)).toBeLessThan(1e-4);
      }
    }
  });

  it('is exactly throughput·intensity (closed form, value-independent)', () => {
    const g = dContribution_dEmissive([0.6, 0.3, 0.9], 2);
    expect(g[0]).toBeCloseTo(1.2, 12);
    expect(g[1]).toBeCloseTo(0.6, 12);
    expect(g[2]).toBeCloseTo(1.8, 12);
  });

  it('emissive does NOT enter evaluateBrdf (sanity: BRDF is emissive-independent)', () => {
    // Confirms the design premise: emission is additive, not a BSDF term — so its
    // adjoint lives at the contribution level, never in dBrdf_*.
    const a = evaluateBrdf([0.5, 0.5, 0.5], 0.4, 0.0, [0, 0, 1], norm([0.2, 0.1, 1]), norm([-0.2, 0.3, 1]));
    // evaluateBrdf has no emissive parameter — there is nothing to perturb; the
    // value is whatever the BRDF computes (assert it's finite + nonzero).
    expect(a.every((x) => Number.isFinite(x))).toBe(true);
    expect(a[0] + a[1] + a[2]).toBeGreaterThan(0);
  });
});

describe('ior adjoint — dFrDielectric_dIor == finite difference', () => {
  // Cases span front (cosThetaI > 0) and back (cosThetaI < 0) faces and a range
  // of dielectric IOR, all clear of the TIR / grazing discontinuities.
  const CASES: { cosThetaI: number; ior: number }[] = [
    { cosThetaI: 1.0, ior: 1.5 }, // normal incidence, glass
    { cosThetaI: 0.7, ior: 1.5 },
    { cosThetaI: 0.3, ior: 1.33 }, // water, steeper angle
    { cosThetaI: 0.9, ior: 2.4 }, // diamond-ish
    // Back faces (eta = 1/ior). These MUST be clear of TIR: sin²θ_t =
    // (1-cos²)·ior² < 1, i.e. |cos| > sqrt(1 - 1/ior²). Both below satisfy it.
    { cosThetaI: -0.85, ior: 1.2 }, // sin²θ_t ≈ 0.40
    { cosThetaI: -0.95, ior: 1.3 }, // sin²θ_t ≈ 0.16
  ];

  it('matches FD of frDielectric to <= 1e-4 over front + back faces', () => {
    const h = 1e-4;
    for (const { cosThetaI, ior } of CASES) {
      const analytic = dFrDielectric_dIor(cosThetaI, ior);
      const fp = frDielectric(cosThetaI, ior + h);
      const fm = frDielectric(cosThetaI, ior - h);
      const fd = (fp - fm) / (2 * h);
      expect(Math.abs(analytic - fd)).toBeLessThan(1e-4);
    }
  });

  it('the back-face chain factor (eta = 1/ior) is exercised, not just front', () => {
    // The back-face branch composes dFr/deta with d(1/ior)/dior = -1/ior². The
    // analytic == FD assertion above already proves the COMPOSITE is correct for a
    // back face; this just guards that the derivative is finite + nonzero there
    // (the branch is live, not silently short-circuited to 0 by a stray guard).
    const back = dFrDielectric_dIor(-0.85, 1.2);
    expect(Number.isFinite(back)).toBe(true);
    expect(Math.abs(back)).toBeGreaterThan(1e-3);
  });

  it('returns 0 on TIR (sin²θ_t ≥ 1) — the frozen-discontinuity convention', () => {
    // Back face, large IOR, grazing: sin²θ_t = (1-cos²)/eta² with eta = 1/ior is
    // (1-cos²)·ior² — push it ≥ 1 to force TIR.
    // cosThetaI = -0.1 ⇒ sin² ≈ 0.99; ior = 2.0 ⇒ eta = 0.5 ⇒ sin²θ_t ≈ 3.96 > 1.
    expect(dFrDielectric_dIor(-0.1, 2.0)).toBe(0);
  });

  it('frDielectric forward mirror: normal-incidence Fresnel matches (n-1)²/(n+1)²', () => {
    // At cosThetaI = 1, frDielectric reduces to the Schlick-exact F0 = ((n-1)/(n+1))².
    const n = 1.5;
    const f0 = ((n - 1) / (n + 1)) ** 2;
    expect(frDielectric(1.0, n)).toBeCloseTo(f0, 6);
  });
});

describe('emissive/ior adjoint — WGSL codegen shape pins (oracle equivalence)', () => {
  const wgsl = PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL;

  it('emits the two new partial functions', () => {
    expect(wgsl).toContain('fn dContribution_dEmissive(throughput: vec3f, emissiveIntensity: f32) -> vec3f');
    expect(wgsl).toContain('fn dFrDielectric_dIor(cosThetaI_in: f32, ior: f32) -> f32');
  });

  it('emissive partial is the diagonal identity × intensity', () => {
    expect(wgsl).toContain('return throughput * emissiveIntensity;');
  });

  it('ior partial mirrors the oracle eta-chain + TIR/grazing guards', () => {
    expect(wgsl).toContain('eta = 1.0 / ior;');
    expect(wgsl).toContain('dEta_dIor = -1.0 / (ior * ior);');
    expect(wgsl).toContain('let dCosT_dEta = s / (cosThetaT * eta * eta * eta);');
    expect(wgsl).toContain('let dFr_dEta = rPar * dRPar_dEta + rPerp * dRPerp_dEta;');
    expect(wgsl).toContain('return dFr_dEta * dEta_dIor;');
    // TIR + grazing both return 0 (frozen-discontinuity convention).
    expect(wgsl).toContain('if (sin2ThetaT >= 1.0) { return 0.0; }');
    expect(wgsl).toContain('if (cosThetaT <= 1e-6) { return 0.0; }');
  });
});

describe('emissive/ior adjoint — GPU FD harness shape', () => {
  it('packs an emissive/ior input into the 8-float record', () => {
    const r = packEmissiveIorAdjointInput([0.6, 0.3, 0.9], 2, 0.7, 1.5);
    expect(r).toHaveLength(ADJOINT_EMISSIVE_IOR_INPUT_FLOATS);
    expect(r.slice(0, 4)).toEqual([0.6, 0.3, 0.9, 2]); // throughput.xyz, intensity
    expect(r.slice(4, 8)).toEqual([0.7, 1.5, 0, 0]); // cosThetaI, ior, pad, pad
  });

  it('bundles the REAL partials + FDs the matching forwards', () => {
    // The partials under test must be the byte-identical production WGSL.
    expect(ADJOINT_EMISSIVE_IOR_FD_WGSL).toContain(PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL);
    expect(ADJOINT_EMISSIVE_IOR_FD_WGSL).toContain('fn frDielectric('); // forward the ior FD differentiates
    expect(ADJOINT_EMISSIVE_IOR_FD_WGSL).toContain('dContribution_dEmissive(e.throughput, e.emissiveIntensity)');
    expect(ADJOINT_EMISSIVE_IOR_FD_WGSL).toContain('dFrDielectric_dIor(e.cosThetaI, e.ior)');
    expect(ADJOINT_EMISSIVE_IOR_FD_WGSL).toContain('@compute @workgroup_size(64)');
    // analytic + FD accumulators for both fields.
    expect(ADJOINT_EMISSIVE_IOR_FD_WGSL).toContain('gradEmAdj');
    expect(ADJOINT_EMISSIVE_IOR_FD_WGSL).toContain('gradEmFd');
    expect(ADJOINT_EMISSIVE_IOR_FD_WGSL).toContain('gradIorAdj');
    expect(ADJOINT_EMISSIVE_IOR_FD_WGSL).toContain('gradIorFd');
  });
});

// ── inverseSession field-set widening (routing + optimizable-field validation) ──

function makeScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'panel',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [0.2, 0.2, 0.2], roughness: 0.5, metallic: 0,
          emissive: [0.1, 0.2, 0.3], emissiveIntensity: 1, ior: 1.5,
          specularColor: [0.7, 0.8, 0.9], specularIntensity: 0.65,
          clearcoat: 0.25, clearcoatRoughness: 0.35,
          sheen: 0.45, sheenColor: [0.15, 0.25, 0.35], sheenRoughness: 0.55,
          iridescence: 0.2, iridescenceIor: 1.4,
          anisotropy: 0.3, anisotropyRotation: 0.4,
        },
      },
    ],
    emitters: [{ kind: 'point', id: 'lamp', color: [1, 1, 1], intensity: 2, position: [0, 1, 0] }],
    environment: { kind: 'none' },
  };
}

function makeHooks(scene: Scene, withAdjoint: boolean): InverseEngineHooks {
  let live = scene;
  const hooks: InverseEngineHooks = {
    getScene: () => live,
    renderAndReadback: async (w, h) => ({ rgb: new Float32Array(w * h * 3), channels: 3 as const }),
    patchMaterial: (id: string, patch: Partial<MaterialSpec>) => {
      live = {
        ...live,
        primitives: live.primitives.map((p) =>
          p.id === id ? { ...p, material: { ...p.material, ...patch } } : p,
        ),
      };
    },
    patchEmitter: (id: string, patch: Partial<SceneEmitter>) => {
      live = {
        ...live,
        emitters: live.emitters.map((e) => (e.id === id ? ({ ...e, ...patch } as SceneEmitter) : e)),
      };
    },
  };
  if (withAdjoint) hooks.computeAdjointGradient = async (req) => new Float32Array(req.gradientLength);
  return hooks;
}

const target = { data: new Float32Array(2 * 2 * 3), width: 2, height: 2, channels: 3 as const };

describe('inverseSession — emissive/ior field-set widening', () => {
  it('resolves an emissive (rgb) material param + seeds it from the scene', () => {
    const session = new PtWebgpuInverseSession(makeHooks(makeScene(), false), {
      target,
      parameters: [{ path: 'materials.panel.emissive', kind: 'rgb' }],
    });
    expect(session.currentValues()[0]).toEqual([
      expect.closeTo(0.1, 6), expect.closeTo(0.2, 6), expect.closeTo(0.3, 6),
    ]);
    session.dispose();
  });

  it('resolves an ior (scalar) material param + seeds it from the scene', () => {
    const session = new PtWebgpuInverseSession(makeHooks(makeScene(), false), {
      target,
      parameters: [{ path: 'materials.panel.ior', kind: 'scalar' }],
    });
    expect(session.currentValues()[0]).toEqual([expect.closeTo(1.5, 6)]);
    session.dispose();
  });

  it('resolves emissive to path-replay (engine emissive scatter GPU-validated end-to-end)', () => {
    const session = new PtWebgpuInverseSession(makeHooks(makeScene(), true), {
      target,
      parameters: [{ path: 'materials.panel.emissive', kind: 'rgb' }],
      method: 'path-replay',
    });
    // emissive is in ADJOINT_ELIGIBLE_FIELDS: the engine adjoint scatters the
    // camera-DIRECT emission at the primary hit (∂loss/∂emissive_c = dLoss_dR_c ·
    // emissiveIntensity, dContribution_dEmissive with throughput = 1), with the
    // fixed emissiveIntensity carried in the descriptor `.w` (bitcast f32). The
    // end-to-end fit converges + sign-matches FD on lavapipe (wsl-gpu
    // tests/v24-emissive-fit.mjs). The earlier divergent trial scattered emissive
    // inside the NEE loop and validated against a barely-visible target.
    expect(session.method).toBe('path-replay');
    session.dispose();
  });

  it('keeps ior on finite-difference even with the hook (NOT adjoint-eligible)', () => {
    const session = new PtWebgpuInverseSession(makeHooks(makeScene(), true), {
      target,
      parameters: [{ path: 'materials.panel.ior', kind: 'scalar' }],
      method: 'path-replay',
    });
    // ior ∂evaluateBrdf/∂ior ≡ 0 in the forward; the single-bounce adjoint pass
    // does not trace transmission, so ior stays FD (no silently-wrong zero grad).
    expect(session.method).toBe('finite-difference');
    session.dispose();
  });

  it('a mixed emissive + ior request degrades the WHOLE step to FD (ior is the holdout)', () => {
    const session = new PtWebgpuInverseSession(makeHooks(makeScene(), true), {
      target,
      parameters: [
        { path: 'materials.panel.emissive', kind: 'rgb' },
        { path: 'materials.panel.ior', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    // emissive IS eligible now, but ior is not — path-replay requires EVERY param
    // eligible, so the whole step degrades to FD on the ior holdout (the all-or-
    // nothing method-resolution contract: no mixed analytic/FD gradient).
    expect(session.method).toBe('finite-difference');
    session.dispose();
  });

  it('ior clamps to the dielectric [1, 2.5] range by default (matches the decoder)', async () => {
    const session = new PtWebgpuInverseSession(makeHooks(makeScene(), false), {
      target,
      parameters: [{ path: 'materials.panel.ior', kind: 'scalar', initial: [3.0] }],
      samplesPerStep: 1,
    });
    // initial 3.0 is above the clamp ceiling; after a step the clamp pins it ≤ 2.5.
    await session.step();
    expect(session.currentValues()[0]![0]!).toBeLessThanOrEqual(2.5);
    session.dispose();
  });

  it('resolves common extension-lobe material params for finite-difference optimization', () => {
    const session = new PtWebgpuInverseSession(makeHooks(makeScene(), false), {
      target,
      parameters: [
        { path: 'materials.panel.specularColor', kind: 'rgb' },
        { path: 'materials.panel.specularIntensity', kind: 'scalar' },
        { path: 'materials.panel.clearcoat', kind: 'scalar' },
        { path: 'materials.panel.clearcoatRoughness', kind: 'scalar' },
        { path: 'materials.panel.sheenColor', kind: 'rgb' },
        { path: 'materials.panel.sheenRoughness', kind: 'scalar' },
        { path: 'materials.panel.iridescence', kind: 'scalar' },
        { path: 'materials.panel.iridescenceIor', kind: 'scalar' },
        { path: 'materials.panel.anisotropy', kind: 'scalar' },
        { path: 'materials.panel.anisotropyRotation', kind: 'scalar' },
      ],
    });

    expect(session.method).toBe('finite-difference');
    expect(session.currentValues()).toEqual([
      [expect.closeTo(0.7, 6), expect.closeTo(0.8, 6), expect.closeTo(0.9, 6)],
      [expect.closeTo(0.65, 6)],
      [expect.closeTo(0.25, 6)],
      [expect.closeTo(0.35, 6)],
      [expect.closeTo(0.15, 6), expect.closeTo(0.25, 6), expect.closeTo(0.35, 6)],
      [expect.closeTo(0.55, 6)],
      [expect.closeTo(0.2, 6)],
      [expect.closeTo(1.4, 6)],
      [expect.closeTo(0.3, 6)],
      [expect.closeTo(0.4, 6)],
    ]);
    session.dispose();
  });

  it('keeps extension-lobe params on finite-difference even when path-replay is requested', () => {
    const session = new PtWebgpuInverseSession(makeHooks(makeScene(), true), {
      target,
      parameters: [
        { path: 'materials.panel.clearcoat', kind: 'scalar' },
        { path: 'materials.panel.specularColor', kind: 'rgb' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('finite-difference');
    session.dispose();
  });
});
