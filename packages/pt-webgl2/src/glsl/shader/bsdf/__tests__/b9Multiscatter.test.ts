/**
 * B9 (road-to-100) — Kulla-Conty GGX multiscatter energy compensation (pt-webgl2).
 *
 * CPU mirror of the GLSL `ggxMultiscatter` / `ggxDirectionalAlbedo` /
 * `ggxAverageAlbedo` fit in `ggx_functions.glsl.js`, plus a white-furnace gate.
 *
 * Two checks:
 *   1. STRUCTURAL — the GLSL exposes the multiscatter helpers and `specularEval`
 *      folds `ggxMultiscatter` into the specular color (so rough metals recover
 *      energy) — the wiring must be present in the emitted shader source.
 *   2. WHITE-FURNACE — for a perfect conductor (F0 = 1 ⇒ Favg = 1) the directional
 *      albedo E(μ) PLUS the cosine-integrated multiscatter lobe recovers ≈ all
 *      incident energy at every roughness. Without compensation a rough conductor
 *      reads dark (E < 1); with the multiscatter term the furnace residual → ~0.
 *
 * The fit is shared CONVENTION-coordinate with the walkaround-hybrid B9 fit, so
 * all backends compensate identically.
 */
import { describe, it, expect } from 'vitest';
// The `.glsl.js` chunks are typed by the wildcard `*.glsl.js` ambient module as a
// default `string` export only (named exports are not declarable in a TS wildcard
// module), so the composer + tests read them via a namespace import cast to
// `Record<string, string>` — the supported pattern. See glsl-modules.d.ts.
import * as GgxFns from '../ggx_functions.glsl.js';
import * as BsdfFns from '../bsdf_functions.glsl.js';

const ggx_functions = (GgxFns as unknown as Record<string, string>)['ggx_functions']!;
const bsdf_functions = (BsdfFns as unknown as Record<string, string>)['bsdf_functions']!;

const PI = Math.PI;

// ── CPU mirror of the GLSL fit (must track ggx_functions.glsl.js exactly) ────
function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}
function ggxDirectionalAlbedo(mu: number, rough: number): number {
  const a = clamp01(rough);
  const c = clamp01(mu);
  const a2 = a * a;
  return clamp01(1 - a2 * (1 - c) * (0.75 + 0.25 * c));
}
function ggxAverageAlbedo(rough: number): number {
  const a = clamp01(rough);
  const a2 = a * a;
  return clamp01(1 - a2 * (7 / 24));
}
/** Scalar (F0 = 1, Favg = 1) multiscatter lobe value, WITHOUT the NdotL cosine
 *  (the GLSL multiplies NdotL in; the furnace integral re-applies it explicitly). */
function ggxMultiscatterScalarNoCos(rough: number, NdotV: number, NdotL: number): number {
  const Eo = ggxDirectionalAlbedo(NdotV, rough);
  const Ei = ggxDirectionalAlbedo(NdotL, rough);
  const Eavg = ggxAverageAlbedo(rough);
  const oneMinusEavg = 1 - Eavg;
  if (!(oneMinusEavg > 0)) return 0;
  const fms = ((1 - Eo) * (1 - Ei)) / (PI * oneMinusEavg);
  // Favg = 1 → Fms = (1·Eavg)/(1 − (1−Eavg)) = Eavg/Eavg = 1.
  return fms; // Fms = 1 for the perfect conductor
}

describe('B9 — pt-webgl2 GGX multiscatter (structural)', () => {
  it('ggx_functions exposes the Kulla-Conty multiscatter helpers', () => {
    expect(ggx_functions).toContain('ggxDirectionalAlbedo');
    expect(ggx_functions).toContain('ggxAverageAlbedo');
    expect(ggx_functions).toContain('ggxMultiscatter');
    expect(ggx_functions).toContain('if ( ! ( oneMinusEavg > 0.0 ) )');
    expect(ggx_functions).not.toContain('oneMinusEavg = max(');
  });

  it('specularEval folds the multiscatter lobe into the specular color', () => {
    // The eval site must ADD ggxMultiscatter(...) to `color` so rough metals
    // recover their dropped energy.
    expect(bsdf_functions).toMatch(/color\s*\+=\s*ggxMultiscatter\(/);
    // Favg uses the Fdez-Agüera cosine-weighted-average Fresnel approximation.
    expect(bsdf_functions).toContain('1.0 / 21.0');
  });
});

describe('B9 — pt-webgl2 GGX multiscatter (white furnace)', () => {
  // Cosine-weighted hemispherical integral of the OUTGOING energy for a perfect
  // white conductor viewed at NdotV = mu:
  //   E_total(mu) = E_single(mu)  +  ∫ fms(mu, μ_i) · μ_i dω_i
  // The single-scatter directional albedo is E_single = ggxDirectionalAlbedo(mu).
  // For energy conservation, E_total should be ≈ 1 at every roughness.
  function furnaceResidual(mu: number, rough: number, withMs: boolean): number {
    const Esingle = ggxDirectionalAlbedo(mu, rough);
    let ms = 0;
    if (withMs) {
      // ∫_hemisphere fms(mu, μ_i) μ_i dω_i, dω_i = sinθ dθ dφ; φ integral = 2π.
      const N = 2000;
      for (let k = 0; k < N; k++) {
        const theta = ((k + 0.5) / N) * (PI / 2);
        const muI = Math.cos(theta);
        const sinT = Math.sin(theta);
        const fms = ggxMultiscatterScalarNoCos(rough, mu, muI);
        ms += fms * muI * sinT * (PI / 2 / N) * 2 * PI;
      }
    }
    return Esingle + ms - 1.0;
  }

  it('without compensation, a rough conductor is dark (energy < 1)', () => {
    // Sanity: the single-scatter lobe alone loses energy at high roughness.
    const r = furnaceResidual(0.5, 0.9, false);
    expect(r).toBeLessThan(-0.1); // noticeably dark
  });

  it('with compensation, energy is recovered to ≈ 1 across roughness', () => {
    for (const rough of [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]) {
      for (const mu of [0.2, 0.5, 0.8, 1.0]) {
        const r = furnaceResidual(mu, rough, true);
        // Kulla-Conty analytic fits are not exact; ±6% is the accepted band and
        // is a large improvement over the uncompensated deficit (up to ~−25%).
        expect(Math.abs(r)).toBeLessThan(0.06);
      }
    }
  });

  it('multiscatter adds zero at roughness 0 (the single lobe is already exact)', () => {
    // At rough → 0 the single-scatter lobe carries all the energy, so the
    // multiscatter term must vanish (E_avg → 1 ⇒ 1−E_avg → 0 guard).
    const ms = ggxMultiscatterScalarNoCos(0.0, 0.7, 0.7);
    expect(ms).toBe(0);
  });
});
