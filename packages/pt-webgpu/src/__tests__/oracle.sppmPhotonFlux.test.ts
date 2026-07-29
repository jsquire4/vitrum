/**
 * SPPM photon-flux energy-conservation oracle — regression verification of the
 * lightSelectInvPdf normalization in the photon-emission pass
 * (packages/pt-webgpu/src/wgsl/pathTrace/sppmBindings.wgsl.ts, sppmEmitPhotons).
 *
 * WHAT IS UNDER AUDIT
 * -------------------
 * sppmEmitPhotons picks ONE shadow-casting photon source uniformly among
 * `availableLightCount` sources and normalizes every photon's flux by
 *   lightSelectInvPdf = f32(availableLightCount)
 * Per-source flux (transcribed below, file:line cited):
 *   point: rad·4π·invPdf/N                                   [L433-435]
 *   rect:  Le·area·π·invPdf/N                                [L500-501]
 *   env (constant one-texel map): L·diskArea·invPdf/(N·envPdf) [L567-568]
 *     with the map's authored direction density envPdf = 1/(4π).
 *
 * ENERGY-CONSERVATION LAW (derived from first principles):
 * With uniform light pick p = 1/K and flux Φ_i = P_pick·K/N, the expected total
 * emitted flux over N photons is
 *   E[ΣΦ] = N · Σ_s (1/K)·P_s·K/N = Σ_s P_s
 * i.e. the SUM of all source powers — NOT the mean. Reference powers are
 * derived independently of the shader formulas:
 *   point  (intensity I, isotropic):           P = ∫_{4π} I dω = 4π·I
 *   rect   (diffuse emitter, radiance Le, A):  P = A·∫_{2π} Le·cosθ dω = π·A·Le
 *   env    (photon-launch convention: photons enter through a disk of area
 *           A_d = π·R² perpendicular to each direction):
 *           P = ∫_{4π} L(ω)·A_d dω = 4π·L0·A_d for a constant sky L0.
 *
 * THE OLD BUG CLASS this catches: a missing lightSelectInvPdf (uniform pick
 * WITHOUT the ×K renormalization) yields E[ΣΦ] = (1/K)·Σ P_s — with K=4
 * sources every light runs at a quarter power (≈1/3 with 3 non-env sources).
 * An over-correction (×K²) would read ×K. The test asserts the ratio ≈ 1 and
 * explicitly excludes the ≈1/K and ≈K failure modes.
 *
 * Scene per the P3 workstream spec: 2 point lights + 1 rect + env → K = 4
 * when all sources cast shadows. If a source has castShadow:false, the photon
 * pass removes it from the source set and renormalizes over the active count.
 */
import { describe, expect, it } from 'vitest';

type V3 = [number, number, number];
type SourceId = 'point0' | 'point1' | 'rect' | 'env';
const PI = Math.PI;

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

// ── scene ─────────────────────────────────────────────────────────────────────
// Two point lights (radiant intensity, pointLights[base+1].rgb in the shader):
const pointRad: [V3, V3] = [
  [2.0, 1.5, 1.0],
  [0.5, 0.8, 1.2],
];
// One rect light: ru=(0.8,0,0), rv=(0,0,0.5) → area = 4·|ru×rv| = 4·0.4 = 1.6
// (area convention at sppmBindings.wgsl.ts:495).
const rectLe: V3 = [3, 2, 1];
const rectArea = 4 * (0.8 * 0.5);
// Constant one-texel environment map with a uniform-sphere direction density.
const skyL0: V3 = [0.3, 0.4, 0.5];
const sceneExtent = 5; // sppmStats.sceneExtent
const diskArea = PI * sceneExtent * sceneExtent; // L566

const ALL_SOURCES = ['point0', 'point1', 'rect', 'env'] as const satisfies readonly SourceId[];
const K = ALL_SOURCES.length; // availableLightCount = 2 points + 1 rect + 1 env.

// ── reference powers (independent derivations, see header) ───────────────────
const CH = [0, 1, 2] as const;
const refPoint: [V3, V3] = [
  [4 * PI * pointRad[0][0], 4 * PI * pointRad[0][1], 4 * PI * pointRad[0][2]],
  [4 * PI * pointRad[1][0], 4 * PI * pointRad[1][1], 4 * PI * pointRad[1][2]],
];
const refRect: V3 = [
  PI * rectArea * rectLe[0],
  PI * rectArea * rectLe[1],
  PI * rectArea * rectLe[2],
];
const refEnv: V3 = [
  4 * PI * skyL0[0] * diskArea,
  4 * PI * skyL0[1] * diskArea,
  4 * PI * skyL0[2] * diskArea,
];

function referenceTotalForSources(sources: readonly SourceId[]): V3 {
  const total: V3 = [0, 0, 0];
  for (const source of sources) {
    for (const c of CH) {
      if (source === 'point0') total[c] += refPoint[0][c];
      else if (source === 'point1') total[c] += refPoint[1][c];
      else if (source === 'rect') total[c] += refRect[c];
      else total[c] += refEnv[c];
    }
  }
  return total;
}

const refTotal: V3 = referenceTotalForSources(ALL_SOURCES);

function attenuatePhotonFluxThroughMedium(
  flux: V3,
  sigmaA: V3,
  scatteringRgb: V3,
  scatteringCoeff: number,
  distance: number,
): V3 {
  const sigmaT: V3 = [
    Math.max(0, sigmaA[0] + Math.max(scatteringRgb[0], scatteringCoeff)),
    Math.max(0, sigmaA[1] + Math.max(scatteringRgb[1], scatteringCoeff)),
    Math.max(0, sigmaA[2] + Math.max(scatteringRgb[2], scatteringCoeff)),
  ];
  return [
    flux[0] * Math.exp(-sigmaT[0] * distance),
    flux[1] * Math.exp(-sigmaT[1] * distance),
    flux[2] * Math.exp(-sigmaT[2] * distance),
  ];
}

// ── transcribed photon seeding (sppmBindings.wgsl.ts:370-571) ─────────────────
// invPdfOverride lets the test simulate the OLD missing-invPdf bug (=1) and an
// over-correction (=K²·(1/K)=K extra) to prove the oracle separates the three.
function emitTotalFlux(
  nPhotons: number,
  seed: number,
  invPdf: number,
  sources: readonly SourceId[] = ALL_SOURCES,
): { total: V3; perClass: [V3, V3, V3] } {
  const rng = mulberry32(seed);
  const k = sources.length;
  const total: V3 = [0, 0, 0];
  const perClass: [V3, V3, V3] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ]; // [points, rect, env]
  for (let i = 0; i < nPhotons; i++) {
    // pick = min(floor(rand·K_active), K_active−1)
    const pick = Math.min(Math.floor(rng() * k), k - 1);
    const source = sources[pick]!;
    let flux: V3;
    let cls: 0 | 1 | 2;
    if (source === 'point0' || source === 'point1') {
      // Point light [L428-439]: Φ = rad·4π·invPdf/N (direction draw does not
      // affect flux; uniformSphere consumes 2 rands — irrelevant to expectation).
      const r = pointRad[source === 'point0' ? 0 : 1];
      flux = [
        (r[0] * 4 * PI * invPdf) / nPhotons,
        (r[1] * 4 * PI * invPdf) / nPhotons,
        (r[2] * 4 * PI * invPdf) / nPhotons,
      ];
      cls = 0;
    } else if (source === 'rect') {
      // Rect light [L472-506]: position xi1/xi2 + cosine-hemisphere direction;
      // flux is INDEPENDENT of both: Φ = Le·area·π·invPdf/N  [L500-501]
      rng(); // xi1  [L485]
      rng(); // xi2  [L486]
      rng(); // hemi u1 [L497]
      rng(); // hemi u2
      flux = [
        (rectLe[0] * rectArea * PI * invPdf) / nPhotons,
        (rectLe[1] * rectArea * PI * invPdf) / nPhotons,
        (rectLe[2] * rectArea * PI * invPdf) / nPhotons,
      ];
      cls = 1;
    } else {
      // Env [L543-571], constant one-texel map: envColor = L0 and the map
      // carries envPdf = 1/(4π). Disk-position draws don't affect flux.
      rng();
      rng(); // uniformSphere [L553]
      rng();
      rng(); // disk r/phi    [L559-560]
      const envPdf = 1 / (4 * PI);
      // L567-568: Φ = envColor·diskArea·invPdf/(N·envPdf)
      flux = [
        (skyL0[0] * diskArea * invPdf) / (nPhotons * envPdf),
        (skyL0[1] * diskArea * invPdf) / (nPhotons * envPdf),
        (skyL0[2] * diskArea * invPdf) / (nPhotons * envPdf),
      ];
      cls = 2;
    }
    for (const c of CH) {
      total[c] += flux[c];
      perClass[cls][c] += flux[c];
    }
  }
  return { total, perClass };
}

describe('SPPM oracle — photon flux energy conservation (lightSelectInvPdf)', () => {
  const N = 400_000;

  it('REFUTED-BIAS check: total emitted flux ≈ Σ source powers (per channel, multi-light)', () => {
    // lightSelectInvPdf = K (sppmBindings.wgsl.ts:394) — the current shader.
    const { total, perClass } = emitTotalFlux(N, 20260611, K);
    for (const c of CH) {
      const ratio = total[c] / refTotal[c];
      // Only multinomial pick-count noise remains (per-photon flux is
      // deterministic per class here): s.e. ≈ 0.4% at N=4·10⁵. 3% band.
      expect(
        ratio,
        `channel ${c}: ΣΦ/ΣP = ${ratio.toFixed(4)} (points ${perClass[0][c].toFixed(2)} vs ` +
          `${(refPoint[0][c] + refPoint[1][c]).toFixed(2)}, rect ${perClass[1][c].toFixed(2)} vs ` +
          `${refRect[c].toFixed(2)}, env ${perClass[2][c].toFixed(2)} vs ${refEnv[c].toFixed(2)})`,
      ).toBeGreaterThan(0.97);
      expect(ratio).toBeLessThan(1.03);
    }
  });

  it('per-class energy split matches each source power (catches per-source formula drift)', () => {
    const { perClass } = emitTotalFlux(N, 555, K);
    const refClasses: [V3, V3, V3] = [
      CH.map((c) => refPoint[0][c] + refPoint[1][c]) as unknown as V3,
      refRect,
      refEnv,
    ];
    for (const cls of [0, 1, 2] as const) {
      for (const c of CH) {
        const ratio = perClass[cls][c] / refClasses[cls][c];
        expect(ratio, `class ${cls} channel ${c}: ${ratio.toFixed(4)}`).toBeGreaterThan(0.96);
        expect(ratio).toBeLessThan(1.04);
      }
    }
  });

  it('renormalizes photon flux over shadow-casting sources when one emitter is disabled', () => {
    const activeSources = ['point0', 'rect', 'env'] as const satisfies readonly SourceId[];
    const activeK = activeSources.length;
    const activeRef = referenceTotalForSources(activeSources);
    const { total } = emitTotalFlux(N, 20260618, activeK, activeSources);
    const wrongAllLightPdf = emitTotalFlux(N, 20260618, K, activeSources).total;
    for (const c of CH) {
      const ratio = total[c] / activeRef[c];
      expect(ratio, `channel ${c}: active-source ΣΦ/ΣP = ${ratio.toFixed(4)}`).toBeGreaterThan(
        0.97,
      );
      expect(ratio).toBeLessThan(1.03);
      const wrongRatio = wrongAllLightPdf[c] / activeRef[c];
      expect(
        wrongRatio,
        `channel ${c}: all-light PDF would over-scale to ${wrongRatio.toFixed(4)}`,
      ).toBeGreaterThan(4 / 3 - 0.08);
      expect(wrongRatio).toBeLessThan(4 / 3 + 0.08);
    }
  });

  it('oracle sensitivity: the old missing-invPdf bug reads ≈1/K and ×K over-correction reads ≈K', () => {
    // Proves the oracle would CATCH the regression it guards against (and an
    // over-correction), i.e. the green result above is not vacuous.
    const buggy = emitTotalFlux(N, 99, 1).total; // missing lightSelectInvPdf
    const over = emitTotalFlux(N, 99, K * K).total; // double-applied invPdf
    for (const c of CH) {
      expect(buggy[c] / refTotal[c]).toBeGreaterThan(1 / K - 0.05);
      expect(buggy[c] / refTotal[c]).toBeLessThan(1 / K + 0.05);
      expect(over[c] / refTotal[c]).toBeGreaterThan(K - 0.5);
      expect(over[c] / refTotal[c]).toBeLessThan(K + 0.5);
    }
  });

  it('attenuates photon flux by Beer-Lambert transmittance inside transmissive media', () => {
    const flux: V3 = [10, 6, 2];
    const sigmaA: V3 = [0.1, 0.2, 0.3];
    const scatteringRgb: V3 = [0.02, 0.04, 0.01];
    const distance = 3.5;
    const out = attenuatePhotonFluxThroughMedium(flux, sigmaA, scatteringRgb, 0.05, distance);

    expect(out[0]).toBeCloseTo(10 * Math.exp(-0.15 * distance), 12);
    expect(out[1]).toBeCloseTo(6 * Math.exp(-0.25 * distance), 12);
    expect(out[2]).toBeCloseTo(2 * Math.exp(-0.35 * distance), 12);

    const noMedium = attenuatePhotonFluxThroughMedium(flux, [0, 0, 0], [0, 0, 0], 0, distance);
    expect(noMedium).toEqual(flux);
  });
});
