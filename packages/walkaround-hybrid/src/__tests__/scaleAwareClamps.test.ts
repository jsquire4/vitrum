/**
 * B15 — scene-scale-aware radiometric clamp defaults.
 *
 * Pins the scaling law + its three invariants:
 *   1. Cornell-scale byte-identity (diagonal ≈ CORNELL_DIAGONAL ⇒ defaults
 *      unchanged).
 *   2. The dimensional law: radiance/irradiance clamps ×1/s², the squared-
 *      distance floor ×s², restirGiWCap NOT scaled.
 *   3. Host overrides are absolute (never scaled).
 */
import { describe, expect, it } from 'vitest';
import type { Scene, ScenePrimitive, Mat4 } from '@vitrum/core';
import {
  CORNELL_DIAGONAL,
  deriveScaleAwareClamps,
  sceneWorldDiagonal,
  type ScaleAwareHostExplicit,
} from '../HybridEngineScaleAwareClamps.js';
import type { Tunables } from '../HybridEngineTuning.js';

const BASE: Tunables = {
  emitterDist2Floor: 0.01,
  directFireflyClamp: 4.0,
  causticBoost: 1.0,
  causticVisClamp: 1.0,
  temporalMClampDI: 20,
  spatialReuseRadiusPx: 30,
  spatialDepthTolFloor: 0.05,
  gtaoRadiusPx: 32,
  gtaoIntensity: 2.0,
  gtaoDepthThreshold: 2.0,
  gtaoBilateralDepthSigma: 0.25,
  adaptiveSamplingThresholdLow: 0.01,
  adaptiveSamplingThresholdHigh: 0.1,
  triIntersectEpsilon: 1e-5,
  glassMixScale: 0.7,
  restirGiWCap: 16.0,
  restirGiIrrClamp: 5.0,
  restirGiMClamp: 50,
  restirGiSpatialRadiusPx: 12.0,
  restirGiSpatialNormalDotMin: 0.906,
  restirGiSpatialCoplanarTol: 0.05,
};
const BASE_INDIRECT: readonly [number, number, number] = [1, 1, 1];

const NO_HOST_OVERRIDES: ScaleAwareHostExplicit = {
  restirGiIrrClamp: false,
  directFireflyClamp: false,
  emitterDist2Floor: false,
  spatialDepthTolFloor: false,
  gtaoDepthThreshold: false,
  gtaoBilateralDepthSigma: false,
  restirGiSpatialCoplanarTol: false,
  indirectFireflyClamp: false,
};

/** Axis-aligned box centred at origin with full extent `size` on each axis. */
function boxScene(extent: number, transform?: Mat4): Scene {
  const h = extent / 2;
  const positions = new Float32Array([
    -h, -h, -h, h, -h, -h, h, h, -h,
    -h, -h, h, h, -h, h, h, h, h,
  ]);
  const prim = {
    kind: 'mesh',
    id: 'b',
    positions,
    normals: new Float32Array(positions.length),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 1, metallic: 0 },
    ...(transform ? { transform } : {}),
  } as unknown as ScenePrimitive;
  return { primitives: [prim], emitters: [], environment: { kind: 'none' } };
}

/** A ±1 Cornell-equivalent box: full extent 2 on each axis ⇒ diagonal 2√3. */
function cornellScene(): Scene {
  return boxScene(2);
}

function inputs(hostExplicit = NO_HOST_OVERRIDES) {
  return { baseTunables: BASE, baseIndirectFireflyClamp: BASE_INDIRECT, hostExplicit };
}

describe('B15 sceneWorldDiagonal', () => {
  it('Cornell ±1 box ⇒ 2√3', () => {
    expect(sceneWorldDiagonal(cornellScene())).toBeCloseTo(CORNELL_DIAGONAL, 6);
  });

  it('empty / null scene ⇒ Cornell diagonal (defaults unchanged)', () => {
    expect(sceneWorldDiagonal(null)).toBe(CORNELL_DIAGONAL);
    expect(sceneWorldDiagonal({ primitives: [], emitters: [], environment: { kind: 'none' } }))
      .toBe(CORNELL_DIAGONAL);
  });

  it('a ×100 box has ×100 diagonal', () => {
    expect(sceneWorldDiagonal(boxScene(200))).toBeCloseTo(CORNELL_DIAGONAL * 100, 4);
  });

  it('retains every finite positive extent below the former 1e-6 floor', () => {
    const diagonal = sceneWorldDiagonal(boxScene(1e-9));
    expect(diagonal).toBeGreaterThan(0);
    expect(diagonal).toBeLessThan(1e-8);
    expect(diagonal).not.toBe(CORNELL_DIAGONAL);
  });

  it('is transform-aware (translation does not change extent; scale does)', () => {
    // Column-major translate by (1000,0,0): extent unchanged ⇒ same diagonal.
    const translate = new Float32Array([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1000, 0, 0, 1,
    ]) as unknown as Mat4;
    expect(sceneWorldDiagonal(boxScene(2, translate))).toBeCloseTo(CORNELL_DIAGONAL, 4);
    // Column-major uniform scale ×5 ⇒ ×5 diagonal.
    const scale5 = new Float32Array([
      5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 1,
    ]) as unknown as Mat4;
    expect(sceneWorldDiagonal(boxScene(2, scale5))).toBeCloseTo(CORNELL_DIAGONAL * 5, 4);
  });
});

describe('B15 deriveScaleAwareClamps — Cornell byte-identity', () => {
  it('Cornell-scale scene ⇒ defaults unchanged (byte-identical)', () => {
    const r = deriveScaleAwareClamps(cornellScene(), inputs());
    expect(r.scaleRatio).toBeCloseTo(1, 6);
    expect(r.tunables.restirGiIrrClamp).toBe(5.0);
    expect(r.tunables.directFireflyClamp).toBe(4.0);
    expect(r.tunables.emitterDist2Floor).toBe(0.01);
    expect(r.tunables.restirGiWCap).toBe(16.0);
    expect(r.indirectFireflyClamp).toEqual([1, 1, 1]);
    expect(r.rayOriginBias).toBeCloseTo(1e-3, 12);
  });

  it('empty scene ⇒ Cornell defaults (no scene yet)', () => {
    const r = deriveScaleAwareClamps(null, inputs());
    expect(r.tunables.restirGiIrrClamp).toBe(5.0);
    expect(r.tunables.emitterDist2Floor).toBe(0.01);
  });
});

describe('B15 deriveScaleAwareClamps — the dimensional law', () => {
  it('×100 scene: radiance/irradiance clamps ×1/s², dist² floor ×s², WCap unchanged', () => {
    const r = deriveScaleAwareClamps(boxScene(200), inputs());
    const s = r.scaleRatio;
    expect(s).toBeCloseTo(100, 3);
    // ×1/s²
    expect(r.tunables.restirGiIrrClamp).toBeCloseTo(5.0 / (s * s), 6);
    expect(r.tunables.directFireflyClamp).toBeCloseTo(4.0 / (s * s), 6);
    expect(r.indirectFireflyClamp[0]).toBeCloseTo(1.0 / (s * s), 6);
    // ×s²
    expect(r.tunables.emitterDist2Floor).toBeCloseTo(0.01 * s * s, 4);
    // world-length tolerances ×s
    expect(r.tunables.spatialDepthTolFloor).toBeCloseTo(0.05 * s, 6);
    expect(r.tunables.gtaoDepthThreshold).toBeCloseTo(2.0 * s, 6);
    expect(r.tunables.gtaoBilateralDepthSigma).toBeCloseTo(0.25 * s, 6);
    expect(r.tunables.restirGiSpatialCoplanarTol).toBeCloseTo(0.05 * s, 6);
    expect(r.rayOriginBias).toBeCloseTo(1e-3 * s, 9);
    // unitless variance cap — unchanged
    expect(r.tunables.restirGiWCap).toBe(16.0);
    // unrelated knobs unchanged
    expect(r.tunables.gtaoRadiusPx).toBe(32);
    expect(r.tunables.temporalMClampDI).toBe(20);
  });

  it('small scene (×0.5): clamps ×4 (1/s²), floor ÷4 (s²)', () => {
    const r = deriveScaleAwareClamps(boxScene(1), inputs());
    expect(r.scaleRatio).toBeCloseTo(0.5, 6);
    expect(r.tunables.restirGiIrrClamp).toBeCloseTo(5.0 / 0.25, 6);
    expect(r.tunables.emitterDist2Floor).toBeCloseTo(0.01 * 0.25, 6);
    expect(r.rayOriginBias).toBeCloseTo(5e-4, 12);
  });

  it('keeps the secondary-ray offset proportional below the former absolute floor', () => {
    const r = deriveScaleAwareClamps(boxScene(1e-9), inputs());
    expect(r.rayOriginBias).toBeGreaterThan(0);
    expect(r.rayOriginBias).toBeLessThan(1e-9);
    expect(r.rayOriginBias).toBeCloseTo(5e-13, 18);
  });
});

describe('B15 deriveScaleAwareClamps — host overrides are absolute', () => {
  it('a host-explicit knob is NEVER scaled, others still scale', () => {
    const hostExplicit: ScaleAwareHostExplicit = {
      restirGiIrrClamp: true,         // host set this one
      directFireflyClamp: false,
      emitterDist2Floor: false,
      spatialDepthTolFloor: true,
      gtaoDepthThreshold: true,
      gtaoBilateralDepthSigma: true,
      restirGiSpatialCoplanarTol: true,
      indirectFireflyClamp: true,      // host set this one too
    };
    const r = deriveScaleAwareClamps(boxScene(200), inputs(hostExplicit));
    const s = r.scaleRatio;
    // host-explicit ⇒ verbatim baseline
    expect(r.tunables.restirGiIrrClamp).toBe(5.0);
    expect(r.tunables.spatialDepthTolFloor).toBe(0.05);
    expect(r.tunables.gtaoDepthThreshold).toBe(2.0);
    expect(r.tunables.gtaoBilateralDepthSigma).toBe(0.25);
    expect(r.tunables.restirGiSpatialCoplanarTol).toBe(0.05);
    expect(r.indirectFireflyClamp).toEqual([1, 1, 1]);
    // non-explicit ⇒ scaled
    expect(r.tunables.directFireflyClamp).toBeCloseTo(4.0 / (s * s), 6);
    expect(r.tunables.emitterDist2Floor).toBeCloseTo(0.01 * s * s, 4);
  });
});
