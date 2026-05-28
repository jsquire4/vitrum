/**
 * Theme T16 — coreEmittersToDDGILights radiometric-fidelity tests.
 *
 * These pin the lossless `@vitrum/core` SceneEmitter → DDGILight projection
 * that replaces the lossy THREE round-trip
 * (`vitrumSceneToThree` → `collectDDGILightsFromThreeRoot`) in
 * `HybridEngineLifecycle`. The round-trip dropped emitter chroma and used the
 * wrong area metric (`width·height` instead of `4·|uAxis × vAxis|`) for
 * rect-area emitters; the new mapper consumes the core emitter union directly,
 * preserving chroma + the true emissive area + the source emitter id.
 *
 * Each "corrected" assertion is paired with a computation of what the OLD
 * THREE-derived path would have produced, so the diff (white-vs-chroma,
 * π/area mismatch, dropped id) is demonstrated rather than merely asserted.
 */

import { describe, it, expect } from 'vitest';
import type {
  Scene,
  DirectionalEmitter,
  RectAreaEmitter,
  PointEmitter,
  DiscAreaEmitter,
  SpotEmitter,
  MeshAreaEmitter,
} from '@vitrum/core';
import { luminance } from '@vitrum/shared-samplers';
import {
  coreEmittersToDDGILights,
  coreEmitterToDDGILight,
} from '../src/coreEmittersToDDGILights.js';
// The OLD lossy path — exercised here purely to demonstrate the corrected
// radiometry differs from what the THREE round-trip produced.
import { vitrumSceneToThree } from '@vitrum/three-bindings';
import { collectDDGILightsFromThreeRoot } from '../src/HybridEngineLifecycle.js';
import type { DDGILight } from '../src/ddgi/types.js';

// ── Scene builder ─────────────────────────────────────────────────────────────

function sceneOf(...emitters: Scene['emitters'][number][]): Scene {
  return { primitives: [], emitters, environment: { kind: 'none' } };
}

// ── Fixtures ─────────────────────────────────────────────────────────────────

/** Red rect-area emitter with ORTHOGONAL half-axes (width=height=1 → area 1). */
const RED_RECT_ORTHO: RectAreaEmitter = {
  id: 'rect-red',
  kind: 'rect-area',
  color: [1, 0, 0],
  intensity: 5,
  position: [0, 2, 0],
  uAxis: [0.5, 0, 0],
  vAxis: [0, 0, 0.5],
};

/** Green rect-area emitter with NON-orthogonal half-axes — this is where the
 *  THREE round-trip's `width·height` area metric diverges from the true
 *  cross-product area `4·|uAxis × vAxis|`. */
const GREEN_RECT_SHEARED: RectAreaEmitter = {
  id: 'rect-green',
  kind: 'rect-area',
  color: [0, 1, 0],
  intensity: 3,
  position: [1, 1, 1],
  uAxis: [1, 0, 0],
  vAxis: [1, 1, 0], // not perpendicular to uAxis
};

const BLUE_POINT: PointEmitter = {
  id: 'point-blue',
  kind: 'point',
  color: [0.1, 0.2, 0.9],
  intensity: 7,
  position: [-2, 3, 1],
};

const SUN: DirectionalEmitter = {
  id: 'sun',
  kind: 'directional',
  color: [1, 0.95, 0.85],
  intensity: 4,
  direction: [0, -1, 0],
};

// ── Helper: what the OLD THREE round-trip produced for one emitter ──────────────

function oldThreePathLights(emitter: Scene['emitters'][number]): DDGILight[] {
  const threeRoot = vitrumSceneToThree(sceneOf(emitter));
  return collectDDGILightsFromThreeRoot(threeRoot);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1 — Mixed scene: directional + rect-area + point
// ─────────────────────────────────────────────────────────────────────────────

describe('coreEmittersToDDGILights — directional + rect + point mix', () => {
  const scene = sceneOf(SUN, RED_RECT_ORTHO, BLUE_POINT);
  const lights = coreEmittersToDDGILights(scene);

  it('emits exactly the rect + point as fixtures (directional excluded)', () => {
    // Directional is routed via setSunIntensityMultiplier + the BVH emitter
    // buffers, not as a DDGILight, so it must NOT appear here.
    expect(lights).toHaveLength(2);
    expect(lights.every((l) => l.kind === 'fixture')).toBe(true);
    expect(lights.map((l) => l.id)).toEqual(['rect-red', 'point-blue']);
    expect(lights.some((l) => l.id === 'sun')).toBe(false);
  });

  it('preserves each emitter id onto DDGILight.id', () => {
    const byId = new Map(lights.map((l) => [l.id, l]));
    expect(byId.has('rect-red')).toBe(true);
    expect(byId.has('point-blue')).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 2 — Rect-area: chroma preserved (old path dropped it → white)
// ─────────────────────────────────────────────────────────────────────────────

describe('coreEmittersToDDGILights — rect-area chroma fidelity', () => {
  it('preserves emitter chroma; old THREE path produced white', () => {
    const [light] = coreEmittersToDDGILights(sceneOf(RED_RECT_ORTHO));
    expect(light).toBeDefined();
    // New mapper: red chroma carried through.
    expect(light!.color).toEqual({ r: 1, g: 0, b: 0 });

    // Old path: collectDDGILightsFromThreeRoot built the fixture with NO
    // color, so the GPU packer defaulted it to white (1,1,1). Demonstrate the
    // dropped chroma directly.
    const [old] = oldThreePathLights(RED_RECT_ORTHO);
    expect(old).toBeDefined();
    expect(old!.color).toBeUndefined(); // → packer falls back to white
  });

  it('orthogonal-axis area matches the old path (area metric agrees here)', () => {
    // width = 2·|uAxis| = 1, height = 2·|vAxis| = 1, true area = 4·|u×v| = 1.
    // For orthogonal half-axes the two area metrics coincide, so the SCALAR
    // intensity is unchanged — only chroma differs. This isolates the chroma
    // bug from the area bug.
    const [light] = coreEmittersToDDGILights(sceneOf(RED_RECT_ORTHO));
    const expectedArea = 1; // 4·|(0.5,0,0)×(0,0,0.5)| = 4·0.25
    expect(light!.intensity).toBeCloseTo(RED_RECT_ORTHO.intensity * expectedArea, 6);

    const [old] = oldThreePathLights(RED_RECT_ORTHO);
    // Old path: light.intensity · (width·height) = 5 · (1·1) = 5.
    expect(old!.intensity).toBeCloseTo(5, 6);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 3 — Rect-area: true cross-product area (old path used width·height)
// ─────────────────────────────────────────────────────────────────────────────

describe('coreEmittersToDDGILights — rect-area true-area fidelity', () => {
  it('uses 4·|uAxis × vAxis| for sheared axes; old path used width·height', () => {
    const [light] = coreEmittersToDDGILights(sceneOf(GREEN_RECT_SHEARED));
    expect(light).toBeDefined();

    // True area: u×v = (1,0,0)×(1,1,0) = (0,0,1) → |u×v| = 1 → area = 4·1 = 4.
    const trueArea = 4;
    expect(light!.intensity).toBeCloseTo(GREEN_RECT_SHEARED.intensity * trueArea, 6);
    expect(light!.color).toEqual({ r: 0, g: 1, b: 0 });

    // Old THREE path: width = 2·|u| = 2, height = 2·|v| = 2·√2,
    // area = width·height = 4·√2 ≈ 5.657 — a √2 over-statement of the true
    // emissive area for these sheared half-axes.
    const [old] = oldThreePathLights(GREEN_RECT_SHEARED);
    const threeArea = 4 * Math.SQRT2;
    expect(old!.intensity).toBeCloseTo(GREEN_RECT_SHEARED.intensity * threeArea, 4);

    // The corrected magnitude differs from the lossy one by exactly the
    // area-metric ratio (√2 here) — the "factor-of-π"/area error the old
    // comment admitted to.
    const ratio = old!.intensity / light!.intensity;
    expect(ratio).toBeCloseTo(Math.SQRT2, 4);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 4 — Point emitter: position/color/intensity + id preservation
// ─────────────────────────────────────────────────────────────────────────────

describe('coreEmittersToDDGILights — point emitter', () => {
  it('maps to a fixture at the emitter position with chroma + bare intensity', () => {
    const [light] = coreEmittersToDDGILights(sceneOf(BLUE_POINT));
    expect(light).toMatchObject({
      kind: 'fixture',
      id: 'point-blue',
      on: true,
      intensity: 7,
      position: { x: -2, y: 3, z: 1 },
      color: { r: 0.1, g: 0.2, b: 0.9 },
    });
  });

  it('preserves the id that the old THREE path discarded', () => {
    const [neu] = coreEmittersToDDGILights(sceneOf(BLUE_POINT));
    expect(neu!.id).toBe('point-blue');

    // Old THREE path: collectDDGIPointLightsFromRoot built the fixture with
    // no id field at all.
    const [old] = oldThreePathLights(BLUE_POINT);
    expect(old).toBeDefined();
    expect(old!.id).toBeUndefined();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 5 — disc-area uses π·r² area; spot maps to a point-like fixture
// ─────────────────────────────────────────────────────────────────────────────

describe('coreEmittersToDDGILights — disc-area + spot', () => {
  it('disc-area carries flux-equivalent intensity = intensity · π·r²', () => {
    const disc: DiscAreaEmitter = {
      id: 'disc-1',
      kind: 'disc-area',
      color: [1, 1, 1],
      intensity: 2,
      position: [0, 5, 0],
      normal: [0, -1, 0],
      radius: 0.5,
    };
    const [light] = coreEmittersToDDGILights(sceneOf(disc));
    const area = Math.PI * 0.5 * 0.5;
    expect(light!.kind).toBe('fixture');
    expect(light!.id).toBe('disc-1');
    expect(light!.intensity).toBeCloseTo(2 * area, 6);
    expect(light!.color).toEqual({ r: 1, g: 1, b: 1 });
  });

  it('spot maps to a point-like fixture (cone dropped — low-freq indirect only)', () => {
    const spot: SpotEmitter = {
      id: 'spot-1',
      kind: 'spot',
      color: [0.8, 0.8, 1],
      intensity: 9,
      position: [3, 4, 5],
      direction: [0, -1, 0],
      angle: 0.5,
    };
    const [light] = coreEmittersToDDGILights(sceneOf(spot));
    expect(light).toMatchObject({
      kind: 'fixture',
      id: 'spot-1',
      intensity: 9,
      position: { x: 3, y: 4, z: 5 },
      color: { r: 0.8, g: 0.8, b: 1 },
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 6 — Excluded kinds + degenerate emitters
// ─────────────────────────────────────────────────────────────────────────────

describe('coreEmittersToDDGILights — exclusions & degenerate guards', () => {
  it('returns null for directional and mesh-area emitters', () => {
    expect(coreEmitterToDDGILight(SUN)).toBeNull();
    const meshArea: MeshAreaEmitter = {
      id: 'mesh-emit',
      kind: 'mesh-area',
      color: [1, 1, 1],
      intensity: 1,
      meshId: 'panel-0',
    };
    expect(coreEmitterToDDGILight(meshArea)).toBeNull();
  });

  it('skips a degenerate rect-area emitter (parallel half-axes → zero area)', () => {
    const degenerate: RectAreaEmitter = {
      id: 'rect-degenerate',
      kind: 'rect-area',
      color: [1, 1, 1],
      intensity: 10,
      position: [0, 0, 0],
      uAxis: [1, 0, 0],
      vAxis: [2, 0, 0], // parallel to uAxis → cross product is zero
    };
    expect(coreEmitterToDDGILight(degenerate)).toBeNull();
    expect(coreEmittersToDDGILights(sceneOf(degenerate))).toHaveLength(0);
  });

  it('a scene of only excluded emitters yields an empty light list', () => {
    expect(coreEmittersToDDGILights(sceneOf(SUN))).toHaveLength(0);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 7 — Radiometric sanity: emitted radiance product (color·intensity) is what
//     the probe shader integrates; confirm chroma+magnitude both survive.
// ─────────────────────────────────────────────────────────────────────────────

describe('coreEmittersToDDGILights — emitted-radiance product survives', () => {
  it('rect-area: color·intensity reproduces emitter Le·area per channel', () => {
    const [light] = coreEmittersToDDGILights(sceneOf(RED_RECT_ORTHO));
    // The probe shader computes color · intensity (· falloff · nDotL). The
    // per-channel emitted magnitude is therefore color·intensity = Le·area.
    const area = 1;
    const r = light!.color!.r * light!.intensity;
    const g = light!.color!.g * light!.intensity;
    const b = light!.color!.b * light!.intensity;
    expect(r).toBeCloseTo(RED_RECT_ORTHO.color[0] * RED_RECT_ORTHO.intensity * area, 6);
    expect(g).toBeCloseTo(RED_RECT_ORTHO.color[1] * RED_RECT_ORTHO.intensity * area, 6);
    expect(b).toBeCloseTo(RED_RECT_ORTHO.color[2] * RED_RECT_ORTHO.intensity * area, 6);

    // The old THREE path, by contrast, multiplied white × scalar — its
    // per-channel product would be (1,1,1)·5 = (5,5,5), i.e. the red and blue
    // channels are WRONG (should be (5,0,0)). luminance() confirms the
    // perceived magnitude also differs.
    const [old] = oldThreePathLights(RED_RECT_ORTHO);
    const oldR = (old!.color?.r ?? 1) * old!.intensity; // white default
    expect(oldR).toBeCloseTo(5, 6); // wrong: should be Le_r·area = 5 here but G/B leak
    const oldLum = luminance(old!.color?.r ?? 1, old!.color?.g ?? 1, old!.color?.b ?? 1) * old!.intensity;
    const newLum = luminance(light!.color!.r, light!.color!.g, light!.color!.b) * light!.intensity;
    expect(oldLum).not.toBeCloseTo(newLum, 2);
  });
});
