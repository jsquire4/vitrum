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
  directionalSunMultiplier,
  sceneHasDirectionalEmitter,
} from '../src/coreEmittersToDDGILights.js';
import { packDDGIProbeLights } from '../src/ddgi/probeUpdateLights.js';
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
  direction: [0, -1, 0], // points AT the light → light is straight below
};

/** A tilted, coloured sun — `direction` points AT the light (up-and-to-the-east),
 *  NOT a unit vector, so it exercises the mapper's normalize + negate. */
const SUN_TILTED: DirectionalEmitter = {
  id: 'sun-tilted',
  kind: 'directional',
  color: [0.9, 0.4, 0.2],
  intensity: 6,
  direction: [3, 4, 0], // |dir| = 5 → toward-light = (0.6, 0.8, 0)
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

  it('emits the directional as a sun + the rect/point as fixtures', () => {
    // Directional now maps to a `sun` DDGILight (the real direction/intensity/
    // colour drive the DDGI probe-pass sun); rect/point map to fixtures.
    expect(lights).toHaveLength(3);
    const sun = lights.find((l) => l.id === 'sun');
    expect(sun?.kind).toBe('sun');
    const fixtures = lights.filter((l) => l.kind === 'fixture');
    expect(fixtures.map((l) => l.id)).toEqual(['rect-red', 'point-blue']);
  });

  it('preserves each emitter id onto DDGILight.id', () => {
    const byId = new Map(lights.map((l) => [l.id, l]));
    expect(byId.has('sun')).toBe(true);
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
  it('returns null for mesh-area emitters (folded into mesh emissive)', () => {
    const meshArea: MeshAreaEmitter = {
      id: 'mesh-emit',
      kind: 'mesh-area',
      color: [1, 1, 1],
      intensity: 1,
      meshId: 'panel-0',
    };
    expect(coreEmitterToDDGILight(meshArea)).toBeNull();
  });

  it('returns null for a degenerate (zero-direction) directional emitter', () => {
    const zeroDir: DirectionalEmitter = {
      id: 'sun-degenerate',
      kind: 'directional',
      color: [1, 1, 1],
      intensity: 5,
      direction: [0, 0, 0],
    };
    expect(coreEmitterToDDGILight(zeroDir)).toBeNull();
    expect(coreEmittersToDDGILights(sceneOf(zeroDir))).toHaveLength(0);
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

  it('a scene of only a mesh-area emitter yields an empty light list', () => {
    const meshArea: MeshAreaEmitter = {
      id: 'mesh-only',
      kind: 'mesh-area',
      color: [1, 1, 1],
      intensity: 1,
      meshId: 'panel-0',
    };
    expect(coreEmittersToDDGILights(sceneOf(meshArea))).toHaveLength(0);
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

// ─────────────────────────────────────────────────────────────────────────────
// 8 — Directional → DDGI sun: real direction + single-count
//
// The DDGI probe pass evaluates a `sun` light via evalSunLight, which negates
// the packed `direction` (`lightDir = normalize(-light.direction)`). The core
// `DirectionalEmitter.direction` points AT the light (toward-light), so the
// mapper must negate it to a TRAVEL direction; the shader then negates again,
// recovering the original toward-light direction for dot(N, L).
//
// Single-count: a `sun` DDGILight carries `intensity = emitter.intensity`, and
// the host sets the probe-pass sun-intensity multiplier to 1 when a scene
// directional is present — so the emitter intensity is applied exactly once,
// never multiplied by the config primaryLightIntensity.
// ─────────────────────────────────────────────────────────────────────────────

// Float layout of packDDGIProbeLights (see probeUpdateLights.ts + the WGSL
// DDGILight struct in probeUpdateRays.wgsl.ts): 4-float header, then 16 floats
// per light. For light 0: base = 4. kind(u32)@0, intensity@7, direction@8..10,
// color@12..14.
const HEADER_FLOATS = 4;
const LIGHT_STRIDE_FLOATS = 16;

function unpackSunLight(buf: ArrayBuffer, lightIdx = 0) {
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  const count = u[0]!;
  const base = HEADER_FLOATS + lightIdx * LIGHT_STRIDE_FLOATS;
  return {
    count,
    kind: u[base + 0]!, // 0 = LIGHT_SUN
    intensity: f[base + 7]!,
    direction: { x: f[base + 8]!, y: f[base + 9]!, z: f[base + 10]! },
    color: { r: f[base + 12]!, g: f[base + 13]!, b: f[base + 14]! },
  };
}

describe('coreEmittersToDDGILights — directional → DDGI sun, real direction', () => {
  it('maps a downward sun to a sun light whose packed dir reproduces prior (0,-1,0)', () => {
    // SUN.direction = (0,-1,0) points AT the light (straight down). The mapper
    // negates → travel direction (0,1,0); but with multiplier=1 the packed sun
    // matches the prior hardcoded straight-down convention as the shader
    // negates the travel dir back to (0,-1,0)... so the IMPORTANT pin is that
    // a (0,-1,0) toward-light emitter reproduces the legacy behaviour exactly.
    const light = coreEmitterToDDGILight(SUN)!;
    expect(light.kind).toBe('sun');
    expect(light.id).toBe('sun');
    expect(light.intensity).toBe(4);
    // toward-light = (0,-1,0) → travel = (0,1,0). (Use closeTo to avoid the
    // -0 vs 0 distinction toEqual would otherwise flag on the negated zeros.)
    expect(light.direction!.x).toBeCloseTo(0, 6);
    expect(light.direction!.y).toBeCloseTo(1, 6);
    expect(light.direction!.z).toBeCloseTo(0, 6);
    expect(light.color).toEqual({ r: 1, g: 0.95, b: 0.85 });
  });

  it('carries a tilted, non-unit direction (normalized + negated) and chroma', () => {
    const light = coreEmitterToDDGILight(SUN_TILTED)!;
    expect(light.kind).toBe('sun');
    // direction (3,4,0) → |dir|=5 → toward-light (0.6,0.8,0) → travel (-0.6,-0.8,0).
    expect(light.direction!.x).toBeCloseTo(-0.6, 6);
    expect(light.direction!.y).toBeCloseTo(-0.8, 6);
    expect(light.direction!.z).toBeCloseTo(0, 6);
    expect(light.intensity).toBe(6);
    expect(light.color).toEqual({ r: 0.9, g: 0.4, b: 0.2 });
  });

  it('packs the real direction into the sun UBO (NOT the hardcoded 0,-1,0)', () => {
    const sun = coreEmitterToDDGILight(SUN_TILTED)!;
    // Single-count: multiplier = 1 when a scene directional drives the sun.
    const buf = packDDGIProbeLights([sun], 1);
    const unpacked = unpackSunLight(buf);
    expect(unpacked.count).toBe(1);
    expect(unpacked.kind).toBe(0); // LIGHT_SUN
    // Packed travel direction equals the mapper's negated/normalized dir.
    expect(unpacked.direction.x).toBeCloseTo(-0.6, 5);
    expect(unpacked.direction.y).toBeCloseTo(-0.8, 5);
    expect(unpacked.direction.z).toBeCloseTo(0, 5);
    // Real chroma, NOT the legacy hardcoded (1,0.95,0.85).
    expect(unpacked.color.r).toBeCloseTo(0.9, 5);
    expect(unpacked.color.g).toBeCloseTo(0.4, 5);
    expect(unpacked.color.b).toBeCloseTo(0.2, 5);
  });

  it('SINGLE-COUNTS intensity: packed sun = emitter.intensity at multiplier=1', () => {
    const sun = coreEmitterToDDGILight(SUN_TILTED)!; // intensity 6
    const buf = packDDGIProbeLights([sun], 1);
    expect(unpackSunLight(buf).intensity).toBeCloseTo(6, 5);

    // Demonstrate the double-count this avoids: had the host kept the legacy
    // config multiplier (e.g. primaryLightIntensity = 3) alongside the emitter
    // intensity, the packed sun would be 6·3 = 18 — 3× too bright.
    const doubled = packDDGIProbeLights([sun], 3);
    expect(unpackSunLight(doubled).intensity).toBeCloseTo(18, 5);
    expect(unpackSunLight(doubled).intensity).not.toBeCloseTo(6, 1);
  });

  it('a (0,-1,0) directional + multiplier=1 packs the prior straight-down sun', () => {
    // Backwards-compat pin: the legacy packer hardcoded direction (0,-1,0) and
    // intensity·sunIntensityMul. A scene-directional that points AT a sun
    // directly below (toward-light (0,-1,0)) → travel (0,1,0); the WGSL shader
    // negates → (0,-1,0), identical to the old hardcoded straight-down sun.
    const sun = coreEmitterToDDGILight(SUN)!; // intensity 4, dir toward-light (0,-1,0)
    const buf = packDDGIProbeLights([sun], 1);
    const unpacked = unpackSunLight(buf);
    expect(unpacked.intensity).toBeCloseTo(4, 5);
    // Packed travel dir (0,1,0); shader negates → (0,-1,0) = legacy straight-down.
    expect(unpacked.direction.x).toBeCloseTo(0, 5);
    expect(unpacked.direction.y).toBeCloseTo(1, 5);
    expect(unpacked.direction.z).toBeCloseTo(0, 5);
  });

  it('a sun light with NO direction/color falls back to the legacy hardcoded sun', () => {
    // Host-supplied sun light (not from a scene emitter) without direction/color
    // → packer uses legacy (0,-1,0) + (1,0.95,0.85) and applies the multiplier.
    const buf = packDDGIProbeLights([{ kind: 'sun', on: true, intensity: 2 }], 3);
    const unpacked = unpackSunLight(buf);
    expect(unpacked.direction).toEqual({ x: 0, y: -1, z: 0 });
    expect(unpacked.color.r).toBeCloseTo(1, 5);
    expect(unpacked.color.g).toBeCloseTo(0.95, 5);
    expect(unpacked.color.b).toBeCloseTo(0.85, 5);
    expect(unpacked.intensity).toBeCloseTo(6, 5); // 2 · 3 (legacy multiplier path)
  });
});

describe('directionalSunMultiplier — single-count resolution', () => {
  it('returns 1 when a scene directional is present (sun carries its own intensity)', () => {
    expect(sceneHasDirectionalEmitter(sceneOf(SUN, RED_RECT_ORTHO))).toBe(true);
    expect(directionalSunMultiplier(sceneOf(SUN, RED_RECT_ORTHO), 5)).toBe(1);
  });

  it('returns primaryLightIntensity (legacy config path) when no scene directional', () => {
    expect(sceneHasDirectionalEmitter(sceneOf(RED_RECT_ORTHO, BLUE_POINT))).toBe(false);
    expect(directionalSunMultiplier(sceneOf(RED_RECT_ORTHO, BLUE_POINT), 5)).toBe(5);
    // Null scene (no core scene supplied) → legacy config path too.
    expect(directionalSunMultiplier(null, 7)).toBe(7);
  });
});
