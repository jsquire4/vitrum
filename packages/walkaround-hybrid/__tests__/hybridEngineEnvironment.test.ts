/**
 * HybridEngine.updateEnvironment() tests (integration-depth review gap B2).
 *
 * The `@vitrum/core` Engine contract has an optional `updateEnvironment(env)`
 * for a runtime env / sky / IBL swap with no engine recreation. The
 * walkaround-hybrid realtime stack has NO IBL baker — its "environment" is the
 * diffuse sky-dome pair `skyTint` (RGB) + `skyIrradiance` (scalar) consumed by
 * the DDGI ProbeUpdate UBO + the shade pass. So `updateEnvironment` MAPS the
 * `SceneEnvironment` onto those scalars, caches it on `_lastScene.environment`,
 * invalidates the DDGI probe cache, and resets the temporal accumulator — the
 * env-map / sky-config sibling of `updateLighting` (see hybridEngineLighting.test).
 *
 * These tests cover:
 *   1. The method exists (capability/ledger agreement is pinned separately in
 *      promiseLedger.test.ts; here we assert it as a public function).
 *   2. `kind:'none'` zeroes `_skyIrradiance` (sky off) and leaves `_skyTint`.
 *   3. `kind:'hdri'` routes `intensity` → `_skyIrradiance` (HDRI is intensity-
 *      only on this baker-less backend) and leaves `_skyTint` (no opaque-ref
 *      sampling), with `intensity` defaulting to 1.
 *   4. The env is cached on `_lastScene.environment`; `null` collapses to
 *      `{ kind:'none' }`.
 *   5. DDGI probe cache invalidation (ddgi._frame→0, _ready→false) + temporal
 *      accumulator reset (pipeline.requestAccumReset).
 *   6. `procedural-sky` (outside supportedEnvironmentKinds) is best-effort:
 *      applies `intensity`, warns once.
 *   7. After dispose() the call is a safe no-op (does NOT touch the torn-down
 *      DDGI / pipeline) — matching the runtime-update siblings + the facade's
 *      'noop' disposed-behaviour for updateEnvironment.
 *
 * No real GPUDevice needed: the constructor only stores fields + instantiates
 * DDGI (no GPU work at construction). `_pipeline` starts null; the accum-reset
 * path is exercised via a mock pipeline injected into the private slot — the
 * same seam hybridEngineLighting.test.ts uses.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import * as THREE from 'three';
import type { Scene, SceneEnvironment } from '@vitrum/core';
import { HybridEngine } from '../src/HybridEngine.js';

// ── Minimal mock GPUDevice (constructor stores it; never called here) ────────

function makeMockDevice(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;
}

function makeEngine(opts: {
  skyTint?: [number, number, number];
  skyIrradiance?: number;
} = {}): HybridEngine {
  return new HybridEngine({
    device:                makeMockDevice(),
    width:                 64,
    height:                64,
    threeScene:            new THREE.Scene(),
    primaryLightDir:       [0, -1, 0],
    primaryLightIntensity: 1.0,
    skyTint:               opts.skyTint       ?? [0.5, 0.6, 1.0],
    skyIrradiance:         opts.skyIrradiance ?? 0.8,
  });
}

type EngineInternals = Record<string, unknown>;

// ─────────────────────────────────────────────────────────────────────────────
// Test 1 — method presence
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine.updateEnvironment — presence', () => {
  it('is exposed as a function (implements the optional Engine.updateEnvironment)', () => {
    const engine = makeEngine();
    try {
      expect(typeof engine.updateEnvironment).toBe('function');
    } finally {
      engine.dispose();
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test 2 — env → sky-scalar mapping
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine.updateEnvironment — sky-scalar mapping', () => {
  it("kind:'none' zeroes _skyIrradiance and leaves _skyTint", () => {
    const engine = makeEngine({ skyTint: [0.5, 0.6, 1.0], skyIrradiance: 0.8 });
    const e = engine as unknown as EngineInternals;
    try {
      engine.updateEnvironment({ kind: 'none' });
      expect(e['_skyIrradiance']).toBe(0);
      // Tint left intact — `none` carries no colour, and the baker-less stack
      // does not invent one.
      expect(e['_skyTint']).toEqual([0.5, 0.6, 1.0]);
    } finally {
      engine.dispose();
    }
  });

  it("kind:'hdri' routes intensity → _skyIrradiance and leaves _skyTint", () => {
    const engine = makeEngine({ skyTint: [0.2, 0.3, 0.4], skyIrradiance: 0.8 });
    const e = engine as unknown as EngineInternals;
    try {
      engine.updateEnvironment({ kind: 'hdri', hdri: {}, intensity: 2.5, rotationY: 1.3 });
      expect(e['_skyIrradiance']).toBe(2.5);
      // HDRI colour / rotation are NOT reflected (no opaque-ref sampling); the
      // host-supplied tint is preserved so an updateLighting({skyTint}) can pair.
      expect(e['_skyTint']).toEqual([0.2, 0.3, 0.4]);
    } finally {
      engine.dispose();
    }
  });

  it("kind:'hdri' defaults a missing intensity to 1", () => {
    const engine = makeEngine({ skyIrradiance: 0.8 });
    const e = engine as unknown as EngineInternals;
    try {
      engine.updateEnvironment({ kind: 'hdri', hdri: {} });
      expect(e['_skyIrradiance']).toBe(1);
    } finally {
      engine.dispose();
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test 3 — env cached on the live scene
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine.updateEnvironment — caches env on _lastScene', () => {
  it('records the new environment on the live scene record', () => {
    const engine = makeEngine();
    const e = engine as unknown as EngineInternals;
    try {
      // Seed a live scene (the engine ctor leaves _lastScene null).
      const scene: Scene = { primitives: [], emitters: [], environment: { kind: 'none' } };
      e['_lastScene'] = scene;

      const env: SceneEnvironment = { kind: 'hdri', hdri: {}, intensity: 3 };
      engine.updateEnvironment(env);

      const stored = (e['_lastScene'] as Scene).environment;
      expect(stored).toEqual(env);
    } finally {
      engine.dispose();
    }
  });

  it('null env collapses to { kind: "none" } on the live scene', () => {
    const engine = makeEngine();
    const e = engine as unknown as EngineInternals;
    try {
      e['_lastScene'] = {
        primitives: [],
        emitters: [],
        environment: { kind: 'hdri', hdri: {}, intensity: 2 },
      } as Scene;

      engine.updateEnvironment(null);

      expect((e['_lastScene'] as Scene).environment).toEqual({ kind: 'none' });
      // null/none ⇒ sky off.
      expect(e['_skyIrradiance']).toBe(0);
    } finally {
      engine.dispose();
    }
  });

  it('is safe (no throw) when no scene has been set yet', () => {
    const engine = makeEngine();
    try {
      // _lastScene is null here — the sky scalar still updates; no scene cache.
      expect(() => engine.updateEnvironment({ kind: 'none' })).not.toThrow();
      expect((engine as unknown as EngineInternals)['_lastScene']).toBeNull();
    } finally {
      engine.dispose();
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test 4 — DDGI invalidation + accumulator reset
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine.updateEnvironment — invalidation + reset', () => {
  it('resets ddgi._frame to 0 and _ready to false', () => {
    const engine = makeEngine();
    const e = engine as unknown as EngineInternals;
    try {
      const ddgi = e['_ddgi'] as Record<string, unknown>;
      ddgi['_frame'] = 16;
      ddgi['_ready'] = true;

      engine.updateEnvironment({ kind: 'hdri', hdri: {}, intensity: 1 });

      expect(ddgi['_frame']).toBe(0);
      expect(ddgi['_ready']).toBe(false);
    } finally {
      engine.dispose();
    }
  });

  it('calls requestAccumReset() on a live pipeline', () => {
    const engine = makeEngine();
    const e = engine as unknown as EngineInternals;
    try {
      // `dispose` is included so the finally-block teardown doesn't choke on
      // the injected mock pipeline (the production pipeline has it).
      const mockPipeline = { requestAccumReset: vi.fn(), dispose: vi.fn() };
      e['_pipeline'] = mockPipeline;

      engine.updateEnvironment({ kind: 'none' });

      expect(mockPipeline.requestAccumReset).toHaveBeenCalledOnce();
    } finally {
      engine.dispose();
    }
  });

  it('is safe when the pipeline is null (engine still initialising)', () => {
    const engine = makeEngine();
    try {
      // _pipeline starts null — must not throw.
      expect(() => engine.updateEnvironment({ kind: 'hdri', hdri: {}, intensity: 1 })).not.toThrow();
    } finally {
      engine.dispose();
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test 5 — procedural-sky (non-native kind) best-effort + warn
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine.updateEnvironment — non-native environment kind', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("applies procedural-sky intensity and warns once (kind outside supportedEnvironmentKinds)", () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = makeEngine({ skyIrradiance: 0.8 });
    const e = engine as unknown as EngineInternals;
    try {
      const env: SceneEnvironment = {
        kind: 'procedural-sky',
        sunDirection: [0, 1, 0],
        turbidity: 2,
        rayleigh: 1,
        mieCoefficient: 0.005,
        mieDirectionalG: 0.8,
        intensity: 1.7,
      };
      engine.updateEnvironment(env);

      expect(e['_skyIrradiance']).toBe(1.7);
      expect(warn).toHaveBeenCalledTimes(1);
      expect(String(warn.mock.calls[0]?.[0])).toContain('procedural-sky');

      // Second call does not re-warn (one-time per engine instance).
      engine.updateEnvironment(env);
      expect(warn).toHaveBeenCalledTimes(1);
    } finally {
      engine.dispose();
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test 6 — disposed no-op
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine.updateEnvironment — disposed no-op', () => {
  it('is a safe no-op after dispose (does not touch torn-down DDGI / pipeline)', () => {
    const engine = makeEngine({ skyIrradiance: 0.8 });
    const e = engine as unknown as EngineInternals;

    // Capture a spy on the DDGI invalidation BEFORE dispose tears it down, so we
    // can prove the disposed call never reaches it.
    const ddgi = e['_ddgi'] as { invalidateProbeCache: (...a: unknown[]) => unknown };
    const invalidateSpy = vi.spyOn(ddgi, 'invalidateProbeCache');

    engine.dispose();
    expect(engine.state).toBe('disposed');
    invalidateSpy.mockClear();

    // The call must not throw and must leave engine state untouched.
    expect(() => engine.updateEnvironment({ kind: 'hdri', hdri: {}, intensity: 9 })).not.toThrow();
    expect(invalidateSpy).not.toHaveBeenCalled();
    // Sky scalar unchanged (the disposed guard early-returns before the mapping).
    expect(e['_skyIrradiance']).toBe(0.8);
  });
});
