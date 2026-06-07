/**
 * SG.C — HybridEngine.updateLighting() tests (stainedGlass audit Gap 1).
 *
 * Tests cover the five behaviours specified in the implementation plan:
 *
 *   1. `_primaryLightDir` field updates after updateLighting({ primaryLightDir }).
 *   2. Temporal-accumulator reset flag is set (pipeline._accumFrameIndex === 0).
 *   3. DDGI atlas-clear flag is set (ddgi._ready === false, ddgi._frame === 0).
 *   4. A partial update (only primaryLightIntensity) does not disturb other
 *      lighting fields.
 *   5. updateLighting({}) is a no-op — DDGI and accumulator are untouched.
 *
 * No real GPUDevice needed: HybridEngine's constructor only stores fields and
 * instantiates DDGI (which does no GPU work at construction time). The pipeline
 * (_pipeline) starts null; the temporal-reset path is exercised via the mock
 * pipeline injected into the engine's private slot.
 *
 * Private field access uses `as any` — acceptable for internal-state
 * verification tests in this codebase (see sprint18-indirectCombine.test.ts,
 * sprint9-10a-welford.test.ts for the same pattern).
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import * as THREE from 'three';
import { HybridEngine } from '../src/HybridEngine.js';
import type { DDGILight } from '../src/ddgi/types.js';
import { LIGHTING_OPTION_KEYS, type LightingOptions } from '../src/HybridEngineOptions.js';

// ── Minimal mock GPUDevice (constructor stores the value; never called here) ─

function makeMockDevice(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;
}

// ── Minimal mock THREE.Scene ───────────────────────────────────────────────────

function makeMockScene(): THREE.Scene {
  const scene = new THREE.Scene();
  return scene;
}

// ── Factory: minimal HybridEngine with lighting parameters ────────────────────

function makeEngine(opts: {
  primaryLightDir?: [number, number, number];
  primaryLightIntensity?: number;
  skyTint?: [number, number, number];
  skyIrradiance?: number;
  lights?: DDGILight[];
} = {}): HybridEngine {
  return new HybridEngine({
    device:                makeMockDevice(),
    width:                 64,
    height:                64,
    threeScene:            makeMockScene(),
    primaryLightDir:       opts.primaryLightDir       ?? [0, -1, 0],
    primaryLightIntensity: opts.primaryLightIntensity ?? 1.0,
    skyTint:               opts.skyTint               ?? [0.5, 0.6, 1.0],
    skyIrradiance:         opts.skyIrradiance         ?? 0.8,
    lights:                opts.lights                ?? [],
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1 — _primaryLightDir updates
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine.updateLighting — field updates', () => {
  it('updates _primaryLightDir when provided', () => {
    const engine = makeEngine({ primaryLightDir: [0, -1, 0] });
    const e = engine as unknown as Record<string, unknown>;

    expect(e['_primaryLightDir']).toEqual([0, -1, 0]);

    engine.updateLighting({ primaryLightDir: [0.5, -0.866, 0] });

    expect(e['_primaryLightDir']).toEqual([0.5, -0.866, 0]);
  });

  it('re-orients DDGI sun lights when primaryLightDir changes', () => {
    const engine = makeEngine({
      primaryLightDir: [0, -1, 0],
      lights: [{
        kind: 'sun',
        intensity: 2,
        on: true,
        direction: { x: 0, y: -1, z: 0 },
      }],
    });
    const ddgi = (engine as unknown as Record<string, unknown>)['_ddgi'] as {
      setLights: (lights: DDGILight[]) => void;
    };
    const setLights = vi.spyOn(ddgi, 'setLights');

    engine.updateLighting({ primaryLightDir: [3, 4, 0] });

    expect(setLights).toHaveBeenCalled();
    const republished = setLights.mock.calls.at(-1)?.[0] as DDGILight[] | undefined;
    const sun = republished?.find((light) => light.kind === 'sun');
    expect(sun?.direction?.x).toBeCloseTo(-0.6);
    expect(sun?.direction?.y).toBeCloseTo(-0.8);
    expect(sun?.direction?.z).toBeCloseTo(0);
  });

  it('updates _primaryLightIntensity when provided', () => {
    const engine = makeEngine({ primaryLightIntensity: 1.0 });
    const e = engine as unknown as Record<string, unknown>;

    engine.updateLighting({ primaryLightIntensity: 3.5 });

    expect(e['_primaryLightIntensity']).toBe(3.5);
  });

  it('updates _skyTint when provided', () => {
    const engine = makeEngine({ skyTint: [0.5, 0.6, 1.0] });
    const e = engine as unknown as Record<string, unknown>;

    engine.updateLighting({ skyTint: [0.8, 0.85, 1.0] });

    expect(e['_skyTint']).toEqual([0.8, 0.85, 1.0]);
  });

  it('updates _skyIrradiance when provided', () => {
    const engine = makeEngine({ skyIrradiance: 0.8 });
    const e = engine as unknown as Record<string, unknown>;

    engine.updateLighting({ skyIrradiance: 2.0 });

    expect(e['_skyIrradiance']).toBe(2.0);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test 2 — Temporal-accumulator reset flag
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine.updateLighting — temporal accumulator reset', () => {
  it('calls requestAccumReset() on the pipeline when one is live', () => {
    const engine = makeEngine();
    const e = engine as unknown as Record<string, unknown>;

    // Inject a mock pipeline with a spy on requestAccumReset.
    const mockPipeline = { requestAccumReset: vi.fn() };
    e['_pipeline'] = mockPipeline;

    engine.updateLighting({ primaryLightDir: [0, -1, 0] });

    expect(mockPipeline.requestAccumReset).toHaveBeenCalledOnce();
  });

  it('is safe when pipeline is null (engine still initialising)', () => {
    const engine = makeEngine();
    // _pipeline starts null — updateLighting must not throw.
    expect(() => engine.updateLighting({ skyIrradiance: 1.5 })).not.toThrow();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test 3 — DDGI atlas-clear flag
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine.updateLighting — DDGI probe cache invalidation', () => {
  it('resets ddgi._frame to 0 and _ready to false after updateLighting', () => {
    const engine = makeEngine();
    const e = engine as unknown as Record<string, unknown>;

    // Simulate a DDGI that has already converged.
    const ddgi = e['_ddgi'] as Record<string, unknown>;
    ddgi['_frame'] = 16;
    ddgi['_ready'] = true;

    engine.updateLighting({ primaryLightDir: [1, -1, 0] });

    expect(ddgi['_frame']).toBe(0);
    expect(ddgi['_ready']).toBe(false);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test 4 — Partial update leaves other fields unchanged
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine.updateLighting — partial update does not disturb other fields', () => {
  it('changing only primaryLightIntensity leaves dir, skyTint, skyIrradiance intact', () => {
    const engine = makeEngine({
      primaryLightDir:       [0.3, -0.9, 0.1],
      primaryLightIntensity: 1.0,
      skyTint:               [0.5, 0.6, 0.9],
      skyIrradiance:         0.7,
    });
    const e = engine as unknown as Record<string, unknown>;

    engine.updateLighting({ primaryLightIntensity: 5.0 });

    expect(e['_primaryLightDir']).toEqual([0.3, -0.9, 0.1]);
    expect(e['_primaryLightIntensity']).toBe(5.0);
    expect(e['_skyTint']).toEqual([0.5, 0.6, 0.9]);
    expect(e['_skyIrradiance']).toBe(0.7);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test 5 — updateLighting({}) is a no-op
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine.updateLighting — empty call is a no-op', () => {
  it('does not invalidate DDGI or reset the accumulator when no fields change', () => {
    const engine = makeEngine();
    const e = engine as unknown as Record<string, unknown>;

    // Simulate a converged DDGI.
    const ddgi = e['_ddgi'] as Record<string, unknown>;
    ddgi['_frame'] = 24;
    ddgi['_ready'] = true;

    // Inject a mock pipeline to detect unwanted accumulator resets.
    const mockPipeline = { requestAccumReset: vi.fn() };
    e['_pipeline'] = mockPipeline;

    engine.updateLighting({});

    // Nothing should have changed.
    expect(ddgi['_frame']).toBe(24);
    expect(ddgi['_ready']).toBe(true);
    expect(mockPipeline.requestAccumReset).not.toHaveBeenCalled();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test 6 — T13 unknown-key guard
//
// `Engine.updateLighting` is contractually opaque (Readonly<Record<string,
// unknown>> in @vitrum/core), so backend-specific keys aren't baked into core.
// To keep that opacity while surfacing silent drops, HybridEngine.updateLighting
// console.warns once per unrecognised key. These tests pin (a) the key-set
// source of truth stays aligned with the LightingOptions interface, (b) valid
// keys don't warn, and (c) an unknown key warns + is ignored without throwing.
// ─────────────────────────────────────────────────────────────────────────────

describe('LIGHTING_OPTION_KEYS — key-set source of truth', () => {
  it('exactly matches the LightingOptions interface fields', () => {
    // `Required<LightingOptions>` forces this object to list every interface
    // field; the round-trip catches drift between the interface and the
    // exported key list consumed by the runtime guard.
    const sample: Required<LightingOptions> = {
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      skyTint: [1, 1, 1],
      skyIrradiance: 1,
    };
    expect([...LIGHTING_OPTION_KEYS].sort()).toEqual(Object.keys(sample).sort());
  });
});

describe('HybridEngine.updateLighting — unknown-key guard (T13)', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('does not warn for valid keys but still applies them', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = makeEngine();
    const e = engine as unknown as Record<string, unknown>;

    engine.updateLighting({
      primaryLightDir: [1, 0, 0],
      primaryLightIntensity: 5,
      skyTint: [0.2, 0.3, 0.4],
      skyIrradiance: 9,
    });

    expect(warn).not.toHaveBeenCalled();
    expect(e['_primaryLightDir']).toEqual([1, 0, 0]);
    expect(e['_primaryLightIntensity']).toBe(5);
    expect(e['_skyTint']).toEqual([0.2, 0.3, 0.4]);
    expect(e['_skyIrradiance']).toBe(9);
  });

  it('warns once per unknown key, ignores it, still applies valid keys, does not throw', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = makeEngine();
    const e = engine as unknown as Record<string, unknown>;

    // `sunIntensity` is a plausible host typo for `primaryLightIntensity`;
    // the opaque core contract lets it through at compile time.
    expect(() =>
      // Cast through Record so the unknown key is accepted (mirrors a host
      // calling via the opaque `Engine.updateLighting` contract surface).
      engine.updateLighting({
        primaryLightIntensity: 7,
        sunIntensity: 7,
      } as unknown as Partial<LightingOptions>),
    ).not.toThrow();

    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn).toHaveBeenCalledWith(
      '[@vitrum/walkaround-hybrid] updateLighting: ignoring unknown key "sunIntensity"',
    );
    // Valid key applied; unknown key did not leak onto engine state.
    expect(e['_primaryLightIntensity']).toBe(7);
    expect(e['sunIntensity']).toBeUndefined();
  });

  it('warns once for each of multiple unknown keys', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = makeEngine();

    engine.updateLighting({
      bogus: 1,
      alsoBad: 'x',
    } as unknown as Partial<LightingOptions>);

    expect(warn).toHaveBeenCalledTimes(2);
    expect(warn).toHaveBeenCalledWith(
      '[@vitrum/walkaround-hybrid] updateLighting: ignoring unknown key "bogus"',
    );
    expect(warn).toHaveBeenCalledWith(
      '[@vitrum/walkaround-hybrid] updateLighting: ignoring unknown key "alsoBad"',
    );
  });

  it('does not warn for an empty object', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = makeEngine();

    engine.updateLighting({});

    expect(warn).not.toHaveBeenCalled();
  });
});
