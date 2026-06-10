import { describe, expect, it, vi } from 'vitest';
import type { SceneEnvironment } from '@vitrum/core';
import {
  resolveHybridEnvironment,
  type HybridEnvironmentMapResolver,
} from '../resolveHybridEnvironment.js';

function expectVecClose(actual: readonly number[] | undefined, expected: readonly number[]): void {
  expect(actual).toBeDefined();
  expect(actual).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actual![i]).toBeCloseTo(expected[i]!);
  }
}

describe('resolveHybridEnvironment', () => {
  it("maps null and kind:'none' to a sky-off result", () => {
    expect(resolveHybridEnvironment(null)).toEqual({
      mode: 'none',
      skyIrradiance: 0,
      warnings: [],
    });
    expect(resolveHybridEnvironment({ kind: 'none' })).toEqual({
      mode: 'none',
      skyIrradiance: 0,
      warnings: [],
    });
  });

  it('maps an opaque HDRI handle to intensity-only with an explicit warning', () => {
    const resolved = resolveHybridEnvironment({
      kind: 'hdri',
      hdri: { texture: 'opaque-host-handle' },
      intensity: 2.5,
      rotationY: 1.3,
    });

    expect(resolved.mode).toBe('hdri-intensity-only');
    expect(resolved.skyIrradiance).toBe(2.5);
    expect(resolved.skyTint).toBeUndefined();
    expect(resolved.warnings).toHaveLength(1);
    expect(resolved.warnings[0]).toContain('opaque');
  });

  it('defaults an HDRI intensity-only fallback to 1', () => {
    const resolved = resolveHybridEnvironment({ kind: 'hdri', hdri: {} });

    expect(resolved.mode).toBe('hdri-intensity-only');
    expect(resolved.skyIrradiance).toBe(1);
  });

  it('derives skyTint and skyIrradiance from a raw RGB HDRI payload', () => {
    const resolved = resolveHybridEnvironment({
      kind: 'hdri',
      intensity: 0.5,
      rotationY: 0.7,
      hdri: {
        width: 2,
        height: 1,
        data: new Float32Array([
          2, 1, 1,
          0, 3, 0,
        ]),
      },
    });

    expect(resolved.mode).toBe('hdri-raw-average');
    // Scalar-tint fallback is UNCHANGED (the B3 directional payload is additive).
    expectVecClose(resolved.skyTint, [0.5, 1, 0.25]);
    expect(resolved.skyIrradiance).toBeCloseTo(1);
    // B3 — a non-black raw map now also yields a directional IBL payload.
    expect(resolved.directional).toBeDefined();
    expect(resolved.directional!.width).toBe(2);
    expect(resolved.directional!.height).toBe(1);
    expect(resolved.directional!.map).toHaveLength(2 * 1 * 4);
    expect(resolved.directional!.totalWeight).toBeGreaterThan(0);
    expect(resolved.rotationY).toBeCloseTo(0.7);
    expect(resolved.directionalIntensity).toBeCloseTo(0.5);
    expect(resolved.warnings.join('\n')).toContain('directional IBL map');
  });

  it('keeps the scalar-only fallback (no directional) for an all-black raw map', () => {
    const resolved = resolveHybridEnvironment({
      kind: 'hdri',
      hdri: { width: 2, height: 2, data: new Float32Array(2 * 2 * 3) },
    });
    expect(resolved.mode).toBe('hdri-raw-average');
    expect(resolved.directional).toBeUndefined();
    expect(resolved.warnings.join('\n')).toContain('solid-angle-weighted average');
  });

  it('supports raw RGBA HDRI payloads by ignoring alpha', () => {
    const resolved = resolveHybridEnvironment({
      kind: 'hdri',
      hdri: {
        width: 1,
        height: 1,
        data: new Float32Array([0.25, 0.5, 1, 0.125]),
      },
    });

    expect(resolved.mode).toBe('hdri-raw-average');
    expectVecClose(resolved.skyTint, [0.25, 0.5, 1]);
    expect(resolved.skyIrradiance).toBeCloseTo(1);
  });

  it('keeps black raw HDRI payloads finite', () => {
    const resolved = resolveHybridEnvironment({
      kind: 'hdri',
      intensity: 4,
      hdri: {
        width: 1,
        height: 1,
        data: new Float32Array([0, 0, 0]),
      },
    });

    expect(resolved.mode).toBe('hdri-raw-average');
    expect(resolved.skyTint).toEqual([1, 1, 1]);
    expect(resolved.skyIrradiance).toBe(0);
  });

  it('falls back cleanly for malformed raw HDRI payloads', () => {
    const resolved = resolveHybridEnvironment({
      kind: 'hdri',
      intensity: 3,
      hdri: {
        width: 4,
        height: 4,
        data: new Float32Array([1, 1, 1]),
      },
    });

    expect(resolved.mode).toBe('hdri-intensity-only');
    expect(resolved.skyIrradiance).toBe(3);
    expect(resolved.warnings.join('\n')).toContain('shorter than width * height * 3');
  });

  it('approximates procedural-sky with diffuse sky scalars and procedural sun fields', () => {
    const env: SceneEnvironment = {
      kind: 'procedural-sky',
      sunDirection: [0, 2, 0],
      turbidity: 2,
      rayleigh: 1,
      mieCoefficient: 0.005,
      mieDirectionalG: 0.8,
      intensity: 1.7,
    };

    const resolved = resolveHybridEnvironment(env);

    expect(resolved.mode).toBe('procedural-sky-approx');
    expectVecClose(resolved.skyTint, [0.855, 0.95, 1]);
    expect(resolved.skyIrradiance).toBe(1.7);
    expect(resolved.proceduralSunDirection).toEqual([0, 1, 0]);
    expect(resolved.proceduralSunIntensity).toBe(1.7);
    expect(resolved.warnings.join('\n')).toContain('approximated');
  });

  it('uses the extension resolver for opaque HDRI handles and applies SceneEnvironment intensity', () => {
    const resolver: HybridEnvironmentMapResolver = vi.fn(() => ({
      skyTint: [0.2, 0.4, 1],
      skyIrradiance: 2,
      warnings: ['host resolver used a precomputed cubemap average'],
    }));
    const hdri = { texture: 'opaque-host-handle' };

    const resolved = resolveHybridEnvironment(
      { kind: 'hdri', hdri, intensity: 3 },
      {
        extensions: {
          'walkaround-hybrid': {
            resolveEnvironmentMap: resolver,
          },
        },
      },
    );

    expect(resolver).toHaveBeenCalledWith(hdri, { kind: 'hdri', hdri, intensity: 3 });
    expect(resolved.mode).toBe('hdri-extension-resolver');
    expect(resolved.skyTint).toEqual([0.2, 0.4, 1]);
    expect(resolved.skyIrradiance).toBe(6);
    expect(resolved.warnings).toContain('host resolver used a precomputed cubemap average');
  });

  it('falls back to intensity-only when the extension resolver declines', () => {
    const resolved = resolveHybridEnvironment(
      { kind: 'hdri', hdri: { texture: 'opaque-host-handle' }, intensity: 1.25 },
      {
        extensions: {
          'walkaround-hybrid': {
            resolveEnvironmentMap: () => undefined,
          },
        },
      },
    );

    expect(resolved.mode).toBe('hdri-intensity-only');
    expect(resolved.skyIrradiance).toBe(1.25);
    expect(resolved.warnings.join('\n')).toContain('returned no result');
  });
});
