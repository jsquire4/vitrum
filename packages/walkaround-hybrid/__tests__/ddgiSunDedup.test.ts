/**
 * DDGI sun de-duplication.
 *
 * Bug: when a host passes `opts.lights` containing a `sun` (manual override)
 * AND the core scene carries a `directional` emitter (which
 * `coreEmittersToDDGILights` converts into a `sun` DDGILight), BOTH suns were
 * spread into DDGI's `setLights` and the probe-update pass evaluated both —
 * double-counting the sun's contribution to every probe.
 *
 * Fix: `mergeDDGILightsDedupSun` drops host-supplied suns whenever the scene
 * contributes a sun (the scene directional is the physical source of truth;
 * the host sun is a manual default the scene overrides). Non-sun lights from
 * both sources are always kept; absent a scene sun, host suns pass through.
 */

import type { MockInstance } from 'vitest';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  createHostSunWarningState,
  mergeDDGILightsDedupSun,
} from '../src/HybridEngineLifecycle.js';
import { coreEmittersToDDGILights } from '../src/coreEmittersToDDGILights.js';
import type { DDGILight } from '../src/ddgi/types.js';
import type { Scene, DirectionalEmitter } from '@vitrum/core';

const HOST_SUN: DDGILight = {
  kind: 'sun',
  intensity: 2,
  on: true,
  direction: { x: 0, y: -1, z: 0 },
  color: { r: 1, g: 1, b: 1 },
};

const HOST_FIXTURE: DDGILight = {
  kind: 'fixture',
  intensity: 3,
  on: true,
  position: { x: 1, y: 2, z: 3 },
};

const SCENE_DIRECTIONAL: DirectionalEmitter = {
  id: 'scene-sun',
  kind: 'directional',
  color: [0.9, 0.4, 0.2],
  intensity: 6,
  direction: [3, 4, 0],
};

function sceneOf(...emitters: Scene['emitters']): Scene {
  return { primitives: [], emitters, environment: { kind: 'none' } };
}

describe('mergeDDGILightsDedupSun', () => {
  let warnSpy: MockInstance<Parameters<typeof console.warn>, void>;

  beforeEach(() => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });
  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('warns once per engine owner, not once per process', () => {
    const sceneLights = coreEmittersToDDGILights(sceneOf(SCENE_DIRECTIONAL));
    expect(sceneLights.filter((l) => l.kind === 'sun')).toHaveLength(1);
    const firstEngine = createHostSunWarningState();
    const secondEngine = createHostSunWarningState();

    mergeDDGILightsDedupSun(
      [HOST_SUN],
      sceneLights,
      { warningState: firstEngine },
    );
    mergeDDGILightsDedupSun(
      [HOST_SUN],
      sceneLights,
      { warningState: firstEngine },
    );
    mergeDDGILightsDedupSun(
      [HOST_SUN],
      sceneLights,
      { warningState: secondEngine },
    );

    // Re-syncing the first engine is quiet; an independent engine still gets
    // its own actionable conflict warning.
    expect(warnSpy).toHaveBeenCalledTimes(2);
  });

  it('drops the host sun when the scene contributes a directional→sun', () => {
    const sceneLights = coreEmittersToDDGILights(sceneOf(SCENE_DIRECTIONAL));
    expect(sceneLights.filter((l) => l.kind === 'sun')).toHaveLength(1);

    const merged = mergeDDGILightsDedupSun([HOST_SUN, HOST_FIXTURE], sceneLights);

    // Exactly ONE sun reaches DDGI — the scene-derived one.
    const suns = merged.filter((l) => l.kind === 'sun');
    expect(suns).toHaveLength(1);
    expect(suns[0]!.id).toBe('scene-sun');
    expect(suns[0]!.intensity).toBe(6);
    // The host's non-sun fixture survives.
    expect(merged.filter((l) => l.kind === 'fixture' && l.intensity === 3)).toHaveLength(1);
  });

  it('keeps the host sun when the scene contributes NO sun (legacy config)', () => {
    const merged = mergeDDGILightsDedupSun([HOST_SUN, HOST_FIXTURE], []);
    const suns = merged.filter((l) => l.kind === 'sun');
    expect(suns).toHaveLength(1);
    expect(suns[0]).toBe(HOST_SUN);
    // No override → no warning.
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('does not drop scene-derived fixtures, and never doubles non-sun lights', () => {
    const sceneLights: DDGILight[] = [
      { kind: 'sun', id: 'scene-sun', intensity: 6, on: true },
      { kind: 'fixture', id: 'scene-fix', intensity: 4, on: true, position: { x: 0, y: 0, z: 0 } },
    ];
    const merged = mergeDDGILightsDedupSun([HOST_SUN, HOST_FIXTURE], sceneLights);
    expect(merged.filter((l) => l.kind === 'sun')).toHaveLength(1);
    // 2 fixtures total: host fixture + scene fixture (neither dropped).
    expect(merged.filter((l) => l.kind === 'fixture')).toHaveLength(2);
  });

  it('drops ALL host suns when multiple are present and the scene has a sun', () => {
    const secondHostSun: DDGILight = { kind: 'sun', intensity: 9, on: true };
    const sceneLights: DDGILight[] = [{ kind: 'sun', id: 'scene-sun', intensity: 6, on: true }];
    const merged = mergeDDGILightsDedupSun([HOST_SUN, secondHostSun], sceneLights);
    const suns = merged.filter((l) => l.kind === 'sun');
    expect(suns).toHaveLength(1);
    expect(suns[0]!.id).toBe('scene-sun');
  });
});
