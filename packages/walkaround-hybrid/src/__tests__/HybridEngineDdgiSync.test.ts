import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning, Scene } from '@vitrum/core';
import type { DDGI } from '../ddgi/DDGI.js';
import type { WalkaroundGPUPipeline } from '../pipeline/WalkaroundGPUPipeline.js';
import { syncDdgiFromCoreScene } from '../HybridEngineDdgiSync.js';

function sceneWithPointAndSpot(): Scene {
  return {
    primitives: [],
    emitters: [
      {
        kind: 'point',
        id: 'point-a',
        position: [0, 2, 0],
        color: [1, 0.8, 0.6],
        intensity: 3,
      },
      {
        kind: 'spot',
        id: 'spot-a',
        position: [1, 3, 2],
        direction: [0, -1, 0],
        angle: Math.PI / 4,
        penumbra: 0.2,
        color: [0.5, 0.5, 1],
        intensity: 2,
      },
    ],
    environment: { kind: 'none' },
  };
}

describe('syncDdgiFromCoreScene', () => {
  it('routes host-sun override through structured warnings and keeps one DDGI sun', () => {
    const scene: Scene = {
      primitives: [],
      emitters: [{
        kind: 'directional',
        id: 'scene-sun',
        direction: [0, -1, 0],
        color: [1, 1, 1],
        intensity: 2,
      }],
      environment: { kind: 'none' },
    };
    const ddgi = {
      setSunIntensityMultiplier: vi.fn(),
      setLights: vi.fn(),
      setEmitterTris: vi.fn(),
    } as unknown as DDGI;
    const pipeline = {
      updateAnalyticLights: vi.fn(),
    } as unknown as WalkaroundGPUPipeline;
    const warnings: EngineWarning[] = [];
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    syncDdgiFromCoreScene({
      ddgi,
      pipeline,
      ctorLights: [{ kind: 'sun', on: true, color: { r: 10, g: 10, b: 10 }, intensity: 99 }],
      primaryLightIntensity: 1,
      onWarning: (warning) => warnings.push(warning),
    }, scene);

    expect(warnSpy).not.toHaveBeenCalled();
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatchObject({
      code: 'walkaround-hybrid.ddgi-host-sun-overridden',
      backend: 'walkaround-hybrid',
      phase: 'setScene',
      method: 'syncDdgiFromCoreScene',
      details: {
        fallback: 'drop-host-sun',
        sourceOfTruth: 'scene-directional-emitter',
      },
    });
    const merged = vi.mocked(ddgi.setLights).mock.calls[0]?.[0] ?? [];
    expect(merged.filter((light) => light.kind === 'sun')).toHaveLength(1);
    expect(merged[0]?.kind).toBe('sun');
    if (merged[0]?.kind === 'sun') {
      expect(merged[0].intensity).toBeCloseTo(2);
    }
    warnSpy.mockRestore();
  });

  it('refreshes the analytic point/spot light upload path with the exact scene', () => {
    const scene = sceneWithPointAndSpot();
    const ddgi = {
      setSunIntensityMultiplier: vi.fn(),
      setLights: vi.fn(),
      setEmitterTris: vi.fn(),
    } as unknown as DDGI;
    const pipeline = {
      updateAnalyticLights: vi.fn(),
    } as unknown as WalkaroundGPUPipeline;

    syncDdgiFromCoreScene({
      ddgi,
      pipeline,
      ctorLights: [],
      primaryLightIntensity: 1,
    }, scene);

    expect(pipeline.updateAnalyticLights).toHaveBeenCalledTimes(1);
    expect(pipeline.updateAnalyticLights).toHaveBeenCalledWith(scene);
    expect(ddgi.setEmitterTris).toHaveBeenCalledTimes(1);
  });

  it('orients DDGI sun lights to the runtime primary-light direction when supplied', () => {
    const scene: Scene = {
      primitives: [],
      emitters: [{
        kind: 'directional',
        id: 'scene-sun',
        direction: [0, -1, 0],
        color: [1, 1, 1],
        intensity: 2,
      }],
      environment: { kind: 'none' },
    };
    const ddgi = {
      setSunIntensityMultiplier: vi.fn(),
      setLights: vi.fn(),
      setEmitterTris: vi.fn(),
    } as unknown as DDGI;

    syncDdgiFromCoreScene({
      ddgi,
      pipeline: null,
      ctorLights: [],
      primaryLightIntensity: 1,
      primaryLightDir: [1, 0, 0],
    }, scene);

    const lights = vi.mocked(ddgi.setLights).mock.calls[0]?.[0] ?? [];
    expect(lights).toHaveLength(1);
    const sun = lights[0];
    expect(sun?.kind).toBe('sun');
    if (sun?.kind === 'sun') {
      expect(sun.direction).toBeDefined();
      if (sun.direction == null) throw new Error('expected oriented sun direction');
      expect(sun.direction.x).toBeCloseTo(-1);
      expect(sun.direction.y).toBeCloseTo(0);
      expect(sun.direction.z).toBeCloseTo(0);
    }
  });
});
