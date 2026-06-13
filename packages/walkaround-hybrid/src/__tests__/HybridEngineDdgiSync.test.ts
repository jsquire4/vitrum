import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
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
});
