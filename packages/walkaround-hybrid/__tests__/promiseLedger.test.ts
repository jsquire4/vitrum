import { describe, expect, it } from 'vitest';
import type { Engine } from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';
import { HybridEngine } from '../src/HybridEngine.js';

function makeMockDevice(): GPUDevice {
  return {
    createCommandEncoder: () => ({ finish: () => ({}) }),
    queue: {
      writeBuffer: () => {},
      submit: () => {},
    },
  } as unknown as GPUDevice;
}

function sorted(values: Iterable<string>): string[] {
  return Array.from(values).sort();
}

describe('walkaround-hybrid promise ledger compliance', () => {
  it('matches declared capability and optional-method promises', () => {
    const expected = BACKEND_PROMISE_LEDGER['walkaround-hybrid'];
    const engine = new HybridEngine({
      device: makeMockDevice(),
      width: 64,
      height: 64,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      skyTint: [1, 1, 1],
      skyIrradiance: 1,
    });
    const engineView = engine as unknown as Engine;
    const caps = engine.capabilities;

    expect(caps.supportsIncrementalScene).toBe(expected.supportsIncrementalScene);
    expect(caps.supportsAuxBuffers).toBe(expected.supportsAuxBuffers);
    expect(caps.accumulates).toBe(expected.accumulates);
    expect(caps.presentationMode).toBe(expected.presentationMode);

    expect(caps.incrementalPatchSupport).toEqual(expected.incrementalPatchSupport);
    expect(sorted(caps.supportedPrimitiveKinds ?? [])).toEqual(sorted(expected.supportedPrimitiveKinds));
    expect(sorted(caps.supportedEmitterKinds)).toEqual(sorted(expected.supportedEmitterKinds));
    expect(sorted(caps.supportedEnvironmentKinds ?? [])).toEqual(sorted(expected.supportedEnvironmentKinds));
    expect(sorted(caps.supportedAnalyticShapes)).toEqual(sorted(expected.supportedAnalyticShapes));
    expect(caps.supportDetails).toEqual(expected.supportDetails);

    expect(typeof engineView.updatePrimitive === 'function').toBe(expected.methodPromises.updatePrimitive);
    expect(typeof engineView.updateEmitter === 'function').toBe(expected.methodPromises.updateEmitter);
    expect(typeof engineView.updateEnvironment === 'function').toBe(expected.methodPromises.updateEnvironment);
    expect(typeof engineView.setSize === 'function').toBe(expected.methodPromises.setSize);
    expect(typeof engineView.updateLighting === 'function').toBe(expected.methodPromises.updateLighting);
    expect(typeof engineView.onFrame === 'function').toBe(expected.methodPromises.onFrame);
    expect(typeof engineView.onProgress === 'function').toBe(expected.methodPromises.onProgress);
    expect(typeof engineView.debug === 'object').toBe(expected.methodPromises.debug);
  });
});
