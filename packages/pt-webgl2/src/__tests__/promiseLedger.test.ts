import { describe, expect, it } from 'vitest';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { createMockGl } from './mockGl.js';

function sorted(values: Iterable<string>): string[] {
  return Array.from(values).sort();
}

function hasFunctionProperty(target: object, key: PropertyKey): boolean {
  return typeof (target as Record<PropertyKey, unknown>)[key] === 'function';
}

describe('pt-webgl2 promise ledger compliance', () => {
  it('matches declared capability and optional-method promises', async () => {
    const expected = BACKEND_PROMISE_LEDGER['pt-webgl2'];
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
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

    expect(caps.supportsAddRemovePrimitive).toBe(expected.supportsAddRemovePrimitive);

    expect(typeof engine.updatePrimitive === 'function').toBe(expected.methodPromises.updatePrimitive);
    expect(typeof engine.updateEmitter === 'function').toBe(expected.methodPromises.updateEmitter);
    expect(typeof engine.updateEnvironment === 'function').toBe(expected.methodPromises.updateEnvironment);
    expect(typeof engine.addPrimitive === 'function').toBe(expected.methodPromises.addPrimitive);
    expect(typeof engine.removePrimitive === 'function').toBe(expected.methodPromises.removePrimitive);
    expect(typeof engine.setSize === 'function').toBe(expected.methodPromises.setSize);
    expect(typeof engine.updateLighting === 'function').toBe(expected.methodPromises.updateLighting);
    expect(typeof engine.onFrame === 'function').toBe(expected.methodPromises.onFrame);
    expect(typeof engine.onProgress === 'function').toBe(expected.methodPromises.onProgress);
    expect(typeof engine.debug === 'object').toBe(expected.methodPromises.debug);
    expect(typeof engine.getScene === 'function').toBe(expected.methodPromises.getScene);
    expect(typeof engine.onError === 'function').toBe(expected.methodPromises.onError);
    expect(typeof engine.onWarning === 'function').toBe(expected.methodPromises.onWarning);
    expect(typeof engine.captureFrame === 'function').toBe(expected.methodPromises.captureFrame);
    expect(typeof engine.createInverseSession === 'function').toBe(expected.methodPromises.createInverseSession);
    expect(typeof engine.getRestirPtResultBuffer === 'function').toBe(
      expected.methodPromises.getRestirPtResultBuffer,
    );
    expect(typeof engine.getPresentationSource === 'function').toBe(
      expected.methodPromises.getPresentationSource,
    );
    expect(typeof engine.getProgressiveSeedTexture === 'function').toBe(
      expected.methodPromises.getProgressiveSeedTexture,
    );
    expect(typeof engine.seedAccumulator === 'function').toBe(expected.methodPromises.seedAccumulator);
    expect(hasFunctionProperty(engine, 'exportGIState') || hasFunctionProperty(engine, 'importGIState')).toBe(
      expected.methodPromises.giStatePersistence,
    );
  });
});
