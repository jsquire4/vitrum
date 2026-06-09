import { describe, expect, it, vi } from 'vitest';
import type { Engine, EngineCapabilities, FrameInput, FrameOutput } from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER, asBackendTexture } from '@vitrum/core';
import { wrapWithIdempotentDispose } from '../src/createEngine.js';

function makeBaseCapabilities(): EngineCapabilities {
  return {
    supportsIncrementalScene: false,
    supportsAuxBuffers: false,
    accumulates: false,
    maxSamplesPerPixel: Infinity,
    maxBounces: 1,
    supportedAnalyticShapes: new Set(),
    supportedEmitterKinds: new Set(),
    causticStrategy: 'none',
  };
}

function makeFakeEngine(capabilities: EngineCapabilities): Engine {
  return {
    state: 'ready',
    capabilities,
    setScene: vi.fn(),
    updatePrimitive: vi.fn(),
    addPrimitive: vi.fn(),
    removePrimitive: vi.fn(),
    updateEmitter: vi.fn(),
    updateEnvironment: vi.fn(),
    setSize: vi.fn(),
    updateLighting: vi.fn(),
    renderFrame: vi.fn((_input: FrameInput): FrameOutput => ({
      kind: 'rendered',
      samplesAccumulated: 1,
      isConverged: false,
      primaryRadiance: asBackendTexture<'test', {}>({}),
    })),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
    onFrame: vi.fn(() => () => {}),
    onProgress: vi.fn(() => () => {}),
    debug: { estimatedGpuMemoryBytes: vi.fn(() => null) },
  };
}

describe('proxy method exposure follows backend promise ledger', () => {
  const ids = ['walkaround-hybrid', 'pt-webgl2', 'pt-webgpu'] as const;

  for (const backendId of ids) {
    it(`matches ${backendId} incremental/optional-method promises`, () => {
      const rec = BACKEND_PROMISE_LEDGER[backendId];
      const engine = makeFakeEngine({
        ...makeBaseCapabilities(),
        supportsIncrementalScene: rec.supportsIncrementalScene,
        supportsAddRemovePrimitive: rec.supportsAddRemovePrimitive,
        supportsAuxBuffers: rec.supportsAuxBuffers,
        accumulates: rec.accumulates,
        incrementalPatchSupport: rec.incrementalPatchSupport,
        supportedAnalyticShapes: new Set(rec.supportedAnalyticShapes),
        supportedEmitterKinds: new Set(rec.supportedEmitterKinds),
        supportedPrimitiveKinds: new Set(rec.supportedPrimitiveKinds),
        supportedEnvironmentKinds: new Set(rec.supportedEnvironmentKinds),
        presentationMode: rec.presentationMode,
        debugSurface: rec.methodPromises.debug,
      });

      if (!rec.methodPromises.updatePrimitive) delete (engine as Partial<Engine>).updatePrimitive;
      if (!rec.methodPromises.addPrimitive) delete (engine as Partial<Engine>).addPrimitive;
      if (!rec.methodPromises.removePrimitive) delete (engine as Partial<Engine>).removePrimitive;
      if (!rec.methodPromises.updateEmitter) delete (engine as Partial<Engine>).updateEmitter;
      if (!rec.methodPromises.updateEnvironment) delete (engine as Partial<Engine>).updateEnvironment;
      if (!rec.methodPromises.setSize) delete (engine as Partial<Engine>).setSize;
      if (!rec.methodPromises.updateLighting) delete (engine as Partial<Engine>).updateLighting;
      if (!rec.methodPromises.onFrame) delete (engine as Partial<Engine>).onFrame;
      if (!rec.methodPromises.onProgress) delete (engine as Partial<Engine>).onProgress;
      if (!rec.methodPromises.debug) delete (engine as Partial<Engine>).debug;

      const proxy = wrapWithIdempotentDispose(engine, () => {});

      expect(typeof proxy.updatePrimitive === 'function').toBe(
        rec.methodPromises.updatePrimitive && (
          rec.incrementalPatchSupport.transform
          || rec.incrementalPatchSupport.positions
          || rec.incrementalPatchSupport.material
          || rec.incrementalPatchSupport.topology
        ),
      );
      expect(typeof proxy.addPrimitive === 'function').toBe(
        rec.methodPromises.addPrimitive && rec.supportsAddRemovePrimitive,
      );
      expect(typeof proxy.removePrimitive === 'function').toBe(
        rec.methodPromises.removePrimitive && rec.supportsAddRemovePrimitive,
      );
      expect(typeof proxy.updateEmitter === 'function').toBe(
        rec.methodPromises.updateEmitter && rec.incrementalPatchSupport.emitter,
      );
      expect(typeof proxy.updateEnvironment === 'function').toBe(rec.methodPromises.updateEnvironment);
      expect(typeof proxy.setSize === 'function').toBe(rec.methodPromises.setSize);
      expect(typeof proxy.updateLighting === 'function').toBe(rec.methodPromises.updateLighting);
      expect(typeof proxy.onFrame === 'function').toBe(rec.methodPromises.onFrame);
      expect(typeof proxy.onProgress === 'function').toBe(rec.methodPromises.onProgress);
      expect(typeof proxy.debug === 'object').toBe(rec.methodPromises.debug);
    });
  }
});

describe('path-tracer optional method omissions follow unsupported ledger rows', () => {
  for (const backendId of ['pt-webgl2', 'pt-webgpu'] as const) {
    it(`omits resize and lighting methods for ${backendId}`, () => {
      const rec = BACKEND_PROMISE_LEDGER[backendId];
      const engine = makeFakeEngine({
        ...makeBaseCapabilities(),
        supportsIncrementalScene: rec.supportsIncrementalScene,
        supportsAddRemovePrimitive: rec.supportsAddRemovePrimitive,
        supportsAuxBuffers: rec.supportsAuxBuffers,
        accumulates: rec.accumulates,
        incrementalPatchSupport: rec.incrementalPatchSupport,
        supportedAnalyticShapes: new Set(rec.supportedAnalyticShapes),
        supportedEmitterKinds: new Set(rec.supportedEmitterKinds),
        supportedPrimitiveKinds: new Set(rec.supportedPrimitiveKinds),
        supportedEnvironmentKinds: new Set(rec.supportedEnvironmentKinds),
        presentationMode: rec.presentationMode,
      });

      delete (engine as Partial<Engine>).setSize;
      delete (engine as Partial<Engine>).updateLighting;

      const proxy = wrapWithIdempotentDispose(engine, () => {});

      expect(rec.supportDetails.mutations.resize).toBe('unsupported');
      expect(rec.supportDetails.mutations.lighting).toBe('unsupported');
      expect(rec.methodPromises.setSize).toBe(false);
      expect(rec.methodPromises.updateLighting).toBe(false);
      expect(proxy.setSize).toBeUndefined();
      expect(proxy.updateLighting).toBeUndefined();
    });
  }
});
