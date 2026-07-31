import { describe, expect, it, vi } from 'vitest';
import type { Engine, Scene } from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';
import { HybridEngine } from '../src/HybridEngine.js';

function makeMockDevice(): GPUDevice {
  return {
    createCommandEncoder: () => ({ finish: () => ({}) }),
    queue: {
      writeBuffer: () => {},
      submit: () => {},
    },
    addEventListener: () => {},
    removeEventListener: () => {},
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function sorted(values: Iterable<string>): string[] {
  return Array.from(values).sort();
}

function hasFunctionProperty(target: object, key: PropertyKey): boolean {
  return typeof (target as Record<PropertyKey, unknown>)[key] === 'function';
}

function transmissionScene(): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'glass',
      positions: new Float32Array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
      ]),
      normals: new Float32Array([
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
      ]),
      material: {
        baseColor: [1, 1, 1],
        roughness: 0.05,
        metallic: 0,
        transmission: 1,
      },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
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
      debug: true,
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
    expect(caps.supportDetails).toEqual({
      ...expected.supportDetails,
      denoisers: {
        ...expected.supportDetails.denoisers,
        neural: 'unsupported',
        'oidn-final': 'unsupported',
      },
    });

    expect(typeof engineView.updatePrimitive === 'function').toBe(expected.methodPromises.updatePrimitive);
    expect(typeof engineView.updateEmitter === 'function').toBe(expected.methodPromises.updateEmitter);
    expect(typeof engineView.updateEnvironment === 'function').toBe(expected.methodPromises.updateEnvironment);
    expect(typeof engineView.addPrimitive === 'function').toBe(expected.methodPromises.addPrimitive);
    expect(typeof engineView.removePrimitive === 'function').toBe(expected.methodPromises.removePrimitive);
    expect(typeof engineView.setSize === 'function').toBe(expected.methodPromises.setSize);
    expect(typeof engineView.updateLighting === 'function').toBe(expected.methodPromises.updateLighting);
    expect(typeof engineView.onFrame === 'function').toBe(expected.methodPromises.onFrame);
    expect(typeof engineView.onProgress === 'function').toBe(expected.methodPromises.onProgress);
    expect(typeof engineView.debug === 'object').toBe(expected.methodPromises.debug);
    expect(typeof engineView.getScene === 'function').toBe(expected.methodPromises.getScene);
    expect(typeof engineView.onError === 'function').toBe(expected.methodPromises.onError);
    expect(typeof engineView.onWarning === 'function').toBe(expected.methodPromises.onWarning);
    expect(typeof engineView.captureFrame === 'function').toBe(expected.methodPromises.captureFrame);
    expect(typeof engineView.createInverseSession === 'function').toBe(expected.methodPromises.createInverseSession);
    expect(typeof engineView.getRestirPtResultBuffer === 'function').toBe(
      expected.methodPromises.getRestirPtResultBuffer,
    );
    expect(typeof engineView.getPresentationSource === 'function').toBe(
      expected.methodPromises.getPresentationSource,
    );
    expect(typeof engineView.getProgressiveSeedTexture === 'function').toBe(
      expected.methodPromises.getProgressiveSeedTexture,
    );
    expect(typeof engineView.seedAccumulator === 'function').toBe(expected.methodPromises.seedAccumulator);
    expect(hasFunctionProperty(engineView, 'exportGIState') || hasFunctionProperty(engineView, 'importGIState')).toBe(
      expected.methodPromises.giStatePersistence,
    );
  });

  it('publishes the debug property and capability only for debug-enabled instances', () => {
    const makeEngine = (debug: boolean): HybridEngine => new HybridEngine({
      device: makeMockDevice(),
      width: 64,
      height: 64,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      skyTint: [1, 1, 1],
      skyIrradiance: 1,
      debug,
    });
    const disabled = makeEngine(false) as unknown as Engine;
    const enabled = makeEngine(true) as unknown as Engine;

    expect(Object.prototype.hasOwnProperty.call(disabled, 'debug')).toBe(false);
    expect(disabled.capabilities.debugSurface).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(enabled, 'debug')).toBe(true);
    expect(typeof enabled.debug).toBe('object');
    expect(enabled.capabilities.debugSurface).toBe(true);
  });

  it('reports requested/effective shading resolution and both scaled reservoir grids', () => {
    const engine = new HybridEngine({
      device: makeMockDevice(),
      width: 2560,
      height: 1440,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      skyTint: [1, 1, 1],
      skyIrradiance: 1,
      debug: true,
    }) as unknown as Engine;

    const resolution = engine.debug?.frameResourceResolution?.();
    expect(resolution).not.toBeNull();
    expect(resolution).toMatchObject({
      presentationWidth: 2560,
      presentationHeight: 1440,
      requestedInternalWidth: 2560,
      requestedInternalHeight: 1440,
      effectiveInternalWidth: 2134,
      effectiveInternalHeight: 1200,
      resolutionDownscale: 1.2,
      restirDiWidth: 533,
      restirDiHeight: 300,
      restirGiWidth: 266,
      restirGiHeight: 150,
      restirReservoirScale: 4,
      policy: 'auto',
    });
  });

  it('forces scale 1 for unshiftable camera transmission and lowers the complete graph in auto mode', () => {
    const engine = new HybridEngine({
      device: makeMockDevice(),
      width: 1920,
      height: 1080,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      skyTint: [1, 1, 1],
      skyIrradiance: 1,
      debug: true,
    });
    try {
      const engineView = engine as unknown as Engine;
      const initCoordinator = (
        engine as unknown as {
          _initCoordinator: { startInit(): void };
        }
      )._initCoordinator;
      vi.spyOn(initCoordinator, 'startInit').mockImplementation(() => {});
      expect(engineView.debug?.frameResourceResolution?.()?.restirReservoirScale)
        .toBe(2);
      engine.setScene(transmissionScene());
      const resolution = engineView.debug?.frameResourceResolution?.();
      expect(resolution?.restirReservoirScale).toBe(1);
      expect(resolution!.effectiveInternalWidth).toBeLessThan(1920);
      expect(resolution!.effectiveInternalHeight).toBeLessThan(1080);
      expect(resolution!.restirDiWidth).toBe(resolution!.effectiveInternalWidth);
      expect(resolution!.restirDiHeight).toBe(resolution!.effectiveInternalHeight);
      expect(resolution!.restirGiWidth)
        .toBe(Math.floor(resolution!.effectiveInternalWidth / 2));
      expect(resolution!.restirGiHeight)
        .toBe(Math.floor(resolution!.effectiveInternalHeight / 2));
      expect(resolution!.resizePeakBytes).toBe(resolution!.persistentBytes);
      const initHost = (
        initCoordinator as unknown as {
          host: { readonly restirReservoirScale?: number };
        }
      ).host;
      expect(initHost.restirReservoirScale).toBe(1);
    } finally {
      engine.dispose();
    }
  });

  it('rejects an explicit coarse scale before accepting a transmission scene', () => {
    const engine = new HybridEngine({
      device: makeMockDevice(),
      width: 64,
      height: 64,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      skyTint: [1, 1, 1],
      skyIrradiance: 1,
      restirReservoirScale: 2,
    });
    try {
      expect(() => engine.setScene(transmissionScene())).toThrow(
        /cannot shift camera-transmission GI prefixes/,
      );
      expect(engine.getScene()).toBeNull();
    } finally {
      engine.dispose();
    }
  });
});
