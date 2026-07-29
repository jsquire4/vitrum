import { describe, expect, it, vi } from 'vitest';
import type {
  BackendSupportManifest,
  Scene,
} from '@vitrum/core';
import { HybridEngine, type HybridEngineOptions } from '../HybridEngine.js';
import type { SceneBVHBuffers } from '../restir/bvhCore.js';

function makeDeviceStub(): GPUDevice {
  const noop = vi.fn();
  return {
    createBuffer: vi.fn(() => ({ destroy: noop })),
    createTexture: vi.fn(() => ({ createView: noop, destroy: noop })),
    queue: { submit: noop, writeBuffer: noop },
    features: new Set<string>(),
    limits: {},
    addEventListener: noop,
    removeEventListener: noop,
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function makeOptions(): HybridEngineOptions {
  return {
    device: makeDeviceStub(),
    width: 16,
    height: 16,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 1,
    skyTint: [1, 1, 1],
    skyIrradiance: 1,
  };
}

function numericReadTrap<T extends Float32Array | Uint32Array>(
  view: T,
  onRead: () => void,
): T {
  return new Proxy(view, {
    get(target, key) {
      if (typeof key === 'string' && /^(?:0|[1-9]\d*)$/.test(key)) {
        onRead();
        throw new Error(`unchanged geometry element ${key} was read`);
      }
      const value = Reflect.get(target, key, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
}

function trappedScene(onGeometryRead: () => void): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'triangle',
      positions: numericReadTrap(
        new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        onGeometryRead,
      ),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      indices: numericReadTrap(new Uint32Array([0, 1, 2]), onGeometryRead),
      material: {
        baseColor: [0.7, 0.7, 0.7],
        roughness: 0.5,
        metallic: 0,
      },
    }],
    emitters: [{
      kind: 'point',
      id: 'lamp',
      position: [0, 2, 0],
      color: [1, 1, 1],
      intensity: 1,
    }],
    environment: { kind: 'none' },
  };
}

interface MutableHybridInternals {
  _state: string;
  _lastScene: Scene | null;
  _renderScene: Scene | null;
  _bvhBuffers: SceneBVHBuffers | null;
  _supportManifest: BackendSupportManifest;
  _routePrimitiveUpdate: () => never;
}

describe('HybridEngine incremental validation scope', () => {
  it('material-only update does not rewalk unchanged position/index elements', () => {
    let geometryReads = 0;
    const scene = trappedScene(() => {
      geometryReads += 1;
    });
    const engine = new HybridEngine(makeOptions());
    const internals = engine as unknown as MutableHybridInternals;
    internals._state = 'ready';
    internals._lastScene = scene;
    internals._renderScene = scene;
    internals._bvhBuffers = null;
    internals._routePrimitiveUpdate = () => {
      throw new Error('targeted-validation-complete');
    };

    try {
      expect(() => engine.updatePrimitive('triangle', {
        material: { roughness: 0.25 },
      } as never)).toThrow('targeted-validation-complete');
      expect(geometryReads).toBe(0);
    } finally {
      engine.dispose();
    }
  });

  it('emitter patch reaches manifest validation without rescanning geometry', () => {
    let geometryReads = 0;
    const scene = trappedScene(() => {
      geometryReads += 1;
    });
    const engine = new HybridEngine(makeOptions());
    const internals = engine as unknown as MutableHybridInternals;
    internals._state = 'ready';
    internals._lastScene = scene;
    internals._renderScene = scene;
    internals._bvhBuffers = {} as SceneBVHBuffers;
    internals._supportManifest = {
      ...internals._supportManifest,
      emitters: {
        ...internals._supportManifest.emitters,
        point: 'unsupported',
      },
    };

    try {
      expect(() => engine.updateEmitter('lamp', { intensity: 2 }))
        .toThrow(/scene capability mismatch/i);
      expect(geometryReads).toBe(0);
    } finally {
      engine.dispose();
    }
  });
});
