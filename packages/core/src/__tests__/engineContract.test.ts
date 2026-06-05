import { describe, expect, it } from 'vitest';
import {
  BACKEND_PROMISE_LEDGER,
  asBackendTexture,
  asBackendTextureFormat,
  narrowToBackendTexture,
  narrowToBackendTextureFormat,
} from '../index.js';
import { asMat4, isMat4 } from '../scene/math.js';

describe('backend promise ledger', () => {
  it('contains all expected backend IDs', () => {
    expect(Object.keys(BACKEND_PROMISE_LEDGER).sort()).toEqual([
      'pt-webgl',
      'pt-webgpu',
      'walkaround-hybrid',
    ]);
  });

  it('keeps incremental patch support exhaustive for every backend', () => {
    for (const rec of Object.values(BACKEND_PROMISE_LEDGER)) {
      expect(typeof rec.incrementalPatchSupport.transform).toBe('boolean');
      expect(typeof rec.incrementalPatchSupport.positions).toBe('boolean');
      expect(typeof rec.incrementalPatchSupport.material).toBe('boolean');
      expect(typeof rec.incrementalPatchSupport.emitter).toBe('boolean');
      expect(typeof rec.incrementalPatchSupport.topology).toBe('boolean');
    }
  });

  it('declares add/remove-primitive support exhaustively and consistently', () => {
    for (const rec of Object.values(BACKEND_PROMISE_LEDGER)) {
      expect(typeof rec.supportsAddRemovePrimitive).toBe('boolean');
      expect(typeof rec.methodPromises.addPrimitive).toBe('boolean');
      expect(typeof rec.methodPromises.removePrimitive).toBe('boolean');
      // The capability flag and the two method promises must agree — a backend
      // that advertises add/remove support must expose both methods, and vice
      // versa (no half-implemented surface).
      expect(rec.methodPromises.addPrimitive).toBe(rec.supportsAddRemovePrimitive);
      expect(rec.methodPromises.removePrimitive).toBe(rec.supportsAddRemovePrimitive);
    }
  });

  it('pins the per-backend add/remove-primitive truth table', () => {
    expect(BACKEND_PROMISE_LEDGER['pt-webgpu'].supportsAddRemovePrimitive).toBe(true);
    expect(BACKEND_PROMISE_LEDGER['pt-webgl'].supportsAddRemovePrimitive).toBe(true);
    expect(BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportsAddRemovePrimitive).toBe(true);
  });

  it('keeps supportDetails aligned with coarse supported-kind sets', () => {
    for (const rec of Object.values(BACKEND_PROMISE_LEDGER)) {
      const primitiveKinds = new Set<string>(rec.supportedPrimitiveKinds);
      for (const [kind, mode] of Object.entries(rec.supportDetails.primitives)) {
        expect(primitiveKinds.has(kind)).toBe(mode !== 'unsupported');
      }

      const emitterKinds = new Set<string>(rec.supportedEmitterKinds);
      for (const [kind, mode] of Object.entries(rec.supportDetails.emitters)) {
        expect(emitterKinds.has(kind)).toBe(mode !== 'unsupported');
      }

      const environmentKinds = new Set<string>(rec.supportedEnvironmentKinds);
      for (const [kind, mode] of Object.entries(rec.supportDetails.environments)) {
        expect(environmentKinds.has(kind)).toBe(mode !== 'unsupported');
      }

      const analyticShapes = new Set<string>(rec.supportedAnalyticShapes);
      for (const [shape, mode] of Object.entries(rec.supportDetails.analyticShapes)) {
        expect(analyticShapes.has(shape)).toBe(mode !== 'unsupported');
      }
    }
  });

  it('keeps supportDetails mutation rows aligned with promised optional methods', () => {
    for (const rec of Object.values(BACKEND_PROMISE_LEDGER)) {
      expect(rec.supportDetails.mutations.emitter !== 'unsupported').toBe(rec.methodPromises.updateEmitter);
      expect(rec.supportDetails.mutations.environment !== 'unsupported').toBe(rec.methodPromises.updateEnvironment);
      expect(rec.supportDetails.mutations.addPrimitive !== 'unsupported').toBe(rec.methodPromises.addPrimitive);
      expect(rec.supportDetails.mutations.removePrimitive !== 'unsupported').toBe(rec.methodPromises.removePrimitive);
      expect(rec.supportDetails.mutations.resize !== 'unsupported').toBe(rec.methodPromises.setSize);
      expect(rec.supportDetails.mutations.lighting !== 'unsupported').toBe(rec.methodPromises.updateLighting);
    }
  });
});

describe('mat4 branding', () => {
  it('accepts exactly 16-element matrices', () => {
    const m = asMat4(new Float32Array(16));
    expect(isMat4(m)).toBe(true);
  });

  it('rejects non-16-element arrays', () => {
    expect(() => asMat4(new Float32Array(9))).toThrow(/Mat4 requires 16 elements/);
  });
});

describe('backend texture branding', () => {
  it('round-trips branded backend texture handles through narrow helpers', () => {
    const h = { texture: true as const };
    const branded = asBackendTexture<'webgpu', { texture: true }>(h);
    expect(narrowToBackendTexture<'webgpu', { texture: true }>(branded)).toBe(h);
  });

  it('round-trips branded backend texture formats through narrow helpers', () => {
    const f = 'rgba16float';
    const branded = asBackendTextureFormat<'webgpu', string>(f);
    expect(narrowToBackendTextureFormat<'webgpu', string>(branded)).toBe(f);
  });
});
