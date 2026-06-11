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
      'pt-webgl2',
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
    expect(BACKEND_PROMISE_LEDGER['pt-webgl2'].supportsAddRemovePrimitive).toBe(true);
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

  it('marks reserved displacement material fields unsupported on every shipping backend', () => {
    for (const rec of Object.values(BACKEND_PROMISE_LEDGER)) {
      expect(rec.supportDetails.materials.displacementMap).toBe('unsupported');
      expect(rec.supportDetails.materials.displacementScale).toBe('unsupported');
      expect(rec.supportDetails.materials.displacementBias).toBe('unsupported');
    }
  });

  it('pins onError: true for all three shipping backends (item 28 — GPU error surface)', () => {
    for (const [id, rec] of Object.entries(BACKEND_PROMISE_LEDGER)) {
      expect(rec.methodPromises.onError).toBe(true);
      // This is also an exhaustive contract check so new backends are forced
      // to decide rather than inherit a silent false default.
      expect(typeof rec.methodPromises.onError).toBe('boolean');
      void id;
    }
  });

  it('pins onWarning: true for all three shipping backends (ENGINE-01 warning surface)', () => {
    for (const [id, rec] of Object.entries(BACKEND_PROMISE_LEDGER)) {
      expect(rec.methodPromises.onWarning).toBe(true);
      expect(typeof rec.methodPromises.onWarning).toBe('boolean');
      void id;
    }
  });

  it('pins captureFrame: true for all three shipping backends (item 31 — pixel readback)', () => {
    for (const [id, rec] of Object.entries(BACKEND_PROMISE_LEDGER)) {
      expect(rec.methodPromises.captureFrame).toBe(true);
      expect(typeof rec.methodPromises.captureFrame).toBe('boolean');
      void id;
    }
  });

  it('pins walkaround mutation fidelity rows that differ from the boolean patch surface', () => {
    const rec = BACKEND_PROMISE_LEDGER['walkaround-hybrid'];

    expect(rec.incrementalPatchSupport).toEqual({
      transform: true,
      positions: true,
      material: true,
      emitter: true,
      topology: true,
    });
    expect(rec.supportDetails.mutations).toEqual({
      transform: 'native',
      positions: 'native',
      material: 'native',
      emitter: 'native',
      topology: 'fallback-rebuild',
      addPrimitive: 'fallback-rebuild',
      removePrimitive: 'fallback-rebuild',
      environment: 'approximate',
      resize: 'native',
      lighting: 'native',
    });
    expect(rec.methodPromises.setSize).toBe(true);
    expect(rec.methodPromises.updateLighting).toBe(true);
  });

  it('pins PT backend resize and lighting as per-frame/offscreen concerns, not optional methods', () => {
    for (const backendId of ['pt-webgl2', 'pt-webgpu'] as const) {
      const rec = BACKEND_PROMISE_LEDGER[backendId];

      expect(rec.presentationMode).toBe('offscreen-texture');
      expect(rec.frameInputPromises).toEqual({
        honorsViewportPerFrame: true,
        requiresSwapChainView: false,
        honorsPerFrameBounces: true,
      });
      expect(rec.supportDetails.mutations.resize).toBe('unsupported');
      expect(rec.supportDetails.mutations.lighting).toBe('unsupported');
      expect(rec.methodPromises.setSize).toBe(false);
      expect(rec.methodPromises.updateLighting).toBe(false);
    }
  });

  it('pins environment fidelity rows for DDGI SH walkaround versus path tracers', () => {
    expect(BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportDetails.environments).toEqual({
      none: 'native',
      // Wave 4 (2026-06-10): promoted approximate→native — env DI NEE candidate
      // in RIS, HDRI in DDGI probe misses + risGiNrc + RC, runtime CDF rebuild
      // on updateEnvironment. Radiometric A/B pending V28-B.
      hdri: 'native',
      // procedural-sky degrades to scalar tint via resolveHybridEnvironment
      // (mode: 'procedural-sky-approx'; turbidity/rayleigh/mie ignored; warn emitted).
      // 'approximate' reflects the degraded-but-functional reality. Item 18c.
      'procedural-sky': 'approximate',
    });
    expect(BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportedEnvironmentKinds).toEqual(['none', 'hdri', 'procedural-sky']);

    expect(BACKEND_PROMISE_LEDGER['pt-webgl2'].supportDetails.environments).toEqual({
      none: 'native',
      hdri: 'native',
      'procedural-sky': 'unsupported',
    });
    expect(BACKEND_PROMISE_LEDGER['pt-webgl2'].supportedEnvironmentKinds).toEqual(['none', 'hdri']);

    // procedural-sky is 'approximate': heuristic tint, not a full Preetham
    // model (turbidity/rayleigh/mieDirectionalG are ignored). Interim per
    // plan/v1-closure-plan-2026-06-10.md; promote to 'native' when the
    // Preetham implementation lands in Wave 2.
    expect(BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.environments).toEqual({
      none: 'native',
      hdri: 'native',
      'procedural-sky': 'approximate',
    });
    expect(BACKEND_PROMISE_LEDGER['pt-webgpu'].supportedEnvironmentKinds).toEqual(['none', 'hdri', 'procedural-sky']);
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
