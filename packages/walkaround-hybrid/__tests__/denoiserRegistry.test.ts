import { describe, it, expect, beforeEach } from 'vitest';
import { normalizeDenoiserConfig, type DenoiserConfig } from '@vitrum/core';
import {
  DenoiserRegistry,
  type Denoiser,
  type DenoiserDispatchContext,
  type DenoiserId,
  type DenoiserInitContext,
} from '../src/pipeline/denoisers/index.js';
import { AtrousDenoiser } from '../src/pipeline/denoisers/atrous.js';
import { AtrousVarianceDenoiser } from '../src/pipeline/denoisers/atrousVariance.js';
import { NeuralDenoiser } from '../src/pipeline/denoisers/neural.js';
import { NoneDenoiser } from '../src/pipeline/denoisers/none.js';
import { OIDNFinalDenoiser } from '../src/pipeline/denoisers/oidnFinal.js';
import { SVGFRealDenoiser } from '../src/pipeline/denoisers/svgfReal.js';
import { registerBuiltinDenoisers } from '../src/pipeline/denoisers/registerBuiltinDenoisers.js';

/**
 * Build a minimal stub Denoiser for registry-shape tests. None of the
 * lifecycle methods run in these tests.
 */
function stubDenoiser(id: DenoiserId, disabled = false): Denoiser {
  return {
    id,
    disabled,
    passLabels: [],
    initialize: async (_ctx: DenoiserInitContext) => undefined,
    dispatch: (_ctx: DenoiserDispatchContext) => null,
    resize: (_w: number, _h: number) => undefined,
    dispose: () => undefined,
  };
}

describe('DenoiserRegistry', () => {
  let reg: DenoiserRegistry;
  beforeEach(() => {
    reg = new DenoiserRegistry();
  });

  describe('register / lookup', () => {
    it('registers and retrieves a denoiser by id', () => {
      const d = stubDenoiser('atrous');
      reg.register(d);
      expect(reg.has('atrous')).toBe(true);
      expect(reg.lookup('atrous')).toBe(d);
      expect(reg.size()).toBe(1);
    });

    it('throws on duplicate id', () => {
      reg.register(stubDenoiser('atrous'));
      expect(() => reg.register(stubDenoiser('atrous'))).toThrow(/duplicate id "atrous"/);
    });

    it('lookup throws on unknown id with a known-ids hint', () => {
      reg.register(stubDenoiser('atrous'));
      reg.register(stubDenoiser('atrous-variance'));
      expect(() => reg.lookup('svgf-real')).toThrow(/unknown denoiser "svgf-real"/);
      expect(() => reg.lookup('svgf-real')).toThrow(/atrous.*atrous-variance/);
    });
  });

  describe('disabled denoiser placeholders', () => {
    it('lookup rejects a disabled denoiser', () => {
      reg.register(stubDenoiser('neural', true));
      expect(() => reg.lookup('neural')).toThrow(/registered but disabled/);
    });

    it('has() returns true for a disabled denoiser (it is registered)', () => {
      reg.register(stubDenoiser('neural', true));
      expect(reg.has('neural')).toBe(true);
    });

    it('ids() includes disabled placeholders for diagnostics', () => {
      reg.register(stubDenoiser('none'));
      reg.register(stubDenoiser('neural', true));
      reg.register(stubDenoiser('oidn-final', true));
      expect(reg.ids()).toEqual(['none', 'neural', 'oidn-final']);
    });

    it('an enabled denoiser remains lookupable after a disabled sibling is registered', () => {
      reg.register(stubDenoiser('atrous'));
      reg.register(stubDenoiser('neural', true));
      expect(reg.lookup('atrous').id).toBe('atrous');
      expect(() => reg.lookup('neural')).toThrow(/disabled/);
    });
  });
});

/**
 * Built-in denoiser entries (W1-R3). These tests exercise only the shape +
 * id + disabled-flag invariants — actual GPU dispatch is exercised by the
 * pipeline-level integration tests once a real WebGPU device is available.
 */
describe('Builtin Denoiser entries', () => {
  it('NoneDenoiser carries id "none" and is enabled by default', () => {
    const d = new NoneDenoiser();
    expect(d.id).toBe('none');
    expect(d.disabled).toBeUndefined();
  });

  it('AtrousDenoiser carries id "atrous" and is enabled by default', () => {
    const d = new AtrousDenoiser();
    expect(d.id).toBe('atrous');
    expect(d.disabled).toBeUndefined();
  });

  it('AtrousVarianceDenoiser carries id "atrous-variance" and is enabled by default', () => {
    const d = new AtrousVarianceDenoiser();
    expect(d.id).toBe('atrous-variance');
    expect(d.disabled).toBeUndefined();
  });

  it('SVGFRealDenoiser carries id "svgf-real" and is enabled by default', () => {
    const d = new SVGFRealDenoiser();
    expect(d.id).toBe('svgf-real');
    expect(d.disabled).toBeUndefined();
  });

  it('NeuralDenoiser carries id "neural" and is disabled (W10 placeholder)', () => {
    const d = new NeuralDenoiser();
    expect(d.id).toBe('neural');
    expect(d.disabled).toBe(true);
  });

  it('OIDNFinalDenoiser carries id "oidn-final" and is disabled (W11 placeholder)', () => {
    const d = new OIDNFinalDenoiser();
    expect(d.id).toBe('oidn-final');
    expect(d.disabled).toBe(true);
  });

  it('NoneDenoiser.dispatch returns null (pass-through, sample raw HDR)', () => {
    const d = new NoneDenoiser();
    expect(d.dispatch({} as DenoiserDispatchContext)).toBeNull();
  });
});

// ────────────────────────────────────────────────────────────────────────────
// W3-D4 — DenoiserConfig discriminated union
// ────────────────────────────────────────────────────────────────────────────

describe('W3-D4 — DenoiserConfig discriminated union', () => {
  describe('normalizeDenoiserConfig — DU passthrough', () => {
    it('passes through { kind: "atrous-variance" } unchanged', () => {
      const cfg: DenoiserConfig = { kind: 'atrous-variance' };
      expect(normalizeDenoiserConfig(cfg)).toEqual({ kind: 'atrous-variance' });
    });

    it('passes through { kind: "neural", weights } and preserves the weights', () => {
      const weights = { layers: [] };
      const cfg: DenoiserConfig = { kind: 'neural', weights };
      const out = normalizeDenoiserConfig(cfg);
      expect(out.kind).toBe('neural');
      if (out.kind === 'neural') {
        expect(out.weights).toBe(weights);
      }
    });

    it('passes through { kind: "oidn-final", modelUrl } and preserves the URL', () => {
      const cfg: DenoiserConfig = { kind: 'oidn-final', modelUrl: '/oidn.onnx' };
      const out = normalizeDenoiserConfig(cfg);
      expect(out.kind).toBe('oidn-final');
      if (out.kind === 'oidn-final') {
        expect(out.modelUrl).toBe('/oidn.onnx');
      }
    });
  });

  describe('normalizeDenoiserConfig — string normalisation', () => {
    it('undefined → atrous-variance default', () => {
      expect(normalizeDenoiserConfig(undefined)).toEqual({ kind: 'atrous-variance' });
    });

    it('plain string ids become DU singletons', () => {
      expect(normalizeDenoiserConfig('none')).toEqual({ kind: 'none' });
      expect(normalizeDenoiserConfig('atrous')).toEqual({ kind: 'atrous' });
      expect(normalizeDenoiserConfig('atrous-variance')).toEqual({ kind: 'atrous-variance' });
      expect(normalizeDenoiserConfig('svgf-real')).toEqual({ kind: 'svgf-real' });
    });

    it('deprecated "svgf" alias normalises to atrous-variance', () => {
      expect(normalizeDenoiserConfig('svgf')).toEqual({ kind: 'atrous-variance' });
    });

    it('bare "neural" string is rejected — weights required', () => {
      expect(() => normalizeDenoiserConfig('neural')).toThrow(/neural.*requires weights/);
    });

    it('bare "oidn-final" string is rejected — modelUrl required', () => {
      expect(() => normalizeDenoiserConfig('oidn-final')).toThrow(/oidn-final.*requires.*modelUrl/);
    });
  });

  describe('per-mode required config is compile-enforced', () => {
    // These tests exist primarily for the @ts-expect-error directives — if
    // a future refactor accidentally widens the DU back to a string union,
    // the directives fail, surfacing the regression.

    it('{ kind: "neural" } without weights is a compile error', () => {
      // @ts-expect-error - weights is required for kind: 'neural'
      const bad: DenoiserConfig = { kind: 'neural' };
      // Belt-and-braces runtime sanity (the DU still has 'kind').
      expect(bad.kind).toBe('neural');
    });

    it('{ kind: "oidn-final" } without modelUrl is a compile error', () => {
      // @ts-expect-error - modelUrl is required for kind: 'oidn-final'
      const bad: DenoiserConfig = { kind: 'oidn-final' };
      expect(bad.kind).toBe('oidn-final');
    });

    it('{ kind: "atrous-variance", weights: x } rejects extraneous fields under strict object literal', () => {
      // Variants without per-mode config don't accept stray keys at the
      // literal site: TS narrows to the no-extras variant.
      // @ts-expect-error - 'weights' is not a property of kind: 'atrous-variance'
      const bad: DenoiserConfig = { kind: 'atrous-variance', weights: {} };
      expect(bad.kind).toBe('atrous-variance');
    });
  });

  describe('DenoiserRegistry.lookupConfig — DU → registered Denoiser', () => {
    it('looks up an enabled denoiser via its DenoiserConfig kind', () => {
      const reg = new DenoiserRegistry();
      registerBuiltinDenoisers(reg);
      expect(reg.lookupConfig({ kind: 'atrous-variance' }).id).toBe('atrous-variance');
      expect(reg.lookupConfig({ kind: 'svgf-real' }).id).toBe('svgf-real');
    });

    it('rejects disabled-placeholder DU variants with the registry error', () => {
      const reg = new DenoiserRegistry();
      registerBuiltinDenoisers(reg);
      expect(() =>
        reg.lookupConfig({ kind: 'neural', weights: { layers: [] } }),
      ).toThrow(/registered but disabled/);
      expect(() =>
        reg.lookupConfig({ kind: 'oidn-final', modelUrl: '/oidn.onnx' }),
      ).toThrow(/registered but disabled/);
    });
  });
});

describe('registerBuiltinDenoisers', () => {
  it('populates a fresh registry with all 6 built-in denoiser ids', () => {
    const reg = new DenoiserRegistry();
    registerBuiltinDenoisers(reg);
    expect(reg.size()).toBe(6);
    expect(reg.ids()).toEqual([
      'none', 'atrous', 'atrous-variance', 'svgf-real', 'neural', 'oidn-final',
    ]);
  });

  it('looks up the enabled denoisers without throwing', () => {
    const reg = new DenoiserRegistry();
    registerBuiltinDenoisers(reg);
    expect(reg.lookup('none').id).toBe('none');
    expect(reg.lookup('atrous').id).toBe('atrous');
    expect(reg.lookup('atrous-variance').id).toBe('atrous-variance');
    expect(reg.lookup('svgf-real').id).toBe('svgf-real');
  });

  it('rejects the disabled placeholders with a clear error', () => {
    const reg = new DenoiserRegistry();
    registerBuiltinDenoisers(reg);
    expect(() => reg.lookup('neural')).toThrow(/registered but disabled/);
    expect(() => reg.lookup('oidn-final')).toThrow(/registered but disabled/);
  });
});
