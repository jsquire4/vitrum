import { describe, it, expect, beforeEach } from 'vitest';
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

describe('registerBuiltinDenoisers', () => {
  it('populates a fresh registry with all 6 built-in denoiser ids', () => {
    const reg = new DenoiserRegistry();
    registerBuiltinDenoisers(reg);
    expect(reg.size()).toBe(6);
    expect(reg.ids()).toEqual([
      'none',
      'atrous',
      'atrous-variance',
      'svgf-real',
      'neural',
      'oidn-final',
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
