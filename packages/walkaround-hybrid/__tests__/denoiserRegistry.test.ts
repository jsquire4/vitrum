import { describe, it, expect, beforeEach } from 'vitest';
import {
  DenoiserRegistry,
  type Denoiser,
  type DenoiserDispatchContext,
  type DenoiserId,
  type DenoiserInitContext,
} from '../src/pipeline/denoisers/index.js';

/**
 * Build a minimal stub Denoiser for registry-shape tests. None of the
 * lifecycle methods run in these tests.
 */
function stubDenoiser(id: DenoiserId, disabled = false): Denoiser {
  return {
    id,
    disabled,
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
