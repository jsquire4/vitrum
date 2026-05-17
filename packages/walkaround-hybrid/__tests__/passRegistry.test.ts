import { describe, it, expect, beforeEach } from 'vitest';
import { PassRegistry } from '../src/pipeline/PassRegistry.js';
import type { Pass, PassDispatchContext, PassGateOptions, PassInitContext } from '../src/pipeline/Pass.js';
import type { PassLabel } from '../src/pipeline/timestampQueries.js';

/**
 * Build a minimal stub Pass for graph-shape tests. The pass is inert — it
 * never gets initialized or dispatched in these tests; only its id,
 * dependencies, and gates matter for ordering + filtering.
 */
function stubPass(
  id: string,
  dependencies: readonly string[] = [],
  gate: (opts: PassGateOptions) => boolean = () => true,
): Pass {
  return {
    id,
    dependencies,
    passLabels: [id as PassLabel],
    gates: gate,
    initialize: async (_ctx: PassInitContext) => undefined,
    dispatch: (_ctx: PassDispatchContext) => undefined,
    dispose: () => undefined,
  };
}

const DEFAULT_OPTS: PassGateOptions = {
  denoiserMode: 'atrous-variance',
  ppgEnabled: false,
};

describe('PassRegistry', () => {
  let reg: PassRegistry;
  beforeEach(() => {
    reg = new PassRegistry();
  });

  describe('register / lookup', () => {
    it('registers and retrieves a pass by id', () => {
      const p = stubPass('foo');
      reg.register(p);
      expect(reg.has('foo')).toBe(true);
      expect(reg.get('foo')).toBe(p);
      expect(reg.size()).toBe(1);
    });

    it('throws on duplicate id', () => {
      reg.register(stubPass('foo'));
      expect(() => reg.register(stubPass('foo'))).toThrow(/duplicate pass id "foo"/);
    });

    it('returns undefined for unknown id without throwing', () => {
      expect(reg.get('nope')).toBeUndefined();
      expect(reg.has('nope')).toBe(false);
    });

    it('ids() reflects registration order', () => {
      reg.register(stubPass('c'));
      reg.register(stubPass('a'));
      reg.register(stubPass('b'));
      expect(reg.ids()).toEqual(['c', 'a', 'b']);
    });
  });

  describe('sortedPasses — topological order', () => {
    it('handles no dependencies (lex order within tier)', () => {
      reg.register(stubPass('c'));
      reg.register(stubPass('a'));
      reg.register(stubPass('b'));
      const sorted = reg.sortedPasses().map((p) => p.id);
      expect(sorted).toEqual(['a', 'b', 'c']);
    });

    it('handles a linear chain A -> B -> C', () => {
      reg.register(stubPass('C', ['B']));
      reg.register(stubPass('A'));
      reg.register(stubPass('B', ['A']));
      expect(reg.sortedPasses().map((p) => p.id)).toEqual(['A', 'B', 'C']);
    });

    it('handles a diamond A -> {B, C} -> D', () => {
      reg.register(stubPass('A'));
      reg.register(stubPass('B', ['A']));
      reg.register(stubPass('C', ['A']));
      reg.register(stubPass('D', ['B', 'C']));
      const sorted = reg.sortedPasses().map((p) => p.id);
      expect(sorted[0]).toBe('A');
      expect(sorted[3]).toBe('D');
      // B and C tie within the same dependency tier; lex order resolves it.
      expect([sorted[1], sorted[2]]).toEqual(['B', 'C']);
    });

    it('is deterministic across multiple sorts of the same registration', () => {
      reg.register(stubPass('zeta', ['alpha']));
      reg.register(stubPass('alpha'));
      reg.register(stubPass('beta', ['alpha']));
      reg.register(stubPass('gamma', ['alpha']));
      const a = reg.sortedPasses().map((p) => p.id);
      const b = reg.sortedPasses().map((p) => p.id);
      expect(a).toEqual(b);
      expect(a).toEqual(['alpha', 'beta', 'gamma', 'zeta']);
    });

    it('caches the sorted result until register invalidates', () => {
      reg.register(stubPass('a'));
      const first = reg.sortedPasses();
      const second = reg.sortedPasses();
      // Same underlying array of refs; values match by content.
      expect(first.map((p) => p.id)).toEqual(second.map((p) => p.id));
      reg.register(stubPass('b'));
      const third = reg.sortedPasses();
      expect(third.map((p) => p.id)).toEqual(['a', 'b']);
    });
  });

  describe('error paths', () => {
    it('throws on missing dependency at sort time', () => {
      reg.register(stubPass('A', ['missing']));
      expect(() => reg.sortedPasses()).toThrow(/unknown dependency "missing"/);
    });

    it('detects a 2-cycle', () => {
      reg.register(stubPass('A', ['B']));
      reg.register(stubPass('B', ['A']));
      expect(() => reg.sortedPasses()).toThrow(/cycle detected/);
    });

    it('detects a 3-cycle', () => {
      reg.register(stubPass('A', ['C']));
      reg.register(stubPass('B', ['A']));
      reg.register(stubPass('C', ['B']));
      expect(() => reg.sortedPasses()).toThrow(/cycle detected/);
    });
  });

  describe('activePasses — gate filtering', () => {
    it('returns only passes whose gate returns true', () => {
      reg.register(stubPass('always'));
      reg.register(stubPass('only-svgf', [], (o) => o.denoiserMode === 'svgf-real'));
      reg.register(stubPass('only-ppg', [], (o) => o.ppgEnabled));
      const active = reg.activePasses(DEFAULT_OPTS).map((p) => p.id);
      expect(active).toEqual(['always']);
    });

    it('respects gate state changes across calls', () => {
      reg.register(stubPass('only-svgf', [], (o) => o.denoiserMode === 'svgf-real'));
      const off = reg.activePasses(DEFAULT_OPTS).map((p) => p.id);
      const on = reg
        .activePasses({ ...DEFAULT_OPTS, denoiserMode: 'svgf-real' })
        .map((p) => p.id);
      expect(off).toEqual([]);
      expect(on).toEqual(['only-svgf']);
    });

    it('preserves topological order in the filtered subset', () => {
      reg.register(stubPass('A'));
      reg.register(stubPass('B', ['A'], () => false));
      reg.register(stubPass('C', ['A']));
      reg.register(stubPass('D', ['C']));
      const active = reg.activePasses(DEFAULT_OPTS).map((p) => p.id);
      expect(active).toEqual(['A', 'C', 'D']);
    });
  });
});
