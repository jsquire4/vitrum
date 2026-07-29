import { describe, it, expect } from 'vitest';
import { buildPassLayout, MAX_PASS_COUNT } from '../src/pipeline/timestampQueries.js';

describe('buildPassLayout — Sprint 9..18 + indirect atrous chain', () => {
  describe('atrous-variance mode (32 slots + trailing regir-build)', () => {
    const layout = buildPassLayout({ denoiserMode: 'atrous-variance' });

    it('prepends sample-budget at slot 0 (runs before RIS)', () => {
      expect(layout.index('sample-budget')).toBe(0);
    });

    it('DI RIS chain at 1..4; GI initial + PPG + reuse at 5..9; shade at 10', () => {
      expect(layout.index('ris')).toBe(1);
      expect(layout.index('temporal')).toBe(2);
      expect(layout.index('spatial-1')).toBe(3);
      expect(layout.index('spatial-2')).toBe(4);
      expect(layout.index('gi-ris')).toBe(5);
      expect(layout.index('ppg-update')).toBe(6);
      expect(layout.index('gi-temporal')).toBe(7);
      expect(layout.index('gi-spatial-1')).toBe(8);
      expect(layout.index('gi-spatial-2')).toBe(9);
      expect(layout.index('shade')).toBe(10);
    });

    it('places motion-vectors + gtao + gtao-upsample at slots 11..13', () => {
      expect(layout.index('motion-vectors')).toBe(11);
      expect(layout.index('gtao')).toBe(12);
      expect(layout.index('gtao-upsample')).toBe(13);
    });

    it('places cb-prefill at slot 14 (between ppg-update and denoiser labels)', () => {
      expect(layout.index('cb-prefill')).toBe(14);
    });

    it('places welford-temporal at slot 15 (after cb-prefill)', () => {
      expect(layout.index('welford-temporal')).toBe(15);
    });

    it('places atrous-variance-variance + 3 atrous-variance-atrous slots in order', () => {
      expect(layout.index('atrous-variance-variance')).toBe(16);
      expect(layout.index('atrous-variance-atrous-0')).toBe(17);
      expect(layout.index('atrous-variance-atrous-2')).toBe(19);
    });

    it('places indirect-temporal-accum, 4 atrous-indirect slots, indirect-combine, ddgi-border, transparent-oit, then temporal+resolve+composite tail', () => {
      expect(layout.index('indirect-temporal-accum')).toBe(20);
      expect(layout.index('atrous-indirect-0')).toBe(21);
      expect(layout.index('atrous-indirect-1')).toBe(22);
      expect(layout.index('atrous-indirect-2')).toBe(23);
      expect(layout.index('atrous-indirect-3')).toBe(24);
      expect(layout.index('indirect-combine')).toBe(25);
      expect(layout.index('ddgi-border-irr')).toBe(26);
      expect(layout.index('ddgi-border-vis')).toBe(27);
      expect(layout.index('transparent-oit')).toBe(28);
      expect(layout.index('temporalAccum')).toBe(29);
      expect(layout.index('resolve')).toBe(30);
      expect(layout.index('composite')).toBe(31);
    });

    it('does not include legacy atrous-0 label', () => {
      expect(() => layout.index('atrous-0')).toThrow(/not active/);
    });

    it('reports 33 slots (32 + the trailing opt-in regir-build slot)', () => {
      expect(layout.slotCount).toBe(33);
      expect(layout.labels).toHaveLength(33);
    });
  });

  describe('legacy atrous mode (31 slots + trailing regir-build)', () => {
    const layout = buildPassLayout({ denoiserMode: 'atrous' });

    it('GI initial + PPG + reuse at 5..9; shade at 10; motion/GTAO at 11..13; shared variance at 15; atrous at 16..18', () => {
      expect(layout.index('sample-budget')).toBe(0);
      expect(layout.index('gi-ris')).toBe(5);
      expect(layout.index('ppg-update')).toBe(6);
      expect(layout.index('gi-spatial-2')).toBe(9);
      expect(layout.index('shade')).toBe(10);
      expect(layout.index('motion-vectors')).toBe(11);
      expect(layout.index('gtao')).toBe(12);
      expect(layout.index('gtao-upsample')).toBe(13);
      expect(layout.index('cb-prefill')).toBe(14);
      expect(layout.index('welford-temporal')).toBe(15);
      expect(layout.index('atrous-0')).toBe(16);
      expect(layout.index('atrous-1')).toBe(17);
      expect(layout.index('atrous-2')).toBe(18);
    });

    it('places indirect-temporal-accum, atrous-indirect-0..3, indirect-combine, ddgi-border, transparent-oit, then temporalAccum/resolve/composite tail', () => {
      expect(layout.index('indirect-temporal-accum')).toBe(19);
      expect(layout.index('atrous-indirect-0')).toBe(20);
      expect(layout.index('atrous-indirect-3')).toBe(23);
      expect(layout.index('indirect-combine')).toBe(24);
      expect(layout.index('ddgi-border-irr')).toBe(25);
      expect(layout.index('ddgi-border-vis')).toBe(26);
      expect(layout.index('transparent-oit')).toBe(27);
      expect(layout.index('temporalAccum')).toBe(28);
      expect(layout.index('resolve')).toBe(29);
      expect(layout.index('composite')).toBe(30);
    });

    it('does not include atrous-variance labels', () => {
      expect(() => layout.index('atrous-variance-variance')).toThrow(/not active/);
    });

    it('reports 32 slots (31 + the trailing opt-in regir-build slot)', () => {
      expect(layout.slotCount).toBe(32);
    });
  });

  describe('MAX_PASS_COUNT invariant', () => {
    it('every layout fits within MAX_PASS_COUNT', () => {
      for (const denoiserMode of ['atrous-variance', 'atrous', 'svgf-real', 'bmfr'] as const) {
        const layout = buildPassLayout({ denoiserMode });
        expect(layout.slotCount).toBeLessThanOrEqual(MAX_PASS_COUNT);
      }
    });

    it('MAX_PASS_COUNT is 37 (includes all-mode variance + the trailing opt-in regir-build slot)', () => {
      expect(MAX_PASS_COUNT).toBe(37);
    });
  });

  describe('labels array matches index() lookup', () => {
    it('layout.labels[i] === label such that layout.index(label) === i', () => {
      const layout = buildPassLayout({ denoiserMode: 'atrous-variance' });
      for (let i = 0; i < layout.slotCount; i++) {
        const label = layout.labels[i];
        if (label === undefined) throw new Error(`undefined label at ${i}`);
        expect(layout.index(label)).toBe(i);
      }
    });

    it('first label is sample-budget; last is regir-build (trailing opt-in slot)', () => {
      const layout = buildPassLayout({ denoiserMode: 'atrous-variance' });
      expect(layout.labels[0]).toBe('sample-budget');
      // `regir-build` is placed LAST in the static order so its timestamp slot
      // is a trailing addition that doesn't shift any existing index; composite
      // keeps its slot just before it. (Dispatch order is independent: the
      // registry runs regir-build FIRST via topo-sort.)
      expect(layout.labels[layout.slotCount - 1]).toBe('regir-build');
      expect(layout.labels[layout.slotCount - 2]).toBe('composite');
    });
  });
});
