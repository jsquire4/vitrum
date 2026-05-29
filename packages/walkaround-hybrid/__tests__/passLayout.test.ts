import { describe, it, expect } from 'vitest';
import { buildPassLayout, MAX_PASS_COUNT } from '../src/pipeline/timestampQueries.js';

describe('buildPassLayout — Sprint 9..18 + indirect atrous chain', () => {
  describe('atrous-variance mode (31 slots)', () => {
    const layout = buildPassLayout({ denoiserMode: 'atrous-variance' });

    it('prepends sample-budget at slot 0 (runs before RIS)', () => {
      expect(layout.index('sample-budget')).toBe(0);
    });

    it('DI RIS chain at 1..4; GI block at 5..8; shade at 9', () => {
      expect(layout.index('ris')).toBe(1);
      expect(layout.index('temporal')).toBe(2);
      expect(layout.index('spatial-1')).toBe(3);
      expect(layout.index('spatial-2')).toBe(4);
      expect(layout.index('gi-ris')).toBe(5);
      expect(layout.index('gi-temporal')).toBe(6);
      expect(layout.index('gi-spatial-1')).toBe(7);
      expect(layout.index('gi-spatial-2')).toBe(8);
      expect(layout.index('shade')).toBe(9);
    });

    it('places motion-vectors + gtao + gtao-upsample + ppg passes at slots 10..14', () => {
      expect(layout.index('motion-vectors')).toBe(10);
      expect(layout.index('gtao')).toBe(11);
      expect(layout.index('gtao-upsample')).toBe(12);
      expect(layout.index('ppg-update')).toBe(13);
      expect(layout.index('ppg-guide')).toBe(14);
    });

    it('places welford-temporal at slot 15 (after ppg)', () => {
      expect(layout.index('welford-temporal')).toBe(15);
    });

    it('places atrous-variance-variance + 3 atrous-variance-atrous slots in order', () => {
      expect(layout.index('atrous-variance-variance')).toBe(16);
      expect(layout.index('atrous-variance-atrous-0')).toBe(17);
      expect(layout.index('atrous-variance-atrous-2')).toBe(19);
    });

    it('places indirect-temporal-accum, 4 atrous-indirect slots, indirect-combine, ddgi-border-irr/vis, then temporal+resolve+composite tail', () => {
      expect(layout.index('indirect-temporal-accum')).toBe(20);
      expect(layout.index('atrous-indirect-0')).toBe(21);
      expect(layout.index('atrous-indirect-1')).toBe(22);
      expect(layout.index('atrous-indirect-2')).toBe(23);
      expect(layout.index('atrous-indirect-3')).toBe(24);
      expect(layout.index('indirect-combine')).toBe(25);
      expect(layout.index('ddgi-border-irr')).toBe(26);
      expect(layout.index('ddgi-border-vis')).toBe(27);
      expect(layout.index('temporalAccum')).toBe(28);
      expect(layout.index('resolve')).toBe(29);
      expect(layout.index('composite')).toBe(30);
    });

    it('does not include legacy atrous-0 label', () => {
      expect(() => layout.index('atrous-0')).toThrow(/not active/);
    });

    it('reports 31 slots', () => {
      expect(layout.slotCount).toBe(31);
      expect(layout.labels).toHaveLength(31);
    });
  });

  describe('legacy atrous mode (29 slots)', () => {
    const layout = buildPassLayout({ denoiserMode: 'atrous' });

    it('GI block at 5..8; shade at 9; motion-vectors + gtao + upsample + ppg at 10..14; atrous-0..2 at 15..17', () => {
      expect(layout.index('sample-budget')).toBe(0);
      expect(layout.index('gi-ris')).toBe(5);
      expect(layout.index('gi-spatial-2')).toBe(8);
      expect(layout.index('shade')).toBe(9);
      expect(layout.index('motion-vectors')).toBe(10);
      expect(layout.index('gtao')).toBe(11);
      expect(layout.index('gtao-upsample')).toBe(12);
      expect(layout.index('ppg-update')).toBe(13);
      expect(layout.index('ppg-guide')).toBe(14);
      expect(layout.index('atrous-0')).toBe(15);
      expect(layout.index('atrous-1')).toBe(16);
      expect(layout.index('atrous-2')).toBe(17);
    });

    it('places indirect-temporal-accum, atrous-indirect-0..3, indirect-combine, ddgi-border-irr/vis, then temporalAccum/resolve/composite tail', () => {
      expect(layout.index('indirect-temporal-accum')).toBe(18);
      expect(layout.index('atrous-indirect-0')).toBe(19);
      expect(layout.index('atrous-indirect-3')).toBe(22);
      expect(layout.index('indirect-combine')).toBe(23);
      expect(layout.index('ddgi-border-irr')).toBe(24);
      expect(layout.index('ddgi-border-vis')).toBe(25);
      expect(layout.index('temporalAccum')).toBe(26);
      expect(layout.index('resolve')).toBe(27);
      expect(layout.index('composite')).toBe(28);
    });

    it('does not include atrous-variance labels', () => {
      expect(() => layout.index('welford-temporal')).toThrow(/not active/);
    });

    it('reports 29 slots', () => {
      expect(layout.slotCount).toBe(29);
    });
  });

  describe('MAX_PASS_COUNT invariant', () => {
    it('every layout fits within MAX_PASS_COUNT', () => {
      for (const denoiserMode of ['atrous-variance', 'atrous', 'svgf-real', 'bmfr'] as const) {
        const layout = buildPassLayout({ denoiserMode });
        expect(layout.slotCount).toBeLessThanOrEqual(MAX_PASS_COUNT);
      }
    });

    it('MAX_PASS_COUNT is 34 (includes motion-vectors pass)', () => {
      expect(MAX_PASS_COUNT).toBe(34);
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

    it('first label is sample-budget; last is composite', () => {
      const layout = buildPassLayout({ denoiserMode: 'atrous-variance' });
      expect(layout.labels[0]).toBe('sample-budget');
      expect(layout.labels[layout.slotCount - 1]).toBe('composite');
    });
  });
});
