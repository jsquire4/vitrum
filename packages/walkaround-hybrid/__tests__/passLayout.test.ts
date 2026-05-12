import { describe, it, expect } from 'vitest';
import { buildPassLayout, MAX_PASS_COUNT } from '../src/pipeline/timestampQueries.js';

describe('buildPassLayout — Sprint 9..18 + indirect atrous chain', () => {
  describe('atrous-variance mode (26 slots)', () => {
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

    it('places gtao + gtao-upsample at slots 10, 11', () => {
      expect(layout.index('gtao')).toBe(10);
      expect(layout.index('gtao-upsample')).toBe(11);
    });

    it('places welford-temporal at slot 12 (after gtao + upsample)', () => {
      expect(layout.index('welford-temporal')).toBe(12);
    });

    it('places atrous-variance-variance + 3 atrous-variance-atrous slots in order', () => {
      expect(layout.index('atrous-variance-variance')).toBe(13);
      expect(layout.index('atrous-variance-atrous-0')).toBe(14);
      expect(layout.index('atrous-variance-atrous-2')).toBe(16);
    });

    it('places indirect-temporal-accum, 4 atrous-indirect slots, indirect-combine, then temporal+resolve+composite tail', () => {
      expect(layout.index('indirect-temporal-accum')).toBe(17);
      expect(layout.index('atrous-indirect-0')).toBe(18);
      expect(layout.index('atrous-indirect-1')).toBe(19);
      expect(layout.index('atrous-indirect-2')).toBe(20);
      expect(layout.index('atrous-indirect-3')).toBe(21);
      expect(layout.index('indirect-combine')).toBe(22);
      expect(layout.index('temporalAccum')).toBe(23);
      expect(layout.index('resolve')).toBe(24);
      expect(layout.index('composite')).toBe(25);
    });

    it('does not include ppg-update', () => {
      expect(() => layout.index('ppg-update')).toThrow(/not active/);
    });

    it('does not include legacy atrous-0 label', () => {
      expect(() => layout.index('atrous-0')).toThrow(/not active/);
    });

    it('reports 26 slots', () => {
      expect(layout.slotCount).toBe(26);
      expect(layout.labels).toHaveLength(26);
    });
  });

  describe('legacy atrous mode (24 slots)', () => {
    const layout = buildPassLayout({ denoiserMode: 'atrous' });

    it('GI block at 5..8; shade at 9; gtao + upsample at 10, 11; atrous-0..2 at 12..14', () => {
      expect(layout.index('sample-budget')).toBe(0);
      expect(layout.index('gi-ris')).toBe(5);
      expect(layout.index('gi-spatial-2')).toBe(8);
      expect(layout.index('shade')).toBe(9);
      expect(layout.index('gtao')).toBe(10);
      expect(layout.index('gtao-upsample')).toBe(11);
      expect(layout.index('atrous-0')).toBe(12);
      expect(layout.index('atrous-1')).toBe(13);
      expect(layout.index('atrous-2')).toBe(14);
    });

    it('places indirect-temporal-accum, atrous-indirect-0..3, indirect-combine, then temporalAccum/resolve/composite tail', () => {
      expect(layout.index('indirect-temporal-accum')).toBe(15);
      expect(layout.index('atrous-indirect-0')).toBe(16);
      expect(layout.index('atrous-indirect-3')).toBe(19);
      expect(layout.index('indirect-combine')).toBe(20);
      expect(layout.index('temporalAccum')).toBe(21);
      expect(layout.index('resolve')).toBe(22);
      expect(layout.index('composite')).toBe(23);
    });

    it('does not include atrous-variance labels', () => {
      expect(() => layout.index('welford-temporal')).toThrow(/not active/);
    });

    it('reports 24 slots', () => {
      expect(layout.slotCount).toBe(24);
    });
  });

  describe('MAX_PASS_COUNT invariant', () => {
    it('every layout fits within MAX_PASS_COUNT', () => {
      for (const denoiserMode of ['atrous-variance', 'atrous'] as const) {
        const layout = buildPassLayout({ denoiserMode });
        expect(layout.slotCount).toBeLessThanOrEqual(MAX_PASS_COUNT);
      }
    });

    it('MAX_PASS_COUNT is 26 (D7 sweep — PPG deleted, atrous-variance is worst-case at 26 slots)', () => {
      expect(MAX_PASS_COUNT).toBe(26);
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
