import { describe, it, expect } from 'vitest';
import { buildPassLayout, MAX_PASS_COUNT } from '../src/pipeline/timestampQueries.js';

describe('buildPassLayout — Sprint 9 + 15 + 16 + 17 (gi-temporal + gi-spatial-1/2)', () => {
  describe('PPG off, SVGF mode (22 slots)', () => {
    const layout = buildPassLayout({ ppgEnabled: false, denoiserMode: 'svgf' });

    it('prepends sample-budget at slot 0 (runs before RIS)', () => {
      expect(layout.index('sample-budget')).toBe(0);
    });

    it('DI RIS chain at 1..4; GI block (ris, temporal, spatial-1, spatial-2) at 5..8; shade at 9', () => {
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

    it('places gtao + gtao-upsample at slots 10, 11 (Sprint 15)', () => {
      expect(layout.index('gtao')).toBe(10);
      expect(layout.index('gtao-upsample')).toBe(11);
    });

    it('places welford-temporal at slot 12 (after gtao + upsample)', () => {
      expect(layout.index('welford-temporal')).toBe(12);
    });

    it('places svgf-variance + 5 svgf-atrous slots in order', () => {
      expect(layout.index('svgf-variance')).toBe(13);
      expect(layout.index('svgf-atrous-0')).toBe(14);
      expect(layout.index('svgf-atrous-4')).toBe(18);
    });

    it('inserts resolve between temporalAccum and composite', () => {
      expect(layout.index('temporalAccum')).toBe(19);
      expect(layout.index('resolve')).toBe(20);
      expect(layout.index('composite')).toBe(21);
    });

    it('does not include ppg-update', () => {
      expect(() => layout.index('ppg-update')).toThrow(/not active/);
    });

    it('does not include legacy atrous labels', () => {
      expect(() => layout.index('atrous-0')).toThrow(/not active/);
    });

    it('reports 22 slots', () => {
      expect(layout.slotCount).toBe(22);
      expect(layout.labels).toHaveLength(22);
    });
  });

  describe('PPG on, SVGF mode (23 slots — full worst-case)', () => {
    const layout = buildPassLayout({ ppgEnabled: true, denoiserMode: 'svgf' });

    it('inserts ppg-update at slot 10 between shade and gtao', () => {
      expect(layout.index('sample-budget')).toBe(0);
      expect(layout.index('gi-ris')).toBe(5);
      expect(layout.index('gi-temporal')).toBe(6);
      expect(layout.index('gi-spatial-1')).toBe(7);
      expect(layout.index('gi-spatial-2')).toBe(8);
      expect(layout.index('shade')).toBe(9);
      expect(layout.index('ppg-update')).toBe(10);
      expect(layout.index('gtao')).toBe(11);
      expect(layout.index('gtao-upsample')).toBe(12);
      expect(layout.index('welford-temporal')).toBe(13);
      expect(layout.index('svgf-variance')).toBe(14);
      expect(layout.index('svgf-atrous-0')).toBe(15);
      expect(layout.index('svgf-atrous-4')).toBe(19);
      expect(layout.index('temporalAccum')).toBe(20);
      expect(layout.index('resolve')).toBe(21);
      expect(layout.index('composite')).toBe(22);
    });

    it('reports 23 slots — matches MAX_PASS_COUNT', () => {
      expect(layout.slotCount).toBe(23);
      expect(layout.slotCount).toBe(MAX_PASS_COUNT);
    });
  });

  describe('PPG off, legacy atrous mode (18 slots)', () => {
    const layout = buildPassLayout({ ppgEnabled: false, denoiserMode: 'atrous' });

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

    it('places temporalAccum, resolve, composite at end', () => {
      expect(layout.index('temporalAccum')).toBe(15);
      expect(layout.index('resolve')).toBe(16);
      expect(layout.index('composite')).toBe(17);
    });

    it('does not include SVGF labels', () => {
      expect(() => layout.index('welford-temporal')).toThrow(/not active/);
      expect(() => layout.index('svgf-atrous-3')).toThrow(/not active/);
    });

    it('reports 18 slots', () => {
      expect(layout.slotCount).toBe(18);
    });
  });

  describe('PPG on, legacy atrous mode (19 slots)', () => {
    const layout = buildPassLayout({ ppgEnabled: true, denoiserMode: 'atrous' });

    it('shade at 9; ppg-update at 10; gtao + upsample at 11, 12; atrous-0..2 at 13..15', () => {
      expect(layout.index('sample-budget')).toBe(0);
      expect(layout.index('gi-ris')).toBe(5);
      expect(layout.index('shade')).toBe(9);
      expect(layout.index('ppg-update')).toBe(10);
      expect(layout.index('gtao')).toBe(11);
      expect(layout.index('gtao-upsample')).toBe(12);
      expect(layout.index('atrous-0')).toBe(13);
      expect(layout.index('atrous-2')).toBe(15);
      expect(layout.index('temporalAccum')).toBe(16);
      expect(layout.index('resolve')).toBe(17);
      expect(layout.index('composite')).toBe(18);
    });

    it('reports 19 slots', () => {
      expect(layout.slotCount).toBe(19);
    });
  });

  describe('MAX_PASS_COUNT invariant', () => {
    it('every layout fits within MAX_PASS_COUNT', () => {
      for (const ppgEnabled of [false, true]) {
        for (const denoiserMode of ['svgf', 'atrous'] as const) {
          const layout = buildPassLayout({ ppgEnabled, denoiserMode });
          expect(layout.slotCount).toBeLessThanOrEqual(MAX_PASS_COUNT);
        }
      }
    });

    it('MAX_PASS_COUNT is 23 (Sprint 17 added gi-temporal + gi-spatial-1 + gi-spatial-2 to the Sprint 16 baseline of 20)', () => {
      expect(MAX_PASS_COUNT).toBe(23);
    });
  });

  describe('labels array matches index() lookup', () => {
    it('layout.labels[i] === label such that layout.index(label) === i', () => {
      const layout = buildPassLayout({ ppgEnabled: true, denoiserMode: 'svgf' });
      for (let i = 0; i < layout.slotCount; i++) {
        const label = layout.labels[i];
        if (label === undefined) throw new Error(`undefined label at ${i}`);
        expect(layout.index(label)).toBe(i);
      }
    });

    it('first label is sample-budget; last is composite', () => {
      const layout = buildPassLayout({ ppgEnabled: false, denoiserMode: 'svgf' });
      expect(layout.labels[0]).toBe('sample-budget');
      expect(layout.labels[layout.slotCount - 1]).toBe('composite');
    });
  });
});
