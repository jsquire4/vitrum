import { describe, it, expect } from 'vitest';
import { buildPassLayout, MAX_PASS_COUNT } from '../src/pipeline/timestampQueries.js';

describe('buildPassLayout — P2-4.6 named-pass slot mapping (Sprint 9 wire-in)', () => {
  describe('PPG off, SVGF mode (16 slots)', () => {
    const layout = buildPassLayout({ ppgEnabled: false, denoiserMode: 'svgf' });

    it('prepends sample-budget at slot 0 (runs before RIS)', () => {
      expect(layout.index('sample-budget')).toBe(0);
    });

    it('shifts the five primary slots by 1', () => {
      expect(layout.index('ris')).toBe(1);
      expect(layout.index('temporal')).toBe(2);
      expect(layout.index('spatial-1')).toBe(3);
      expect(layout.index('spatial-2')).toBe(4);
      expect(layout.index('shade')).toBe(5);
    });

    it('places welford-temporal at slot 6 (denoiseBase when ppg off)', () => {
      expect(layout.index('welford-temporal')).toBe(6);
    });

    it('places svgf-variance + 5 svgf-atrous slots in order', () => {
      expect(layout.index('svgf-variance')).toBe(7);
      expect(layout.index('svgf-atrous-0')).toBe(8);
      expect(layout.index('svgf-atrous-4')).toBe(12);
    });

    it('inserts resolve between temporalAccum and composite', () => {
      expect(layout.index('temporalAccum')).toBe(13);
      expect(layout.index('resolve')).toBe(14);
      expect(layout.index('composite')).toBe(15);
    });

    it('does not include ppg-update', () => {
      expect(() => layout.index('ppg-update')).toThrow(/not active/);
    });

    it('does not include legacy atrous labels', () => {
      expect(() => layout.index('atrous-0')).toThrow(/not active/);
    });

    it('reports 16 slots', () => {
      expect(layout.slotCount).toBe(16);
      expect(layout.labels).toHaveLength(16);
    });
  });

  describe('PPG on, SVGF mode (17 slots — full worst-case)', () => {
    const layout = buildPassLayout({ ppgEnabled: true, denoiserMode: 'svgf' });

    it('inserts ppg-update at slot 6, pushing welford to 7', () => {
      expect(layout.index('sample-budget')).toBe(0);
      expect(layout.index('shade')).toBe(5);
      expect(layout.index('ppg-update')).toBe(6);
      expect(layout.index('welford-temporal')).toBe(7);
      expect(layout.index('svgf-variance')).toBe(8);
      expect(layout.index('svgf-atrous-0')).toBe(9);
      expect(layout.index('svgf-atrous-4')).toBe(13);
      expect(layout.index('temporalAccum')).toBe(14);
      expect(layout.index('resolve')).toBe(15);
      expect(layout.index('composite')).toBe(16);
    });

    it('reports 17 slots — matches MAX_PASS_COUNT', () => {
      expect(layout.slotCount).toBe(17);
      expect(layout.slotCount).toBe(MAX_PASS_COUNT);
    });
  });

  describe('PPG off, legacy atrous mode (12 slots)', () => {
    const layout = buildPassLayout({ ppgEnabled: false, denoiserMode: 'atrous' });

    it('sample-budget at 0; atrous-0..2 starting at slot 6', () => {
      expect(layout.index('sample-budget')).toBe(0);
      expect(layout.index('atrous-0')).toBe(6);
      expect(layout.index('atrous-1')).toBe(7);
      expect(layout.index('atrous-2')).toBe(8);
    });

    it('places temporalAccum, resolve, composite at end', () => {
      expect(layout.index('temporalAccum')).toBe(9);
      expect(layout.index('resolve')).toBe(10);
      expect(layout.index('composite')).toBe(11);
    });

    it('does not include SVGF labels', () => {
      expect(() => layout.index('welford-temporal')).toThrow(/not active/);
      expect(() => layout.index('svgf-atrous-3')).toThrow(/not active/);
    });

    it('reports 12 slots', () => {
      expect(layout.slotCount).toBe(12);
    });
  });

  describe('PPG on, legacy atrous mode (13 slots)', () => {
    const layout = buildPassLayout({ ppgEnabled: true, denoiserMode: 'atrous' });

    it('places sample-budget at 0; ppg-update at slot 6, atrous-0..2 at 7..9', () => {
      expect(layout.index('sample-budget')).toBe(0);
      expect(layout.index('ppg-update')).toBe(6);
      expect(layout.index('atrous-0')).toBe(7);
      expect(layout.index('atrous-2')).toBe(9);
      expect(layout.index('temporalAccum')).toBe(10);
      expect(layout.index('resolve')).toBe(11);
      expect(layout.index('composite')).toBe(12);
    });

    it('reports 13 slots', () => {
      expect(layout.slotCount).toBe(13);
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

    it('MAX_PASS_COUNT is 17 (Sprint 9 added sample-budget + resolve to the base 15)', () => {
      expect(MAX_PASS_COUNT).toBe(17);
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
