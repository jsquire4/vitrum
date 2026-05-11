import { describe, it, expect } from 'vitest';
import { buildPassLayout, MAX_PASS_COUNT } from '../src/pipeline/timestampQueries.js';

describe('buildPassLayout — P2-4.6 named-pass slot mapping', () => {
  describe('PPG off, SVGF mode (14 slots)', () => {
    const layout = buildPassLayout({ ppgEnabled: false, denoiserMode: 'svgf' });

    it('assigns the five primary slots first', () => {
      expect(layout.index('ris')).toBe(0);
      expect(layout.index('temporal')).toBe(1);
      expect(layout.index('spatial-1')).toBe(2);
      expect(layout.index('spatial-2')).toBe(3);
      expect(layout.index('shade')).toBe(4);
    });

    it('places welford-temporal at slot 5 (denoiseBase when ppg off)', () => {
      expect(layout.index('welford-temporal')).toBe(5);
    });

    it('places svgf-variance + 5 svgf-atrous slots in order', () => {
      expect(layout.index('svgf-variance')).toBe(6);
      expect(layout.index('svgf-atrous-0')).toBe(7);
      expect(layout.index('svgf-atrous-4')).toBe(11);
    });

    it('places temporalAccum and composite at the end', () => {
      expect(layout.index('temporalAccum')).toBe(12);
      expect(layout.index('composite')).toBe(13);
    });

    it('does not include ppg-update', () => {
      expect(() => layout.index('ppg-update')).toThrow(/not active/);
    });

    it('does not include legacy atrous labels', () => {
      expect(() => layout.index('atrous-0')).toThrow(/not active/);
    });

    it('reports 14 slots', () => {
      expect(layout.slotCount).toBe(14);
      expect(layout.labels).toHaveLength(14);
    });
  });

  describe('PPG on, SVGF mode (15 slots — full worst-case)', () => {
    const layout = buildPassLayout({ ppgEnabled: true, denoiserMode: 'svgf' });

    it('inserts ppg-update at slot 5, pushing denoiseBase to 6', () => {
      expect(layout.index('ppg-update')).toBe(5);
      expect(layout.index('welford-temporal')).toBe(6);
      expect(layout.index('svgf-variance')).toBe(7);
      expect(layout.index('svgf-atrous-0')).toBe(8);
      expect(layout.index('svgf-atrous-4')).toBe(12);
      expect(layout.index('temporalAccum')).toBe(13);
      expect(layout.index('composite')).toBe(14);
    });

    it('reports 15 slots — matches MAX_PASS_COUNT', () => {
      expect(layout.slotCount).toBe(15);
      expect(layout.slotCount).toBe(MAX_PASS_COUNT);
    });
  });

  describe('PPG off, legacy atrous mode (10 slots)', () => {
    const layout = buildPassLayout({ ppgEnabled: false, denoiserMode: 'atrous' });

    it('places atrous-0..2 starting at slot 5', () => {
      expect(layout.index('atrous-0')).toBe(5);
      expect(layout.index('atrous-1')).toBe(6);
      expect(layout.index('atrous-2')).toBe(7);
    });

    it('places temporalAccum and composite at end', () => {
      expect(layout.index('temporalAccum')).toBe(8);
      expect(layout.index('composite')).toBe(9);
    });

    it('does not include SVGF labels', () => {
      expect(() => layout.index('welford-temporal')).toThrow(/not active/);
      expect(() => layout.index('svgf-atrous-3')).toThrow(/not active/);
    });

    it('reports 10 slots', () => {
      expect(layout.slotCount).toBe(10);
    });
  });

  describe('PPG on, legacy atrous mode (11 slots)', () => {
    const layout = buildPassLayout({ ppgEnabled: true, denoiserMode: 'atrous' });

    it('places ppg-update at slot 5, atrous-0..2 at 6..8', () => {
      expect(layout.index('ppg-update')).toBe(5);
      expect(layout.index('atrous-0')).toBe(6);
      expect(layout.index('atrous-2')).toBe(8);
      expect(layout.index('temporalAccum')).toBe(9);
      expect(layout.index('composite')).toBe(10);
    });

    it('reports 11 slots', () => {
      expect(layout.slotCount).toBe(11);
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
  });
});
