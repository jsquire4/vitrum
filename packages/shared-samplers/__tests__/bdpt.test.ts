/**
 * bdpt.test.ts — Unit tests for BDPT vertex pack/unpack and MIS weight computation.
 *
 * Tests cover:
 *   - BDPTVertex pack/unpack round-trip (all fields, all kind values)
 *   - BDPT kind constants are distinct and in the right range
 *   - BDPT_VERTEX_FLOATS / BDPT_VERTEX_BYTES size constants
 *   - Strategy PDF table length = s + t + 1
 *   - MIS weight sums to 1 across all strategies (when all PDFs > 0)
 *   - Power heuristic with β=1 reduces to balance heuristic for 2-strategy case
 *   - Edge cases: empty light subpath, empty eye subpath, all-zero PDFs
 *   - Out-of-range strategy index returns 0
 *   - Single-strategy returns weight 1.0
 */

import { describe, it, expect } from 'vitest';
import {
  packBDPTVertex,
  unpackBDPTVertex,
  BDPT_KIND_LIGHT,
  BDPT_KIND_EYE,
  BDPT_KIND_CONNECTION,
  BDPT_KIND_INVALID,
  BDPT_VERTEX_FLOATS,
  BDPT_VERTEX_BYTES,
  BDPT_MAX_LIGHT_BOUNCES,
  BDPT_MAX_EYE_BOUNCES,
} from '../src/bdptVertex.js';
import type { BDPTVertex } from '../src/bdptVertex.js';
import { bdptConnectionMIS, buildBDPTStrategyPDFs } from '../src/bdptMIS.js';

// ── Helper factories ──────────────────────────────────────────────────────────

function makeVertex(overrides: Partial<BDPTVertex> = {}): BDPTVertex {
  return {
    position: [1, 2, 3],
    kind: BDPT_KIND_LIGHT,
    normal: [0, 1, 0],
    pdfFwd: 0.5,
    throughput: [0.8, 0.6, 0.4],
    pdfRev: 0.25,
    ...overrides,
  };
}

function roundTrip(v: BDPTVertex): BDPTVertex {
  const buf = new Float32Array(BDPT_VERTEX_FLOATS);
  packBDPTVertex(v, buf, 0);
  return unpackBDPTVertex(buf, 0);
}

// ── Constants ─────────────────────────────────────────────────────────────────

describe('BDPT constants', () => {
  it('kind constants are distinct', () => {
    const kinds = new Set([
      BDPT_KIND_LIGHT,
      BDPT_KIND_EYE,
      BDPT_KIND_CONNECTION,
      BDPT_KIND_INVALID,
    ]);
    expect(kinds.size).toBe(4);
  });

  it('kind constants are in the range 0–3', () => {
    for (const k of [BDPT_KIND_LIGHT, BDPT_KIND_EYE, BDPT_KIND_CONNECTION, BDPT_KIND_INVALID]) {
      expect(k).toBeGreaterThanOrEqual(0);
      expect(k).toBeLessThanOrEqual(3);
    }
  });

  it('BDPT_KIND_LIGHT is 0', () => expect(BDPT_KIND_LIGHT).toBe(0));
  it('BDPT_KIND_EYE is 1', () => expect(BDPT_KIND_EYE).toBe(1));
  it('BDPT_KIND_CONNECTION is 2', () => expect(BDPT_KIND_CONNECTION).toBe(2));
  it('BDPT_KIND_INVALID is 3', () => expect(BDPT_KIND_INVALID).toBe(3));

  it('BDPT_VERTEX_FLOATS is 12 (3 RGBA32F texels)', () => {
    expect(BDPT_VERTEX_FLOATS).toBe(12);
  });

  it('BDPT_VERTEX_BYTES is 48 (12 floats × 4 bytes)', () => {
    expect(BDPT_VERTEX_BYTES).toBe(48);
  });

  it('BDPT_MAX_LIGHT_BOUNCES is 3 (per roadmap DoD)', () => {
    expect(BDPT_MAX_LIGHT_BOUNCES).toBe(3);
  });

  it('BDPT_MAX_EYE_BOUNCES is 12 (matches engine default maxBounces)', () => {
    expect(BDPT_MAX_EYE_BOUNCES).toBe(12);
  });
});

// ── Pack / unpack round-trip ──────────────────────────────────────────────────

describe('packBDPTVertex / unpackBDPTVertex', () => {
  it('round-trips a light-kind vertex with arbitrary values', () => {
    const v = makeVertex({ kind: BDPT_KIND_LIGHT });
    const rt = roundTrip(v);
    // Float32Array round-trip — use toBeCloseTo (Float32 has ~7 decimal digits precision)
    expect(rt.position[0]).toBeCloseTo(v.position[0], 5);
    expect(rt.position[1]).toBeCloseTo(v.position[1], 5);
    expect(rt.position[2]).toBeCloseTo(v.position[2], 5);
    expect(rt.kind).toBe(BDPT_KIND_LIGHT);
    expect(rt.normal[0]).toBeCloseTo(v.normal[0], 5);
    expect(rt.normal[1]).toBeCloseTo(v.normal[1], 5);
    expect(rt.normal[2]).toBeCloseTo(v.normal[2], 5);
    expect(rt.pdfFwd).toBeCloseTo(v.pdfFwd, 5);
    expect(rt.throughput[0]).toBeCloseTo(v.throughput[0], 5);
    expect(rt.throughput[1]).toBeCloseTo(v.throughput[1], 5);
    expect(rt.throughput[2]).toBeCloseTo(v.throughput[2], 5);
    expect(rt.pdfRev).toBeCloseTo(v.pdfRev, 5);
  });

  it('round-trips a BDPT_KIND_EYE vertex', () => {
    const v = makeVertex({ kind: BDPT_KIND_EYE, position: [10, -3, 0.5] });
    expect(roundTrip(v).kind).toBe(BDPT_KIND_EYE);
    expect(roundTrip(v).position[0]).toBeCloseTo(10);
  });

  it('round-trips a BDPT_KIND_CONNECTION vertex', () => {
    const v = makeVertex({ kind: BDPT_KIND_CONNECTION });
    expect(roundTrip(v).kind).toBe(BDPT_KIND_CONNECTION);
  });

  it('round-trips a BDPT_KIND_INVALID vertex', () => {
    const v = makeVertex({ kind: BDPT_KIND_INVALID });
    expect(roundTrip(v).kind).toBe(BDPT_KIND_INVALID);
  });

  it('preserves throughput = [0, 0, 0] (black vertex)', () => {
    const v = makeVertex({ throughput: [0, 0, 0] });
    const rt = roundTrip(v);
    expect(rt.throughput[0]).toBeCloseTo(0);
    expect(rt.throughput[1]).toBeCloseTo(0);
    expect(rt.throughput[2]).toBeCloseTo(0);
  });

  it('preserves extreme float values without sign flip', () => {
    const v = makeVertex({
      pdfFwd: 1e-7,
      pdfRev: 1e7,
      throughput: [1e3, 1e-4, 0.999],
    });
    const rt = roundTrip(v);
    // Float32 relative error ≤ ~1.2e-7; check sign and rough magnitude only
    expect(Math.sign(rt.pdfFwd)).toBe(1);
    expect(rt.pdfFwd).toBeGreaterThan(0);
    expect(rt.pdfFwd).toBeLessThan(1e-5);
    expect(rt.pdfRev).toBeCloseTo(1e7, -1); // within ±10% of 1e7
  });

  it('packs into the correct float offsets (spot-check via raw buffer)', () => {
    const v = makeVertex({
      position: [7, 8, 9],
      kind: BDPT_KIND_EYE,
      normal: [1, 0, 0],
      pdfFwd: 0.3,
      throughput: [0.1, 0.2, 0.3],
      pdfRev: 0.4,
    });
    const buf = new Float32Array(BDPT_VERTEX_FLOATS);
    packBDPTVertex(v, buf, 0);
    // Texel 0
    expect(buf[0]).toBeCloseTo(7);   // position.x
    expect(buf[1]).toBeCloseTo(8);   // position.y
    expect(buf[2]).toBeCloseTo(9);   // position.z
    expect(buf[3]).toBeCloseTo(BDPT_KIND_EYE); // kind
    // Texel 1
    expect(buf[4]).toBeCloseTo(1);   // normal.x
    expect(buf[5]).toBeCloseTo(0);   // normal.y
    expect(buf[6]).toBeCloseTo(0);   // normal.z
    expect(buf[7]).toBeCloseTo(0.3); // pdfFwd
    // Texel 2
    expect(buf[8]).toBeCloseTo(0.1);  // throughput.x
    expect(buf[9]).toBeCloseTo(0.2);  // throughput.y
    expect(buf[10]).toBeCloseTo(0.3); // throughput.z
    expect(buf[11]).toBeCloseTo(0.4); // pdfRev
  });

  it('respects a non-zero offset in the target buffer', () => {
    const v = makeVertex({ position: [5, 6, 7] });
    const buf = new Float32Array(BDPT_VERTEX_FLOATS * 2);
    packBDPTVertex(v, buf, BDPT_VERTEX_FLOATS); // write at slot 1
    const rt = unpackBDPTVertex(buf, BDPT_VERTEX_FLOATS);
    expect(rt.position[0]).toBeCloseTo(5);
    expect(rt.position[1]).toBeCloseTo(6);
    expect(rt.position[2]).toBeCloseTo(7);
  });

  it('invalid stored kind value coerces to BDPT_KIND_INVALID on unpack', () => {
    const buf = new Float32Array(BDPT_VERTEX_FLOATS);
    buf[3] = 99; // invalid kind
    const v = unpackBDPTVertex(buf, 0);
    expect(v.kind).toBe(BDPT_KIND_INVALID);
  });
});

// ── buildBDPTStrategyPDFs ─────────────────────────────────────────────────────

describe('buildBDPTStrategyPDFs', () => {
  it('strategy k=s (interior) is the only non-zero row when s>0 and t>0 (simplified PDF model)', () => {
    const light = [makeVertex({ pdfFwd: 0.6 }), makeVertex({ pdfFwd: 0.7 })];
    const eye = [makeVertex({ pdfFwd: 0.5 }), makeVertex({ pdfFwd: 0.4 })];
    const pdfs = buildBDPTStrategyPDFs(light, eye);
    const s = 2;
    expect(pdfs.length).toBe(s + 2 + 1);
    for (let k = 0; k < pdfs.length; k++) {
      if (k === s) {
        expect(pdfs[k]).toBeCloseTo(0.6 * 0.7 * 0.5 * 0.4, 6);
      } else {
        expect(pdfs[k]).toBe(0);
      }
    }
  });

  it('returns length s+t+1 for non-trivial subpaths', () => {
    const light = [makeVertex(), makeVertex()]; // s=2
    const eye = [makeVertex(), makeVertex(), makeVertex()]; // t=3
    const pdfs = buildBDPTStrategyPDFs(light, eye);
    expect(pdfs.length).toBe(6); // s+t+1 = 2+3+1 = 6
  });

  it('returns length 1 when both subpaths are empty', () => {
    const pdfs = buildBDPTStrategyPDFs([], []);
    expect(pdfs.length).toBe(1);
  });

  it('returns length s+1 when eye subpath is empty', () => {
    const light = [makeVertex(), makeVertex()]; // s=2
    const pdfs = buildBDPTStrategyPDFs(light, []);
    expect(pdfs.length).toBe(3); // s+t+1 = 2+0+1 = 3
  });

  it('returns length t+1 when light subpath is empty', () => {
    const eye = [makeVertex(), makeVertex(), makeVertex()]; // t=3
    const pdfs = buildBDPTStrategyPDFs([], eye);
    expect(pdfs.length).toBe(4); // s+t+1 = 0+3+1 = 4
  });

  it('all entries are non-negative', () => {
    const light = [makeVertex({ pdfFwd: 0.5 }), makeVertex({ pdfFwd: 0.3 })];
    const eye = [makeVertex({ pdfFwd: 0.8 }), makeVertex({ pdfFwd: 0.2 })];
    const pdfs = buildBDPTStrategyPDFs(light, eye);
    for (const p of pdfs) {
      expect(p).toBeGreaterThanOrEqual(0);
    }
  });

  it('strategy k=0 (pure eye tracing) uses only eye subpath PDFs', () => {
    // k=0: 0 light vertices, t eye vertices → product of all eye pdfFwds
    const eye = [
      makeVertex({ pdfFwd: 0.5 }),
      makeVertex({ pdfFwd: 0.4 }),
    ];
    const pdfs = buildBDPTStrategyPDFs([], eye);
    // k=0: eye product = 0.5 * 0.4 = 0.2
    expect(pdfs[0]).toBeCloseTo(0.5 * 0.4, 6);
  });

  it('strategy k=s (max light vertices) uses only light subpath PDFs', () => {
    const light = [
      makeVertex({ pdfFwd: 0.6 }),
      makeVertex({ pdfFwd: 0.7 }),
    ];
    const pdfs = buildBDPTStrategyPDFs(light, []);
    // k=s=2: light product = 0.6 * 0.7 = 0.42
    expect(pdfs[2]).toBeCloseTo(0.6 * 0.7, 6);
  });
});

// ── bdptConnectionMIS ─────────────────────────────────────────────────────────

describe('bdptConnectionMIS', () => {
  it('MIS weights sum to 1 across all strategies when all PDFs > 0', () => {
    // Full Veach / power-heuristic sanity: each strategy gets p_i^β / Σ p_j^β.
    // With buildBDPTStrategyPDFs most entries are zero; see tests above for sparsity.
    const pdfs = [0.1, 0.5, 0.3, 0.2, 0.4];
    let sum = 0;
    for (let k = 0; k < pdfs.length; k++) {
      sum += bdptConnectionMIS(pdfs, k);
    }
    expect(sum).toBeCloseTo(1.0, 5);
  });

  it('single strategy always returns weight 1.0', () => {
    expect(bdptConnectionMIS([0.7], 0)).toBeCloseTo(1.0);
    expect(bdptConnectionMIS([0.001], 0)).toBeCloseTo(1.0);
  });

  it('with β=1 (balance heuristic), 2-strategy case matches balance formula', () => {
    const p1 = 0.3;
    const p2 = 0.7;
    const w1 = bdptConnectionMIS([p1, p2], 0, 1);
    // balance heuristic: p1 / (p1 + p2)
    expect(w1).toBeCloseTo(p1 / (p1 + p2), 6);
    const w2 = bdptConnectionMIS([p1, p2], 1, 1);
    expect(w2).toBeCloseTo(p2 / (p1 + p2), 6);
    expect(w1 + w2).toBeCloseTo(1.0, 6);
  });

  it('default β=2 (power heuristic) amplifies dominant strategy more than balance', () => {
    const pdfs = [0.1, 0.9]; // strategy 1 is dominant
    const wBalance = pdfs[1]! / (pdfs[0]! + pdfs[1]!);
    const wPower = bdptConnectionMIS(pdfs, 1, 2);
    expect(wPower).toBeGreaterThan(wBalance);
  });

  it('all-zero PDFs returns 0 gracefully (no NaN/Infinity)', () => {
    const result = bdptConnectionMIS([0, 0, 0], 1);
    expect(result).toBe(0);
    expect(Number.isFinite(result)).toBe(true);
  });

  it('out-of-range strategy index returns 0', () => {
    expect(bdptConnectionMIS([0.5, 0.5], -1)).toBe(0);
    expect(bdptConnectionMIS([0.5, 0.5], 2)).toBe(0);
    expect(bdptConnectionMIS([0.5, 0.5], 100)).toBe(0);
  });

  it('weights sum to 1 across strategies built from real subpaths', () => {
    const light = [makeVertex({ pdfFwd: 0.5 }), makeVertex({ pdfFwd: 0.3 })];
    const eye = [makeVertex({ pdfFwd: 0.8 }), makeVertex({ pdfFwd: 0.4 })];
    const pdfs = buildBDPTStrategyPDFs(light, eye);

    // Only non-zero strategies should contribute; sum of their weights = 1
    const nonZero = Array.from(pdfs).filter(p => p > 0);
    if (nonZero.length > 0) {
      let sum = 0;
      for (let k = 0; k < pdfs.length; k++) {
        if (pdfs[k]! > 0) {
          sum += bdptConnectionMIS(Array.from(pdfs), k);
        }
      }
      expect(sum).toBeCloseTo(1.0, 4);
    }
  });

  it('empty light subpath — all weight goes to pure eye-tracing strategy', () => {
    const eye = [makeVertex({ pdfFwd: 0.5 }), makeVertex({ pdfFwd: 0.5 })];
    const pdfs = buildBDPTStrategyPDFs([], eye);
    // Only k=0 (pure eye tracing) is non-zero — weight should be 1
    const w = bdptConnectionMIS(Array.from(pdfs), 0);
    expect(w).toBeCloseTo(1.0, 5);
  });

  it('empty eye subpath — all weight goes to pure light-tracing strategy', () => {
    const light = [makeVertex({ pdfFwd: 0.5 }), makeVertex({ pdfFwd: 0.3 })];
    const pdfs = buildBDPTStrategyPDFs(light, []);
    // Only k=s=2 (pure light tracing) is non-zero — weight should be 1
    const w = bdptConnectionMIS(Array.from(pdfs), light.length);
    expect(w).toBeCloseTo(1.0, 5);
  });
});
