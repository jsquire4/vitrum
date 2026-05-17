/**
 * bdpt.test.ts — Unit tests for BDPT vertex pack/unpack.
 *
 * Tests cover:
 *   - BDPTVertex pack/unpack round-trip (all fields, all kind values)
 *   - BDPT kind constants are distinct and in the right range
 *   - BDPT_VERTEX_FLOATS / BDPT_VERTEX_BYTES size constants
 *
 * NOTE: Tests for the deprecated `_partial` MIS helpers were removed in W7
 * alongside the helpers themselves. Coverage for the canonical `_full`
 * variants lives in bdptVeachFull.test.ts.
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

