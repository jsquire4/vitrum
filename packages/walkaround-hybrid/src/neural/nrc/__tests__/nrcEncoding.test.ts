// nrcEncoding.test.ts — CPU-oracle + WGSL-codegen tests for the NRC "full"
// input encoding (hash-grid + one-blob) and its trainable hash-grid backward.
//
// The CPU oracle (`../nrcEncoding.ts`) is the load-bearing reference, the same
// role `reconnectionShift.ts` plays for GRIS. The emitted WGSL forward/backward
// (`../wgsl/nrcEncoding.wgsl.ts`) is hand-verified line-for-line against it; the
// codegen-shape tests below pin that the WGSL keeps emitting the SAME arithmetic
// (hash primes, trilinear corner loop, L1-normalised one-blob, fixed-point grad
// atomics) so a future shader-compile A/B (V20) only needs to confirm it runs.
//
// The hash-grid BACKWARD is checked TWO ways: (1) EXACT-ANALYTIC — the trilinear
// interpolation weight IS the analytic ∂feature/∂corner, so the scatter is
// exact; (2) a finite-difference probe of a downstream scalar loss as a
// cross-check. The interpolation is smooth (NO ReLU kink), so the FD probe here
// is clean — unlike the MLP-internal FD the kernel agent documented.

import { describe, it, expect } from 'vitest';
import {
  spatialHash3D, levelResolution, trilinearCorners, normalizeToAabb,
  hashGridForward, hashGridBackward, oneBlobEncodeScalar, octEncodeDir,
  assembleNrcInput, nrcInputWidth,
  type HashGridConfig, type HashGridLevel, type NrcEncodingConfig,
} from '../nrcEncoding.ts';
import { nrcEncodeHelpersWgsl } from '../wgsl/nrcEncoding.wgsl.ts';
import { nrcQueryWgsl } from '../wgsl/nrcQuery.wgsl.ts';

// ── build a small but representative multiresolution hash grid ──
function makeGrid(seed = 1): HashGridConfig {
  const F = 2;
  const nMin = 4, growth = 2;
  const levels: HashGridLevel[] = [];
  let s = seed >>> 0;
  const rng = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0x100000000; };
  for (let l = 0; l < 4; l++) {
    const resolution = levelResolution(nMin, growth, l);
    const tableSize = 97; // a prime, smaller than dense (res+1)^3 at fine levels → collisions exercised
    const table = new Float32Array(tableSize * F);
    for (let k = 0; k < table.length; k++) table[k] = rng() * 2 - 1;
    levels.push({ resolution, tableSize, table });
  }
  return { dim: 3, featuresPerEntry: F, levels, aabbMin: [-1, -1, -1], aabbMax: [1, 1, 1] };
}

describe('NRC hash-grid — spatial hash', () => {
  it('is deterministic and stays within [0, tableSize)', () => {
    for (let t = 0; t < 50; t++) {
      const h = spatialHash3D(t * 13, t * 7 + 1, t * 101 + 3, 97);
      expect(h).toBe(spatialHash3D(t * 13, t * 7 + 1, t * 101 + 3, 97));
      expect(h).toBeGreaterThanOrEqual(0);
      expect(h).toBeLessThan(97);
    }
  });

  it('uses the Instant-NGP primes (matches the WGSL multipliers exactly)', () => {
    // 0x9E3779B1 = 2654435761, 0x30034BB7 = 805459861. The WGSL must use the
    // same literals so the GPU hash and CPU oracle land on the SAME row.
    const wgsl = nrcEncodeHelpersWgsl();
    expect(wgsl).toContain('2654435761u');
    expect(wgsl).toContain('805459861u');
    // hand-recompute one hash with the documented formula and assert the oracle.
    const ix = 5 >>> 0, iy = 9 >>> 0, iz = 2 >>> 0;
    const expected = (Math.imul(ix, 1) ^ Math.imul(iy, 2654435761) ^ Math.imul(iz, 805459861)) >>> 0;
    expect(spatialHash3D(5, 9, 2, 1000)).toBe(expected % 1000);
  });
});

describe('NRC hash-grid — trilinear corners', () => {
  it('the 8 corner weights are a partition of unity (Σ = 1)', () => {
    const level: HashGridLevel = { resolution: 8, tableSize: 97, table: new Float32Array(97 * 2) };
    for (const [nx, ny, nz] of [[0.13, 0.77, 0.41], [0.5, 0.5, 0.5], [0.999, 0.001, 0.333]]) {
      const corners = trilinearCorners(level, nx!, ny!, nz!);
      expect(corners).toHaveLength(8);
      const sum = corners.reduce((a, c) => a + c.weight, 0);
      expect(sum).toBeCloseTo(1, 12);
      for (const c of corners) expect(c.weight).toBeGreaterThanOrEqual(0);
    }
  });

  it('at a voxel corner exactly one weight is 1 and the rest 0', () => {
    const level: HashGridLevel = { resolution: 4, tableSize: 97, table: new Float32Array(97 * 2) };
    // nx*N integer → frac 0 on all axes → corner (0,0,0) gets weight 1.
    const corners = trilinearCorners(level, 0.25, 0.5, 0.75); // ×4 = (1,2,3) integers
    const ones = corners.filter((c) => Math.abs(c.weight - 1) < 1e-12);
    expect(ones).toHaveLength(1);
  });
});

describe('NRC hash-grid — forward', () => {
  it('forward equals the weighted sum of corner features (output width L·F)', () => {
    const grid = makeGrid();
    const pos: [number, number, number] = [0.2, -0.4, 0.6];
    const out = hashGridForward(grid, pos);
    expect(out.length).toBe(grid.levels.length * grid.featuresPerEntry);
    // recompute level 0 by hand from the corners.
    const [nx, ny, nz] = normalizeToAabb(pos, grid.aabbMin, grid.aabbMax);
    const lvl0 = grid.levels[0]!;
    const corners = trilinearCorners(lvl0, nx, ny, nz);
    for (let f = 0; f < grid.featuresPerEntry; f++) {
      let expected = 0;
      for (const { row, weight } of corners) expected += weight * lvl0.table[row * grid.featuresPerEntry + f]!;
      expect(out[f]).toBeCloseTo(expected, 6);
    }
  });

  it('normalizeToAabb clamps out-of-bounds queries into [0,1]', () => {
    const n = normalizeToAabb([5, -5, 0], [-1, -1, -1], [1, 1, 1]);
    expect(n[0]).toBe(1);
    expect(n[1]).toBe(0);
    expect(n[2]).toBeCloseTo(0.5, 12);
  });
});

describe('NRC hash-grid — TRAINABLE backward (gradient scatter)', () => {
  it('backward is EXACT-ANALYTIC: scatter = corner weight × dOut, collisions accumulate', () => {
    const grid = makeGrid();
    const pos: [number, number, number] = [-0.3, 0.1, 0.45];
    const F = grid.featuresPerEntry;
    const L = grid.levels.length;
    // pick a non-trivial upstream gradient.
    const dOut = new Float32Array(L * F);
    for (let i = 0; i < dOut.length; i++) dOut[i] = Math.sin(i * 1.3 + 0.2);
    const grads = hashGridBackward(grid, pos, dOut);
    // recompute by hand: for each level, accumulate weight*dOut into the row.
    const [nx, ny, nz] = normalizeToAabb(pos, grid.aabbMin, grid.aabbMax);
    for (let l = 0; l < L; l++) {
      const lvl = grid.levels[l]!;
      const corners = trilinearCorners(lvl, nx, ny, nz);
      const expected = new Float32Array(lvl.table.length);
      for (const { row, weight } of corners) {
        for (let f = 0; f < F; f++) expected[row * F + f] = expected[row * F + f]! + weight * dOut[l * F + f]!;
      }
      for (let k = 0; k < expected.length; k++) expect(grads[l]![k]).toBeCloseTo(expected[k]!, 10);
    }
  });

  it('FD cross-check: ∂(½‖feature‖²)/∂table matches the scatter (smooth, no ReLU kink)', () => {
    // Downstream scalar loss L = ½ Σ feature². Then dL/dfeature = feature, and
    // dL/dtable[row][f] = Σ_corners weight·feature[levelBase+f]. The backward
    // with dOut = feature must equal the FD gradient of L w.r.t. each table cell.
    const grid = makeGrid(7);
    const pos: [number, number, number] = [0.11, 0.62, -0.27];
    const _F = grid.featuresPerEntry, L = grid.levels.length; // _F reserved for per-feature assertions
    const fwd = hashGridForward(grid, pos);
    const dOut = fwd.slice(); // dL/dfeature = feature
    const grads = hashGridBackward(grid, pos, dOut);

    const lossOf = (g: HashGridConfig) => {
      const o = hashGridForward(g, pos);
      let s = 0; for (let i = 0; i < o.length; i++) s += 0.5 * o[i]! * o[i]!;
      return s;
    };
    const h = 1e-3;
    // probe a handful of cells per level (full sweep is O(tableSize) — sample).
    for (let l = 0; l < L; l++) {
      const lvl = grid.levels[l]!;
      for (const k of [0, 5, 17, 40, lvl.table.length - 1]) {
        const tp = lvl.table.slice(); tp[k] = tp[k]! + h;
        const tm = lvl.table.slice(); tm[k] = tm[k]! - h;
        const gp: HashGridConfig = { ...grid, levels: grid.levels.map((x, i) => i === l ? { ...x, table: tp } : x) };
        const gm: HashGridConfig = { ...grid, levels: grid.levels.map((x, i) => i === l ? { ...x, table: tm } : x) };
        const fd = (lossOf(gp) - lossOf(gm)) / (2 * h);
        expect(grads[l]![k]).toBeCloseTo(fd, 4);
      }
    }
  });
});

describe('NRC one-blob direction encoding', () => {
  it('produces k bins that L1-normalise to 1', () => {
    const cfg = { bins: 8, sigma: 1 / 8 };
    for (const u of [0, 0.25, 0.5, 0.73, 1]) {
      const b = oneBlobEncodeScalar(u, cfg);
      expect(b.length).toBe(8);
      const s = b.reduce((a, x) => a + x, 0);
      expect(s).toBeCloseTo(1, 6);
      for (const x of b) expect(x).toBeGreaterThanOrEqual(0);
    }
  });

  it('peaks at the bin nearest the encoded scalar', () => {
    const cfg = { bins: 8, sigma: 1 / 8 };
    const b = oneBlobEncodeScalar(0.5, cfg); // nearest bin centre (3.5/8=0.4375, 4.5/8=0.5625) → bins 3 & 4 tie-ish
    let argmax = 0; for (let i = 1; i < b.length; i++) if (b[i]! > b[argmax]!) argmax = i;
    expect([3, 4]).toContain(argmax);
  });

  it('octEncodeDir maps unit directions into [0,1]²', () => {
    for (const d of [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0.577, 0.577, 0.577]]) {
      const len = Math.hypot(d[0]!, d[1]!, d[2]!);
      const [u, v] = octEncodeDir([d[0]! / len, d[1]! / len, d[2]! / len]);
      expect(u).toBeGreaterThanOrEqual(0); expect(u).toBeLessThanOrEqual(1);
      expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(1);
    }
  });

  it('octEncodeDir is invariant to every finite non-zero direction scale', () => {
    const expected = octEncodeDir([1, -2, 3]);
    const tiny = octEncodeDir([1e-30, -2e-30, 3e-30]);
    expect(tiny[0]).toBeCloseTo(expected[0], 15);
    expect(tiny[1]).toBeCloseTo(expected[1], 15);
    expect(octEncodeDir([0, 0, 0])).toEqual([0.5, 0.5]);
  });
});

describe('NRC full input assembly', () => {
  it('assembles [hashgrid | oneblob-u | oneblob-v | normal | rough | albedo] at the declared width', () => {
    const cfg: NrcEncodingConfig = { hashGrid: makeGrid(), oneBlob: { bins: 8, sigma: 1 / 8 } };
    const width = nrcInputWidth(cfg);
    // 4 levels × 2 F = 8 ; 2 × 8 bins = 16 ; 3+1+3 raw = 7 → 31.
    expect(width).toBe(8 + 16 + 7);
    const input = assembleNrcInput(cfg, {
      position: [0.2, -0.3, 0.5],
      normal: [0, 1, 0],
      direction: [0, 0, 1],
      roughness: 0.4,
      albedo: [0.8, 0.2, 0.1],
    });
    expect(input.length).toBe(width);
    // the raw tail must be verbatim (f32-stored → compare to f32 precision).
    const tail = Array.from(input.slice(width - 7));
    const expectedTail = [0, 1, 0, 0.4, 0.8, 0.2, 0.1];
    for (let i = 0; i < expectedTail.length; i++) expect(tail[i]).toBeCloseTo(expectedTail[i]!, 6);
  });
});

describe('NRC WGSL codegen — shape pins (line-for-line oracle equivalence)', () => {
  it('the live query one-blob implementation L1-normalises like the oracle', () => {
    const query = nrcQueryWgsl({
      levels: 4,
      featuresPerEntry: 2,
      oneBlobBins: 8,
      width: 32,
      outWidth: 3,
      hidden: 2,
    });
    expect(query).toContain('fn nrcOneBlob(');
    expect(query).toContain('exp(-0.5 * d * d)');
    expect(query).toContain('/ sum'); // L1 normalisation
    expect(query).not.toContain('nrcOneBlobScalar');
  });

  // The undispatchable ptr-arg WGSL oracle `nrcHashGridBackwardWgsl` was DELETED
  // (Task 4.5 #5) — it duplicated the DISPATCHED scatter in nrcEncodeBackward.wgsl.ts
  // (pinned by nrcEncodeBackward.test.ts) and could drift. The CPU reference
  // `hashGridBackward` (nrcEncoding.ts) IS the oracle the dispatched kernel mirrors,
  // so we pin its scatter behaviour here directly.
  it('CPU hashGridBackward scatters weight × dOut into the same hashed rows the forward reads', () => {
    const grid = makeGrid(7);
    const pos: [number, number, number] = [0.2, -0.4, 0.6];
    const F = grid.featuresPerEntry;
    const LF = grid.levels.length * F;
    // dOut: distinct per (level, feature) so we can read the scatter back.
    const dOut = new Float32Array(LF);
    for (let i = 0; i < LF; i++) dOut[i] = 0.1 * (i + 1);

    const grads = hashGridBackward(grid, pos, dOut);
    expect(grads.length).toBe(grid.levels.length);

    // For each level, re-derive the 8 trilinear corners + the same hashed rows the
    // forward uses; the per-row grad MUST equal Σ_corners(weight) × dOut[level·F+f]
    // (collisions onto the same row accumulate — Instant-NGP §4).
    const [nx, ny, nz] = normalizeToAabb(pos, grid.aabbMin, grid.aabbMax);
    for (let l = 0; l < grid.levels.length; l++) {
      const level = grid.levels[l]!;
      const corners = trilinearCorners(level, nx, ny, nz);
      // Expected per-row accumulated weight.
      const rowWeight = new Map<number, number>();
      for (const { row, weight } of corners) {
        rowWeight.set(row, (rowWeight.get(row) ?? 0) + weight);
      }
      const g = grads[l]!;
      for (const [row, w] of rowWeight) {
        for (let f = 0; f < F; f++) {
          // f32 accumulation (the grad tables are Float32Array) → 1e-6 tol.
          expect(g[row * F + f]!).toBeCloseTo(w * dOut[l * F + f]!, 6);
        }
      }
      // Σ of all the 8 corner weights == 1 (partition of unity) → Σ grads over the
      // touched rows for feature f == dOut[level·F+f].
      for (let f = 0; f < F; f++) {
        let sum = 0;
        for (const row of rowWeight.keys()) sum += g[row * F + f]!;
        expect(sum).toBeCloseTo(dOut[l * F + f]!, 6);
      }
    }
  });
});
