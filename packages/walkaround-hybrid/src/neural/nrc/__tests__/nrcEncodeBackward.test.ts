// nrcEncodeBackward.test.ts — TDD correctness gate for WS3 (NRC hash-grid
// encode-backward: making the multiresolution feature tables actually LEARN).
//
// Müller, Rousselle, Novák, Keller 2021 (NRC) + Müller, Evans, Schied, Keller
// 2022 (Instant-NGP multiresolution hash encoding §4: the interpolation weight
// is the analytic ∂feature/∂corner; collisions accumulate).
//
// This file proves, on a CPU oracle (vitest runs in node — no GPU), the math the
// GPU kernels (fusedMlp backward dL/dX + nrcEncodeBackward scatter + table Adam)
// implement, and pins the WGSL codegen so a future lavapipe A/B (V20) only has to
// confirm it RUNS. The chain is:
//   encoded input  ──MLP──▶ loss  ──backprop──▶ dL/dX  ──scatter──▶ dL/dtable
// The CRITICAL gate is: analytic dL/dX == finite-difference of the loss w.r.t.
// the input (≤1e-4), and the chained table grad == FD of the loss w.r.t. each
// table cell.
//
// NRC IS BIASED — we assert NOTHING about converged mean equality anywhere.

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import {
  hashGridForward, hashGridBackward, normalizeToAabb, trilinearCorners,
  type HashGridConfig, type HashGridLevel,
  levelResolution,
} from '../nrcEncoding.ts';
import { nrcEncodeBackwardWgsl } from '../wgsl/nrcEncodeBackward.wgsl.ts';
import { fusedBackwardWgsl, type FusedMlpWgslOptions } from '../wgsl/fusedMlp.wgsl.ts';

// ── A tiny dense ReLU MLP CPU oracle (mirrors fusedMlpHarness.cpuGrads layout:
// node-layers padded to W; hidden ReLU; linear output). Returns the forward
// activations + the per-node deltas so we can read off dL/dX (the input grad).
interface TinyNet {
  W: number; outW: number; hidden: number; inW: number;
  // concatenated weights/biases, per-layer offsets
  w: Float32Array; b: Float32Array;
  wOff: number[]; bOff: number[]; lin: number[]; lout: number[]; wlayers: number;
}

function planTiny(inW: number, W: number, outW: number, hidden: number): TinyNet {
  const widths = [W];
  for (let h = 0; h < hidden; h++) widths.push(W);
  widths.push(outW);
  const wlayers = widths.length - 1;
  const wOff: number[] = [], bOff: number[] = [], lin: number[] = [], lout: number[] = [];
  let tw = 0, tb = 0;
  for (let l = 0; l < wlayers; l++) {
    const i = widths[l]!, o = widths[l + 1]!;
    wOff.push(tw); bOff.push(tb); lin.push(i); lout.push(o);
    tw += o * i; tb += o;
  }
  const w = new Float32Array(tw), b = new Float32Array(tb);
  let s = 4242 >>> 0;
  const rng = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0x100000000; };
  for (let l = 0; l < wlayers; l++) {
    const scale = Math.sqrt(2 / lin[l]!);
    for (let k = 0; k < lin[l]! * lout[l]!; k++) w[wOff[l]! + k] = (rng() * 2 - 1) * scale;
  }
  for (let i = 0; i < b.length; i++) b[i] = 0.1;
  return { W, outW, hidden, inW, w, b, wOff, bOff, lin, lout, wlayers };
}

// Forward MSE loss against a target, plus dL/dX (gradient w.r.t. the raw input).
// x has length inW (padded to W internally). Returns { loss, dX } where dX has
// length inW. The input layer is LINEAR (no ReLU), so dL/dX = Σ_o W[0][o,i]·δ₁[o].
function forwardBackward(net: TinyNet, x: Float32Array, y: Float32Array): { loss: number; dX: Float32Array } {
  const { W, outW, wlayers, w, b, wOff, bOff, lin, lout, inW } = net;
  const node = net.hidden + 2;
  const a: number[][] = [], z: number[][] = [];
  for (let nl = 0; nl < node; nl++) { a.push(new Array(W).fill(0)); z.push(new Array(W).fill(0)); }
  for (let i = 0; i < W; i++) a[0]![i] = i < inW ? x[i]! : 0;
  for (let l = 0; l < wlayers; l++) {
    const iN = lin[l]!, oN = lout[l]!, isOut = l === wlayers - 1;
    for (let o = 0; o < oN; o++) {
      let acc = b[bOff[l]! + o]!;
      for (let i = 0; i < iN; i++) acc += w[wOff[l]! + o * iN + i]! * a[l]![i]!;
      z[l + 1]![o] = acc;
      a[l + 1]![o] = isOut ? acc : Math.max(0, acc);
    }
  }
  // loss = ½ Σ (pred - y)²   (B=1 → no /B)
  let loss = 0;
  for (let o = 0; o < outW; o++) { const e = a[node - 1]![o]! - y[o]!; loss += 0.5 * e * e; }
  // backprop
  const delta: number[][] = [];
  for (let nl = 0; nl < node; nl++) delta.push(new Array(W).fill(0));
  for (let o = 0; o < outW; o++) delta[node - 1]![o] = a[node - 1]![o]! - y[o]!;
  for (let l = wlayers - 1; l >= 1; l--) {
    const iN = lin[l]!, oN = lout[l]!;
    for (let i = 0; i < iN; i++) {
      let acc = 0;
      for (let o = 0; o < oN; o++) acc += w[wOff[l]! + o * iN + i]! * delta[l + 1]![o]!;
      delta[l]![i] = acc * (z[l]![i]! > 0 ? 1 : 0);
    }
  }
  // dL/dX = Σ_o W[0][o,i]·δ₁[o]  (input layer linear — NO relu')
  const dX = new Float32Array(inW);
  { const l = 0, iN = lin[0]!, oN = lout[0]!;
    for (let i = 0; i < Math.min(iN, inW); i++) {
      let acc = 0;
      for (let o = 0; o < oN; o++) acc += w[wOff[l]! + o * iN + i]! * delta[1]![o]!;
      dX[i] = acc;
    }
  }
  return { loss, dX };
}

// build a small multiresolution grid (collisions exercised at fine levels)
function makeGrid(seed = 3): HashGridConfig {
  const F = 2, nMin = 4, growth = 2;
  const levels: HashGridLevel[] = [];
  let s = seed >>> 0;
  const rng = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0x100000000; };
  for (let l = 0; l < 4; l++) {
    const resolution = levelResolution(nMin, growth, l);
    const tableSize = 53; // prime, < dense at fine levels → collisions
    const table = new Float32Array(tableSize * F);
    for (let k = 0; k < table.length; k++) table[k] = (rng() * 2 - 1) * 0.5;
    levels.push({ resolution, tableSize, table });
  }
  return { dim: 3, featuresPerEntry: F, levels, aabbMin: [-1, -1, -1], aabbMax: [1, 1, 1] };
}

// assemble a full encoded input vector with the hash-grid features at the FRONT
// (matches nrcEncoding.ts assembleNrcInput layout); the tail is fixed raw values.
function assemble(grid: HashGridConfig, pos: [number, number, number]): { x: Float32Array; LF: number } {
  const hg = hashGridForward(grid, pos);
  const LF = hg.length;
  const tail = [0.3, 0.2, 0.7, 0.4, 0.8, 0.1, 0.5]; // dir-blob-stand-in + raw (fixed)
  const x = new Float32Array(LF + tail.length);
  x.set(hg, 0); x.set(tail, LF);
  return { x, LF };
}

describe('NRC encode-backward — dL/dX analytic == finite difference (CRITICAL GATE)', () => {
  it('input gradient matches central finite-difference of the loss (≤1e-4)', () => {
    const net = planTiny(15 /*inW*/, 16 /*W*/, 3 /*outW*/, 3 /*hidden*/);
    const grid = makeGrid(11);
    const { x } = assemble(grid, [0.13, -0.42, 0.55]);
    expect(x.length).toBe(net.inW);
    const y = new Float32Array([0.4, 0.55, 0.2]);
    const { dX } = forwardBackward(net, x, y);

    const h = 1e-4;
    let maxAbsErr = 0;
    for (let i = 0; i < net.inW; i++) {
      const xp = x.slice(); xp[i] = xp[i]! + h;
      const xm = x.slice(); xm[i] = xm[i]! - h;
      const fd = (forwardBackward(net, xp, y).loss - forwardBackward(net, xm, y).loss) / (2 * h);
      maxAbsErr = Math.max(maxAbsErr, Math.abs(dX[i]! - fd));
      expect(dX[i]!).toBeCloseTo(fd, 4);
    }
    // tight gate — the analytic input grad is the load-bearing new quantity.
    expect(maxAbsErr).toBeLessThan(1e-4);
  });
});

describe('NRC encode-backward — chained encode→forward→backward table grad == FD', () => {
  it('dL/dtable from the scatter matches FD of the loss through the whole chain (≤1e-3)', () => {
    const net = planTiny(15, 16, 3, 3);
    const grid = makeGrid(5);
    const pos: [number, number, number] = [-0.3, 0.22, 0.41];
    const y = new Float32Array([0.5, 0.3, 0.6]);

    // loss(grid) = MLP loss with the hash-grid features (front L·F of the input).
    const lossOf = (g: HashGridConfig) => forwardBackward(net, assemble(g, pos).x, y).loss;

    // analytic: dL/dfeature = first L·F of dL/dX; scatter via hashGridBackward.
    const { x, LF } = assemble(grid, pos);
    const { dX } = forwardBackward(net, x, y);
    const dFeature = dX.slice(0, LF); // hash-grid features are at the FRONT
    const grads = hashGridBackward(grid, pos, dFeature);

    const hh = 1e-3;
    for (let l = 0; l < grid.levels.length; l++) {
      const lvl = grid.levels[l]!;
      for (const k of [0, 7, 19, lvl.table.length - 1]) {
        const tp = lvl.table.slice(); tp[k] = tp[k]! + hh;
        const tm = lvl.table.slice(); tm[k] = tm[k]! - hh;
        const gp: HashGridConfig = { ...grid, levels: grid.levels.map((v, i) => i === l ? { ...v, table: tp } : v) };
        const gm: HashGridConfig = { ...grid, levels: grid.levels.map((v, i) => i === l ? { ...v, table: tm } : v) };
        const fd = (lossOf(gp) - lossOf(gm)) / (2 * hh);
        expect(grads[l]![k]!).toBeCloseTo(fd, 3);
      }
    }
  });

  it('the scatter recomputes the SAME trilinear corners the encode-backward kernel uses', () => {
    // The kernel recomputes corners from the stored world pos + AABB; this pins
    // that the CPU oracle (which the kernel mirrors) does the same normalisation.
    const grid = makeGrid(2);
    const pos: [number, number, number] = [0.2, -0.1, 0.33];
    const [nx, ny, nz] = normalizeToAabb(pos, grid.aabbMin, grid.aabbMax);
    for (const lvl of grid.levels) {
      const corners = trilinearCorners(lvl, nx, ny, nz);
      const sum = corners.reduce((a, c) => a + c.weight, 0);
      expect(sum).toBeCloseTo(1, 12); // partition of unity (scatter conserves dOut)
    }
  });
});

// ── Adam step formula (the table optimizer reuses ADAM_WGSL) ──
function adamStep(p: number, g: number, m: number, v: number, t: number, lr: number) {
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const mi = b1 * m + (1 - b1) * g;
  const vi = b2 * v + (1 - b2) * g * g;
  const bc1 = 1 - Math.pow(b1, t), bc2 = 1 - Math.pow(b2, t);
  const mhat = mi / bc1, vhat = vi / bc2;
  return { p: p - lr * mhat / (Math.sqrt(vhat) + eps), m: mi, v: vi };
}

describe('NRC table Adam — step formula + liveness', () => {
  it('matches the closed-form Adam update for one step (the ADAM_WGSL math)', () => {
    const r = adamStep(0.5, 0.2, 0, 0, 1, 0.1);
    // m1 = 0.1·0.2=0.02; v1=0.001·0.04=4e-5; bc1=0.1,bc2=0.001
    // mhat=0.2, vhat=0.04 → step = 0.1·0.2/(0.2+1e-8) ≈ 0.0999999...
    expect(r.m).toBeCloseTo(0.02, 9);
    expect(r.v).toBeCloseTo(4e-5, 12);
    expect(r.p).toBeCloseTo(0.5 - 0.1 * 0.2 / (0.2 + 1e-8), 9);
  });

  it('LIVENESS: table entries CHANGE by >1e-6 after N Adam steps with non-zero grad', () => {
    // Guards the silent no-write failure: if the scatter produced zero grad (the
    // old frozen-table bug) the table would never move. With a real dL/dfeature
    // the touched rows must move measurably. CPU simulation of the GPU pipeline.
    const net = planTiny(15, 16, 3, 3);
    const grid = makeGrid(9);
    const pos: [number, number, number] = [0.05, 0.5, -0.25];
    const y = new Float32Array([0.7, 0.2, 0.9]);
    const lvl0 = grid.levels[0]!;
    const before = lvl0.table.slice();
    const m = new Float32Array(lvl0.table.length), v = new Float32Array(lvl0.table.length);
    let moved = false;
    for (let step = 1; step <= 8; step++) {
      const { x, LF } = assemble(grid, pos);
      const { dX } = forwardBackward(net, x, y);
      const grads = hashGridBackward(grid, pos, dX.slice(0, LF));
      const g0 = grads[0]!;
      for (let k = 0; k < lvl0.table.length; k++) {
        const r = adamStep(lvl0.table[k]!, g0[k]!, m[k]!, v[k]!, step, 0.1);
        lvl0.table[k] = r.p; m[k] = r.m; v[k] = r.v;
      }
    }
    let maxDelta = 0;
    for (let k = 0; k < lvl0.table.length; k++) maxDelta = Math.max(maxDelta, Math.abs(lvl0.table[k]! - before[k]!));
    moved = maxDelta > 1e-6;
    expect(moved).toBe(true);
    expect(maxDelta).toBeGreaterThan(1e-6);
  });
});

describe('NRC encode-backward — WGSL codegen pins (line-for-line oracle equivalence)', () => {
  const opts = { levels: 4, featuresPerEntry: 2, inWidth: 31 };

  it('binds gradTablesFx at MODULE scope (no storage ptr arg) and uses i32 atomics', () => {
    const wgsl = nrcEncodeBackwardWgsl(opts);
    expect(wgsl).toContain('var<storage, read_write> gradTablesFx : array<atomic<i32>>');
      expect(wgsl).toContain('atomicCompareExchangeWeak(&gradTablesFx[gradIndex]');
      expect(wgsl).toContain('NRC_DIAG_DROPPED_UPDATE');
    // it must NOT take the storage buffer as a function pointer parameter.
    expect(wgsl).not.toContain('ptr<storage, array<atomic<i32>>');
  });

  it('keeps the executable harness binding-complete for diagnostics', () => {
    const harness = readFileSync(
      new URL('../nrcEncodeBackwardHarness.ts', import.meta.url),
      'utf8',
    );
    expect(harness).toContain('{ binding: 5, resource: { buffer: diagnosticsBuf } }');
    expect(harness).toContain('size: NRC_DIAGNOSTIC_BYTES');
  });

  it('inlines the 8-corner trilinear scatter with the same product-of-axes weight', () => {
    const wgsl = nrcEncodeBackwardWgsl(opts);
    expect(wgsl).toContain('for (var c: u32 = 0u; c < 8u');
    expect(wgsl).toContain('let weight = wx * wy * wz;');
    expect(wgsl).toContain('gradInputF[rowBase + outBase + f]'); // dL/dfeature read
  });

  it('uses the Instant-NGP hash primes (matches the forward + CPU oracle)', () => {
    const wgsl = nrcEncodeBackwardWgsl(opts);
    expect(wgsl).toContain('2654435761u');
    expect(wgsl).toContain('805459861u');
  });

  it('uses the SAME fixed-point scale (2^20) as the fused MLP grad atomics', () => {
    const wgsl = nrcEncodeBackwardWgsl(opts);
    expect(wgsl).toContain('1048576.0');
    expect(wgsl).toContain('NRC_GRAD_FP');
  });

  it('one invocation per active sample, guarded by numActive', () => {
    const wgsl = nrcEncodeBackwardWgsl(opts);
    expect(wgsl).toContain('@workgroup_size(64, 1, 1)');
    expect(wgsl).toContain('if (s >= p.numActive) { return; }');
  });
});

describe('fused MLP backward — now emits dL/dX into gradInputFx (the upstream signal)', () => {
  const MULLER: FusedMlpWgslOptions = { useF16: false, W: 64, OUT_W: 3, HIDDEN: 6, TILE_B: 32 };

    it('declares the gradInputFx atomic binding and safely writes it at the l==0 step', () => {
      const wgsl = fusedBackwardWgsl(MULLER);
      expect(wgsl).toContain('gradInputFx : array<atomic<i32>>');
      // Row stride is the RAW input width p.inW, and the bounded CAS helper
      // prevents non-finite conversion and signed-i32 accumulator overflow.
      expect(wgsl).toContain('nrcAddGradInput(S * p.inW + col, acc);');
      expect(wgsl).toContain('fn nrcAddGradInput(index: u32, value: f32)');
      expect(wgsl).toContain('NRC_DIAG_DROPPED_UPDATE');
      const l0branch = wgsl.slice(wgsl.indexOf('// l == 0: emit dL/dX'));
      expect(l0branch).toContain('nrcAddGradInput(');
    });
});
