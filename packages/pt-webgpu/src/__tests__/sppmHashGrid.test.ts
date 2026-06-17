/**
 * sppmHashGrid.test.ts — unit tests for the A4 SPPM photon hash-grid and
 * progressive-radius math.
 *
 * Tests:
 *  1. Hash-grid cell math (TS): sppmCellIndex insert/query round-trip and
 *     3×3×3 neighbourhood coverage.
 *  2. Progressive radius schedule (TS): sppmRadiusAtFrame against closed form,
 *     α = 2/3, frames 0/1/10/100.
 *  3. Scale-aware initial radius (TS): sppmInitialRadius — Cornell, large scene,
 *     floor clamp at 1e-3.
 *  4. Structural WGSL assertions:
 *     - SPPM_GROUP3_BINDINGS_WGSL contains sppmGatherProgressive, sppmInsertPhoton,
 *       the @group(3) @binding(6/7/8) declarations.
 *     - SPPM_GROUP3_BINDINGS_WGSL does NOT contain the old approximation artefacts
 *       (gatherRadius = 0.35, strategyScale, 1.25).
 *     - SPPM_PHOTON_PASS_WGSL contains the sppmEmitPhotons entry point.
 *     - Both modules use the POINT_LIGHT_VEC4_STRIDE / SPOT_LIGHT_VEC4_STRIDE
 *       named constants (no bare numeric literals at stride sites).
 *
 * Provenance: Hachisuka & Jensen 2009 "Stochastic Progressive Photon Mapping".
 */

import { describe, expect, it } from 'vitest';
import {
  SPPM_MAX_CELLS,
  SPPM_CELL_CAPACITY,
  SPPM_PHOTON_RECORD_BYTES,
  SPPM_PHOTON_CELLS_BYTES,
  SPPM_CELL_COUNTERS_BYTES,
  SPPM_STATS_BYTES,
  SPPM_ALPHA,
  SPPM_PIXEL_STATS_BYTES_PER_PIXEL,
  sppmRadiusAtFrame,
  sppmInitialRadius,
  SPPM_GROUP3_BINDINGS_WGSL,
  SPPM_PHOTON_PASS_WGSL,
} from '../wgsl/pathTrace/sppmBindings.wgsl.js';

// ── 1. Hash-grid cell math ─────────────────────────────────────────────────────
//
// We mirror the WGSL `sppmCellIndex` hash in TypeScript to verify round-trip
// stability and neighbourhood coverage.  The prime-multiplied hash is:
//   ix = floor(pos.x / r),  iy = ...,  iz = ...
//   ux = bitcast_u32(ix), etc.
//   h = (ux * 1223u) ^ (uy * 7919u) ^ (uz * 1049u)
//   cell = h % SPPM_MAX_CELLS
//
// JavaScript lacks unsigned 32-bit integer bitcast, but Math.imul + >>> 0
// gives the same wrapping arithmetic on 32 bits.

function sppmCellIndexTS(posX: number, posY: number, posZ: number, radius: number): number {
  const r = Math.max(radius, 1e-6);
  const ix = Math.floor(posX / r) | 0;
  const iy = Math.floor(posY / r) | 0;
  const iz = Math.floor(posZ / r) | 0;
  const ux = ix >>> 0; // reinterpret as u32 (two's complement)
  const uy = iy >>> 0;
  const uz = iz >>> 0;
  const h = (Math.imul(ux, 1223) ^ Math.imul(uy, 7919) ^ Math.imul(uz, 1049)) >>> 0;
  return h % SPPM_MAX_CELLS;
}

function sppmNeighbourhoodCellsTS(
  posX: number,
  posY: number,
  posZ: number,
  offsetRadius: number,
  hashRadius: number,
): Set<number> {
  const cells = new Set<number>();
  for (let dz = -1; dz <= 1; dz++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        cells.add(sppmCellIndexTS(
          posX + dx * offsetRadius,
          posY + dy * offsetRadius,
          posZ + dz * offsetRadius,
          hashRadius,
        ));
      }
    }
  }
  return cells;
}

describe('SPPM hash-grid cell math (TS mirror of sppmCellIndex WGSL)', () => {
  it('constants are consistent with each other', () => {
    // SPPM_MAX_CELLS must be a prime-ish value in a sensible range (≥ 1000).
    expect(SPPM_MAX_CELLS).toBe(65521);
    // R7a (2026-06-10): 64, not 128 — 128 made the cells buffer 402 MiB,
    // exceeding WebGPU's DEFAULT maxBufferSize (256 MiB); photon-map failed
    // buffer validation on every default-limit device. 64 ≈ 201 MiB fits.
    expect(SPPM_CELL_CAPACITY).toBe(32);
    expect(SPPM_PHOTON_RECORD_BYTES).toBe(48); // 3 × vec4f
    expect(SPPM_PHOTON_CELLS_BYTES).toBe(SPPM_MAX_CELLS * SPPM_CELL_CAPACITY * SPPM_PHOTON_RECORD_BYTES);
    // The whole grid must fit the WebGPU DEFAULT maxBufferSize so photon-map
    // works without negotiated limits.
    expect(SPPM_PHOTON_CELLS_BYTES).toBeLessThanOrEqual(128 * 1024 * 1024); // default maxStorageBufferBindingSize — the tighter limit
    expect(SPPM_CELL_COUNTERS_BYTES).toBe(SPPM_MAX_CELLS * 4);
    expect(SPPM_STATS_BYTES).toBe(32);
  });

  it('returns a cell index in [0, SPPM_MAX_CELLS)', () => {
    const cases: [number, number, number, number][] = [
      [0, 0, 0, 0.1],
      [1.23, -0.5, 2.7, 0.05],
      [-100, 50, -30, 0.5],
      [0.001, 0.001, 0.001, 0.01],
    ];
    for (const [px, py, pz, r] of cases) {
      const cell = sppmCellIndexTS(px, py, pz, r);
      expect(cell).toBeGreaterThanOrEqual(0);
      expect(cell).toBeLessThan(SPPM_MAX_CELLS);
    }
  });

  it('same position + radius always maps to the same cell (deterministic)', () => {
    const cell1 = sppmCellIndexTS(1.5, 2.5, 3.5, 0.2);
    const cell2 = sppmCellIndexTS(1.5, 2.5, 3.5, 0.2);
    expect(cell1).toBe(cell2);
  });

  it('positions within the same cell-voxel hash to the same cell', () => {
    const r = 0.1;
    // Both 0.01 and 0.09 are in the same cell [0..0.1) along x.
    const cell1 = sppmCellIndexTS(0.01, 0.0, 0.0, r);
    const cell2 = sppmCellIndexTS(0.09, 0.0, 0.0, r);
    expect(cell1).toBe(cell2);
  });

  it('3×3×3 neighbourhood covers adjacent cells (straddling photon captured)', () => {
    const r = 0.1;
    const pos = { x: 0.5, y: 0.5, z: 0.5 };
    const hitCells = sppmNeighbourhoodCellsTS(pos.x, pos.y, pos.z, r, r);

    // Must hit at least the central cell and some neighbours (≥ 4 distinct cells
    // for a point not at a cell boundary, always ≥ 1).
    expect(hitCells.size).toBeGreaterThanOrEqual(1);

    // A photon exactly at pos is in the same cell as the hit-point lookup.
    const photonCell = sppmCellIndexTS(pos.x, pos.y, pos.z, r);
    expect(hitCells.has(photonCell)).toBe(true);
  });

  it('gather queries the stable insertion grid after progressive radius shrink', () => {
    const insertionRadius = 1.0;
    const shrunkGatherRadius = 0.1;
    const photon = { x: 0.9, y: 0.0, z: 0.0 };
    const receiver = { x: 0.95, y: 0.0, z: 0.0 };
    const insertedCell = sppmCellIndexTS(photon.x, photon.y, photon.z, insertionRadius);

    // Old bug: after R shrank, gather also hashed probes by R, so it looked in
    // cells 8/9/10... even though photons were inserted into the r0 grid cell 0.
    const oldShrunkGridCells = sppmNeighbourhoodCellsTS(
      receiver.x,
      receiver.y,
      receiver.z,
      shrunkGatherRadius,
      shrunkGatherRadius,
    );
    // Fixed path: query the same r0 grid used by insertion, while the later
    // dist2 <= R^2 filter still enforces the physical progressive gather disk.
    const fixedInsertionGridCells = sppmNeighbourhoodCellsTS(
      receiver.x,
      receiver.y,
      receiver.z,
      insertionRadius,
      insertionRadius,
    );
    const dist2 =
      (photon.x - receiver.x) ** 2 +
      (photon.y - receiver.y) ** 2 +
      (photon.z - receiver.z) ** 2;

    expect(oldShrunkGridCells.has(insertedCell)).toBe(false);
    expect(fixedInsertionGridCells.has(insertedCell)).toBe(true);
    expect(dist2).toBeLessThanOrEqual(shrunkGatherRadius * shrunkGatherRadius);
  });
});

// ── 2. Progressive radius schedule ────────────────────────────────────────────
//
// r(n) = r₀ × sqrt((n × α + α) / (n + 1))  for n ≥ 1
// r(0) = r₀ × sqrt(α)                        (first frame)
// α = 2/3 (Hachisuka & Jensen 2009, Eq. 4)

describe('SPPM progressive radius schedule (sppmRadiusAtFrame)', () => {
  it('SPPM_ALPHA is exactly 2/3', () => {
    expect(SPPM_ALPHA).toBeCloseTo(2 / 3, 10);
  });

  it('frame 0: r(0) = r₀ × sqrt(α)', () => {
    const r0 = 0.1;
    const expected = r0 * Math.sqrt(SPPM_ALPHA);
    expect(sppmRadiusAtFrame(r0, 0)).toBeCloseTo(expected, 10);
  });

  it('frame 1: r(1) = r₀ × sqrt((1×α+α)/(1+1)) = r₀ × sqrt(α)', () => {
    // (α + α) / 2 = α, sqrt(α) — same as frame 0 due to the formula symmetry.
    const r0 = 0.1;
    const expected = r0 * Math.sqrt((1 * SPPM_ALPHA + SPPM_ALPHA) / (1 + 1));
    expect(sppmRadiusAtFrame(r0, 1)).toBeCloseTo(expected, 10);
  });

  it('frame 10: r(10) = r₀ × sqrt((10α+α)/11)', () => {
    const r0 = 0.05;
    const n = 10;
    const expected = r0 * Math.sqrt((n * SPPM_ALPHA + SPPM_ALPHA) / (n + 1));
    expect(sppmRadiusAtFrame(r0, n)).toBeCloseTo(expected, 10);
  });

  it('frame 100: radius shrinks relative to r₀ but stays positive', () => {
    const r0 = 0.1;
    const r100 = sppmRadiusAtFrame(r0, 100);
    expect(r100).toBeGreaterThan(0);
    expect(r100).toBeLessThan(r0);
  });

  it('radius is monotonically non-increasing over many frames', () => {
    const r0 = 0.1;
    let prev = r0;
    for (let n = 0; n <= 200; n++) {
      const curr = sppmRadiusAtFrame(r0, n);
      // Allow a tiny floating-point tolerance.
      expect(curr).toBeLessThanOrEqual(prev + 1e-12);
      prev = curr;
    }
  });
});

// ── 3. Scale-aware initial radius ─────────────────────────────────────────────
//
// r₀ = max(diagonal / 100, 1e-3)

describe('SPPM scale-aware initial radius (sppmInitialRadius)', () => {
  it('Cornell box: diagonal ~1.73 m → r₀ ≈ 0.0173 (well above 1e-3)', () => {
    // Standard Cornell box is approximately 1×1×1 m.
    const r0 = sppmInitialRadius([0, 0, 0], [1, 1, 1]);
    const diagonal = Math.sqrt(3); // ≈ 1.732
    const expected = diagonal / 100;
    expect(r0).toBeCloseTo(expected, 5);
    expect(r0).toBeGreaterThan(1e-3);
  });

  it('large scene (100m): r₀ = diagonal/100 ≈ 1.732', () => {
    const r0 = sppmInitialRadius([0, 0, 0], [100, 100, 100]);
    const diagonal = Math.sqrt(100 * 100 * 3); // 100√3 ≈ 173.2
    const expected = diagonal / 100; // ≈ 1.732
    expect(r0).toBeCloseTo(expected, 4);
  });

  it('micro scene: floor clamp at 1e-3', () => {
    // A 0.001 m box has diagonal ≈ 1.73e-3 m; diagonal/100 ≈ 1.73e-5 < 1e-3.
    const r0 = sppmInitialRadius([0, 0, 0], [0.001, 0.001, 0.001]);
    expect(r0).toBeCloseTo(1e-3, 10);
  });

  it('degenerate: single-axis box uses that axis', () => {
    // Only x-extent = 10, y=z=0 → diagonal = 10 → r₀ = 0.1
    const r0 = sppmInitialRadius([0, 0, 0], [10, 0, 0]);
    expect(r0).toBeCloseTo(10 / 100, 6);
  });
});

// ── 4. Structural WGSL assertions ─────────────────────────────────────────────

describe('SPPM WGSL structural assertions (A4)', () => {
  it('SPPM_GROUP3_BINDINGS_WGSL declares @group(3) @binding(6/7/8)', () => {
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('@group(3) @binding(6)');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('@group(3) @binding(7)');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('@group(3) @binding(8)');
    // Group 4 must NOT appear (lavapipe only supports maxBindGroups=4, i.e. 0-3).
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('@group(4)');
  });

  // D9.9 — sppmGather deleted 2026-06-10 (superseded by sppmGatherProgressive).
  it('SPPM_GROUP3_BINDINGS_WGSL contains sppmGatherProgressive and sppmInsertPhoton', () => {
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('fn sppmGather('); // dead — deleted D9.9
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('fn sppmGatherProgressive(');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('fn sppmInsertPhoton(');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('fn sppmCellIndex(');
  });

  it('SPPM_GROUP3_BINDINGS_WGSL does NOT contain old approximation artefacts', () => {
    // The old 32-photon per-pixel approximation had these artefacts:
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('gatherRadius = 0.35');
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('strategyScale');
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('1.25');
  });

  it('sppmGatherProgressive queries the stable insertion grid, not the shrunk gather radius', () => {
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('let gridRadius = max(sppmStats.currentRadius, 1e-6);');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('* gridRadius');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('sppmCellIndex(probe, gridRadius)');
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('sppmCellIndex(probe, r)');
  });

  it('SPPM_PHOTON_PASS_WGSL contains the sppmEmitPhotons entry point', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('@compute @workgroup_size(64, 1, 1)');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('fn sppmEmitPhotons(');
  });

  it('photon-emission pass uses POINT_LIGHT_VEC4_STRIDE (not a bare literal)', () => {
    expect(SPPM_PHOTON_PASS_WGSL).not.toMatch(/let pointBase\s*=\s*pointIdx\s*\*\s*\d+u/);
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let pointBase = pointIdx * POINT_LIGHT_VEC4_STRIDE');
  });

  it('photon-emission pass uses SPOT_LIGHT_VEC4_STRIDE (not a bare literal)', () => {
    expect(SPPM_PHOTON_PASS_WGSL).not.toMatch(/let spotBase\s*=\s*spotIdx\s*\*\s*\d+u/);
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let spotBase = spotIdx * SPOT_LIGHT_VEC4_STRIDE');
  });

  it('photon-emission spot axis has no negation (forward emission axis)', () => {
    // The packed spot direction is the forward emission axis; negating it emits
    // photons away from the scene.
    expect(SPPM_PHOTON_PASS_WGSL).not.toMatch(/let spotAxis\s*=\s*safe_normalize\(-/);
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let spotAxis = safe_normalize(saxisVec.xyz)');
  });

  it('spmmGather uses the π r² density estimator (no hardcoded fudge)', () => {
    // The gather must divide by PI * r² — standard SPPM estimator.
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('PI * r2');
    // No hardcoded scale factor: not ×1.25 or similar.
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toMatch(/\*\s*1\.25/);
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toMatch(/\*\s*1\.5/);
  });
});

// ── 5. A4-progressive: SPPM_PIXEL_STATS_BYTES_PER_PIXEL constant ─────────────

describe('SPPM_PIXEL_STATS_BYTES_PER_PIXEL constant (A4-progressive)', () => {
  it('equals 32 (8 × f32)', () => {
    // SppmPixelStats = tau.rgb (f32×3) + radius2 (f32) + N (f32) + _pad×3 (f32×3) = 8 f32.
    expect(SPPM_PIXEL_STATS_BYTES_PER_PIXEL).toBe(32);
  });

  it('buffer size for a 1920×1080 frame fits inside maxStorageBufferBindingSize default (128 MiB)', () => {
    const w = 1920, h = 1080;
    const bytes = w * h * SPPM_PIXEL_STATS_BYTES_PER_PIXEL;
    // 1920 × 1080 × 32 = 66 355 200 bytes ≈ 63 MiB
    expect(bytes).toBeLessThan(128 * 1024 * 1024);
  });
});

// ── 6. A4-progressive: per-pixel recurrence vs closed form ───────────────────
//
// TypeScript mirror of the WGSL sppmGatherProgressive update rule.
// Given a constant M photons per frame (idealized), the per-pixel stats evolve
// as a first-order recurrence.  We compare the TS recurrence against the
// closed-form N(n) and R²(n) derived from the update rule:
//
//   N(0) = 0,  N(k+1) = N(k) + α·M
//   ⟹ N(k) = k·α·M  (closed form, linear for M > 0)
//
//   R²(0) = r0²
//   R²(k+1) = R²(k) · (N(k)+α·M) / (N(k)+M)   [when M > 0]
//           = R²(k) · N(k+1) / (N(k)+M)
//
// The asymptotic behaviour is R²(k) ~ k^(α-1) · r₀² / (α·M)^(1-α) as k→∞.
// We test that the recurrence matches the TS step-by-step computation to
// within floating-point precision (no WGSL involved — pure TS math).

/** TS mirror of one sppmGatherProgressive update step (no rendering). */
function sppmUpdateStep(
  tau: [number, number, number],
  radius2: number,
  N: number,
  M: number,
  phiM: [number, number, number],
  alpha: number,
): { tau: [number, number, number]; radius2: number; N: number } {
  const Nprime = N + alpha * M;
  const NplusM = N + M;
  const ratio = M < 0.5 ? 1.0 : Nprime / NplusM;
  const r2prime = radius2 * ratio;
  const tauPrime: [number, number, number] = [
    (tau[0] + phiM[0]) * ratio,
    (tau[1] + phiM[1]) * ratio,
    (tau[2] + phiM[2]) * ratio,
  ];
  return { tau: tauPrime, radius2: r2prime, N: Nprime };
}

describe('A4-progressive SPPM recurrence (TS mirror vs closed form)', () => {
  it('N(k) = k·α·M after k steps with constant M (linear accumulation)', () => {
    const M = 5.0; // constant photons per frame
    let N = 0.0;
    for (let k = 1; k <= 100; k++) {
      const result = sppmUpdateStep([0, 0, 0], 1.0, N, M, [0, 0, 0], SPPM_ALPHA);
      N = result.N;
      const closedForm = k * SPPM_ALPHA * M;
      expect(N).toBeCloseTo(closedForm, 8);
    }
  });

  it('R² shrinks monotonically for M > 0', () => {
    let r2 = 0.01; // r₀² = 0.1²
    let N = 0.0;
    let prevR2 = r2;
    const M = 3.0;
    for (let k = 0; k < 100; k++) {
      const result = sppmUpdateStep([0, 0, 0], r2, N, M, [0, 0, 0], SPPM_ALPHA);
      r2 = result.radius2;
      N = result.N;
      expect(r2).toBeLessThanOrEqual(prevR2 + 1e-15);
      expect(r2).toBeGreaterThan(0);
      prevR2 = r2;
    }
  });

  it('tau accumulates then shrinks (converging integral)', () => {
    // Inject a constant phiM each frame and verify tau/Ne·π·R² converges.
    const r0 = 0.1;
    let r2 = r0 * r0;
    let N = 0.0;
    let tau: [number, number, number] = [0, 0, 0];
    const phiM: [number, number, number] = [1.0, 0.5, 0.25]; // constant per-frame
    const M = 1.0;
    const photonCount = 10000;
    let prevEstimate = Infinity;
    // After many frames the estimate should stabilise (not blow up).
    for (let k = 1; k <= 200; k++) {
      const result = sppmUpdateStep(tau, r2, N, M, phiM, SPPM_ALPHA);
      tau = result.tau;
      r2 = result.radius2;
      N = result.N;
      const Ne = k * photonCount;
      const estimate = tau[0] / (Ne * Math.PI * r2);
      if (k > 50) {
        // After warm-up the estimate must not diverge.
        expect(estimate).toBeLessThan(prevEstimate * 2.0);
      }
      prevEstimate = estimate;
    }
    // Final estimate must be finite and positive.
    expect(prevEstimate).toBeGreaterThan(0);
    expect(Number.isFinite(prevEstimate)).toBe(true);
  });

  it('M=0 stability: stats are unchanged when no photons hit the pixel', () => {
    const r2 = 0.01;
    const N = 42.0;
    const tau: [number, number, number] = [3.0, 2.0, 1.0];
    // M=0 → ratio=1 → r2, tau, N all unchanged (except tau gets scaled by 1.0).
    const result = sppmUpdateStep(tau, r2, N, 0.0, [0, 0, 0], SPPM_ALPHA);
    expect(result.radius2).toBeCloseTo(r2, 15);
    expect(result.N).toBeCloseTo(N, 15);
    // tau' = (tau + 0) × 1 = tau
    expect(result.tau[0]).toBeCloseTo(tau[0], 15);
    expect(result.tau[1]).toBeCloseTo(tau[1], 15);
    expect(result.tau[2]).toBeCloseTo(tau[2], 15);
  });

  it('first-frame initialization: radius2=0 → seeds from r0² (reset behavior)', () => {
    // When the buffer is GPU-cleared (zero-initialised), radius2 == 0.
    // The WGSL uses: let r2 = select(pxStats.radius2, r0*r0, isFirstFrame)
    // which produces r0² when radius2 ≤ 0.  After one frame with M>0 the
    // radius must be strictly < r0² (it shrunk).
    const r0 = 0.1;
    const r0sq = r0 * r0;
    const M = 5.0;
    // Simulate the first-frame logic: seed r2 from r0² (radius2=0 → isFirstFrame).
    const result = sppmUpdateStep([0, 0, 0], r0sq, 0.0, M, [0.5, 0.5, 0.5], SPPM_ALPHA);
    // After first frame radius must be ≤ r0² (shrunk by ratio < 1 for M>0).
    expect(result.radius2).toBeLessThan(r0sq);
    expect(result.radius2).toBeGreaterThan(0);
    // N after first frame = 0 + α·M
    expect(result.N).toBeCloseTo(SPPM_ALPHA * M, 10);
  });
});

// ── 7. A4-progressive: WGSL structural assertions for binding(9) ─────────────

describe('A4-progressive WGSL structural assertions (binding 9 + progressive fn)', () => {
  it('SPPM_GROUP3_BINDINGS_WGSL declares @group(3) @binding(9) for sppmPixelStats', () => {
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('@group(3) @binding(9)');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('sppmPixelStats');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('array<SppmPixelStats>');
  });

  it('SPPM_GROUP3_BINDINGS_WGSL declares the SppmPixelStats struct with tau/radius2/N fields', () => {
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('struct SppmPixelStats');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('tau     : vec3f');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('radius2 : f32');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('N       : f32');
  });

  it('SPPM_GROUP3_BINDINGS_WGSL contains sppmGatherProgressive (A4 entry point)', () => {
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('fn sppmGatherProgressive(');
  });

  it('sppmGatherProgressive writes all three per-pixel stats fields back', () => {
    // The update rule must persist tau', radius2', and N' after each frame.
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('sppmPixelStats[pixelIndex].tau');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('sppmPixelStats[pixelIndex].radius2');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('sppmPixelStats[pixelIndex].N');
  });

  it('sppmGatherProgressive contains the Hachisuka ratio guard (M=0 stability)', () => {
    // The WGSL must guard M=0 to avoid 0/0.
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('select(Nprime / NplusM, 1.0, M < 0.5)');
  });

  it('SPPM_ALPHA_WGSL is interpolated into the bindings string', () => {
    // The alpha constant must be baked into the composed WGSL.
    expect(SPPM_GROUP3_BINDINGS_WGSL).toMatch(/SPPM_ALPHA_WGSL\s*=\s*[\d.]+f/);
  });
});
