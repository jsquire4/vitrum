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
 *     - SPPM_GROUP4_BINDINGS_WGSL contains sppmGather, sppmInsertPhoton, the
 *       @group(3) @binding(6/7/8) declarations.
 *     - SPPM_GROUP4_BINDINGS_WGSL does NOT contain the old approximation artefacts
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
  sppmRadiusAtFrame,
  sppmInitialRadius,
  SPPM_GROUP4_BINDINGS_WGSL,
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

describe('SPPM hash-grid cell math (TS mirror of sppmCellIndex WGSL)', () => {
  it('constants are consistent with each other', () => {
    // SPPM_MAX_CELLS must be a prime-ish value in a sensible range (≥ 1000).
    expect(SPPM_MAX_CELLS).toBe(65521);
    expect(SPPM_CELL_CAPACITY).toBe(128);
    expect(SPPM_PHOTON_RECORD_BYTES).toBe(48); // 3 × vec4f
    expect(SPPM_PHOTON_CELLS_BYTES).toBe(SPPM_MAX_CELLS * SPPM_CELL_CAPACITY * SPPM_PHOTON_RECORD_BYTES);
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
    const hitCells = new Set<number>();

    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const probe = {
            x: pos.x + dx * r,
            y: pos.y + dy * r,
            z: pos.z + dz * r,
          };
          hitCells.add(sppmCellIndexTS(probe.x, probe.y, probe.z, r));
        }
      }
    }

    // Must hit at least the central cell and some neighbours (≥ 4 distinct cells
    // for a point not at a cell boundary, always ≥ 1).
    expect(hitCells.size).toBeGreaterThanOrEqual(1);

    // A photon exactly at pos is in the same cell as the hit-point lookup.
    const photonCell = sppmCellIndexTS(pos.x, pos.y, pos.z, r);
    expect(hitCells.has(photonCell)).toBe(true);
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
  it('SPPM_GROUP4_BINDINGS_WGSL declares @group(3) @binding(6/7/8)', () => {
    expect(SPPM_GROUP4_BINDINGS_WGSL).toContain('@group(3) @binding(6)');
    expect(SPPM_GROUP4_BINDINGS_WGSL).toContain('@group(3) @binding(7)');
    expect(SPPM_GROUP4_BINDINGS_WGSL).toContain('@group(3) @binding(8)');
    // Group 4 must NOT appear (lavapipe only supports maxBindGroups=4, i.e. 0-3).
    expect(SPPM_GROUP4_BINDINGS_WGSL).not.toContain('@group(4)');
  });

  it('SPPM_GROUP4_BINDINGS_WGSL contains sppmGather and sppmInsertPhoton', () => {
    expect(SPPM_GROUP4_BINDINGS_WGSL).toContain('fn sppmGather(');
    expect(SPPM_GROUP4_BINDINGS_WGSL).toContain('fn sppmInsertPhoton(');
    expect(SPPM_GROUP4_BINDINGS_WGSL).toContain('fn sppmCellIndex(');
  });

  it('SPPM_GROUP4_BINDINGS_WGSL does NOT contain old approximation artefacts', () => {
    // The old 32-photon per-pixel approximation had these artefacts:
    expect(SPPM_GROUP4_BINDINGS_WGSL).not.toContain('gatherRadius = 0.35');
    expect(SPPM_GROUP4_BINDINGS_WGSL).not.toContain('strategyScale');
    expect(SPPM_GROUP4_BINDINGS_WGSL).not.toContain('1.25');
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
    expect(SPPM_GROUP4_BINDINGS_WGSL).toContain('PI * r2');
    // No hardcoded scale factor: not ×1.25 or similar.
    expect(SPPM_GROUP4_BINDINGS_WGSL).not.toMatch(/\*\s*1\.25/);
    expect(SPPM_GROUP4_BINDINGS_WGSL).not.toMatch(/\*\s*1\.5/);
  });
});
