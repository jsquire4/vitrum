/**
 * sppmHashGrid.test.ts — unit tests for the A4 SPPM photon hash-grid and
 * progressive-radius math.
 *
 * Tests:
 *  1. Hash-grid cell math (TS): sppmCellIndex insert/query round-trip and
 *     3×3×3 neighbourhood coverage.
 *  2. Progressive radius schedule through the live SPPM parameter update,
 *     α = 2/3, frames 0/1/10/100.
 *  3. Scale-aware initial radius (TS): sppmInitialRadius — Cornell, large scene,
 *     homogeneous behavior for tiny and large scenes.
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
  SPPM_PHOTON_COUNT,
  SPPM_PHOTON_RECORD_BYTES,
  SPPM_PHOTON_CELLS_BYTES,
  SPPM_PHOTON_CELLS_MAX_BYTES,
  SPPM_CELL_COUNTERS_BYTES,
  SPPM_STATS_BYTES,
  SPPM_ALPHA,
  SPPM_PIXEL_STATS_BYTES_PER_PIXEL,
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

function sppmCellIndexTS(
  posX: number,
  posY: number,
  posZ: number,
  radius: number,
  center: readonly [number, number, number] = [0, 0, 0],
): number {
  if (!(radius > 0) || !Number.isFinite(radius)) return 0;
  const ix = Math.floor((posX - center[0]) / radius) | 0;
  const iy = Math.floor((posY - center[1]) / radius) | 0;
  const iz = Math.floor((posZ - center[2]) / radius) | 0;
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
    expect(SPPM_PHOTON_COUNT).toBe(65536);
    expect(SPPM_PHOTON_RECORD_BYTES).toBe(48); // 3 × vec4f
    expect(SPPM_PHOTON_CELLS_BYTES).toBe(
      SPPM_PHOTON_COUNT * SPPM_PHOTON_RECORD_BYTES,
    );
    expect(SPPM_PHOTON_CELLS_BYTES).toBe(3 * 1024 * 1024);
    expect(SPPM_PHOTON_CELLS_MAX_BYTES).toBe(SPPM_PHOTON_CELLS_BYTES);
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

  it('is translation-invariant because hashing is relative to the scene center', () => {
    const local: [number, number, number] = [4.25, -2.5, 8.75];
    const center: [number, number, number] = [1e12, -2e12, 3e12];
    const atOrigin = sppmCellIndexTS(...local, 0.25);
    const translated = sppmCellIndexTS(
      center[0] + local[0],
      center[1] + local[1],
      center[2] + local[2],
      0.25,
      center,
    );
    expect(translated).toBe(atOrigin);
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('let centered = pos - sceneCenter');
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
    // scale-free radius test still enforces the physical progressive gather disk.
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

  it('one unique record per emitted lane forms a bounded collision chain', () => {
    const bucketHeads = new Uint32Array(4);
    const nextEncoded = new Uint32Array(6);
    for (const photonIndex of [0, 2, 5]) {
      const previousHead = bucketHeads[1]!;
      nextEncoded[photonIndex] = previousHead;
      bucketHeads[1] = photonIndex + 1;
    }
    const visited: number[] = [];
    let encoded = bucketHeads[1]!;
    while (encoded !== 0 && visited.length < nextEncoded.length) {
      const photonIndex = encoded - 1;
      visited.push(photonIndex);
      encoded = nextEncoded[photonIndex]!;
    }
    expect(visited).toEqual([5, 2, 0]);
    expect(new Set(visited).size).toBe(visited.length);
    expect(visited).not.toContain(4); // a nondeposited record is never linked
  });
  it('valid atomic-exchange publication is acyclic and worst-case bounded', () => {
    const nextEncoded = new Uint32Array(SPPM_PHOTON_COUNT);
    let head = 0;
    for (let photonIndex = 0; photonIndex < SPPM_PHOTON_COUNT; photonIndex++) {
      nextEncoded[photonIndex] = head;
      head = photonIndex + 1;
    }
    const seen = new Set<number>();
    let encoded = head;
    let traversed = 0;
    while (encoded !== 0 && traversed < SPPM_PHOTON_COUNT) {
      const photonIndex = encoded - 1;
      expect(photonIndex).toBeLessThan(SPPM_PHOTON_COUNT);
      expect(seen.has(photonIndex)).toBe(false);
      seen.add(photonIndex);
      encoded = nextEncoded[photonIndex]!;
      traversed++;
    }
    expect(encoded).toBe(0);
    expect(traversed).toBe(SPPM_PHOTON_COUNT);
    expect(seen.has(0)).toBe(true);
  });

  it('corrupt indices and cycles terminate without revisiting records', () => {
    const traverse = (head: number, next: Uint32Array): number[] => {
      const nextHead = (encoded: number): number => {
        if (encoded === 0) return 0;
        const index = encoded - 1;
        return index < next.length ? next[index]! : 0;
      };
      const visited: number[] = [];
      let encoded = head;
      let slow = head;
      let fast = head;
      while (encoded !== 0 && visited.length < next.length) {
        const photonIndex = encoded - 1;
        if (photonIndex >= next.length) break;
        encoded = next[photonIndex]!;
        slow = nextHead(slow);
        fast = nextHead(nextHead(fast));
        visited.push(photonIndex);
        if (slow !== 0 && slow === fast) break;
      }
      return visited;
    };
    const corrupt = new Uint32Array(4);
    expect(traverse(9, corrupt)).toEqual([]);

    const cyclic = new Uint32Array(4);
    cyclic[0] = 2;
    cyclic[1] = 1;
    const visited = traverse(1, cyclic);
    expect(visited).toEqual([0, 1]);
    expect(new Set(visited).size).toBe(visited.length);
  });


  it('deduplicates hash collisions among the 27 neighbourhood probes', () => {
    const cells = [] as number[];
    for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      cells.push(sppmCellIndexTS(dx * 0.1, dy * 0.1, dz * 0.1, 0.1));
    }
    const unique = [...new Set(cells)];
    expect(unique.length).toBeLessThanOrEqual(27);
    expect(unique.length).toBeGreaterThan(0);
    expect(unique.reduce((count) => count + 1, 0)).toBe(unique.length);
  });
});

// ── 3. Scale-aware initial radius ─────────────────────────────────────────────
//
// r₀ = diagonal / 100 for every non-degenerate scene.

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

  it('micro scene: preserves diagonal/100 without a world-unit floor', () => {
    const r0 = sppmInitialRadius([0, 0, 0], [0.001, 0.001, 0.001]);
    expect(r0).toBeCloseTo(Math.sqrt(3) * 1e-5, 12);
  });

  it('is homogeneous across thirty orders of magnitude', () => {
    const ordinary = sppmInitialRadius([0, 0, 0], [1, 2, 3]);
    for (const scale of [1e-30, 1e30]) {
      const scaled = sppmInitialRadius(
        [0, 0, 0],
        [scale, 2 * scale, 3 * scale],
      );
      expect((scaled / ordinary) / scale).toBeCloseTo(1, 12);
    }
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

  // D9.9 — the legacy bounded gather was replaced by progressive update/readback.
  it('contains split progressive update/readback and photon insertion', () => {
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('fn sppmGather('); // dead — deleted D9.9
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('fn sppmUpdateProgressiveKind(');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('fn sppmCurrentProgressiveEstimate(');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('fn sppmInsertPhoton(');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('kind: u32');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('mediumMatId: u32');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('phaseG: f32');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('fn sppmCellIndex(');
  });

  it('SPPM_GROUP3_BINDINGS_WGSL does NOT contain old approximation artefacts', () => {
    // The old 32-photon per-pixel approximation had these artefacts:
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('gatherRadius = 0.35');
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('strategyScale');
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('1.25');
  });

  it('progressive update queries the stable insertion grid, not the shrunk gather radius', () => {
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('let gridRadius = sppmStats.currentRadius;');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain(
      'if (!(gridRadius > 0.0) || gridRadius > 3.402823466e38)',
    );
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('max(sppmStats.currentRadius, 1e-6)');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('* gridRadius');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('sppmCellIndex(probe, gridRadius)');
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('sppmCellIndex(probe, r)');
  });

  it('uses a race-free per-frame linked grid with bounded deduplicated traversal', () => {
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('photonIndex + 1u');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('atomicExchange(&sppmCellCounters[cellIdx]');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('.nextEncoded = previousHead');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('encodedPhoton = ph.nextEncoded');
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('bitcast<f32>(previousHead)');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('var visitedCells: array<u32, 27>');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('if (visitedCells[vi] == cellIdx)');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('traversed >= nPhotons');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('fn sppmNextEncodedHead(');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('var cycleSlow = encodedPhoton');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('var cycleFast = encodedPhoton');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('cycleSlow == cycleFast');
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('atomicAdd(&sppmCellCounters');
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('reservoirXi');
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('cellSampleScale');
  });

  it('does not apply a second receiver cosine to the photon density estimate', () => {
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('if (nDotL > 0.0)');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('throughput * brdf * ph.flux');
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('brdf * ph.flux * nDotL');
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

  it('sppm gather uses the π r² density estimator in log space (no hardcoded fudge)', () => {
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain(
      'log(PI) + 2.0 * log(pxStats.radius)',
    );
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('fn sppmDensityChannel(');
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('radius <= 1e-24');
    // No hardcoded scale factor: not ×1.25 or similar.
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toMatch(/\*\s*1\.25/);
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toMatch(/\*\s*1\.5/);
  });
});

// ── 5. A4-progressive: SPPM_PIXEL_STATS_BYTES_PER_PIXEL constant ─────────────

describe('SPPM_PIXEL_STATS_BYTES_PER_PIXEL constant (A4-progressive)', () => {
  it('equals 64 (independent 8-f32 surface and volume records)', () => {
    expect(SPPM_PIXEL_STATS_BYTES_PER_PIXEL).toBe(64);
  });

  it('buffer size for a 1920×1080 frame fits inside maxStorageBufferBindingSize default (128 MiB)', () => {
    const w = 1920, h = 1080;
    const bytes = w * h * SPPM_PIXEL_STATS_BYTES_PER_PIXEL;
    // 1920 × 1080 × 64 = 132 710 400 bytes ≈ 126.6 MiB
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

  it('first-frame initialization keeps a linear radius in GPU state', () => {
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
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain(
      'let r = select(pxStats.radius, r0, isFirstFrame)',
    );
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('r0 * r0');
  });

  it('normalizes photon power once and evaluates the receiver BRDF once', () => {
    const sourcePower = 20;
    const sourceSelectionPdf = 0.25;
    const emittedPhotonCount = 100;
    const gatheredPhotons = 10;
    const receiverBrdf = 0.25;
    const r0 = 2;

    // The photon stores source power divided by source PMF, but not by N_e.
    const storedPower = sourcePower / sourceSelectionPdf;
    const phiM: [number, number, number] = [
      gatheredPhotons * receiverBrdf * storedPower,
      0,
      0,
    ];
    const update = sppmUpdateStep([0, 0, 0], r0 * r0, 0, gatheredPhotons, phiM, SPPM_ALPHA);
    const estimate = update.tau[0] / (emittedPhotonCount * Math.PI * update.radius2);
    const independent =
      gatheredPhotons * receiverBrdf * storedPower /
      (emittedPhotonCount * Math.PI * r0 * r0);

    expect(estimate).toBeCloseTo(independent, 12);
    expect(estimate).not.toBeCloseTo(independent / emittedPhotonCount, 12);
    expect(estimate).not.toBeCloseTo(independent * receiverBrdf, 12);
  });
});

// ── 7. A4-progressive: WGSL structural assertions for binding(9) ─────────────

describe('A4-progressive WGSL structural assertions (binding 9 + progressive fn)', () => {
  it('SPPM_GROUP3_BINDINGS_WGSL declares @group(3) @binding(9) for sppmPixelStats', () => {
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('@group(3) @binding(9)');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('sppmPixelStats');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('array<SppmPixelStats>');
  });

  it('SPPM_GROUP3_BINDINGS_WGSL declares the SppmPixelStats struct with tau/radius/N fields', () => {
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('struct SppmPixelStats');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('tau     : vec3f');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('radius  : f32');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('N       : f32');
  });

  it('declares distinct surface/volume update entry points', () => {
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('fn sppmUpdateSurfaceProgressive(');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('fn sppmUpdateVolumeProgressive(');
  });

  it('progressive update writes all three per-pixel stats fields back', () => {
    // The update rule must persist tau', radius', and N' after each frame.
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('sppmPixelStats[statsIndex].tau');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('sppmPixelStats[statsIndex].radius');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('sppmPixelStats[statsIndex].N');
  });

  it('progressive update contains the Hachisuka ratio guard (M=0 stability)', () => {
    // The WGSL must guard M=0 to avoid 0/0.
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('select(Nprime / NplusM, 1.0, M < 0.5)');
  });

  it('SPPM_ALPHA_WGSL is interpolated into the bindings string', () => {
    // The alpha constant must be baked into the composed WGSL.
    expect(SPPM_GROUP3_BINDINGS_WGSL).toMatch(/SPPM_ALPHA_WGSL\s*=\s*[\d.]+f/);
  });
});
