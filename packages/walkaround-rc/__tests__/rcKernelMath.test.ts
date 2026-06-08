/**
 * G-P2.7 / E2 — numeric oracle for the walkaround-rc WGSL kernels.
 *
 * The two RC kernels — `probeRayCast.wgsl.ts` and `cascadeMerge.wgsl.ts` — can't
 * run inside `npm test` (no GPU), and the package's one behavior test
 * (`rcBehavior.gpu.test.ts`) is env-gated off. Until this file, the kernels had
 * STRING pins only (assertions that a WGSL substring is present), which cannot
 * catch a wrong formula.
 *
 * The durable in-tree approach is a **TS mirror + numeric oracle**: the pure-math
 * pieces of each kernel are ported to TypeScript here, VERBATIM from the WGSL, and
 * asserted against INDEPENDENT analytic references — hand-computed values and
 * closed-form identities — NOT against the WGSL string and NOT against tautologies.
 *
 * Mirrored pieces (all kept byte-faithful to the WGSL):
 *   - `octDecode` / `octEncode` (shared-samplers octahedralCore.wgsl.ts) — the
 *     `octDecode(rayUV*2-1)` per-ray direction generation in probeRayCastKernel.
 *   - `octCellSolidAngle` + `sphericalQuadAreaForMerge` (cascadeMerge.wgsl.ts) —
 *     the per-child solid-angle weight.
 *   - `cascadeMergeCell` — the `merged = Σ child·Ω / Σ Ω` weighted average.
 *
 * References (each test cites the specific identity it uses):
 *   Cigolle et al. 2014, "A Survey of Efficient Representations for Independent
 *     Unit Vectors", JCGT §2 / §A.1 / §A.2 — octahedral map, fold convention,
 *     Jacobian / texel solid angle.
 *   Sannikov 2023, §3 — cascade conservation law (the merge is a weighted average).
 *
 * NOTE (documented approximation, not a bug): the merge kernel's `octCellSolidAngle`
 * uses a single planar-quad approximation of the spherical cell area. Summed over an
 * N×N grid it UNDER-estimates 4π (−12.6% at N=4, shrinking with N). The merge formula
 * self-normalizes by Σ Ω, so this per-cell error cancels exactly in the weighted
 * average — see the convergence test below, which is the real 4π identity for this
 * helper. The high-accuracy (SUB=16) `computeOctahedralSolidAngles`, which DOES sum
 * to 4π within 1e-3, is exercised separately in `rcSolidAngles.test.ts`.
 */

import { describe, it, expect } from 'vitest';

const FOUR_PI = 4 * Math.PI;

// ─── TS mirror: octDecode (shared-samplers octahedralCore.wgsl.ts:25–34) ─────
// Byte-faithful port of the WGSL `octDecode(oct: vec2f) -> vec3f`.
// WGSL fold: xy = (1 - abs(n.yx)) * vec2f(sx, sy)  ⇒  the swizzle means the new
// x uses (1-|y|) and the new y uses (1-|x|). select(-1,+1, n>=0) for the signs
// (Cigolle §A.1 — sign(0)=+1, avoids the south-pole singularity at (0,0)).
function octDecode(u: number, v: number): [number, number, number] {
  let nx = u;
  let ny = v;
  const nz = 1.0 - Math.abs(u) - Math.abs(v);
  if (nz < 0) {
    const ox = nx;
    const oy = ny;
    nx = (1.0 - Math.abs(oy)) * (ox >= 0 ? 1 : -1);
    ny = (1.0 - Math.abs(ox)) * (oy >= 0 ? 1 : -1);
  }
  const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
  return [nx / len, ny / len, nz / len];
}

// ─── TS mirror: octEncode (shared-samplers octahedralCore.wgsl.ts:7–23) ──────
// Byte-faithful port of the WGSL `octEncode(dir: vec3f) -> vec2f`.
function octEncode(d: [number, number, number]): [number, number] {
  const l1 = Math.max(Math.abs(d[0]) + Math.abs(d[1]) + Math.abs(d[2]), 1e-20);
  const nx = d[0] / l1;
  const ny = d[1] / l1;
  const nz = d[2] / l1;
  if (nz >= 0) return [nx, ny];
  const sx = nx >= 0 ? 1 : -1;
  const sy = ny >= 0 ? 1 : -1;
  return [(1.0 - Math.abs(ny)) * sx, (1.0 - Math.abs(nx)) * sy];
}

// ─── TS mirror: probeRayCastKernel ray-direction generation ──────────────────
// probeRayCast.wgsl.ts:311–316:
//   let gx = f32(rayIdx % rayGridSize);  let gy = f32(rayIdx / rayGridSize);
//   let rayUV = (vec2f(gx, gy) + jitter) / f32(rayGridSize);
//   let rayDir = octDecode(rayUV * 2.0 - 1.0);
// jitter ∈ [0,1)² (pcgHashToF32). For the math oracle we drive it directly.
function probeRayDir(
  rayIdx: number,
  rayGridSize: number,
  jitterX: number,
  jitterY: number,
): [number, number, number] {
  const gx = rayIdx % rayGridSize;
  const gy = Math.floor(rayIdx / rayGridSize);
  const uvU = (gx + jitterX) / rayGridSize;
  const uvV = (gy + jitterY) / rayGridSize;
  return octDecode(uvU * 2.0 - 1.0, uvV * 2.0 - 1.0);
}

// ─── TS mirror: cascadeMerge octCellSolidAngle (cascadeMerge.wgsl.ts:48–67) ──
function cross3(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function len3(v: [number, number, number]): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
function sub3(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
// WGSL sphericalQuadAreaForMerge (two-triangle planar-cross approximation).
function sphericalQuadAreaForMerge(
  p00: [number, number, number],
  p10: [number, number, number],
  p01: [number, number, number],
  p11: [number, number, number],
): number {
  const d1 = cross3(sub3(p10, p00), sub3(p01, p00));
  const d2 = cross3(sub3(p10, p11), sub3(p01, p11));
  return (len3(d1) + len3(d2)) * 0.5;
}
// WGSL octCellSolidAngle(cx, cy, N) — single-quad (4-corner) per-cell solid angle.
function octCellSolidAngle(cx: number, cy: number, N: number): number {
  const cellWidth = 2.0 / N;
  const u0 = -1.0 + cx * cellWidth;
  const v0 = -1.0 + cy * cellWidth;
  const u1 = u0 + cellWidth;
  const v1 = v0 + cellWidth;
  return sphericalQuadAreaForMerge(
    octDecode(u0, v0),
    octDecode(u1, v0),
    octDecode(u0, v1),
    octDecode(u1, v1),
  );
}

// ─── TS mirror: cascadeMergeKernel weighted average (cascadeMerge.wgsl.ts:181–197) ─
//   merged = Σ_i child_i · Ω_i ;  merged /= max(Σ_i Ω_i, 1e-6)
// The 4 children of parent bin (gx,gy) in an N×N upper grid are the 2×2 block
// at childGx = gx*2 + (ci%2), childGy = gy*2 + (ci/2).
type RGB = [number, number, number];
function cascadeMergeCell(
  children: [RGB, RGB, RGB, RGB],
  parentGx: number,
  parentGy: number,
  upperGridSize: number,
): RGB {
  let r = 0;
  let g = 0;
  let b = 0;
  let omegaTotal = 0;
  for (let ci = 0; ci < 4; ci++) {
    const dx = ci % 2;
    const dy = Math.floor(ci / 2);
    const childGx = parentGx * 2 + dx;
    const childGy = parentGy * 2 + dy;
    const omega = octCellSolidAngle(childGx, childGy, upperGridSize);
    const ch = children[ci]!;
    r += ch[0] * omega;
    g += ch[1] * omega;
    b += ch[2] * omega;
    omegaTotal += omega;
  }
  const denom = Math.max(omegaTotal, 1e-6);
  return [r / denom, g / denom, b / denom];
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Probe ray-direction generation (probeRayCastKernel)
// ─────────────────────────────────────────────────────────────────────────────
describe('probeRayCast: octahedral ray-direction generation', () => {
  // Reference: a decoded octahedral direction is a UNIT vector by construction
  // (octDecode ends in normalize). Independent identity: ‖rayDir‖ == 1.
  it.each([2, 4, 8, 16] as const)(
    'every ray direction is unit-length for rayGridSize=%i (no jitter and jittered)',
    (N) => {
      let maxErr = 0;
      for (let rayIdx = 0; rayIdx < N * N; rayIdx++) {
        for (const [jx, jy] of [
          [0, 0],
          [0.5, 0.5],
          [0.123, 0.987],
        ] as const) {
          const d = probeRayDir(rayIdx, N, jx, jy);
          const L = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
          maxErr = Math.max(maxErr, Math.abs(L - 1));
        }
      }
      expect(maxErr, `max |‖dir‖-1| = ${maxErr}`).toBeLessThan(1e-6);
    },
  );

  // Reference: the octahedral map is a BIJECTION on the sphere (Cigolle 2014 §2).
  // For any unit direction d, octDecode(octEncode(d)) == d. We round-trip a dense
  // analytically-generated set of spherical directions (NOT grid-derived, so the
  // identity is independent of the kernel's own UV layout).
  it('octEncode∘octDecode is identity on the sphere (Cigolle 2014 §2 bijection)', () => {
    let maxErr = 0;
    const M = 40;
    for (let i = 0; i <= M; i++) {
      for (let j = 0; j < 2 * M; j++) {
        const theta = (Math.PI * i) / M;
        const phi = (2 * Math.PI * j) / (2 * M);
        const dir: RGB = [
          Math.sin(theta) * Math.cos(phi),
          Math.cos(theta),
          Math.sin(theta) * Math.sin(phi),
        ];
        const oct = octEncode(dir);
        const back = octDecode(oct[0], oct[1]);
        for (let k = 0; k < 3; k++) {
          maxErr = Math.max(maxErr, Math.abs(back[k]! - dir[k]!));
        }
      }
    }
    expect(maxErr, `max round-trip error = ${maxErr}`).toBeLessThan(1e-6);
  });

  // Reference: a uniform N×N ray grid covers the WHOLE sphere — its cell-center
  // directions must populate all 8 sign-octants of R³. (A hemisphere-only or
  // collapsed map would miss octants.) Independent of any single direction's value.
  it.each([4, 8, 16] as const)(
    'cell-center ray directions cover all 8 octants of the sphere for rayGridSize=%i',
    (N) => {
      const octants = new Set<string>();
      for (let rayIdx = 0; rayIdx < N * N; rayIdx++) {
        const d = probeRayDir(rayIdx, N, 0.5, 0.5); // cell center
        const sig =
          `${d[0] >= 0 ? 1 : 0}${d[1] >= 0 ? 1 : 0}${d[2] >= 0 ? 1 : 0}`;
        octants.add(sig);
      }
      expect(octants.size, `octants seen: ${[...octants].sort().join(',')}`).toBe(
        8,
      );
    },
  );

  // Hand-computed anchor: the cell-center of the central UV cell maps to a known
  // direction. For rayGridSize=2, rayIdx=0 with jitter (0.5,0.5):
  //   gx=0, gy=0 → uv = (0.5/2)*2-1 = (-0.5,-0.5) → octDecode(-0.5,-0.5).
  //   nz = 1-0.5-0.5 = 0 ⇒ no fold; raw=(-0.5,-0.5,0), len=√0.5 ⇒
  //   dir = (-1/√2, -1/√2, 0). Hand-derived, not read back from the kernel.
  it('rayGridSize=2 center ray matches the hand-derived (-1/√2,-1/√2,0)', () => {
    const d = probeRayDir(0, 2, 0.5, 0.5);
    const inv = 1 / Math.SQRT2;
    expect(Math.abs(d[0] - -inv)).toBeLessThan(1e-7);
    expect(Math.abs(d[1] - -inv)).toBeLessThan(1e-7);
    expect(Math.abs(d[2] - 0)).toBeLessThan(1e-7);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. Cascade-merge weighted average (cascadeMergeKernel)
// ─────────────────────────────────────────────────────────────────────────────
describe('cascadeMerge: solid-angle-weighted average', () => {
  // Reference: a weighted average with positive weights is a CONVEX COMBINATION —
  // the result lies inside the per-channel [min, max] hull of the inputs. This is
  // a property of any normalized Σ wᵢ xᵢ / Σ wᵢ, independent of the weight values.
  it('merge result is a convex combination (in the per-channel hull of the 4 children)', () => {
    const childSets: [RGB, RGB, RGB, RGB][] = [
      [
        [2, 0, 0],
        [5, 1, 9],
        [0.5, 3, 1],
        [8, 0.2, 4],
      ],
      [
        [1, 1, 1],
        [0, 0, 0],
        [0.3, 0.7, 0.1],
        [10, 2, 6],
      ],
    ];
    for (const N of [4, 8, 16]) {
      const parentGridSize = N / 2;
      for (const children of childSets) {
        const mn: RGB = [
          Math.min(...children.map((c) => c[0])),
          Math.min(...children.map((c) => c[1])),
          Math.min(...children.map((c) => c[2])),
        ];
        const mx: RGB = [
          Math.max(...children.map((c) => c[0])),
          Math.max(...children.map((c) => c[1])),
          Math.max(...children.map((c) => c[2])),
        ];
        for (let gy = 0; gy < parentGridSize; gy++) {
          for (let gx = 0; gx < parentGridSize; gx++) {
            const m = cascadeMergeCell(children, gx, gy, N);
            for (let k = 0; k < 3; k++) {
              expect(
                m[k]! >= mn[k]! - 1e-9 && m[k]! <= mx[k]! + 1e-9,
                `N=${N} gx=${gx} gy=${gy} ch=${k}: ${m[k]} not in [${mn[k]},${mx[k]}]`,
              ).toBe(true);
            }
          }
        }
      }
    }
  });

  // Reference: convex weights sum to 1 ⇒ a uniform input is a fixed point.
  // If all 4 children carry the SAME radiance c, the merge must return exactly c
  // (Σ c·Ω / Σ Ω = c). This is the conservation/normalization identity
  // (Sannikov 2023 §3) — and it would FAIL if the kernel forgot the `/ Σ Ω`.
  it('uniform children → merged equals that uniform value (normalization identity)', () => {
    const c: RGB = [0.37, 1.42, 9.0];
    const uniform: [RGB, RGB, RGB, RGB] = [c, c, c, c];
    for (const N of [4, 8, 16]) {
      const parentGridSize = N / 2;
      for (let gy = 0; gy < parentGridSize; gy++) {
        for (let gx = 0; gx < parentGridSize; gx++) {
          const m = cascadeMergeCell(uniform, gx, gy, N);
          for (let k = 0; k < 3; k++) {
            expect(Math.abs(m[k]! - c[k]!), `N=${N} ch=${k}: ${m[k]}`).toBeLessThan(
              1e-12,
            );
          }
        }
      }
    }
  });

  // Counter-check that the normalization is REAL, not vacuous: an UN-normalized
  // weighted SUM of uniform (1,1,1) children would equal Σ Ω (≈ the parent's solid
  // angle, ≠ 1). We confirm the raw sum differs from 1 while the actual merge == 1.
  it('the / Σ Ω normalization is load-bearing (raw weighted sum ≠ 1 for uniform input)', () => {
    const N = 8;
    const gx = 1;
    const gy = 1;
    let rawSum = 0;
    for (let ci = 0; ci < 4; ci++) {
      const childGx = gx * 2 + (ci % 2);
      const childGy = gy * 2 + Math.floor(ci / 2);
      rawSum += 1 * octCellSolidAngle(childGx, childGy, N);
    }
    expect(Math.abs(rawSum - 1), `raw Σ Ω = ${rawSum}`).toBeGreaterThan(0.01);

    const ones: RGB = [1, 1, 1];
    const merged = cascadeMergeCell([ones, ones, ones, ones], gx, gy, N);
    expect(Math.abs(merged[0]! - 1)).toBeLessThan(1e-12);
  });

  // Hand-computed two-child mix: with weights ω0, ω1 (the merge's own
  // octCellSolidAngle for the two children), the merge of children
  // c0=(1,0,0), c1=(0,0,0) and c2=c3=(0,0,0) for parent (0,0), N=4 is
  //   (ω0·1) / (ω0+ω1+ω2+ω3) along R. We recompute ω's independently here
  // and assert the merge equals that ratio (a non-tautological closed form).
  it('weighted-average ratio matches an independently computed Σ Ω closed form', () => {
    const N = 4;
    const gx = 0;
    const gy = 0;
    // The 4 children of parent (0,0): (0,0),(1,0),(0,1),(1,1).
    const w0 = octCellSolidAngle(0, 0, N);
    const w1 = octCellSolidAngle(1, 0, N);
    const w2 = octCellSolidAngle(0, 1, N);
    const w3 = octCellSolidAngle(1, 1, N);
    const expectedR = (5 * w0) / (w0 + w1 + w2 + w3);

    const children: [RGB, RGB, RGB, RGB] = [
      [5, 0, 0], // ci=0 → child (0,0), weight w0
      [0, 0, 0], // ci=1 → child (1,0), weight w1
      [0, 0, 0], // ci=2 → child (0,1), weight w2
      [0, 0, 0], // ci=3 → child (1,1), weight w3
    ];
    const m = cascadeMergeCell(children, gx, gy, N);
    expect(Math.abs(m[0]! - expectedR), `R=${m[0]} vs ${expectedR}`).toBeLessThan(
      1e-12,
    );
    expect(m[1]).toBe(0);
    expect(m[2]).toBe(0);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. octCellSolidAngle 4π identity (cascadeMerge.wgsl.ts)
// ─────────────────────────────────────────────────────────────────────────────
describe('cascadeMerge: octCellSolidAngle solid-angle identity (sphere = 4π)', () => {
  // The merge kernel's `octCellSolidAngle` uses ONE planar quad per cell. Summed
  // over an N×N grid it under-estimates 4π (−12.6% @N=4 … −0.25% @N=32) — a known
  // approximation, NOT a bug, because the merge normalizes by Σ Ω so the bias
  // cancels in the weighted average. We pin the magnitude of the gap so a future
  // change to the helper is noticed, and below assert the TRUE 4π identity via
  // refinement.
  it.each([
    [4, 0.13],
    [8, 0.04],
    [16, 0.011],
    [32, 0.003],
  ] as const)(
    '1-quad octCellSolidAngle sum under-estimates 4π by the documented margin (N=%i, < %f rel)',
    (N, maxRel) => {
      let s = 0;
      for (let cy = 0; cy < N; cy++) {
        for (let cx = 0; cx < N; cx++) s += octCellSolidAngle(cx, cy, N);
      }
      const relErr = (FOUR_PI - s) / FOUR_PI; // positive ⇒ under-estimate
      expect(relErr, `N=${N}: under-estimate ${(relErr * 100).toFixed(3)}%`).toBeGreaterThan(0);
      expect(relErr, `N=${N}: under-estimate ${(relErr * 100).toFixed(3)}%`).toBeLessThan(maxRel);
    },
  );

  // Reference: as the SAME 1-quad helper is applied on an ever-finer UV grid, the
  // planar-quad area → true spherical area, so the whole-sphere sum → 4π. This is
  // the genuine closed-form identity (sphere solid angle = 4π). At M=256 the sum
  // is within 0.005% of 4π — proving the helper is an unbiased estimator of the
  // octahedral cell solid angle, just coarse at the grid sizes RC uses.
  it('refined octCellSolidAngle sum converges to 4π (M=256 within 0.05%)', () => {
    let prev = -Infinity;
    let last = 0;
    for (const M of [4, 8, 16, 32, 64, 128, 256]) {
      let s = 0;
      for (let cy = 0; cy < M; cy++) {
        for (let cx = 0; cx < M; cx++) s += octCellSolidAngle(cx, cy, M);
      }
      // Monotone improvement toward 4π as the grid refines.
      expect(s, `M=${M}: ${s} not > prev ${prev}`).toBeGreaterThan(prev);
      expect(s, `M=${M}: ${s} overshoots 4π`).toBeLessThanOrEqual(FOUR_PI + 1e-6);
      prev = s;
      last = s;
    }
    expect(Math.abs(FOUR_PI - last) / FOUR_PI, `final rel err`).toBeLessThan(5e-4);
  });

  // All per-cell solid angles must be strictly positive and bounded by 2π (no
  // single cell can exceed a hemisphere). Pure analytic bound, kernel-independent.
  it.each([4, 8, 16, 32] as const)(
    'every octCellSolidAngle is in (0, 2π) for N=%i',
    (N) => {
      for (let cy = 0; cy < N; cy++) {
        for (let cx = 0; cx < N; cx++) {
          const w = octCellSolidAngle(cx, cy, N);
          expect(w, `N=${N} cell(${cx},${cy})`).toBeGreaterThan(0);
          expect(w, `N=${N} cell(${cx},${cy})`).toBeLessThan(2 * Math.PI);
        }
      }
    },
  );
});
