/**
 * HYB-DDGI-01 — independent CPU oracle for DDGI visibility-moment poisoning by
 * sky-miss rays (plan/road-to-100-gap-ledger-2026-06-11.md §HYB-DDGI-01).
 *
 * THE SUSPECTED MECHANISM (verified by code read):
 *   - Probe miss rays store hitDistance = BVH_INTERSECT_INFINITY
 *     (probeUpdateRays.wgsl.ts:644; the constant is 1e20, declared at
 *     shared-bvh/src/wgsl/bvhIntersect.wgsl.ts:192).
 *   - The visibility blend (probeUpdateBlend.wgsl.ts:209-240,
 *     probeUpdateBlendVisibility) skips ONLY backface (d < 0, L217) and
 *     self-intersection (d < 0.05, L225) rays. A sky miss (d = 1e20) passes
 *     both guards and is accumulated into the depth / depth² moments:
 *       newDepth   += d·w        [L233]
 *       newDepthSq += d·d·w      [L234]
 *     In f32, d·d = 1e40 OVERFLOWS (f32 max ≈ 3.4e38) → newDepthSq = +Inf.
 *   - The blended moments are stored to an rgba16float atlas (L176, L247);
 *     f16 max ≈ 65504, so even the depth MEAN (≈1e20·w/ΣW) quantizes to +Inf.
 *   - The receiver Chebyshev test (ddgiSampleWgsl.ts:139-147):
 *       mean      = vis.x
 *       variance  = abs(vis.y − mean²)
 *       chebyshev = probeDist <= mean ? 1.0
 *                 : variance / (variance + (probeDist − mean)²)
 *     With mean = +Inf (or simply "astronomically large"), EVERY probeDist
 *     satisfies probeDist <= mean → chebyshev = 1.0 → the probe NEVER occludes,
 *     even directly behind a wall the other rays hit — light leaks through.
 *
 * ORACLE: a faithful TS transcription of the blend accumulation (f32 semantics
 * via Math.fround) + f16 storage quantization + the Chebyshev consumption,
 * driven by a synthetic probe: a 3×3 wall at z = 2 catching a ray cone around
 * +Z, with the cone's outermost ray missing the wall (sky, d = 1e20). The
 * query point sits BEHIND the wall at z = 3 (ground truth visibility = 0).
 *
 *   - CONTROL (all rays hit): Chebyshev ≈ 0 → occluded. Proves the
 *     transcription + scenario are sound, so the poisoned result is
 *     attributable to the miss-ray handling alone.
 *   - POISONED (one miss among 32): pins chebyshev == 1.0 (full leak).
 *   - it.skip sibling: with finite/skipped miss semantics the same probe
 *     occludes the behind-wall point (< 0.05). Un-skip with the fix.
 *
 * NOTE on hysteresis: blended = mix(new, prev, 0.97) (L246) is a convex blend
 * converging to `new`; the oracle evaluates the steady state (prev = new),
 * which is the most favorable case for the shader — transient frames are at
 * least as poisoned.
 * NOTE on RAYS_PER_PROBE: production uses 192 rays (ddgiConstants.ts:17); the
 * oracle uses a 32-ray cone — the poisoning law is independent of ray count
 * (ONE miss with weight w contributes w·1e20 to a denominator of order Σw).
 */
import { describe, expect, it } from 'vitest';

const f32 = Math.fround;
const BVH_INTERSECT_INFINITY = 1e20; // shared-bvh bvhIntersect.wgsl.ts:192
const DDGI_VISIBILITY_OPEN_SKY_MOMENT = 65504;

// IEEE-754 binary16 round-trip (rgba16float storage). Round-to-nearest-even,
// overflow → ±Infinity. Only magnitude behavior matters for this oracle.
function f16(x: number): number {
  if (!Number.isFinite(x)) return x;
  if (x === 0) return 0;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  if (ax >= 65520) return sign * Infinity; // > f16 max after rounding
  if (ax < 2 ** -24 / 2) return sign * 0;
  // subnormal range
  if (ax < 2 ** -14) {
    const q = Math.round(ax / 2 ** -24) * 2 ** -24;
    return sign * q;
  }
  const e = Math.floor(Math.log2(ax));
  const ulp = 2 ** (e - 10);
  return sign * Math.round(ax / ulp) * ulp;
}

type V3 = [number, number, number];
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: V3): V3 => {
  const l = Math.sqrt(dot(a, a));
  return [a[0] / l, a[1] / l, a[2] / l];
};

interface ProbeRay {
  direction: V3;
  hitDistance: number;
}

// ── transcription: probeUpdateBlendVisibility moment accumulation ────────────
// probeUpdateBlend.wgsl.ts:204-240 for ONE atlas texel whose octDecode
// direction is `texelDir` (the oracle picks the texel aligned with the
// surface→probe query direction, which is the texel Chebyshev reads).
// f32 semantics modelled with Math.fround on every accumulate.
function blendVisibilityTexel(
  texelDir: V3,
  rays: ProbeRay[],
  options: { skipMisses?: boolean } = {},
): { mean: number; meanSq: number } {
  const skipMisses = options.skipMisses ?? true;
  let newDepth = 0;
  let newDepthSq = 0;
  let totalWeight = 0;
  for (const ray of rays) {
    if (ray.hitDistance < 0.0) continue; // L217 backface skip
    if (ray.hitDistance < 0.05) continue; // L225 self-intersection skip
    if (skipMisses && ray.hitDistance >= BVH_INTERSECT_INFINITY * 0.1) continue;
    const w = Math.max(0, dot(texelDir, ray.direction)); // L226
    if (w < 1e-3) continue; // L227
    const weight = f32(w * w); // L231 pow(w, 2.0)
    const d = f32(ray.hitDistance); // L232
    newDepth = f32(newDepth + f32(d * weight)); // L233
    newDepthSq = f32(newDepthSq + f32(f32(d * d) * weight)); // L234  (d² = 1e40 → +Inf in f32)
    totalWeight = f32(totalWeight + weight); // L235
  }
  if (totalWeight > 1e-5) {
    newDepth = f32(newDepth / totalWeight); // L238
    newDepthSq = f32(newDepthSq / totalWeight); // L239
  } else {
    newDepth = f32(DDGI_VISIBILITY_OPEN_SKY_MOMENT);
    newDepthSq = f32(DDGI_VISIBILITY_OPEN_SKY_MOMENT);
  }
  // Steady-state hysteresis (header note) + rgba16float storage (L247).
  return { mean: f16(newDepth), meanSq: f16(newDepthSq) };
}

// ── transcription: receiver Chebyshev visibility ─────────────────────────────
// ddgiSampleWgsl.ts:139-147 (the texture read is replaced by the blended texel).
function chebyshevVisibility(mean: number, meanSq: number, probeDist: number): number {
  const variance = Math.abs(meanSq - mean * mean); // L141
  const cheb =
    probeDist <= mean
      ? 1.0 // L144-145
      : variance / (variance + Math.max(0, probeDist - mean) * Math.max(0, probeDist - mean)); // L143
  return Math.max(cheb, 0); // L147 (w = w * max(chebyshev, 0))
}

// ── synthetic probe scenario ──────────────────────────────────────────────────
// Probe at origin. Wall: 3×3 plane at z = 2 (normal −Z). 32 rays in a cone
// around +Z (half-angle ≈ 25°); ray k hits the wall at distance 2/dir.z.
// The MISS variant replaces one outer cone ray with a sky miss (the wall is
// finite; a slightly wider ray slips past the edge → BVH_INTERSECT_INFINITY).
function coneRays(includeMiss: boolean): ProbeRay[] {
  const rays: ProbeRay[] = [];
  const n = 32;
  for (let k = 0; k < n; k++) {
    const phi = (2 * Math.PI * k) / n;
    const tilt = 0.15 + 0.3 * ((k % 4) / 4); // spread within the cone
    const dir = norm([tilt * Math.cos(phi), tilt * Math.sin(phi), 1]);
    rays.push({ direction: dir, hitDistance: 2 / dir[2] });
  }
  if (includeMiss) {
    // One ray just past the wall edge → sky miss (probeUpdateRays.wgsl.ts:644).
    rays[0] = { direction: norm([0.3, 0, 1]), hitDistance: BVH_INTERSECT_INFINITY };
  }
  return rays;
}

function openSkyRays(): ProbeRay[] {
  const rays: ProbeRay[] = [];
  const n = 32;
  for (let k = 0; k < n; k++) {
    const phi = (2 * Math.PI * k) / n;
    const dir = norm([0.35 * Math.cos(phi), 0.35 * Math.sin(phi), 1]);
    rays.push({ direction: dir, hitDistance: BVH_INTERSECT_INFINITY });
  }
  return rays;
}

const texelDir: V3 = [0, 0, 1]; // surface→probe query direction = +Z texel
const probeDist = 3; // query point at z = 3, BEHIND the wall at z = 2

describe('HYB-DDGI-01 oracle — sky-miss rays poison visibility moments', () => {
  it('CONTROL: with all rays hitting the wall, the behind-wall point is occluded', () => {
    const { mean, meanSq } = blendVisibilityTexel(texelDir, coneRays(false));
    // mean ≈ 2.0…2.2 (wall distance over the cone), tight variance.
    expect(mean).toBeGreaterThan(1.9);
    expect(mean).toBeLessThan(2.4);
    const cheb = chebyshevVisibility(mean, meanSq, probeDist);
    // Chebyshev correctly crushes visibility behind the wall.
    expect(cheb, `control chebyshev = ${cheb}`).toBeLessThan(0.05);
  });

  it('historical characterization: ONE sky miss among 32 rays → chebyshev = 1.0 (full leak)', () => {
    const { mean, meanSq } = blendVisibilityTexel(texelDir, coneRays(true), { skipMisses: false });
    // d² = 1e40 overflows f32 → meanSq is +Inf BEFORE storage quantization:
    expect(meanSq, 'depthSq moment overflows f32 (1e20² = 1e40 > 3.4e38)').toBe(Infinity);
    // The depth mean is ~1e20·w/ΣW ≈ 1e19 in f32 → +Inf after rgba16float
    // quantization (f16 max 65504):
    expect(mean, 'depth mean overflows the rgba16float atlas').toBe(Infinity);
    const cheb = chebyshevVisibility(mean, meanSq, probeDist);
    // probeDist <= mean(=Inf) → the L144 early branch returns 1.0: the probe
    // can NEVER occlude anything in this texel direction. Light leaks through
    // the wall for every receiver behind it.
    expect(cheb, `poisoned chebyshev = ${cheb} (ground-truth visibility = 0)`).toBe(1.0);
  });

  it('poisoning persists even WITHOUT f16/f32 overflow (the law is arithmetic, not precision)', () => {
    // Same scenario evaluated in f64 with no storage quantization: the miss
    // ray still drags the mean to ~1e18-1e19 >> probeDist, so the L144 branch
    // still returns 1.0. A "use rg32float" fix would NOT close this defect —
    // the miss distance must be excluded or given finite semantics.
    const rays = coneRays(true);
    let newDepth = 0;
    let newDepthSq = 0;
    let totalWeight = 0;
    for (const ray of rays) {
      if (ray.hitDistance < 0.05) continue;
      const w = Math.max(0, dot(texelDir, ray.direction));
      if (w < 1e-3) continue;
      const weight = w * w;
      newDepth += ray.hitDistance * weight;
      newDepthSq += ray.hitDistance * ray.hitDistance * weight;
      totalWeight += weight;
    }
    const mean = newDepth / totalWeight;
    const meanSq = newDepthSq / totalWeight;
    expect(mean).toBeGreaterThan(1e17);
    const cheb = chebyshevVisibility(mean, meanSq, probeDist);
    expect(cheb).toBe(1.0);
  });

  it('REGRESSION HYB-DDGI-01: one sky miss does not disable occlusion', () => {
    const { mean, meanSq } = blendVisibilityTexel(texelDir, coneRays(true));
    const cheb = chebyshevVisibility(mean, meanSq, probeDist);
    expect(cheb).toBeLessThan(0.05);
  });

  it('REGRESSION HYB-DDGI-01: an all-sky visibility texel stays open instead of becoming a zero-depth occluder', () => {
    const { mean, meanSq } = blendVisibilityTexel(texelDir, openSkyRays());
    expect(mean).toBe(DDGI_VISIBILITY_OPEN_SKY_MOMENT);
    expect(meanSq).toBe(DDGI_VISIBILITY_OPEN_SKY_MOMENT);
    const cheb = chebyshevVisibility(mean, meanSq, probeDist);
    expect(cheb).toBe(1.0);
  });
});
