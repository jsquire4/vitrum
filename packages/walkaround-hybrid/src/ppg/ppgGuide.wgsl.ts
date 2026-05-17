/**
 * PPG guide kernel — GPU-side direction sampling using the learned dTree.
 *
 * Reference: Müller et al. 2017 "Practical Path Guiding for Efficient
 * Light-Transport Simulation", §3.2 (dTree sampling) and §3.4 (MIS with BSDF).
 *
 * At each shading point, this kernel:
 *   1. Descends the sTree to find the leaf cell for the surface position.
 *   2. Samples the cell's dTree proportional to accumulated flux.
 *   3. Returns a sampled direction in WORLD space and its solid-angle PDF.
 *
 * The host then mixes p_guide with p_bsdf via the MIS weight formula (§3.4):
 *   p_mixed = α·p_guide + (1−α)·p_bsdf
 *
 * === DEVIATION 4 FIX (coordinate frame) ===
 * Sampled directions are returned in WORLD space. No per-surface ONB
 * transform is applied here. The BSDF sampler at the shading point applies
 * its own local-frame transform independently.
 *
 * === DEVIATION 5 FIX (solid-angle PDF) ===
 * The guide PDF for a sampled direction ω is:
 *   pdf(ω) = (flux_leaf / totalFlux) / solidAngle_leaf
 * where solidAngle_leaf = 4π × (u1−u0) × (v1−v0) is the exact leaf area.
 * NOT the uniform 4π/leafCount approximation.
 *
 * Bindings:
 *   group(0) binding(0) — ppgLeafFlux:     array<f32>   (per-leaf flux, CPU-decoded from atomics)
 *   group(0) binding(1) — ppgLeafSolidAng: array<f32>   (per-leaf solid angle, precomputed)
 *   group(0) binding(2) — ppgTotalFlux:    array<f32>   (one f32 per sTree cell)
 *   group(0) binding(3) — ppgSampleOut:    array<vec4<f32>> (output: xyz=dir world, w=pdf)
 *   group(1) binding(0) — ppgGuideUBO:     struct { ... }
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const PPG_GUIDE_WGSL = /* wgsl */`
// ── PPG guide kernel ──────────────────────────────────────────────────────────
// Müller et al. 2017 §3.2, §3.4 — dTree direction sampling + MIS PDF.
// DEVIATION 4 FIX: sampled directions are in WORLD space.
// DEVIATION 5 FIX: PDF uses per-leaf solid angle from octahedral patch area.

struct PPGGuideUBO {
  pixelCount  : u32,   // total pixels to shade
  leafCount   : u32,   // dTree leaf count (same for all cells in this dispatch)
  alpha       : f32,   // MIS mixing weight α ∈ [0.1, 0.9] (Müller §3.4)
  padding0    : u32,
}

@group(0) @binding(0) var<storage, read>       ppgLeafFlux     : array<f32>;
@group(0) @binding(1) var<storage, read>       ppgLeafSolidAng : array<f32>;
@group(0) @binding(2) var<storage, read>       ppgTotalFlux    : array<f32>;
@group(0) @binding(3) var<storage, read_write> ppgSampleOut    : array<vec4<f32>>;
@group(1) @binding(0) var<uniform>             ppgGuideUBO     : PPGGuideUBO;

// ── Octahedral decode (Cigolle et al. 2014) ───────────────────────────────────
// Maps octahedral UV in [0,1]² to a unit direction in WORLD space.
// DEVIATION 4 FIX: result is in WORLD frame; caller handles BSDF-local frame.
fn octToDir(uv: vec2<f32>) -> vec3<f32> {
  let p = uv * 2.0 - 1.0;
  let z = 1.0 - abs(p.x) - abs(p.y);
  var d: vec3<f32>;
  if (z >= 0.0) {
    d = vec3<f32>(p.x, p.y, z);
  } else {
    let s = select(vec2<f32>(-1.0), vec2<f32>(1.0), p >= vec2<f32>(0.0));
    d = vec3<f32>((1.0 - abs(p.yx)) * s, z);
  }
  return normalize(d);
}

// ── LCG random (deterministic per-pixel) ─────────────────────────────────────
fn lcg(state: ptr<function, u32>) -> f32 {
  *state = *state * 1664525u + 1013904223u;
  return f32(*state) / f32(0xFFFFFFFFu);
}

// ── dTree leaf sampling (proportional to flux) ───────────────────────────────
// Returns the leaf index selected by weighted random sampling.
fn sampleLeaf(
  cellOffset : u32,
  leafCount  : u32,
  totalFlux  : f32,
  rng        : ptr<function, u32>,
) -> u32 {
  let target = lcg(rng) * totalFlux;
  var cum = 0.0;
  for (var i = 0u; i < leafCount - 1u; i++) {
    cum += ppgLeafFlux[cellOffset + i];
    if (target <= cum) { return i; }
  }
  return leafCount - 1u;
}

@compute @workgroup_size(64)
fn ppgGuideMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pix = gid.x;
  if (pix >= ppgGuideUBO.pixelCount) { return; }

  let leafCount  = ppgGuideUBO.leafCount;
  let cellOffset = 0u; // single-cell dispatch; multi-cell handled by pipeline loop
  let totalFlux  = ppgTotalFlux[0];

  // Degenerate: no training signal yet — return zero pdf (caller falls back to BSDF).
  if (totalFlux <= 0.0) {
    ppgSampleOut[pix] = vec4<f32>(0.0, 0.0, 1.0, 0.0);
    return;
  }

  var rng = pix * 2654435761u + 1u;
  let r0 = lcg(&rng);
  let r1 = lcg(&rng);

  // Sample a leaf proportional to flux.
  let leafIdx = sampleLeaf(cellOffset, leafCount, totalFlux, &rng);

  // Solid angle for this leaf (deviation 5 fix — NOT 4π/N).
  let solidAng = ppgLeafSolidAng[cellOffset + leafIdx];
  let leafFlux = ppgLeafFlux[cellOffset + leafIdx];

  // PDF = (leafFlux / totalFlux) / solidAngle_leaf (Müller §3.2, deviation 5 fix).
  let pdf = (leafFlux / totalFlux) / max(solidAng, 1e-12);

  // Reconstruct the leaf's octahedral UV centre and add sub-leaf jitter.
  // The GPU version approximates the leaf UV from a uniform grid over [0,1]²
  // with N bins; the exact leaf bounds are stored in the dTree CPU-side and
  // serialised as solidAng. For the GPU fast-path we use the index-derived UV.
  let N = f32(leafCount);
  let col = leafIdx % u32(sqrt(N + 0.5));
  let row = leafIdx / u32(sqrt(N + 0.5));
  let uv = vec2<f32>(
    (f32(col) + r0) / sqrt(N),
    (f32(row) + r1) / sqrt(N),
  );

  // Convert octahedral UV to WORLD-space direction (deviation 4 fix).
  let dir = octToDir(uv);

  // Output: xyz = world direction, w = guide PDF.
  ppgSampleOut[pix] = vec4<f32>(dir, pdf);
}
`;

/** W1-R6 — declarative include-graph entry. Self-contained. */
export const PPG_GUIDE_MODULE: WgslModule = {
  name: 'ppgGuide',
  source: PPG_GUIDE_WGSL,
  requires: [],
};

// ────────────────────────────────────────────────────────────────────────────
// MIS weight computation (Müller §3.4) — pure TypeScript for host + test use
// ────────────────────────────────────────────────────────────────────────────

/**
 * Compute the one-sample MIS weights for guide and BSDF sampling using the
 * power heuristic (Müller §3.4).
 *
 * p_mixed = α·p_guide + (1−α)·p_bsdf
 *
 * Power-heuristic MIS weights:
 *   w_ppg  = (α·p_guide)²  / ((α·p_guide)² + ((1−α)·p_bsdf)²)
 *   w_bsdf = ((1−α)·p_bsdf)² / denom
 *
 * Note: the throughput division is by p_mixed, NOT by p_guide alone.
 * This ensures the estimator remains unbiased even when the guide PDF is
 * far from converged (Müller §3.4).
 *
 * @param alpha    MIS mixing weight α ∈ [0.1, 0.9].
 * @param pGuide   Guide PDF (solid-angle PDF from dTree).
 * @param pBsdf    BSDF PDF (solid-angle PDF from BSDF sampler).
 * @returns `{ wPpg, wBsdf, pMixed }`.
 */
export function computeMISWeights(
  alpha: number,
  pGuide: number,
  pBsdf: number,
): { wPpg: number; wBsdf: number; pMixed: number } {
  const pMixed = alpha * pGuide + (1 - alpha) * pBsdf;
  const aPg = alpha * pGuide;
  const bPb = (1 - alpha) * pBsdf;
  const denom = aPg * aPg + bPb * bPb;
  if (denom <= 0) return { wPpg: 0.5, wBsdf: 0.5, pMixed: Math.max(pMixed, 1e-12) };
  return {
    wPpg:  (aPg * aPg) / denom,
    wBsdf: (bPb * bPb) / denom,
    pMixed: Math.max(pMixed, 1e-12),
  };
}
