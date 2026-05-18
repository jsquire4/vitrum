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
 * === W9 — REAL FLAT-BUFFER TRAVERSAL ===
 * The pre-W9 kernel approximated the dTree as a uniform N-bin grid
 * (`col = leafIdx % u32(sqrt(N))`), discarding the adaptive refinement the
 * CPU side worked hard to learn. This module replaces that approximation
 * with a **flat-buffer descent that mirrors `findDTreeLeaf` exactly**:
 *   - sTree leaf lookup via kd-tree traversal (binary descent).
 *   - dTree leaf sampling via flux-proportional quadtree descent.
 *   - Read the leaf's exact (u0,u1)·(v0,v1) rectangle and solidAngle.
 *
 * The serialiser in `serialise.ts` produces a Float32Array layout that the
 * kernel below indexes via constant offsets (DTREE_HEADER_F32 = 4,
 * DTREE_NODE_STRIDE = 8, STREE_HEADER_F32 = 4, STREE_NODE_STRIDE = 16).
 * If those constants change there, they must change here in lock-step.
 *
 * Bindings:
 *   group(0) binding(0) — ppgSTreeBuf:    array<f32>   (serialised sTree nodes)
 *   group(0) binding(1) — ppgDTreeBuf:    array<f32>   (serialised dTree nodes, concatenated)
 *   group(0) binding(2) — ppgDTreeOffsets: array<u32>  (dTreeIndex → f32 offset into ppgDTreeBuf)
 *   group(0) binding(3) — ppgSampleOut:   array<vec4<f32>> (output: xyz=dir world, w=pdf)
 *   group(1) binding(0) — ppgGuideUBO:    struct { ... }
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const PPG_GUIDE_WGSL = /* wgsl */`
// ── PPG guide kernel ──────────────────────────────────────────────────────────
// Müller et al. 2017 §3.2, §3.4 — dTree direction sampling + MIS PDF.
// DEVIATION 4 FIX: sampled directions are in WORLD space.
// DEVIATION 5 FIX: PDF uses per-leaf solid angle from octahedral patch area.
// W9: real flat-buffer traversal (no more uniform-grid stub).

struct PPGGuideUBO {
  pixelCount    : u32,   // total pixels to shade
  imgWidth      : u32,   // dimension to recover (x, y) from flat pixel id
  alpha         : f32,   // MIS mixing weight α ∈ [0.1, 0.9] (Müller §3.4)
  frameSeed     : u32,   // RNG salt — varies per frame for stratified sampling
  sceneMinX     : f32,   // cell-position oracle (world-space) — TODO: replace
  sceneMinY     : f32,   //   with a per-pixel position buffer once primary-hit
  sceneMinZ     : f32,   //   readback lands (W9 Phase 1 uses scene-centre as
  sceneMaxX     : f32,   //   a placeholder; the kernel still produces valid
  sceneMaxY     : f32,   //   samples — they just all share one sTree cell
  sceneMaxZ     : f32,   //   until the position buffer is wired).
  _pad0         : u32,
  _pad1         : u32,
}

// W9 — serialised tree bindings (layout documented in ppg/serialise.ts).
//   ppgSTreeBuf: [nodeCount, dTreeCount, _, _] + N × 16 f32 sNode records
//   ppgDTreeBuf: per-cell dTree blocks concatenated; each is [N, leafN, total, _]
//                followed by N × 8 f32 dNode records.
//   ppgDTreeOffsets: ppgDTreeOffsets[k] = f32 base offset of cell k's dTree.
@group(0) @binding(0) var<storage, read>       ppgSTreeBuf     : array<f32>;
@group(0) @binding(1) var<storage, read>       ppgDTreeBuf     : array<f32>;
@group(0) @binding(2) var<storage, read>       ppgDTreeOffsets : array<u32>;
@group(0) @binding(3) var<storage, read_write> ppgSampleOut    : array<vec4<f32>>;
@group(1) @binding(0) var<uniform>             ppgGuideUBO     : PPGGuideUBO;

// Layout constants — MUST stay in sync with serialise.ts.
const DTREE_HEADER_F32 : u32 = 4u;
const DTREE_NODE_STRIDE: u32 = 8u;
const STREE_HEADER_F32 : u32 = 4u;
const STREE_NODE_STRIDE: u32 = 16u;

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

// ── sTree descent ────────────────────────────────────────────────────────────
// Mirror of serialise.gpuTraverseSTreeLeaf — returns the f32 base offset of
// the sNode whose AABB contains 'pos'. If the tree has zero cells the kernel
// already early-returned in main(); we never call this on an empty tree.
fn sTreeFindLeafBase(pos: vec3<f32>) -> u32 {
  let nodeCount = u32(ppgSTreeBuf[0]);
  var idx: u32 = 0u;
  // Cap iterations defensively: the sTree depth is bounded by log2(cells)
  // ≤ log2(16384) = 14, so 32 is a generous safety margin.
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    let base = STREE_HEADER_F32 + idx * STREE_NODE_STRIDE;
    let splitAxisF = ppgSTreeBuf[base + 7u];
    // Leaf when splitAxis < 0 (encoded -1 in the producer).
    if (splitAxisF < 0.0) { return base; }
    let splitVal = ppgSTreeBuf[base + 3u];
    let leftChildF  = ppgSTreeBuf[base + 8u];
    let rightChildF = ppgSTreeBuf[base + 9u];
    let axis = u32(splitAxisF);
    var queryAxis: f32 = 0.0;
    if (axis == 0u)      { queryAxis = pos.x; }
    else if (axis == 1u) { queryAxis = pos.y; }
    else                 { queryAxis = pos.z; }
    if (queryAxis < splitVal) { idx = u32(leftChildF); }
    else                      { idx = u32(rightChildF); }
    if (idx >= nodeCount) { return base; } // defensive fallback
  }
  return STREE_HEADER_F32;
}

// ── dTree flux-proportional sampling ─────────────────────────────────────────
// Walk the per-cell dTree quadtree; at each interior node pick a child with
// probability proportional to its accumulated flux, descending until a leaf.
// Returns the leaf's f32 base offset within ppgDTreeBuf.
fn dTreeSampleLeafBase(dTreeOffset: u32, rng: ptr<function, u32>) -> u32 {
  let totalFlux = ppgDTreeBuf[dTreeOffset + 2u];
  var idx: u32 = 0u;
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    let base = dTreeOffset + DTREE_HEADER_F32 + idx * DTREE_NODE_STRIDE;
    let isLeafFlag = ppgDTreeBuf[base + 7u];
    if (isLeafFlag > 0.5) { return base; }
    let firstChildF = ppgDTreeBuf[base + 6u];
    if (firstChildF < 0.0) { return base; } // defensive fallback
    let firstChild = u32(firstChildF);

    // Sum children's flux; pick proportional. If all children have zero
    // flux (cold start), pick a child uniformly at random — this preserves
    // the uniform-fallback contract documented in dTree.dTreeSample.
    var sum: f32 = 0.0;
    var cFlux: array<f32, 4>;
    for (var ci: u32 = 0u; ci < 4u; ci = ci + 1u) {
      let cBase = dTreeOffset + DTREE_HEADER_F32 + (firstChild + ci) * DTREE_NODE_STRIDE;
      cFlux[ci] = ppgDTreeBuf[cBase + 4u];
      sum = sum + cFlux[ci];
    }
    let r = lcg(rng);
    var pick: u32 = 3u; // default to last child if target reaches sum
    if (sum <= 0.0) {
      // Uniform fallback.
      pick = min(u32(r * 4.0), 3u);
    } else {
      let target = r * sum;
      var cum: f32 = 0.0;
      // WGSL doesn't permit break-with-value in arbitrary positions; we use
      // the "first match" pattern by guarding subsequent updates with
      // (pick == 3u) so only the FIRST child crossing the threshold sticks.
      for (var ci: u32 = 0u; ci < 4u; ci = ci + 1u) {
        cum = cum + cFlux[ci];
        if (target <= cum && pick == 3u) {
          pick = ci;
        }
      }
    }
    idx = firstChild + pick;
  }
  return dTreeOffset + DTREE_HEADER_F32; // unreachable on a well-formed tree
  // Note: 'totalFlux' is used by the PDF formula in the caller (main below).
  // It is read here only for completeness; the compiler will not emit a
  // load if WGSL DCEs it. We rely on the caller to read it directly.
}

@compute @workgroup_size(64)
fn ppgGuideMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pix = gid.x;
  if (pix >= ppgGuideUBO.pixelCount) { return; }

  // sTree cell: W9 Phase 1 uses scene-centre as a placeholder for the per-pixel
  // surface position (Phase 2 will read this from a primary-hit position buffer).
  // With a single-cell sTree this is a no-op; with a split sTree the kernel
  // still picks SOME cell deterministically and learns its directional dist.
  let sceneCentre = vec3<f32>(
    0.5 * (ppgGuideUBO.sceneMinX + ppgGuideUBO.sceneMaxX),
    0.5 * (ppgGuideUBO.sceneMinY + ppgGuideUBO.sceneMaxY),
    0.5 * (ppgGuideUBO.sceneMinZ + ppgGuideUBO.sceneMaxZ),
  );
  let sBase = sTreeFindLeafBase(sceneCentre);
  let dTreeIndex = u32(ppgSTreeBuf[sBase + 10u]);
  // dTreeOffsets is u32; safe to read with default 0 if the table is degenerate.
  let dOff = ppgDTreeOffsets[dTreeIndex];
  let totalFlux = ppgDTreeBuf[dOff + 2u];

  // Degenerate: no training signal yet — write zero PDF (host falls back to BSDF).
  if (totalFlux <= 0.0) {
    ppgSampleOut[pix] = vec4<f32>(0.0, 0.0, 1.0, 0.0);
    return;
  }

  // Per-pixel RNG salted by frameSeed so successive frames don't lock-step.
  var rng = pix * 2654435761u ^ ppgGuideUBO.frameSeed;
  // Sample the leaf proportional to flux.
  let leafBase = dTreeSampleLeafBase(dOff, &rng);
  let u0 = ppgDTreeBuf[leafBase + 0u];
  let v0 = ppgDTreeBuf[leafBase + 1u];
  let u1 = ppgDTreeBuf[leafBase + 2u];
  let v1 = ppgDTreeBuf[leafBase + 3u];
  let leafFlux = ppgDTreeBuf[leafBase + 4u];
  let solidAng = ppgDTreeBuf[leafBase + 5u];

  // Jitter inside the leaf's exact (u0..u1) × (v0..v1) rectangle.
  let r0 = lcg(&rng);
  let r1 = lcg(&rng);
  let uv = vec2<f32>(u0 + r0 * (u1 - u0), v0 + r1 * (v1 - v0));

  // PDF = (leafFlux / totalFlux) / solidAng (Müller §3.2 deviation-5 fix).
  let pdf = (leafFlux / totalFlux) / max(solidAng, 1e-12);

  // Convert octahedral UV to WORLD-space direction (deviation 4 fix).
  let dir = octToDir(uv);

  // Output: xyz = world direction, w = guide PDF.
  ppgSampleOut[pix] = vec4<f32>(dir, max(pdf, 1e-12));
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
