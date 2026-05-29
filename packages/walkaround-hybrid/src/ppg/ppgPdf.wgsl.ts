/**
 * PPG pdf-eval + guided-sampling helpers for the gi-ris RIS source pdf.
 *
 * Reference: Müller et al. 2017 "Practical Path Guiding for Efficient
 * Light-Transport Simulation", §3.2 (dTree sampling), §3.4 (MIS mixture).
 *
 * === Why this module exists ===
 * `ppgGuide.wgsl.ts` already samples the dTree, but its descent helpers read
 * the PPG trees through `@group(0)` bindings owned by the standalone PPG guide
 * pipeline. The gi-ris pass compiles against the SHARED hybrid pipeline layout
 * (frame/scene/ubo/hybridLayers), so it cannot reach those `@group(0)` buffers.
 * This module re-declares the three PPG tree buffers on the `hybridLayers`
 * group (group 3, bindings 6/7/8 — packed there because the adapter caps
 * `maxBindGroups = 4`) and provides:
 *
 *   1. `ppgEvalPdf(pos, wi)`        — the SOLID-ANGLE guide pdf p_guide(ωi) for
 *                                     an ARBITRARY world direction. Mirrors the
 *                                     CPU `dTreePdf` (dTree.ts) EXACTLY so the
 *                                     defensive MIS weight stays unbiased.
 *   2. `ppgSampleGuidedDir(pos,&rng)` — draw a world direction from the learned
 *                                     dTree (flux-proportional descent + leaf
 *                                     jitter), mirroring `dTreeSampleLeafBase`
 *                                     in ppgGuide.wgsl.
 *
 * Both descend the SAME serialised layout the producer (`serialise.ts`) and the
 * training kernels (`ppgUpdate.wgsl`, `ppgGuide.wgsl`) use — if the layout
 * constants change there, they must change here in lock-step.
 *
 * === Octahedral convention ===
 * The dTree stores oct UV in [0,1]². A world direction maps to that UV via
 *   uv = octEncode(dir) * 0.5 + 0.5          (octEncode → [-1,1]²)
 * which is byte-identical to the producer's `dirToOct` (ppgUpdate.wgsl) and the
 * inverse of `octDecode(uv * 2 - 1)` (== ppgGuide's `octToDir`). Reusing the
 * canonical `octEncode`/`octDecode` from @vitrum/shared-samplers
 * (requires: ['octahedralCore']) keeps the encode/decode in one place.
 *
 * === Bindings (group 3 = hybridLayers — see bindGroupLayouts.ts) ===
 *   @group(3) @binding(6) ppgSTreeBuf_gi     : array<f32>  (serialised sTree)
 *   @group(3) @binding(7) ppgDTreeBuf_gi     : array<f32>  (concatenated dTrees)
 *   @group(3) @binding(8) ppgDTreeOffsets_gi : array<u32>  (cell → dTree offset)
 *
 * These are gated: gi-ris only calls into this module when `ubo.ppgEnabled == 1`,
 * so the 16-byte placeholders bound when PPG is off are never dereferenced.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const PPG_PDF_WGSL = /* wgsl */ `
// ── PPG guided-sampling + pdf-eval (gi-ris) ─────────────────────────────────
// Müller 2017 §3.2/§3.4. Reads the learned sTree/dTree from group(3).

@group(3) @binding(6) var<storage, read> ppgSTreeBuf_gi     : array<f32>;
@group(3) @binding(7) var<storage, read> ppgDTreeBuf_gi     : array<f32>;
@group(3) @binding(8) var<storage, read> ppgDTreeOffsets_gi : array<u32>;

// Layout constants — MUST stay in sync with serialise.ts / ppgGuide.wgsl.
const PPG_DTREE_HEADER_F32 : u32 = 4u;
const PPG_DTREE_NODE_STRIDE: u32 = 8u;
const PPG_STREE_HEADER_F32 : u32 = 4u;
const PPG_STREE_NODE_STRIDE: u32 = 16u;
const PPG_FOUR_PI          : f32 = 12.566370614359172; // 4π — uniform-fallback pdf = 1/4π

// ── World direction → dTree [0,1]² octahedral UV ────────────────────────────
// octEncode (octahedralCore) returns [-1,1]²; remap to [0,1]² to match the
// producer's dirToOct convention. Inverse of octDecode(uv * 2 - 1).
fn ppgDirToOctUv(dir: vec3<f32>) -> vec2<f32> {
  return octEncode(dir) * 0.5 + 0.5;
}

// ── sTree descent (mirror of serialise.gpuTraverseSTreeLeaf) ────────────────
// Returns the f32 base offset of the sNode whose AABB contains 'pos'.
fn ppgSTreeFindLeafBase(pos: vec3<f32>) -> u32 {
  let nodeCount = u32(ppgSTreeBuf_gi[0]);
  var idx: u32 = 0u;
  // sTree depth ≤ log2(16384) = 14; 32 is a generous safety cap.
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    let base = PPG_STREE_HEADER_F32 + idx * PPG_STREE_NODE_STRIDE;
    let splitAxisF = ppgSTreeBuf_gi[base + 7u];
    if (splitAxisF < 0.0) { return base; } // leaf
    let splitVal = ppgSTreeBuf_gi[base + 3u];
    let leftChildF  = ppgSTreeBuf_gi[base + 8u];
    let rightChildF = ppgSTreeBuf_gi[base + 9u];
    let axis = u32(splitAxisF);
    var queryAxis: f32 = 0.0;
    if (axis == 0u)      { queryAxis = pos.x; }
    else if (axis == 1u) { queryAxis = pos.y; }
    else                 { queryAxis = pos.z; }
    if (queryAxis < splitVal) { idx = u32(leftChildF); }
    else                      { idx = u32(rightChildF); }
    if (idx >= nodeCount) { return base; } // defensive fallback
  }
  return PPG_STREE_HEADER_F32;
}

// ── dTree descent to the leaf containing an arbitrary UV ────────────────────
// Mirror of dTree.findDTreeLeaf: at each interior node, descend the quadrant
// of octUV. Returns the leaf's f32 base offset within ppgDTreeBuf_gi.
fn ppgDTreeFindLeafBase(dTreeOffset: u32, octUV: vec2<f32>) -> u32 {
  var idx: u32 = 0u;
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    let base = dTreeOffset + PPG_DTREE_HEADER_F32 + idx * PPG_DTREE_NODE_STRIDE;
    let isLeafFlag = ppgDTreeBuf_gi[base + 7u];
    if (isLeafFlag > 0.5) { return base; }
    let firstChildF = ppgDTreeBuf_gi[base + 6u];
    if (firstChildF < 0.0) { return base; } // defensive: malformed interior
    let firstChild = u32(firstChildF);
    let u0 = ppgDTreeBuf_gi[base + 0u];
    let v0 = ppgDTreeBuf_gi[base + 1u];
    let u1 = ppgDTreeBuf_gi[base + 2u];
    let v1 = ppgDTreeBuf_gi[base + 3u];
    let uMid = (u0 + u1) * 0.5;
    let vMid = (v0 + v1) * 0.5;
    // firstChild ordering: 0=NW, 1=NE, 2=SW, 3=SE (consecutive children).
    let goRight = octUV.x >= uMid;
    let goDown  = octUV.y >= vMid;
    var off: u32 = 0u;
    if (goDown)  { off = off + 2u; }
    if (goRight) { off = off + 1u; }
    idx = firstChild + off;
  }
  return dTreeOffset + PPG_DTREE_HEADER_F32; // unreachable on a well-formed tree
}

// ── Guide pdf for an arbitrary world direction (Müller §3.2/§3.4) ───────────
// p_guide(ωi) = (leafFlux / totalFlux) / solidAngle_leaf, with a 1/(4π)
// uniform fallback when the cell has no training flux yet. Mirrors the CPU
// dTreePdf (dTree.ts) EXACTLY — this is the defensive evaluation that keeps
// the gi-ris mixture estimator unbiased.
fn ppgEvalPdf(pos: vec3<f32>, wi: vec3<f32>) -> f32 {
  let sBase = ppgSTreeFindLeafBase(pos);
  let dTreeIndex = u32(ppgSTreeBuf_gi[sBase + 10u]);
  let dOff = ppgDTreeOffsets_gi[dTreeIndex];
  let totalFlux = ppgDTreeBuf_gi[dOff + 2u];
  if (totalFlux <= 0.0) { return 1.0 / PPG_FOUR_PI; }
  let octUV = ppgDirToOctUv(wi);
  let leafBase = ppgDTreeFindLeafBase(dOff, octUV);
  let leafFlux = ppgDTreeBuf_gi[leafBase + 4u];
  let solidAng = ppgDTreeBuf_gi[leafBase + 5u];
  return (leafFlux / totalFlux) / max(solidAng, 1e-12);
}

// ── Flux-proportional dTree leaf sampler (mirror of ppgGuide.dTreeSampleLeafBase) ─
fn ppgDTreeSampleLeafBase(dTreeOffset: u32, rng: ptr<function, u32>) -> u32 {
  var idx: u32 = 0u;
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    let base = dTreeOffset + PPG_DTREE_HEADER_F32 + idx * PPG_DTREE_NODE_STRIDE;
    let isLeafFlag = ppgDTreeBuf_gi[base + 7u];
    if (isLeafFlag > 0.5) { return base; }
    let firstChildF = ppgDTreeBuf_gi[base + 6u];
    if (firstChildF < 0.0) { return base; }
    let firstChild = u32(firstChildF);

    var sum: f32 = 0.0;
    var cFlux: array<f32, 4>;
    for (var ci: u32 = 0u; ci < 4u; ci = ci + 1u) {
      let cBase = dTreeOffset + PPG_DTREE_HEADER_F32 + (firstChild + ci) * PPG_DTREE_NODE_STRIDE;
      cFlux[ci] = ppgDTreeBuf_gi[cBase + 4u];
      sum = sum + cFlux[ci];
    }
    let r = rand_f32(rng);
    var pick: u32 = 3u;
    if (sum <= 0.0) {
      // Uniform fallback when the cell's children carry no flux (cold start).
      pick = min(u32(r * 4.0), 3u);
    } else {
      let target = r * sum;
      var cum: f32 = 0.0;
      for (var ci: u32 = 0u; ci < 4u; ci = ci + 1u) {
        cum = cum + cFlux[ci];
        if (target <= cum && pick == 3u) { pick = ci; }
      }
    }
    idx = firstChild + pick;
  }
  return dTreeOffset + PPG_DTREE_HEADER_F32;
}

// ── Draw a guided world direction from the learned dTree ────────────────────
// Returns a unit world direction sampled ∝ leaf flux, jittered within the
// chosen leaf's octahedral patch. The caller evaluates the mixture pdf for the
// returned direction via ppgEvalPdf (and a cosine pdf) — this routine does NOT
// return a pdf, because the RIS source pdf is the α-mixture, not p_guide alone.
fn ppgSampleGuidedDir(pos: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
  let sBase = ppgSTreeFindLeafBase(pos);
  let dTreeIndex = u32(ppgSTreeBuf_gi[sBase + 10u]);
  let dOff = ppgDTreeOffsets_gi[dTreeIndex];
  let totalFlux = ppgDTreeBuf_gi[dOff + 2u];
  // Degenerate cell (no training flux): fall back to a uniform-sphere sample
  // so the returned direction is still valid (its mixture pdf is dominated by
  // the cosine term anyway, and p_guide reduces to the 1/4π uniform fallback).
  if (totalFlux <= 0.0) {
    let z = rand_f32(rng) * 2.0 - 1.0;
    let phi = rand_f32(rng) * 6.283185307179586;
    let rxy = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(rxy * cos(phi), rxy * sin(phi), z);
  }
  let leafBase = ppgDTreeSampleLeafBase(dOff, rng);
  let u0 = ppgDTreeBuf_gi[leafBase + 0u];
  let v0 = ppgDTreeBuf_gi[leafBase + 1u];
  let u1 = ppgDTreeBuf_gi[leafBase + 2u];
  let v1 = ppgDTreeBuf_gi[leafBase + 3u];
  let r0 = rand_f32(rng);
  let r1 = rand_f32(rng);
  let uv = vec2<f32>(u0 + r0 * (u1 - u0), v0 + r1 * (v1 - v0));
  // [0,1]² UV → [-1,1]² → world dir (inverse of ppgDirToOctUv; == octToDir).
  return octDecode(uv * 2.0 - 1.0);
}
`;

/** W1-R6 — declarative include-graph entry.
 *  Requires `octahedralCore` for octEncode/octDecode. The `rand_f32` RNG +
 *  the group(3) PPG bindings are provided by the gi-ris compilation unit
 *  (sharedPrimitives supplies `rand_f32`; risGi declares it requires this
 *  module). The bindings live here because only gi-ris consumes them. */
export const PPG_PDF_MODULE: WgslModule = {
  name: 'ppgPdf',
  source: PPG_PDF_WGSL,
  requires: ['octahedralCore'],
};
