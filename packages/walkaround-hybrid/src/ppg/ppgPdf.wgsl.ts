/**
 * PPG pdf-eval + guided-sampling helpers for the gi-ris RIS source pdf.
 *
 * Reference: Müller et al. 2017 "Practical Path Guiding for Efficient
 * Light-Transport Simulation", §3.2 (dTree sampling), §3.4 (MIS mixture).
 *
 * === Why this module exists ===
 * gi-ris compiles against the shared hybrid pipeline layout
 * (frame/scene/ubo/hybridLayers), so the guide sampler must read the learned
 * PPG trees from one packed query arena in the `hybridLayers` group (group 3,
 * binding 6 — there because the adapter caps `maxBindGroups = 4`) and provides:
 *
 *   1. `ppgEvalPdf(pos, wi)`        — the SOLID-ANGLE guide pdf p_guide(ωi) for
 *                                     an ARBITRARY world direction. Mirrors the
 *                                     CPU `dTreePdf` (dTree.ts) EXACTLY so the
 *                                     defensive MIS weight stays unbiased.
 *   2. `ppgSampleGuidedDir(pos,&rng)` — draw a world direction from the learned
 *                                     dTree (flux-proportional descent + leaf
 *                                     jitter), mirroring the CPU dTree sampler.
 *
 * Both descend the SAME serialised layout the producer (`serialise.ts`) and the
 * training kernel (`ppgUpdate.wgsl`) use — if the layout constants change there,
 * they must change here in lock-step.
 *
 * === Directional parametrisation (cylindrical equal-area) ===
 * The dTree stores UV in [0,1]². A world direction maps to that UV via the
 * cylindrical EQUAL-AREA map `ppgDirToUv` (inline below; u = (1−z)/2 uniform in
 * z, v = azimuth/2π) — the parametrisation Müller 2017 §3.2 actually uses, so
 * the dTree's `solidAngle = 4π·uvArea` is exact and the guide pdf equals the
 * uniform-in-UV sampling density (unbiased). The 2026-06-09 fix replaced the
 * earlier non-equal-area Cigolle octahedral map (which dropped a varying
 * Jacobian → biased MIS source pdf). Train (`ppgUpdate.wgsl`), pdf, and the
 * sampler (`ppgUvToDir`) all share this SAME map in lock-step. No octahedralCore
 * dependency: the shared octEncode/octDecode are no longer referenced here.
 *
 * === Bindings (group 3 = hybridLayers — see bindGroupLayouts.ts) ===
 *   @group(3) @binding(6) ppgQueryArena_gi : array<u32>
 *     (header + serialised sTree + concatenated dTrees + cell offsets)
 *
 * These are gated: gi-ris only calls into this module when `ubo.ppgEnabled == 1`,
 * so the 16-byte placeholders bound when PPG is off are never dereferenced.
 */

import type { WgslModule } from '../wgslTypes.js';
import {
  PPG_QUERY_ARENA_MAGIC,
  PPG_QUERY_ARENA_SCHEMA,
  PPG_QUERY_ARENA_VERSION,
} from './ppgQueryArena.js';

export const PPG_PDF_WGSL = /* wgsl */ `
// ── PPG guided-sampling + pdf-eval (gi-ris) ─────────────────────────────────
// Müller 2017 §3.2/§3.4. Reads the learned sTree/dTree from group(3).

@group(3) @binding(6) var<storage, read> ppgQueryArena_gi : array<u32>;

const PPG_QUERY_MAGIC_GI: u32 = ${PPG_QUERY_ARENA_MAGIC}u;
const PPG_QUERY_VERSION_GI: u32 = ${PPG_QUERY_ARENA_VERSION}u;
const PPG_QUERY_SCHEMA_GI: u32 = ${PPG_QUERY_ARENA_SCHEMA}u;
const PPG_QUERY_HEADER_WORDS_GI: u32 = 16u;
const PPG_INVALID_OFFSET_GI: u32 = 0xffffffffu;
fn ppgArenaRangeValidGi(offset: u32, length: u32, arenaWords: u32) -> bool {
  return offset <= arenaWords && length <= arenaWords - offset;
}
fn ppgQueryArenaValidGi() -> bool {
  let arenaWords = arrayLength(&ppgQueryArena_gi);
  if (arenaWords < PPG_QUERY_HEADER_WORDS_GI) { return false; }
  if (
    ppgQueryArena_gi[0] != PPG_QUERY_MAGIC_GI ||
    ppgQueryArena_gi[1] != PPG_QUERY_VERSION_GI ||
    ppgQueryArena_gi[2] == 0u ||
    ppgQueryArena_gi[3] != PPG_QUERY_SCHEMA_GI ||
    ppgQueryArena_gi[15] != arenaWords
  ) { return false; }

  let sOffset = ppgQueryArena_gi[4];
  let sLength = ppgQueryArena_gi[5];
  let sCapacity = ppgQueryArena_gi[6];
  let dOffset = ppgQueryArena_gi[7];
  let dLength = ppgQueryArena_gi[8];
  let dCapacity = ppgQueryArena_gi[9];
  let oOffset = ppgQueryArena_gi[10];
  let oLength = ppgQueryArena_gi[11];
  let oCapacity = ppgQueryArena_gi[12];
  if (
    sLength < STREE_HEADER_F32 + STREE_NODE_STRIDE ||
    dLength < DTREE_HEADER_F32 + DTREE_NODE_STRIDE ||
    oLength == 0u ||
    sLength > sCapacity ||
    dLength > dCapacity ||
    oLength > oCapacity ||
    !ppgArenaRangeValidGi(sOffset, sCapacity, arenaWords) ||
    !ppgArenaRangeValidGi(dOffset, dCapacity, arenaWords) ||
    !ppgArenaRangeValidGi(oOffset, oCapacity, arenaWords) ||
    ppgQueryArena_gi[13] == 0u ||
    ppgQueryArena_gi[14] == 0u ||
    oLength > ppgQueryArena_gi[13]
  ) { return false; }
  return true;
}
fn ppgArenaLoadSTreeF32(word: u32) -> f32 {
  return bitcast<f32>(ppgQueryArena_gi[ppgQueryArena_gi[4] + word]);
}
fn ppgArenaLoadDTreeF32(word: u32) -> f32 {
  return bitcast<f32>(ppgQueryArena_gi[ppgQueryArena_gi[7] + word]);
}
fn ppgArenaLoadDTreeOffset(word: u32) -> u32 {
  return ppgQueryArena_gi[ppgQueryArena_gi[10] + word];
}

// Layout constants provided by ppgTreeLayout (DTREE_HEADER_F32, DTREE_NODE_STRIDE,
// STREE_HEADER_F32, STREE_NODE_STRIDE — shared with ppgUpdate).
const PPG_FOUR_PI: f32 = 12.566370614359172; // 4π — uniform-fallback pdf = 1/4π
const PPG_REPRESENTED_BUCKETS: u32 = 16777216u;
const PPG_INV_REPRESENTED_BUCKETS: f32 = 5.960464477539063e-8;

// ── World direction ↔ dTree [0,1]² UV (cylindrical EQUAL-AREA, Müller 2017 §3.2) ──
// FIX 2026-06-09: this used octEncode (Cigolle 2014, a NON-equal-area octahedral
// map) while dTree.ts stores solidAngle = 4π·uvArea ASSUMING equal-area. So the
// guide pdf (leafFlux/totalFlux)/solidAngle did NOT equal the uniform-in-UV
// sampling density (it dropped the octahedral Jacobian) → the gi-ris MIS source
// pdf was biased → guided GI GAINED energy, growing as the dTree refined into the
// distorted diagonal regions (g-p11: PPG-on 2.36× over-bright, Δ grows with frames).
// The cylindrical map (u = (1-z)/2 uniform in z — Archimedes' hat-box) IS equal-area,
// so solidAngle = 4π·uvArea is exact and uniform-in-UV = uniform-in-solid-angle.
// This is the parametrisation Müller's paper actually uses. Train (ppgUpdate.wgsl),
// pdf, and the sampler below MUST all use this SAME map — keep them in lock-step.
fn ppgDirToUv(dir: vec3<f32>) -> vec2<f32> {
  let u = (1.0 - clamp(dir.z, -1.0, 1.0)) * 0.5;            // z∈[+1,−1] → u∈[0,1]
  let v = atan2(dir.y, dir.x) * 0.15915494309189535 + 0.5; // azimuth/(2π) + 0.5 → [0,1]
  return vec2<f32>(u, clamp(v, 0.0, 1.0));
}
fn ppgUvToDir(uv: vec2<f32>) -> vec3<f32> {
  let z = 1.0 - 2.0 * uv.x;
  let r = sqrt(max(0.0, 1.0 - z * z));
  let phi = 6.283185307179586 * (uv.y - 0.5);              // inverse of ppgDirToUv azimuth
  return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

// ── sTree descent (mirror of serialise.gpuTraverseSTreeLeaf) ────────────────
// Returns the f32 base offset of the sNode whose AABB contains 'pos'.
// MUST-MATCH: this descent body is semantically identical to sTreeFindLeafBase
// in ppgUpdate.wgsl.ts — only the buffer name differs (ppgSTreeBuf_gi here vs
// ppgSTreeBuf there). If you edit the logic here, mirror the change there,
// and vice versa. The ppgDescentDrift vitest gate enforces this automatically.
fn ppgSTreeFindLeafBase(pos: vec3<f32>) -> u32 {
  let nodeCount = u32(ppgArenaLoadSTreeF32(0));
  var idx: u32 = 0u;
  // sTree depth ≤ log2(16384) = 14; 32 is a generous safety cap.
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    let base = STREE_HEADER_F32 + idx * STREE_NODE_STRIDE;
    let splitAxisF = ppgArenaLoadSTreeF32(base + 7u);
    if (splitAxisF < 0.0) { return base; } // leaf
    let splitVal = ppgArenaLoadSTreeF32(base + 3u);
    let leftChildF  = ppgArenaLoadSTreeF32(base + 8u);
    let rightChildF = ppgArenaLoadSTreeF32(base + 9u);
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

// Resolve one spatial leaf to a dTree block without ever trusting a partial
// query-arena publication. The producer validates the complete topology before
// publishing it; these checks keep a torn/corrupted arena on the same uniform
// fallback path in both the sampler and PDF evaluator.
fn ppgResolveDTreeOffsetGi(pos: vec3<f32>) -> u32 {
  let sLength = ppgQueryArena_gi[5];
  if ((sLength - STREE_HEADER_F32) % STREE_NODE_STRIDE != 0u) {
    return PPG_INVALID_OFFSET_GI;
  }
  let packedNodeCount = (sLength - STREE_HEADER_F32) / STREE_NODE_STRIDE;
  let nodeCountF = ppgArenaLoadSTreeF32(0u);
  let dTreeCountF = ppgArenaLoadSTreeF32(1u);
  if (
    !(nodeCountF >= 1.0) || nodeCountF != floor(nodeCountF) ||
    nodeCountF != f32(packedNodeCount) ||
    !(dTreeCountF >= 1.0) || dTreeCountF != floor(dTreeCountF) ||
    dTreeCountF != f32(ppgQueryArena_gi[11])
  ) { return PPG_INVALID_OFFSET_GI; }
  let nodeCount = u32(nodeCountF);

  var idx: u32 = 0u;
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    if (idx >= nodeCount) { return PPG_INVALID_OFFSET_GI; }
    let base = STREE_HEADER_F32 + idx * STREE_NODE_STRIDE;
    let splitAxisF = ppgArenaLoadSTreeF32(base + 7u);
    if (splitAxisF == -1.0) {
      let dTreeIndexF = ppgArenaLoadSTreeF32(base + 10u);
      if (
        !(dTreeIndexF >= 0.0) || dTreeIndexF != floor(dTreeIndexF) ||
        dTreeIndexF >= dTreeCountF
      ) { return PPG_INVALID_OFFSET_GI; }
      let dOff = ppgArenaLoadDTreeOffset(u32(dTreeIndexF));
      let dLength = ppgQueryArena_gi[8];
      if (
        dOff > dLength ||
        DTREE_HEADER_F32 + DTREE_NODE_STRIDE > dLength - dOff
      ) { return PPG_INVALID_OFFSET_GI; }
      return dOff;
    }
    if (!(splitAxisF == 0.0 || splitAxisF == 1.0 || splitAxisF == 2.0)) {
      return PPG_INVALID_OFFSET_GI;
    }
    let splitVal = ppgArenaLoadSTreeF32(base + 3u);
    let leftChildF = ppgArenaLoadSTreeF32(base + 8u);
    let rightChildF = ppgArenaLoadSTreeF32(base + 9u);
    if (
      !(splitVal >= -3.402823466e38 && splitVal <= 3.402823466e38) ||
      !(leftChildF >= 0.0) || leftChildF != floor(leftChildF) || leftChildF >= nodeCountF ||
      !(rightChildF >= 0.0) || rightChildF != floor(rightChildF) || rightChildF >= nodeCountF
    ) { return PPG_INVALID_OFFSET_GI; }
    let axis = u32(splitAxisF);
    var queryAxis: f32 = pos.z;
    if (axis == 0u) { queryAxis = pos.x; }
    else if (axis == 1u) { queryAxis = pos.y; }
    idx = select(u32(rightChildF), u32(leftChildF), queryAxis < splitVal);
  }
  return PPG_INVALID_OFFSET_GI;
}

// ── dTree descent to the leaf containing an arbitrary UV ────────────────────
// Mirror of dTree.findDTreeLeaf: at each interior node, descend the quadrant
// of octUV. Returns the leaf's f32 base offset within ppgDTreeBuf_gi.
// MUST-MATCH: this descent body is semantically identical to dTreeFindLeafBase
// in ppgUpdate.wgsl.ts — only the buffer name differs (ppgDTreeBuf_gi here vs
// ppgDTreeBuf there). If you edit the logic here, mirror the change there,
// and vice versa. The ppgDescentDrift vitest gate enforces this automatically.
fn ppgDTreeFindLeafBase(dTreeOffset: u32, octUV: vec2<f32>) -> u32 {
  var idx: u32 = 0u;
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    let base = dTreeOffset + DTREE_HEADER_F32 + idx * DTREE_NODE_STRIDE;
    let isLeafFlag = ppgArenaLoadDTreeF32(base + 7u);
    if (isLeafFlag > 0.5) { return base; }
    let u0 = ppgArenaLoadDTreeF32(base + 0u);
    let v0 = ppgArenaLoadDTreeF32(base + 1u);
    let u1 = ppgArenaLoadDTreeF32(base + 2u);
    let v1 = ppgArenaLoadDTreeF32(base + 3u);
    let uMid = (u0 + u1) * 0.5;
    let vMid = (v0 + v1) * 0.5;
    let firstChildF = ppgArenaLoadDTreeF32(base + 6u);
    if (firstChildF < 0.0) { return base; }
    let firstChild = u32(firstChildF);
    var off: u32 = 0u;
    if (octUV.x >= uMid) { off = off + 1u; }
    if (octUV.y >= vMid) { off = off + 2u; }
    idx = firstChild + off;
  }
  return dTreeOffset + DTREE_HEADER_F32;
}

// ── Guide pdf for an arbitrary world direction (Müller §3.2/§3.4) ───────────
// Query-view bucket ABI (built by ppgRepresentedProposal.ts):
//   interior lane 5 = exact represented subtree buckets
//   leaf lane 6     = exact represented leaf buckets
// Both lanes are otherwise unused by the query/update traversal at that node
// kind. Counts are exact f32 integers in [0, 2^24].
fn ppgDTreeNodeBucketCount(base: u32) -> u32 {
  let isLeaf = ppgArenaLoadDTreeF32(base + 7u) > 0.5;
  let encoded = select(
    ppgArenaLoadDTreeF32(base + 5u),
    ppgArenaLoadDTreeF32(base + 6u),
    isLeaf,
  );
  if (!(encoded >= 0.0) || encoded > f32(PPG_REPRESENTED_BUCKETS) || encoded != floor(encoded)) {
    return 0u;
  }
  return u32(encoded);
}

fn ppgDTreeNodeKindValid(base: u32) -> bool {
  let encoded = ppgArenaLoadDTreeF32(base + 7u);
  return encoded == 0.0 || encoded == 1.0;
}

fn ppgDTreeNodeBucketEncodingValid(base: u32) -> bool {
  if (!ppgDTreeNodeKindValid(base)) { return false; }
  let isLeaf = ppgArenaLoadDTreeF32(base + 7u) == 1.0;
  let encoded = select(
    ppgArenaLoadDTreeF32(base + 5u),
    ppgArenaLoadDTreeF32(base + 6u),
    isLeaf,
  );
  return encoded >= 0.0 &&
    encoded <= f32(PPG_REPRESENTED_BUCKETS) &&
    encoded == floor(encoded);
}

fn ppgDTreePatchValid(base: u32) -> bool {
  let u0 = ppgArenaLoadDTreeF32(base + 0u);
  let v0 = ppgArenaLoadDTreeF32(base + 1u);
  let u1 = ppgArenaLoadDTreeF32(base + 2u);
  let v1 = ppgArenaLoadDTreeF32(base + 3u);
  return u0 >= 0.0 && v0 >= 0.0 && u1 <= 1.0 && v1 <= 1.0 &&
    u1 > u0 && v1 > v0;
}

fn ppgDTreePatchSolidAngle(base: u32) -> f32 {
  let u0 = ppgArenaLoadDTreeF32(base + 0u);
  let v0 = ppgArenaLoadDTreeF32(base + 1u);
  let u1 = ppgArenaLoadDTreeF32(base + 2u);
  let v1 = ppgArenaLoadDTreeF32(base + 3u);
  return PPG_FOUR_PI * (u1 - u0) * (v1 - v0);
}

fn ppgDTreeNodeCountGi(dTreeOffset: u32) -> u32 {
  let dLength = ppgQueryArena_gi[8];
  if (
    dTreeOffset > dLength ||
    DTREE_HEADER_F32 + DTREE_NODE_STRIDE > dLength - dTreeOffset
  ) { return 0u; }
  let encoded = ppgArenaLoadDTreeF32(dTreeOffset);
  let maxNodesByLength = (dLength - dTreeOffset - DTREE_HEADER_F32) / DTREE_NODE_STRIDE;
  if (
    !(encoded >= 1.0) || encoded != floor(encoded) ||
    encoded > f32(ppgQueryArena_gi[14]) || encoded > f32(maxNodesByLength)
  ) { return 0u; }
  let nodeCount = u32(encoded);
  let blockWords = DTREE_HEADER_F32 + nodeCount * DTREE_NODE_STRIDE;
  if (blockWords > dLength - dTreeOffset) { return 0u; }
  return nodeCount;
}

// Validate the exact local quadtree partition and represented-mass invariant
// before either consumer descends. If a child pointer, patch, kind, or bucket
// count is partially corrupted, both consumers stop at the same current patch;
// its already-published subtree buckets then define a normalized local-uniform
// fallback rather than silently changing the proposal seen by MIS.
fn ppgDTreeChildrenValidGi(
  dTreeOffset: u32,
  nodeCount: u32,
  nodeIndex: u32,
  base: u32,
) -> bool {
  if (!ppgDTreePatchValid(base) || !ppgDTreeNodeBucketEncodingValid(base)) {
    return false;
  }
  let firstChildF = ppgArenaLoadDTreeF32(base + 6u);
  if (
    !(firstChildF >= 0.0) || firstChildF != floor(firstChildF) ||
    firstChildF <= f32(nodeIndex) || nodeCount < 4u ||
    firstChildF > f32(nodeCount - 4u)
  ) { return false; }
  let firstChild = u32(firstChildF);
  let u0 = ppgArenaLoadDTreeF32(base + 0u);
  let v0 = ppgArenaLoadDTreeF32(base + 1u);
  let u1 = ppgArenaLoadDTreeF32(base + 2u);
  let v1 = ppgArenaLoadDTreeF32(base + 3u);
  let uMid = (u0 + u1) * 0.5;
  let vMid = (v0 + v1) * 0.5;
  var childBucketSum: u32 = 0u;
  for (var ci: u32 = 0u; ci < 4u; ci = ci + 1u) {
    let childBase = dTreeOffset + DTREE_HEADER_F32 + (firstChild + ci) * DTREE_NODE_STRIDE;
    if (!ppgDTreePatchValid(childBase) || !ppgDTreeNodeBucketEncodingValid(childBase)) {
      return false;
    }
    let expectedU0 = select(u0, uMid, (ci & 1u) != 0u);
    let expectedU1 = select(uMid, u1, (ci & 1u) != 0u);
    let expectedV0 = select(v0, vMid, (ci & 2u) != 0u);
    let expectedV1 = select(vMid, v1, (ci & 2u) != 0u);
    if (
      ppgArenaLoadDTreeF32(childBase + 0u) != expectedU0 ||
      ppgArenaLoadDTreeF32(childBase + 1u) != expectedV0 ||
      ppgArenaLoadDTreeF32(childBase + 2u) != expectedU1 ||
      ppgArenaLoadDTreeF32(childBase + 3u) != expectedV1
    ) { return false; }
    childBucketSum = childBucketSum + ppgDTreeNodeBucketCount(childBase);
  }
  return childBucketSum == ppgDTreeNodeBucketCount(base);
}

// Query-only descent. Unlike ppgDTreeFindLeafBase (which intentionally mirrors
// the training kernel's raw-topology walk), this returns either a terminal leaf
// or the first locally-invalid interior patch used by the symmetric fallback.
fn ppgDTreeFindDistributionBase(dTreeOffset: u32, octUV: vec2<f32>) -> u32 {
  let nodeCount = ppgDTreeNodeCountGi(dTreeOffset);
  if (nodeCount == 0u) { return PPG_INVALID_OFFSET_GI; }
  var idx: u32 = 0u;
  var fallbackBase = dTreeOffset + DTREE_HEADER_F32;
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    if (idx >= nodeCount) { return PPG_INVALID_OFFSET_GI; }
    let base = dTreeOffset + DTREE_HEADER_F32 + idx * DTREE_NODE_STRIDE;
    fallbackBase = base;
    if (!ppgDTreePatchValid(base) || !ppgDTreeNodeBucketEncodingValid(base)) {
      return PPG_INVALID_OFFSET_GI;
    }
    if (ppgArenaLoadDTreeF32(base + 7u) == 1.0) { return base; }
    if (!ppgDTreeChildrenValidGi(dTreeOffset, nodeCount, idx, base)) {
      return base;
    }
    let u0 = ppgArenaLoadDTreeF32(base + 0u);
    let v0 = ppgArenaLoadDTreeF32(base + 1u);
    let u1 = ppgArenaLoadDTreeF32(base + 2u);
    let v1 = ppgArenaLoadDTreeF32(base + 3u);
    let uMid = (u0 + u1) * 0.5;
    let vMid = (v0 + v1) * 0.5;
    let firstChild = u32(ppgArenaLoadDTreeF32(base + 6u));
    var off: u32 = 0u;
    if (octUV.x >= uMid) { off = off + 1u; }
    if (octUV.y >= vMid) { off = off + 2u; }
    idx = firstChild + off;
  }
  return fallbackBase;
}

fn ppgDTreeRootValidGi(dTreeOffset: u32) -> bool {
  if (ppgDTreeNodeCountGi(dTreeOffset) == 0u) { return false; }
  let rootBase = dTreeOffset + DTREE_HEADER_F32;
  return ppgDTreeNodeBucketEncodingValid(rootBase) &&
    ppgDTreePatchValid(rootBase) &&
    ppgArenaLoadDTreeF32(rootBase + 0u) == 0.0 &&
    ppgArenaLoadDTreeF32(rootBase + 1u) == 0.0 &&
    ppgArenaLoadDTreeF32(rootBase + 2u) == 1.0 &&
    ppgArenaLoadDTreeF32(rootBase + 3u) == 1.0 &&
    ppgDTreeNodeBucketCount(rootBase) == PPG_REPRESENTED_BUCKETS;
}

// p_guide(ωi) = (leafBuckets / 2^24) / solidAngle_leaf, with a 1/(4π)
// uniform fallback only when the complete cell has no represented training
// mass. A zero-bucket leaf in a live guide has density zero: cosine sampling
// may reach it, but the guide proposal cannot.
fn ppgEvalPdf(pos: vec3<f32>, wi: vec3<f32>) -> f32 {
  if (!ppgQueryArenaValidGi()) { return 1.0 / PPG_FOUR_PI; }
  let dOff = ppgResolveDTreeOffsetGi(pos);
  // Sampling uses the same uniform-sphere fallback for both a genuinely cold
  // cell and a malformed root/block. Publish that actual fallback density here
  // too so defensive corruption handling cannot desynchronise the sampler and
  // its MIS PDF.
  if (dOff == PPG_INVALID_OFFSET_GI || !ppgDTreeRootValidGi(dOff)) {
    return 1.0 / PPG_FOUR_PI;
  }
  let octUV = ppgDirToUv(wi);
  let distributionBase = ppgDTreeFindDistributionBase(dOff, octUV);
  if (distributionBase == PPG_INVALID_OFFSET_GI) { return 1.0 / PPG_FOUR_PI; }
  let distributionBuckets = ppgDTreeNodeBucketCount(distributionBase);
  let solidAng = ppgDTreePatchSolidAngle(distributionBase);
  if (distributionBuckets == 0u) { return 0.0; }
  if (!(solidAng > 0.0)) { return 0.0; }
  return (f32(distributionBuckets) * PPG_INV_REPRESENTED_BUCKETS) / solidAng;
}

// ── Exact represented dTree leaf sampler ─────────────────────────────────────
fn ppgDTreeSampleLeafBase(dTreeOffset: u32, rng: ptr<function, u32>) -> u32 {
  let nodeCount = ppgDTreeNodeCountGi(dTreeOffset);
  if (nodeCount == 0u) { return PPG_INVALID_OFFSET_GI; }
  let rootBase = dTreeOffset + DTREE_HEADER_F32;
  let rootLeaf = ppgArenaLoadDTreeF32(rootBase + 7u) == 1.0;
  if (rootLeaf) { return rootBase; }

  // One 24-bit root integer chooses the terminal leaf. Descending by exact
  // subtree counts is the inverse CDF of the global represented distribution;
  // no per-level rounded-f32 proposal or extra random draw is involved.
  var remaining = pcgNext(rng) >> 8u;
  var idx: u32 = 0u;
  var fallbackBase = rootBase;
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    if (idx >= nodeCount) { return PPG_INVALID_OFFSET_GI; }
    let base = dTreeOffset + DTREE_HEADER_F32 + idx * DTREE_NODE_STRIDE;
    fallbackBase = base;
    if (ppgArenaLoadDTreeF32(base + 7u) == 1.0) { return base; }
    if (!ppgDTreeChildrenValidGi(dTreeOffset, nodeCount, idx, base)) {
      return base;
    }
    let firstChild = u32(ppgArenaLoadDTreeF32(base + 6u));
    var selected = false;
    for (var ci: u32 = 0u; ci < 4u; ci = ci + 1u) {
      let cBase = dTreeOffset + DTREE_HEADER_F32 + (firstChild + ci) * DTREE_NODE_STRIDE;
      let childBuckets = ppgDTreeNodeBucketCount(cBase);
      if (remaining < childBuckets) {
        idx = firstChild + ci;
        selected = true;
        break;
      }
      remaining = remaining - childBuckets;
    }
    if (!selected) { return base; }
  }
  return fallbackBase;
}

// ── Draw a guided world direction from the learned dTree ────────────────────
// Returns a unit world direction sampled from the exact represented leaf PMF,
// jittered within the chosen equal-area cylindrical patch. The caller evaluates
// the mixture pdf for the returned direction via ppgEvalPdf (and a cosine
// pdf). This routine does not return one because the RIS source PDF is the
// alpha mixture, not p_guide alone.
fn ppgSampleGuidedDir(pos: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
  if (!ppgQueryArenaValidGi()) {
    let z = rand_f32(rng) * 2.0 - 1.0;
    let phi = rand_f32(rng) * 6.283185307179586;
    let rxy = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(rxy * cos(phi), rxy * sin(phi), z);
  }
  let dOff = ppgResolveDTreeOffsetGi(pos);
  // Degenerate cell (no training flux): fall back to a uniform-sphere sample
  // so the returned direction is still valid (its mixture pdf is dominated by
  // the cosine term anyway, and p_guide reduces to the 1/4π uniform fallback).
  if (dOff == PPG_INVALID_OFFSET_GI || !ppgDTreeRootValidGi(dOff)) {
    let z = rand_f32(rng) * 2.0 - 1.0;
    let phi = rand_f32(rng) * 6.283185307179586;
    let rxy = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(rxy * cos(phi), rxy * sin(phi), z);
  }
  let leafBase = ppgDTreeSampleLeafBase(dOff, rng);
  if (leafBase == PPG_INVALID_OFFSET_GI) {
    let z = rand_f32(rng) * 2.0 - 1.0;
    let phi = rand_f32(rng) * 6.283185307179586;
    let rxy = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(rxy * cos(phi), rxy * sin(phi), z);
  }
  let u0 = ppgArenaLoadDTreeF32(leafBase + 0u);
  let v0 = ppgArenaLoadDTreeF32(leafBase + 1u);
  let u1 = ppgArenaLoadDTreeF32(leafBase + 2u);
  let v1 = ppgArenaLoadDTreeF32(leafBase + 3u);
  let r0 = rand_f32(rng);
  let r1 = rand_f32(rng);
  let uv = vec2<f32>(u0 + r0 * (u1 - u0), v0 + r1 * (v1 - v0));
  // [0,1]² UV → world dir via the cylindrical equal-area map (inverse of ppgDirToUv).
  return ppgUvToDir(uv);
}
`;

/** W1-R6 — declarative include-graph entry.
 *  Requires `ppgTreeLayout` for the shared DTREE_/STREE_ layout constants.
 *  No octahedralCore require: the 2026-06-09 equal-area fix replaced
 *  octEncode/octDecode with the inline cylindrical map (`ppgDirToUv`/
 *  `ppgUvToDir`), so the shared octahedral helpers are no longer referenced.
 *  The `rand_f32` RNG + the group(3) PPG bindings are provided by the gi-ris
 *  compilation unit (sharedPrimitives supplies `rand_f32`; risGi declares it
 *  requires this module). The bindings live here because only gi-ris consumes them. */
export const PPG_PDF_MODULE: WgslModule = {
  name: 'ppgPdf',
  source: PPG_PDF_WGSL,
  requires: ['ppgTreeLayout'],
};
