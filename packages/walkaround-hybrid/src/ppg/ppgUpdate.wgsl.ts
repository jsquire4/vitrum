/**
 * PPG update kernel — GPU-side flux accumulation from training samples.
 *
 * Reference: Müller et al. 2017 "Practical Path Guiding for Efficient
 * Light-Transport Simulation", §3.3 (training signal) and §5 (GPU update).
 *
 * Called once per frame after the shade pass. Reads per-path training samples
 * `(samplePos, sampleDir, Li)` and atomically increments the appropriate
 * dTree leaf flux counter.
 *
 * === DEVIATION 3 FIX (training signal) ===
 * The input binding is named `ppgLiSamples` (L_i — INCOMING radiance at the
 * sample point). The deleted implementation consumed Lo (outgoing radiance
 * from the shade pass, post-BRDF and post-clamp). This kernel READS FROM
 * THE L_i BINDING exclusively. See the `@group(0) @binding(2)` declaration
 * below.
 *
 * === DEVIATION 4 FIX (coordinate frame) ===
 * Both samplePos and sampleDir are in WORLD space. The octahedral encoding
 * `dirToOct(dir)` is applied to the WORLD-FRAME direction; no per-surface
 * ONB transform is performed.
 *
 * === W9 — REAL FLAT-BUFFER LEAF LOCATION ===
 * The pre-W9 kernel approximated the dTree leaf by `uIdx = uv.x * leafCount`,
 * discarding the adaptive structure and the `vIdx` axis entirely (the trailing
 * `_ = vIdx` line acknowledged this as a stub). This module replaces that
 * approximation with a **real flat-buffer descent that mirrors
 * `findDTreeLeaf` exactly**:
 *   1. Walk the sTree to find the spatial cell for the sample.
 *   2. Walk that cell's dTree to find the leaf containing the (u, v) of the
 *      incoming direction.
 *   3. Atomically increment the leaf's flux counter (fixed-point f32 → u32).
 *
 * The CPU reads back the atomic buffer at the end of each rebuild cycle and
 * calls `refineDTree` / `splitOverflowLeaves` (the topology changes are
 * CPU-side per Müller §5).
 *
 * Bindings:
 *   group(0) binding(0) — ppgSamplesPos:    array<vec4<f32>>  (xyz=pos, w=unused)
 *   group(0) binding(1) — ppgSamplesDir:    array<vec4<f32>>  (xyz=dir WORLD, w=unused)
 *   group(0) binding(2) — ppgLiSamples:     array<vec4<f32>>  (xyz=Li, w=pathThroughput)
 *   group(0) binding(3) — ppgFluxAtomics:   array<atomic<u32>> (per-dTree-leaf accumulator;
 *                                                              index = (dTreeOffset + leafBase)/8)
 *   group(0) binding(4) — ppgSTreeBuf:      array<f32>   (serialised sTree)
 *   group(0) binding(5) — ppgDTreeBuf:      array<f32>   (serialised dTree blocks)
 *   group(0) binding(6) — ppgDTreeOffsets:  array<u32>   (dTreeIndex → f32 offset)
 *   group(1) binding(0) — ppgUBO: struct { sampleCount: u32, _pad: u32, ... }
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const PPG_UPDATE_WGSL = /* wgsl */`
// ── PPG update kernel ─────────────────────────────────────────────────────────
// Müller et al. 2017 §3.3 — training on INCOMING radiance (L_i).
// DEVIATION 3 FIX: ppgLiSamples is the L_i binding, not Lo.
// DEVIATION 4 FIX: all directions are in WORLD space.
// W9: real flat-buffer leaf location (no more uniform-grid stub).

struct PPGUpdateUBO {
  sampleCount : u32,  // total path samples this frame
  fluxBudget  : u32,  // number of u32 slots in ppgFluxAtomics (bounds check)
  padding0    : u32,
  padding1    : u32,
}

@group(0) @binding(0) var<storage, read>           ppgSamplesPos   : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>           ppgSamplesDir   : array<vec4<f32>>;
// ── DEVIATION 3 FIX ──────────────────────────────────────────────────────────
// ppgLiSamples holds the INCOMING radiance L_i at each sample point.
// The shade pass records L_i BEFORE the BSDF multiply (path throughput
// estimate at the next-bounce side). Do NOT substitute Lo (outgoing radiance)
// here — a white and a black wall with identical illumination must produce the
// same guide PDF.
@group(0) @binding(2) var<storage, read>           ppgLiSamples    : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write>     ppgFluxAtomics  : array<atomic<u32>>;
// W9 — serialised tree bindings.
@group(0) @binding(4) var<storage, read>           ppgSTreeBuf     : array<f32>;
@group(0) @binding(5) var<storage, read>           ppgDTreeBuf     : array<f32>;
@group(0) @binding(6) var<storage, read>           ppgDTreeOffsets : array<u32>;
@group(1) @binding(0) var<uniform>                 ppgUBO          : PPGUpdateUBO;

// Layout constants — MUST stay in sync with serialise.ts.
const DTREE_HEADER_F32 : u32 = 4u;
const DTREE_NODE_STRIDE: u32 = 8u;
const STREE_HEADER_F32 : u32 = 4u;
const STREE_NODE_STRIDE: u32 = 16u;

// ── Octahedral encoding (Cigolle et al. 2014) ─────────────────────────────────
// Maps a unit direction in WORLD space to [0,1]² octahedral UV.
// DEVIATION 4 FIX: direction is in WORLD frame; no ONB rotation is applied.
fn dirToOct(n: vec3<f32>) -> vec2<f32> {
  let p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
  if (n.z < 0.0) {
    let s = select(vec2<f32>(-1.0), vec2<f32>(1.0), p >= vec2<f32>(0.0));
    return (1.0 - abs(p.yx)) * s * 0.5 + 0.5;
  }
  return p * 0.5 + 0.5;
}

// ── Fixed-point encode (1/65536 ULP resolution) ──────────────────────────────
const FLUX_SCALE: f32 = 65536.0;

fn encodeFlux(f: f32) -> u32 {
  return u32(clamp(f * FLUX_SCALE, 0.0, f32(0xFFFFFFFFu)));
}

// ── sTree descent — same shape as ppgGuide.wgsl.sTreeFindLeafBase ────────────
fn sTreeFindLeafBase(pos: vec3<f32>) -> u32 {
  let nodeCount = u32(ppgSTreeBuf[0]);
  var idx: u32 = 0u;
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    let base = STREE_HEADER_F32 + idx * STREE_NODE_STRIDE;
    let splitAxisF = ppgSTreeBuf[base + 7u];
    if (splitAxisF < 0.0) { return base; } // leaf
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
    if (idx >= nodeCount) { return base; }
  }
  return STREE_HEADER_F32;
}

// ── dTree descent — locate the leaf containing (u, v) in a given cell ────────
// Mirror of dTree.findDTreeLeaf: at each interior node compare uv against the
// quadrant midpoints, descend to firstChild + (goDown ? 2 : 0) + (goRight ? 1 : 0).
fn dTreeFindLeafBase(dTreeOffset: u32, uv: vec2<f32>) -> u32 {
  var idx: u32 = 0u;
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    let base = dTreeOffset + DTREE_HEADER_F32 + idx * DTREE_NODE_STRIDE;
    let isLeafFlag = ppgDTreeBuf[base + 7u];
    if (isLeafFlag > 0.5) { return base; }
    let u0 = ppgDTreeBuf[base + 0u];
    let v0 = ppgDTreeBuf[base + 1u];
    let u1 = ppgDTreeBuf[base + 2u];
    let v1 = ppgDTreeBuf[base + 3u];
    let uMid = (u0 + u1) * 0.5;
    let vMid = (v0 + v1) * 0.5;
    let firstChildF = ppgDTreeBuf[base + 6u];
    if (firstChildF < 0.0) { return base; }
    let firstChild = u32(firstChildF);
    var off: u32 = 0u;
    if (uv.x >= uMid) { off = off + 1u; }
    if (uv.y >= vMid) { off = off + 2u; }
    idx = firstChild + off;
  }
  return dTreeOffset + DTREE_HEADER_F32;
}

// ── Dispatch ──────────────────────────────────────────────────────────────────
// One workgroup per 64 samples. Each invocation:
//   1. Reads its (pos, dir, Li) triple.
//   2. Computes the incoming-direction octahedral UV (world frame).
//   3. Walks the sTree to its cell, then the cell's dTree to a leaf.
//   4. Atomically increments the leaf's flux accumulator.
@compute @workgroup_size(64)
fn ppgUpdateMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= ppgUBO.sampleCount) { return; }

  let pos = ppgSamplesPos[idx].xyz;
  let dirRaw = ppgSamplesDir[idx].xyz;
  let dirLen2 = dot(dirRaw, dirRaw);
  if (dirLen2 < 1e-12) { return; }
  let dir = dirRaw * inverseSqrt(dirLen2);

  // L_i: incoming radiance — luminance of the path contribution.
  // DEVIATION 3 FIX: read from ppgLiSamples (L_i binding), not from any
  // clamped outgoing-radiance buffer (Lo binding).
  let Li  = ppgLiSamples[idx].xyz;
  let lum = dot(Li, vec3<f32>(0.2126, 0.7152, 0.0722));
  if (lum <= 0.0) { return; }

  // Octahedral UV of the incoming direction in WORLD space (deviation 4 fix).
  let uv = dirToOct(dir);

  // Walk the sTree to the spatial cell for this sample.
  let sBase = sTreeFindLeafBase(pos);
  let dTreeIndex = u32(ppgSTreeBuf[sBase + 10u]);
  let dOff = ppgDTreeOffsets[dTreeIndex];

  // Walk the dTree to the leaf for this direction.
  let leafBase = dTreeFindLeafBase(dOff, uv);

  // Bin index for the atomic accumulator: one u32 per dTree node slot. The
  // CPU side decodes this via the dTree's node ordering (refineDTree reads
  // dTree.nodes[i].flux). We map leafBase (an f32 offset) to a slot index by
  // dividing by DTREE_NODE_STRIDE (the leaf's node index within the buffer's
  // f32 layout: (leafBase − HEADER) / STRIDE within the cell's dTree, but
  // we want a GLOBAL slot here for the readback).
  // The atomic buffer's slot k corresponds to the f32 offset
  //   dTreeOffset + HEADER + k * STRIDE.
  // For W9 Phase 1 we accumulate per-node-slot indexed by
  //   slot = (leafBase) / STRIDE, capped at fluxBudget.
  let slot = leafBase / DTREE_NODE_STRIDE;
  if (slot >= ppgUBO.fluxBudget) { return; }

  // Encode lum as fixed-point and accumulate atomically.
  atomicAdd(&ppgFluxAtomics[slot], encodeFlux(lum));
}
`;

/** W1-R6 — declarative include-graph entry. Self-contained. */
export const PPG_UPDATE_MODULE: WgslModule = {
  name: 'ppgUpdate',
  source: PPG_UPDATE_WGSL,
  requires: [],
};
