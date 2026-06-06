/**
 * PPG update kernel — GPU-side flux accumulation from training samples.
 *
 * Reference: Müller et al. 2017 "Practical Path Guiding for Efficient
 * Light-Transport Simulation", §3.3 (training signal) and §5 (GPU update).
 *
 * Called once per frame after the GI reservoir passes. Reads accepted
 * ReSTIR-GI reservoirs and atomically increments the appropriate dTree leaf
 * flux counter.
 *
 * === DEVIATION 3 FIX (training signal) ===
 * The training signal is the reservoir's stored incoming-radiance proxy `Lo`
 * at the reconnection vertex. The sample direction is reconstructed from the
 * accepted edge `xv -> xs`; no synthetic per-pixel training buffers are used.
 *
 * === DEVIATION 4 FIX (coordinate frame) ===
 * Both reservoir endpoints are in WORLD space. The octahedral encoding
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
 *   group(0) binding(0) — ppgReservoirGiCurrent: array<u32> (ReSTIR-GI reservoirs)
 *   group(0) binding(1) — ppgFluxAtomics:   array<atomic<u32>> (per-dTree-leaf accumulator;
 *                                                              index = (dTreeOffset + leafBase)/8)
 *   group(0) binding(2) — ppgSTreeBuf:      array<f32>   (serialised sTree)
 *   group(0) binding(3) — ppgDTreeBuf:      array<f32>   (serialised dTree blocks)
 *   group(0) binding(4) — ppgDTreeOffsets:  array<u32>   (dTreeIndex → f32 offset)
 *   group(1) binding(0) — ppgUBO: struct { sampleCount: u32, _pad: u32, ... }
 */

import { RESERVOIR_GI_STRIDE } from './ppgConstants.js';
import type { WgslModule } from '../pipeline/wgslComposer.js';

export const PPG_UPDATE_WGSL = /* wgsl */`
// ── PPG update kernel ─────────────────────────────────────────────────────────
// Muller et al. 2017 section 3.3 - training from accepted GI reservoir samples.
// The training tuple is (xv, normalize(xs - xv), Lo) from reservoirGiCurrent.
// DEVIATION 4 FIX: all directions are in WORLD space.
// W9: real flat-buffer leaf location (no more uniform-grid stub).

struct PPGUpdateUBO {
  sampleCount : u32,  // total half-res GI reservoir entries this frame
  fluxBudget  : u32,  // number of u32 slots in ppgFluxAtomics (bounds check)
  padding0    : u32,
  padding1    : u32,
}

@group(0) @binding(0) var<storage, read>           ppgReservoirGiCurrent : array<u32>;
@group(0) @binding(1) var<storage, read_write>     ppgFluxAtomics        : array<atomic<u32>>;
// W9 — serialised tree bindings.
@group(0) @binding(2) var<storage, read>           ppgSTreeBuf           : array<f32>;
@group(0) @binding(3) var<storage, read>           ppgDTreeBuf           : array<f32>;
@group(0) @binding(4) var<storage, read>           ppgDTreeOffsets       : array<u32>;
@group(1) @binding(0) var<uniform>                 ppgUBO                : PPGUpdateUBO;

// Layout constants provided by ppgTreeLayout (DTREE_HEADER_F32, DTREE_NODE_STRIDE,
// STREE_HEADER_F32, STREE_NODE_STRIDE). ppgUpdate-specific constant below.
// Must match allocatePPGResources default (resourceManager.ts).
const MAX_DTREE_NODES_PER_CELL : u32 = 341u;
const RESERVOIR_GI_STRIDE_LOCAL : u32 = ${RESERVOIR_GI_STRIDE}u;

// ── Fixed-point encode (1/65536 ULP resolution) ──────────────────────────────
const FLUX_SCALE: f32 = 65536.0;

fn encodeFlux(f: f32) -> u32 {
  return u32(clamp(f * FLUX_SCALE, 0.0, f32(0xFFFFFFFFu)));
}

// ── sTree descent — same serialised layout as ppgPdf.gi-ris sampling ─────────
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
// One workgroup per 64 half-res GI reservoir entries. Each invocation:
//   1. Reads an accepted reservoir's (xv, normalize(xs - xv), Lo) triple.
//   2. Computes the incoming-direction octahedral UV (world frame).
//   3. Walks the sTree to its cell, then the cell's dTree to a leaf.
//   4. Atomically increments the leaf's flux accumulator.
@compute @workgroup_size(64)
fn ppgUpdateMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= ppgUBO.sampleCount) { return; }

  let b = idx * RESERVOIR_GI_STRIDE_LOCAL;
  let reservoirM = ppgReservoirGiCurrent[b + 15u];
  if (reservoirM == 0u) { return; }

  let pos = vec3f(
    bitcast<f32>(ppgReservoirGiCurrent[b + 0u]),
    bitcast<f32>(ppgReservoirGiCurrent[b + 1u]),
    bitcast<f32>(ppgReservoirGiCurrent[b + 2u])
  );
  let samplePoint = vec3f(
    bitcast<f32>(ppgReservoirGiCurrent[b + 8u]),
    bitcast<f32>(ppgReservoirGiCurrent[b + 9u]),
    bitcast<f32>(ppgReservoirGiCurrent[b + 10u])
  );
  let dirRaw = samplePoint - pos;
  let dirLen2 = dot(dirRaw, dirRaw);
  if (dirLen2 < 1e-12) { return; }
  let dir = dirRaw * inverseSqrt(dirLen2);

  // Training radiance proxy from the chosen reconnection vertex.
  let Li = vec3f(
    bitcast<f32>(ppgReservoirGiCurrent[b + 16u]),
    bitcast<f32>(ppgReservoirGiCurrent[b + 17u]),
    bitcast<f32>(ppgReservoirGiCurrent[b + 18u])
  );
  let lum = luminance(Li);
  if (lum <= 0.0) { return; }

  // Octahedral UV of the incoming direction in WORLD space (deviation 4 fix).
  // Equivalent to the removed dirToOct: octEncode returns [-1,1]², remapped
  // to [0,1]² by *0.5+0.5, matching the producer's dirToOct convention.
  let uv = octEncode(dir) * 0.5 + 0.5;

  // Walk the sTree to the spatial cell for this sample.
  let sBase = sTreeFindLeafBase(pos);
  let dTreeIndex = u32(ppgSTreeBuf[sBase + 10u]);
  let dOff = ppgDTreeOffsets[dTreeIndex];

  // Walk the dTree to the leaf for this direction.
  let leafBase = dTreeFindLeafBase(dOff, uv);

  // Global atomic slot: fixed grid of maxDTreeNodesPerCell slots per spatial cell.
  let nodeIdx = (leafBase - dOff - DTREE_HEADER_F32) / DTREE_NODE_STRIDE;
  let slot = dTreeIndex * MAX_DTREE_NODES_PER_CELL + nodeIdx;
  if (slot >= ppgUBO.fluxBudget) { return; }

  // Encode lum as fixed-point and accumulate atomically.
  atomicAdd(&ppgFluxAtomics[slot], encodeFlux(lum));
}
`;

/** W1-R6 — declarative include-graph entry. Requires the canonical
 *  Rec.709 luminance helper, ppgTreeLayout for the shared layout constants,
 *  and octahedralCore for octEncode (replaces removed inline dirToOct —
 *  byte-equivalent: octEncode(n)*0.5+0.5). */
export const PPG_UPDATE_MODULE: WgslModule = {
  name: 'ppgUpdate',
  source: PPG_UPDATE_WGSL,
  requires: ['luminance', 'ppgTreeLayout', 'octahedralCore'],
};
