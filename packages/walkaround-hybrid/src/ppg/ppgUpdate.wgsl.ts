/**
 * PPG update kernel — GPU-side flux accumulation from training samples.
 *
 * Reference: Müller et al. 2017 "Practical Path Guiding for Efficient
 * Light-Transport Simulation", §3.3 (training signal) and §5 (GPU update).
 *
 * Called once per frame immediately after the initial GI RIS pass, before temporal or spatial reuse. Reads accepted
 * ReSTIR-GI reservoirs and atomically increments the appropriate dTree leaf
 * flux counter.
 *
 * === TRAINING ESTIMATOR ===
 * For M independent RIS candidates Xi~q with weights wi=pHat(Xi)/q(Xi),
 * represented WRS chooses Y with its persisted represented selection
 * probability. For any directional bin B, `1[Y in B] * exp2(H)`, where
 * `H=log2(w_selected / selectionProbability / M)`, has expectation
 * `integral_B pHat(w) dw`. The update pass therefore runs before reuse and
 * deposits `exp2(H)` at the selected direction. This is an unbiased histogram
 * estimator for arbitrary bins of the exact receiver contribution targeted by
 * the RIS producer; raw Lo and capped logW are not valid substitutes.
 *
 * === DEVIATION 4 FIX (coordinate frame) ===
 * Both reservoir endpoints are in WORLD space. The cylindrical equal-area UV
 * map (`ppgDirToUv`, inline below) is applied to the WORLD-FRAME direction; no
 * per-surface ONB transform is performed.
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
 *   3. Atomically increment the leaf's flux counter (atomic f32 CAS through u32 bits).
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
 *   group(0) binding(5) — ppgCellSampleCounts: array<atomic<u32>> (A2 — per-cell
 *                          training-record counter, indexed by dTreeIndex; the
 *                          coordinator reads it back to drive splitOverflowLeaves)
 *   group(1) binding(0) — ppgUBO: struct { sampleCount, fluxBudget, sampleCountBudget, _pad }
 */

import {
  PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL,
  PPG_DEFAULT_SPATIAL_CELLS,
} from './ppgConstants.js';
import { RESERVOIR_GI_STRIDE_U32 } from '../gi/giLayout.js';
import type { WgslModule } from '../wgslTypes.js';
import {
  PPG_QUERY_ARENA_MAGIC,
  PPG_QUERY_ARENA_SCHEMA,
  PPG_QUERY_ARENA_VERSION,
} from './ppgQueryArena.js';

/**
 * Default spatial-cell count used by allocatePPGResources and the dTree stride
 * calculation. Placed here (beside the stride builder) so both the compiler
 * and allocator can import it from a single source.
 *
 * H29 — the GPU stride constant `MAX_DTREE_NODES_PER_CELL` is now
 * template-interpolated by `buildPpgUpdateWgsl(maxDTreeNodesPerCell)` instead
 * of being hardcoded in the WGSL source. pipelineCompiler.ts passes the live
 * allocation value so both the shader and the host agree on the per-cell stride
 * at every configuration.
 */
export { PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL, PPG_DEFAULT_SPATIAL_CELLS };

/**
 * Build the PPG update kernel WGSL source for a given per-cell dTree node cap.
 *
 * H29 — converts the formerly static `PPG_UPDATE_WGSL` string into a builder
 * so the `MAX_DTREE_NODES_PER_CELL` constant is single-sourced: it is
 * template-interpolated from the live `maxDTreeNodesPerCell` value that
 * `allocatePPGResources` used to size the GPU flux buffer.  This eliminates
 * the silent divergence where the host could allocate with a non-default cap
 * while the GPU kernel still used the hardcoded 341.
 *
 * Byte-identical at the default cap (341): the produced WGSL contains `341u`
 * exactly as before.
 *
 * @param maxDTreeNodesPerCell  Per-cell node cap baked into the GPU flux
 *   buffer (= `fluxAtomicsBuf.size / 4 / maxSpatialCells`).  Default 341
 *   (depth-4 full quadtree: 1+4+16+64+256).
 */
export function buildPpgUpdateWgsl(
  maxDTreeNodesPerCell: number = PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL,
  reservoirGiStrideU32: number = RESERVOIR_GI_STRIDE_U32,
): string {
  if (reservoirGiStrideU32 !== RESERVOIR_GI_STRIDE_U32) {
    throw new TypeError(
      'buildPpgUpdateWgsl: PPG training requires the live 28-u32 generalized GI ABI.',
    );
  }
  return /* wgsl */`
// ── PPG update kernel ─────────────────────────────────────────────────────────
// Muller et al. 2017 section 3.3 - training from accepted GI reservoir samples.
// The training tuple is (xv, persisted wi_recon, exp2(H)) from the initial reservoir.
// DEVIATION 4 FIX: all directions are in WORLD space.
// W9: real flat-buffer leaf location (no more uniform-grid stub).

struct PPGUpdateUBO {
  sampleCount       : u32,  // total half-res GI reservoir entries this frame
  fluxBudget        : u32,  // number of u32 slots in ppgFluxAtomics (bounds check)
  sampleCountBudget : u32,  // A2 — slots in ppgCellSampleCounts (= maxSpatialCells)
  padding1          : u32,
}

@group(0) @binding(0) var<storage, read>           ppgReservoirGiCurrent : array<u32>;
@group(0) @binding(1) var<storage, read_write>     ppgFluxAtomics        : array<atomic<u32>>;
// W9 — serialised tree bindings.
@group(0) @binding(2) var<storage, read>           ppgQueryArena         : array<u32>;
// A2 — per-spatial-cell training-sample counter (one atomic u32 per cell,
// indexed by dTreeIndex). Drives the CPU sTree split decision.
@group(0) @binding(3) var<storage, read_write>     ppgCellSampleCounts   : array<atomic<u32>>;
@group(1) @binding(0) var<uniform>                 ppgUBO                : PPGUpdateUBO;

const PPG_QUERY_MAGIC_UPDATE: u32 = ${PPG_QUERY_ARENA_MAGIC}u;
const PPG_QUERY_VERSION_UPDATE: u32 = ${PPG_QUERY_ARENA_VERSION}u;
const PPG_QUERY_SCHEMA_UPDATE: u32 = ${PPG_QUERY_ARENA_SCHEMA}u;
fn ppgQueryArenaValidUpdate() -> bool {
  return ppgQueryArena[0] == PPG_QUERY_MAGIC_UPDATE &&
    ppgQueryArena[1] == PPG_QUERY_VERSION_UPDATE &&
    ppgQueryArena[2] != 0u &&
    ppgQueryArena[3] == PPG_QUERY_SCHEMA_UPDATE;
}
fn ppgArenaLoadSTreeF32Update(word: u32) -> f32 {
  return bitcast<f32>(ppgQueryArena[ppgQueryArena[4] + word]);
}
fn ppgArenaLoadDTreeF32Update(word: u32) -> f32 {
  return bitcast<f32>(ppgQueryArena[ppgQueryArena[7] + word]);
}
fn ppgArenaLoadDTreeOffsetUpdate(word: u32) -> u32 {
  return ppgQueryArena[ppgQueryArena[10] + word];
}

// Layout constants provided by ppgTreeLayout (DTREE_HEADER_F32, DTREE_NODE_STRIDE,
// STREE_HEADER_F32, STREE_NODE_STRIDE). ppgUpdate-specific constant below.
  // H29: MAX_DTREE_NODES_PER_CELL is now single-sourced — pipelineCompiler.ts
  // passes the live allocatePPGResources value to buildPpgUpdateWgsl().
  // H24: RESERVOIR_GI_STRIDE_LOCAL is single-sourced from the sole live
  // generalized-reuse ABI.
  const MAX_DTREE_NODES_PER_CELL : u32 = ${maxDTreeNodesPerCell}u;
  const RESERVOIR_GI_STRIDE_LOCAL : u32 = ${reservoirGiStrideU32}u;

// ── Atomic f32 accumulation through u32 compare/exchange ────────────────────
// WebGPU has no atomic<f32>. Storing IEEE-754 bits in atomic<u32> avoids the
// quantisation and per-sample 65536-radiance saturation of the old fixed-point
// counter. Retry until this deposit is published: a fixed retry cap biases the
// learned distribution under contention by discarding arbitrary samples while
// still counting them toward topology confidence. This is lock-free (there is
// no held lock); every interfering successful CAS advances the shared sum and
// that writer exits. Saturation is the only terminal storage limit.
const MAX_FINITE_F32: f32 = 3.402823466e+38;
const PPG_RESERVOIR_LOG_ZERO: f32 = -3.402823466e+38;
// Correctly-rounded f32 endpoint policy. 2^-150 is exactly half of the
// smallest subnormal and therefore rounds to zero (ties-to-even); every log in
// (-150, 128) is still allowed to reach exp2, including subnormal results.
const PPG_LOG2_ROUND_TO_ZERO: f32 = -150.0;
const PPG_LOG2_OVERFLOW: f32 = 128.0;

fn atomicAddFlux(slot: u32, value: f32) {
  if (!(value > 0.0) || value > MAX_FINITE_F32) { return; }
  var oldBits = atomicLoad(&ppgFluxAtomics[slot]);
  loop {
    let oldValue = bitcast<f32>(oldBits);
    if (oldValue == MAX_FINITE_F32) { return; }
    // Repair an impossible negative/NaN/Inf accumulator defensively. Fresh
    // training buffers are zeroed, so this branch only handles corruption.
    var nextValue = value;
    if (oldValue >= 0.0 && oldValue <= MAX_FINITE_F32) {
      nextValue = oldValue + value;
      if (!(nextValue >= oldValue) || nextValue > MAX_FINITE_F32) {
        nextValue = MAX_FINITE_F32;
      }
    }
    let exchanged = atomicCompareExchangeWeak(
      &ppgFluxAtomics[slot], oldBits, bitcast<u32>(nextValue),
    );
    if (exchanged.exchanged) { return; }
    oldBits = exchanged.old_value;
  }
}

// ── sTree descent
// MUST-MATCH: this descent body is semantically identical to ppgSTreeFindLeafBase
// in ppgPdf.wgsl.ts — only the buffer name differs (ppgSTreeBuf here vs
// ppgSTreeBuf_gi there). If you edit the logic here, mirror the change there,
// and vice versa. The ppgDescentDrift vitest gate enforces this automatically.
fn sTreeFindLeafBase(pos: vec3<f32>) -> u32 {
  let nodeCount = u32(ppgArenaLoadSTreeF32Update(0));
  var idx: u32 = 0u;
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    let base = STREE_HEADER_F32 + idx * STREE_NODE_STRIDE;
    let splitAxisF = ppgArenaLoadSTreeF32Update(base + 7u);
    if (splitAxisF < 0.0) { return base; } // leaf
    let splitVal = ppgArenaLoadSTreeF32Update(base + 3u);
    let leftChildF  = ppgArenaLoadSTreeF32Update(base + 8u);
    let rightChildF = ppgArenaLoadSTreeF32Update(base + 9u);
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
// MUST-MATCH: this descent body is semantically identical to ppgDTreeFindLeafBase
// in ppgPdf.wgsl.ts — only the buffer name differs (ppgDTreeBuf here vs
// ppgDTreeBuf_gi there). If you edit the logic here, mirror the change there,
// and vice versa. The ppgDescentDrift vitest gate enforces this automatically.
fn dTreeFindLeafBase(dTreeOffset: u32, uv: vec2<f32>) -> u32 {
  var idx: u32 = 0u;
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    let base = dTreeOffset + DTREE_HEADER_F32 + idx * DTREE_NODE_STRIDE;
    let isLeafFlag = ppgArenaLoadDTreeF32Update(base + 7u);
    if (isLeafFlag > 0.5) { return base; }
    let u0 = ppgArenaLoadDTreeF32Update(base + 0u);
    let v0 = ppgArenaLoadDTreeF32Update(base + 1u);
    let u1 = ppgArenaLoadDTreeF32Update(base + 2u);
    let v1 = ppgArenaLoadDTreeF32Update(base + 3u);
    let uMid = (u0 + u1) * 0.5;
    let vMid = (v0 + v1) * 0.5;
    let firstChildF = ppgArenaLoadDTreeF32Update(base + 6u);
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
//   1. Reads an accepted reservoir's (xv, persisted wi_recon, H, M) tuple.
//   2. Computes the incoming-direction octahedral UV (world frame).
//   3. Walks the sTree to its cell, then the cell's dTree to a leaf.
//   4. Atomically increments the leaf's flux accumulator.
@compute @workgroup_size(64)
fn ppgUpdateMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (!ppgQueryArenaValidUpdate()) { return; }
  let idx = gid.x;
  if (idx >= ppgUBO.sampleCount) { return; }

  let b = idx * RESERVOIR_GI_STRIDE_LOCAL;
  let reservoirM = ppgReservoirGiCurrent[b + 15u];
  if (reservoirM == 0u) { return; }
  // Camera-prefix glass samples carry receiver-specific throughput and are
  // deliberately non-shiftable. Training the shared world-space guide from
  // that view-specific target would contaminate other receivers.
  if (ppgReservoirGiCurrent[b + 24u] != 1u) { return; }

  let pos = vec3f(
    bitcast<f32>(ppgReservoirGiCurrent[b + 0u]),
    bitcast<f32>(ppgReservoirGiCurrent[b + 1u]),
    bitcast<f32>(ppgReservoirGiCurrent[b + 2u])
  );
  // The producer already persists a robust receiver-to-sample direction.
  // Reconstructing xs-xv here loses precision in translated scenes, collapses
  // short edges, and gives environment proxies a needless distance dependence.
  let dirRaw = vec3f(
    bitcast<f32>(ppgReservoirGiCurrent[b + 20u]),
    bitcast<f32>(ppgReservoirGiCurrent[b + 21u]),
    bitcast<f32>(ppgReservoirGiCurrent[b + 22u])
  );
  let dirLen2 = dot(dirRaw, dirRaw);
  if (!(dirLen2 > 1e-12) || dirLen2 > MAX_FINITE_F32) { return; }
  let dir = dirRaw * inverseSqrt(dirLen2);

  // Unbiased represented-RIS histogram mass for the selected initial sample.
  // For arbitrary B: E[1(Y in B) * exp2(H)] = integral_B pHat(w) dw.
  // Convert only at the atomic buffer endpoint. Values at/below the f32
  // round-to-zero midpoint deposit zero; values at/above exp2's overflow
  // boundary saturate. Never floor a tiny mass upward or truncate a finite log.
  let reservoirH = bitcast<f32>(ppgReservoirGiCurrent[b + 11u]);
  if (!(reservoirH > PPG_RESERVOIR_LOG_ZERO) || reservoirH > MAX_FINITE_F32) {
    return;
  }
  if (reservoirH <= PPG_LOG2_ROUND_TO_ZERO) { return; }
  var trainingMass = MAX_FINITE_F32;
  if (reservoirH < PPG_LOG2_OVERFLOW) {
    trainingMass = min(exp2(reservoirH), MAX_FINITE_F32);
  }
  if (!(trainingMass > 0.0) || trainingMass > MAX_FINITE_F32) { return; }

  // Cylindrical EQUAL-AREA UV of the incoming WORLD direction (Müller 2017 §3.2).
  // MUST be byte-identical to ppgPdf.wgsl's ppgDirToUv (train↔pdf↔sample lock-step).
  // FIX 2026-06-09: was octEncode (Cigolle 2014, NON-equal-area) — that made the
  // guide pdf (which assumes solidAngle = 4π·uvArea) mismatch the uniform-in-UV
  // sampling density → biased MIS source pdf → guided GI gained energy. The
  // cylindrical map is equal-area, so the dTree solidAngle is exact. See ppgPdf.wgsl.
  let uvU = (1.0 - clamp(dir.z, -1.0, 1.0)) * 0.5;
  let uvV = atan2(dir.y, dir.x) * 0.15915494309189535 + 0.5;
  let uv = vec2<f32>(uvU, clamp(uvV, 0.0, 1.0));

  // Walk the sTree to the spatial cell for this sample.
  let sBase = sTreeFindLeafBase(pos);
  let dTreeIndex = u32(ppgArenaLoadSTreeF32Update(sBase + 10u));
  let dOff = ppgArenaLoadDTreeOffsetUpdate(dTreeIndex);

  // A2 — count ONE training record for this spatial cell (drives the CPU sTree
  // split decision in splitOverflowLeaves). Bounded by the per-cell counter
  // buffer length (maxSpatialCells = sampleCountBudget).
  if (dTreeIndex < ppgUBO.sampleCountBudget) {
    atomicAdd(&ppgCellSampleCounts[dTreeIndex], 1u);
  }

  // Walk the dTree to the leaf for this direction.
  let leafBase = dTreeFindLeafBase(dOff, uv);

  // Global atomic slot: fixed per-cell node stride of maxDTreeNodesPerCell slots per spatial cell.
  let nodeIdx = (leafBase - dOff - DTREE_HEADER_F32) / DTREE_NODE_STRIDE;
  let slot = dTreeIndex * MAX_DTREE_NODES_PER_CELL + nodeIdx;
  if (slot >= ppgUBO.fluxBudget) { return; }

  atomicAddFlux(slot, trainingMass);
}
`;
}

/**
 * H29 — default module instance (maxDTreeNodesPerCell = 341).
 *
 * Used by wgslModules.ts / wgslCompose tests that import PPG_UPDATE_MODULE
 * directly (e.g. the WGSL-compose byte-identity gate). pipelineCompiler.ts
 * MUST call buildPpgUpdateWgsl() with the live allocation value instead of
 * using this constant when compiling the actual GPU pipeline.
 */
export const PPG_UPDATE_WGSL: string = buildPpgUpdateWgsl();

/** W1-R6 — declarative include-graph entry. Requires ppgTreeLayout for the
 *  shared layout constants.
 *  No octahedralCore require: the 2026-06-09 equal-area fix replaced the
 *  octEncode call with the inline cylindrical map (`ppgDirToUv`), so the
 *  shared octahedral helpers are no longer referenced by this kernel.
 *
 *  H29: pipelineCompiler.ts builds its own WgslModule via buildPpgUpdateWgsl()
 *  with the live allocation cap. This module instance uses the default 341 for
 *  compose-tests and module-registry purposes.
 */
export const PPG_UPDATE_MODULE: WgslModule = {
  name: 'ppgUpdate',
  source: PPG_UPDATE_WGSL,
  requires: ['ppgTreeLayout'],
};
