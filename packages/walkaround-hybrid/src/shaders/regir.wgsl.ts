/**
 * ReGIR — Reservoir-based Grid Importance Resampling for ReSTIR-DI light SELECTION.
 *
 * ReGIR (Boksansky, Wyman, Benty 2021, "Rendering Many Lights with Grid-Based
 * Reservoirs", Ray Tracing Gems II ch. 23) decouples the per-pixel light-
 * selection cost from the light count. Each frame a compute pass fills a
 * WORLD-SPACE GRID of light reservoirs: per cell, a weighted-reservoir stream
 * (WRS) pre-resamples lights by power × proximity, SEEDED BY THE LIGHT TREE
 * (`sampleLightTree` at the cell centroid). ReSTIR-DI's initial-candidate light
 * selection then draws from the containing cell's reservoir instead of
 * traversing the tree per pixel — so the per-pixel cost is O(1) regardless of
 * light count.
 *
 * ─── Buffer co-location (storage-budget constraint) ──────────────────────────
 * The RIS pipeline is already at the guaranteed
 * `maxStorageBuffersPerShaderStage = 8` floor (4 frame + 3 versioned scene
 * arenas + 1 light-tree group(3) buffer). Adding a second group(3) storage
 * buffer for the grid would push RIS to 9. So the grid is co-located in the
 * same `array<f32>` buffer
 * as the light tree (`@group(3) @binding(0)`): the tree nodes occupy floats
 * `[0 .. lightTreeNodeCount*16)`, the grid cells occupy floats
 * `[regirGridFloatOffset ..)`. RIS reads the combined buffer read-only (still 1
 * group(3) buffer ⇒ still 8). The grid-build pass binds the SAME buffer as
 * read_write in its own dedicated bind group and writes only the grid region;
 * the tree region is uploaded once and never written on the GPU.
 *
 * ─── Per-cell layout ─────────────────────────────────────────────────────────
 * A cell stores `K = regirSurvivorsPerCell` independent WRS sub-reservoir
 * survivors, each `REGIR_FLOATS_PER_SURVIVOR = 2` floats:
 *   survivor j of cell c at float `regirGridFloatOffset + (c*K + j)*2`:
 *     [+0] emitterIndex (f32; < 0 ⇒ empty slot)
 *     [+1] log2PSel — log2 represented effective source density
 *
 * ─── Unbiasedness (CRITICAL) ─────────────────────────────────────────────────
 * The cell target is the packed light-tree leaf importance
 * `q̂_c(e) = power_e / max(dist²(x_c, e.aabb), floor)` (with the same orientation
 * cone term as tree traversal). `power_e` is the CPU emitter-distribution power:
 * scalar for uniform emitters, UV-local/map-aware for emissive-map micro-emitters.
 * This is power × proximity at the cell centroid `x_c`, NO BRDF (the receiver BRDF
 * is unknown at grid-build time; it enters later in the RIS p̂). Each sub-reservoir
 * runs WRS over `M` tree draws: candidate occurrence `i` is drawn with source
 * PMF `p_tree_i` and gets weight `q̂_c(e_i)/p_tree_i`. Integer branch buckets
 * realise an occurrence probability `r_i`, tracked through every replace/keep
 * decision. The stored density is `log2PSel_i = log2(M · r_i · p_tree_i)`.
 * Thus `Σ_i r_i F_i/pSel_i = (1/M) Σ_i F_i/p_tree_i` for every realised
 * candidate batch. Positive-power leaves whose centroid cone target is zero
 * receive minimum-normal q̂ support. The `1/K` does not enter the weight because
 * RIS draws one of K iid survivors.
 *
 * Mirrors `regirBuildSurvivorCPU` / `regirCellPmfExact` in
 * `@vitrum/shared-samplers/src/regir.ts` byte-for-byte.
 *
 * Gate: `ubo.regirEnabled == 0u` ⇒ the grid-build pass early-returns and RIS
 * falls back to the light-tree path BIT-IDENTICALLY (it reads the tree region of
 * the combined buffer, never the grid region).
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
// D5.5/I2.4: template-interpolate REGIR_FLOATS_PER_SURVIVOR from the shared-samplers
// TS constant so the WGSL value is always in sync with the host value.
import {
  REGIR_FLOATS_PER_SURVIVOR,
  REGIR_LOG2_PSEL_INVALID,
  lightTreeTraversalWgsl,
} from '@vitrum/shared-samplers';

// D8-6 (complexity-sweep 2026-07-20, T4-3): the cell-index → world-centroid math
// is byte-identical between the REGIR read kernel (`regir_cell_centroid`) and the
// REGIR build kernel (`rb_cell_centroid`). It reads `ubo` (a CONSUMER binding), so
// per the composeWgsl ordering rule it is shared as a RAW-STRING template
// interpolated into each consumer body — NOT a WgslModule. The function name is
// the single parameterized slot; the body is emitted byte-for-byte identically so
// both composed shaders stay byte-identical.
function regirCellCentroidWgsl(fnName: string): string {
  return /* wgsl */ `fn ${fnName}(cellIdx: u32) -> vec3f {
  let dimsXY = ubo.regirDims.x * ubo.regirDims.y;
  let cz = cellIdx / dimsXY;
  let rem = cellIdx % dimsXY;
  let cy = rem / ubo.regirDims.x;
  let cx = rem % ubo.regirDims.x;
  let cellSize = 1.0 / ubo.regirInvCellSize;
  return ubo.regirOrigin + (vec3f(f32(cx), f32(cy), f32(cz)) + vec3f(0.5)) * cellSize;
}`;
}

export const REGIR_WGSL = /* wgsl */ `// ============================================================
// ReGIR — grid build + cell sampling. Reuses the combined light-tree storage
// buffer (lightTree @group(3) @binding(0) on RIS; the grid-build pass binds
// the same buffer read_write as regirGridRW in its own group).
// ============================================================

const REGIR_FLOATS_PER_SURVIVOR: u32 = ${REGIR_FLOATS_PER_SURVIVOR}u;
const REGIR_LOG2_PSEL_INVALID: f32 = ${REGIR_LOG2_PSEL_INVALID};
const REGIR_F32_MAX: f32 = 3.402823466e38;

struct ReGIRCellSample {
  emitterIndex: i32,
  log2PSel:     f32,
};

// World position → flat cell index. Clamps to the grid AABB so points just
// outside the padded bounds still map to a border cell (never out-of-range).
fn regir_cell_index(p: vec3f) -> u32 {
  let rel = (p - ubo.regirOrigin) * ubo.regirInvCellSize;
  let cx = u32(clamp(floor(rel.x), 0.0, f32(ubo.regirDims.x) - 1.0));
  let cy = u32(clamp(floor(rel.y), 0.0, f32(ubo.regirDims.y) - 1.0));
  let cz = u32(clamp(floor(rel.z), 0.0, f32(ubo.regirDims.z) - 1.0));
  return (cz * ubo.regirDims.y + cy) * ubo.regirDims.x + cx;
}

// Cell centroid in world space (cell-index → centre).
${regirCellCentroidWgsl('regir_cell_centroid')}

// Float offset of survivor j of cell c in the combined buffer.
fn regir_survivor_base(cellIdx: u32, j: u32) -> u32 {
  return ubo.regirGridFloatOffset
       + (cellIdx * ubo.regirSurvivorsPerCell + j) * REGIR_FLOATS_PER_SURVIVOR;
}

// ── Cell sampling (RIS read path) ────────────────────────────────────────────
// Pick one of the cell K survivors uniformly and return its emitter + log2PSel.
// Empty slots are identified only by emitterIndex < 0. A negative log density
// is valid; malformed/sentinel log lanes are rejected by a separate check.
fn regir_sample_cell(p: vec3f, rng: ptr<function, u32>) -> ReGIRCellSample {
  var out: ReGIRCellSample;
  out.emitterIndex = -1;
  out.log2PSel = REGIR_LOG2_PSEL_INVALID;
  let k = ubo.regirSurvivorsPerCell;
  if (k == 0u) { return out; }
  let cellIdx = regir_cell_index(p);
  // Exactly uniform among K survivors, including non-divisors of 2^24.
  let j = rand_bounded_u32(rng, k);
  let base = regir_survivor_base(cellIdx, j);
  let emitterF = lightTree[base + 0u];
  let log2PSel = lightTree[base + 1u];
  if (!(
    emitterF >= 0.0 && emitterF <= 16777215.0 && floor(emitterF) == emitterF
  )) { return out; }
  if (!(log2PSel > REGIR_LOG2_PSEL_INVALID && log2PSel <= REGIR_F32_MAX)) {
    return out;
  }
  out.emitterIndex = i32(emitterF);
  out.log2PSel = log2PSel;
  return out;
}

`;

/**
 * RIS read-side include-graph entry. `regir` reads the SAME `lightTree`
 * storage binding + the `WalkaroundUBO` (regir* fields), both supplied by
 * `lightTree` → `common`; declaring `requires: ['lightTree']` pulls in the
 * `@group(3)` binding + `sampleLightTree` so RIS composes both seamlessly.
 */
export const REGIR_MODULE: WgslModule = {
  name: 'regir',
  source: REGIR_WGSL,
  requires: ['lightTree'],
};

// Instantiate the same represented-support traversal used by RIS and
// pt-webgpu, but point it at ReGIR's read_write alias of the combined buffer.
// ReGIR's wrapper below only adds qHat recovery from the returned leaf index.
const REGIR_BUILD_LIGHT_TREE_WGSL = lightTreeTraversalWgsl({
  storageVariable: 'regirGridRW',
  helperPrefix: 'rb_lt',
  strideConstantName: 'RB_LIGHT_TREE_STRIDE',
  sampleStructName: 'RBLightTreeSelection',
  sampleFunctionName: 'rb_sampleLightTree',
});

// ─────────────────────────────────────────────────────────────────────────────
// Grid-build compute pass (write path). Standalone module: its own bind group
// declarations (the combined buffer as read_write + ubo), the WRS
// helpers, and the `regirBuildMain` entry point. The read path references the
// READ-ONLY `lightTree` binding, which the build pass binds read_write under a
// different name, so the build kernel reads tree leaf importance from its own
// `regirGridRW` binding and recomputes the survivor base offset locally.
// ─────────────────────────────────────────────────────────────────────────────

export const REGIR_BUILD_WGSL = /* wgsl */ `// ============================================================
// ReGIR grid-build kernel. One workgroup-invocation per (cell × survivor):
// dispatched as a 1-D grid of size numCells × K. Each invocation runs one WRS
// sub-reservoir over the light tree seeded at the cell centroid and writes its
// survivor (emitterIndex + log2PSel) into the grid region of the combined buffer.
// ============================================================

const REGIR_FLOATS_PER_SURVIVOR: u32 = ${REGIR_FLOATS_PER_SURVIVOR}u;
const REGIR_LOG2_PSEL_INVALID: f32 = ${REGIR_LOG2_PSEL_INVALID};

// The combined light-tree + grid buffer, bound READ_WRITE here (the build pass
// reads the tree region for sampleLightTree, writes the grid region). RIS binds
// the SAME buffer read-only as lightTree — different access, same GPUBuffer.
@group(0) @binding(0) var<storage, read_write> regirGridRW: array<f32>;
@group(0) @binding(1) var<uniform>             ubo:         WalkaroundUBO;

${REGIR_BUILD_LIGHT_TREE_WGSL}

// The generated traversal returns the exact leaf node index. ReGIR adds only
// the leaf target evaluation; selection and qHat therefore share the same
// canonical AABB/cone/saturation implementation.
struct RBTreeSample { emitterIndex: i32, pdf: f32, qHat: f32, nodeIndex: u32 };
fn rb_sample_tree(p: vec3f, dist2Floor: f32, nodeCount: u32, rng: ptr<function, u32>) -> RBTreeSample {
  let draw = rb_sampleLightTree(p, dist2Floor, nodeCount, rng);
  var result: RBTreeSample;
  result.emitterIndex = draw.emitterIndex;
  result.pdf = draw.pdf;
  result.qHat = rb_lt_importance(draw.nodeIndex * RB_LIGHT_TREE_STRIDE, p, dist2Floor);
  result.nodeIndex = draw.nodeIndex;
  return result;
}

${regirCellCentroidWgsl('rb_cell_centroid')}

@compute @workgroup_size(64, 1, 1)
fn regirBuildMain(@builtin(global_invocation_id) gid: vec3u) {
  if (ubo.regirEnabled == 0u) { return; }
  let k = ubo.regirSurvivorsPerCell;
  let numCells = ubo.regirDims.x * ubo.regirDims.y * ubo.regirDims.z;
  let total = numCells * k;
  let flatIdx = gid.x;
  if (flatIdx >= total) { return; }

  let cellIdx = flatIdx / k;
  let survivorJ = flatIdx % k;
  let xc = rb_cell_centroid(cellIdx);
  let dist2Floor = ubo.emitterDist2Floor;
  let M = max(ubo.regirCandidatesPerCell, 1u);

  // Independent RNG per (cell, survivor) so the K survivors are decorrelated.
  var rng = pcgInit(
    cellIdx ^ (ubo.frameSeed * 2654435761u),
    survivorJ ^ (ubo.frameSeed * 40503u),
    ubo.frameSeed ^ 0x9e3779b9u);

  // Represented WRS over M tree draws. The generic primitive records the actual
  // integer-bucket occurrence probability in compensated log2 form.
  var wrs = representedWrsInit();
  var chosen: i32 = -1;
  var chosenTreePmf: f32 = 0.0;
  for (var i: u32 = 0u; i < M; i = i + 1u) {
    let draw = rb_sample_tree(xc, dist2Floor, ubo.lightTreeNodeCount, &rng);
    if (draw.emitterIndex < 0 || draw.pdf <= 0.0) { continue; }
    let leafBase = draw.nodeIndex * RB_LIGHT_TREE_STRIDE;
    let leafPower = regirGridRW[leafBase + 1u];
    if (!(leafPower > 0.0)) { continue; }
    // The represented tree reserves support for this leaf even when its cone
    // importance is zero at xc, so WRS must not discard that reachable draw.
    let qHat = max(draw.qHat, rb_lt_F32_MIN_NORMAL);
    let logWeight = log2(qHat) - log2(draw.pdf);
    if (representedWrsUpdate(&wrs, logWeight, &rng)) {
      chosen = draw.emitterIndex;
      chosenTreePmf = draw.pdf;
    }
  }

  // Persist the represented effective density in log2 form.
  let base = ubo.regirGridFloatOffset
           + (cellIdx * k + survivorJ) * REGIR_FLOATS_PER_SURVIVOR;
  if (chosen < 0 || !wrs.hasSelection || !(chosenTreePmf > 0.0)) {
    regirGridRW[base + 0u] = -1.0; // empty slot
    regirGridRW[base + 1u] = REGIR_LOG2_PSEL_INVALID;
    return;
  }
  let occurrenceLog = representedWrsLogSelectionProbabilityParts(wrs);
  var densityLog = representedWrsAddLogTerm(
    occurrenceLog.x,
    occurrenceLog.y,
    log2(f32(M)),
  );
  densityLog = representedWrsAddLogTerm(
    densityLog.x,
    densityLog.y,
    log2(chosenTreePmf),
  );
  let log2PSel = densityLog.x + densityLog.y;
  if (!(log2PSel > REGIR_LOG2_PSEL_INVALID && log2PSel <= rb_lt_F32_MAX)) {
    regirGridRW[base + 0u] = -1.0;
    regirGridRW[base + 1u] = REGIR_LOG2_PSEL_INVALID;
    return;
  }
  regirGridRW[base + 0u] = f32(chosen);
  regirGridRW[base + 1u] = log2PSel;
}

`;

/**
 * Grid-build include-graph entry. Requires `common` for `WalkaroundUBO`,
 * `EmitterTri`, `pcgInit`/`rand_f32`. Declares its OWN @group(0)
 * bindings (combined buffer read_write + emitters + ubo), so it does NOT
 * require the `lightTree` / `regir` read modules (which would re-declare a
 * conflicting read-only `lightTree` binding).
 */
export const REGIR_BUILD_MODULE: WgslModule = {
  name: 'regirBuild',
  source: REGIR_BUILD_WGSL,
  requires: ['common'],
};
