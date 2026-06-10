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
 * The RIS pipeline is already at the `maxStorageBuffersPerShaderStage = 16`
 * full-tier floor (4 frame + 11 scene + 1 light-tree group(3) buffers). Adding
 * a SECOND group(3) storage buffer for the grid would push RIS to 17 and fail
 * pipeline creation. So the grid is co-located in the SAME `array<f32>` buffer
 * as the light tree (`@group(3) @binding(0)`): the tree nodes occupy floats
 * `[0 .. lightTreeNodeCount*12)`, the grid cells occupy floats
 * `[regirGridFloatOffset ..)`. RIS reads the combined buffer read-only (still 1
 * group(3) buffer ⇒ still 16). The grid-build pass binds the SAME buffer as
 * read_write in its own dedicated bind group and writes only the grid region;
 * the tree region is uploaded once and never written on the GPU.
 *
 * ─── Per-cell layout ─────────────────────────────────────────────────────────
 * A cell stores `K = regirSurvivorsPerCell` independent WRS sub-reservoir
 * survivors, each `REGIR_FLOATS_PER_SURVIVOR = 2` floats:
 *   survivor j of cell c at float `regirGridFloatOffset + (c*K + j)*2`:
 *     [+0] emitterIndex (f32; < 0 ⇒ empty slot)
 *     [+1] pSel — the EFFECTIVE per-cell selection pmf of that emitter
 *
 * ─── Unbiasedness (CRITICAL) ─────────────────────────────────────────────────
 * The cell target is `q̂_c(e) = luminance(Le_e)·area_e / max(dist²(x_c, e), floor)`
 * — power × proximity at the cell centroid `x_c`, NO BRDF (the receiver BRDF is
 * unknown at grid-build time; it enters later in the RIS p̂). Each sub-reservoir
 * runs WRS over `M` tree draws: candidate `e_i` is drawn with source pdf
 * `p_tree(e_i | x_c)`, RIS weight `w_i = q̂_c(e_i)/p_tree(e_i|x_c)`. The survivor
 * `e*` is (in expectation) distributed ∝ `q̂_c`, and `wSum/M` is an unbiased
 * estimate `Ŝ` of `S_c = Σ_e q̂_c(e)`. The stored selection pmf is therefore
 *     pSel(e*) = q̂_c(e*) / Ŝ = q̂_c(e*) · M / wSum,
 * the standard RIS relation (Bitterli 2020 §3): a WRS reservoir is an importance
 * sampler whose effective pdf is `target / normalisation-estimate`. Σ_e pSel(e)
 * → 1 (a valid pmf). When RIS draws candidate `e*` from cell `c`, it uses EXACTLY
 * this `pSel` as the source pmf in `w = p̂ / (pSel · pdfArea)` — the SAME unbiased
 * discipline as the light-tree path, just with a different (grid-amortised)
 * source distribution. The `1/K` does NOT enter the weight: RIS draws ONE
 * candidate from ONE uniformly-chosen survivor; it is not summing over the K.
 *
 * Mirrors `regirBuildSurvivorCPU` / `regirCellPmfExact` in
 * `@vitrum/shared-samplers/src/lightTree.ts` byte-for-byte.
 *
 * Gate: `ubo.regirEnabled == 0u` ⇒ the grid-build pass early-returns and RIS
 * falls back to the light-tree path BIT-IDENTICALLY (it reads the tree region of
 * the combined buffer, never the grid region).
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const REGIR_WGSL = /* wgsl */ `// ============================================================
// ReGIR — grid build + cell sampling. Reuses the combined light-tree storage
// buffer (lightTree @group(3) @binding(0) on RIS; the grid-build pass binds
// the same buffer read_write as regirGridRW in its own group).
// ============================================================

const REGIR_FLOATS_PER_SURVIVOR: u32 = 2u;

struct ReGIRCellSample {
  emitterIndex: i32,
  pSel:         f32,   // effective per-cell selection pmf of the chosen emitter
};

// Cell target q̂_c(e) for the cell centroid x_c: power × proximity, NO BRDF.
// power = luminance(Le)·area is the SAME quantity the light tree + flat CDF use,
// and the dist² floor is the SAME ubo.emitterDist2Floor the tree descent +
// shade geometry term use — so the grid target, tree descent, and RIS p̂ all
// agree on near-light behaviour.
fn regir_cell_target(lid: u32, xc: vec3f, dist2Floor: f32) -> f32 {
  let e = emitters[lid];
  let power = luminance(e.Le) * e.area;
  if (power <= 0.0) { return 0.0; }
  let centroid = (e.vA + e.vB + e.vC) / 3.0;
  let toC = centroid - xc;
  let d2 = max(dot(toC, toC), dist2Floor);
  return power / d2;
}

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
fn regir_cell_centroid(cellIdx: u32) -> vec3f {
  let dimsXY = ubo.regirDims.x * ubo.regirDims.y;
  let cz = cellIdx / dimsXY;
  let rem = cellIdx % dimsXY;
  let cy = rem / ubo.regirDims.x;
  let cx = rem % ubo.regirDims.x;
  let cellSize = 1.0 / ubo.regirInvCellSize;
  return ubo.regirOrigin + (vec3f(f32(cx), f32(cy), f32(cz)) + vec3f(0.5)) * cellSize;
}

// Float offset of survivor j of cell c in the combined buffer.
fn regir_survivor_base(cellIdx: u32, j: u32) -> u32 {
  return ubo.regirGridFloatOffset
       + (cellIdx * ubo.regirSurvivorsPerCell + j) * REGIR_FLOATS_PER_SURVIVOR;
}

// ── Cell sampling (RIS read path) ────────────────────────────────────────────
// Pick one of the cell K survivors uniformly and return its emitter + pSel.
// The 1/K factor does NOT enter pSel (RIS draws ONE candidate; see header). An
// empty survivor (pSel <= 0) returns emitterIndex < 0 so the caller skips it.
fn regir_sample_cell(p: vec3f, rng: ptr<function, u32>) -> ReGIRCellSample {
  var out: ReGIRCellSample;
  out.emitterIndex = -1;
  out.pSel = 0.0;
  let k = ubo.regirSurvivorsPerCell;
  if (k == 0u) { return out; }
  let cellIdx = regir_cell_index(p);
  // Uniform among K survivors.
  let j = min(u32(rand_f32(rng) * f32(k)), k - 1u);
  let base = regir_survivor_base(cellIdx, j);
  let emitterF = lightTree[base + 0u];
  let pSel = lightTree[base + 1u];
  if (emitterF < 0.0 || pSel <= 0.0) { return out; }
  out.emitterIndex = i32(emitterF);
  out.pSel = pSel;
  return out;
}

`;

/**
 * RIS read-side include-graph entry. `regir` reads the SAME `lightTree`
 * storage binding + the `WalkaroundUBO` (regir* fields) + `emitters` /
 * `luminance` for the target eval, all of which `lightTree` → `common`
 * already provide; declaring `requires: ['lightTree']` pulls in the
 * `@group(3)` binding + `sampleLightTree` so RIS composes both seamlessly.
 */
export const REGIR_MODULE: WgslModule = {
  name: 'regir',
  source: REGIR_WGSL,
  requires: ['lightTree'],
};

// ─────────────────────────────────────────────────────────────────────────────
// Grid-build compute pass (write path). Standalone module: its own bind group
// declarations (the combined buffer as read_write + emitters + ubo), the WRS
// helpers, and the `regirBuildMain` entry point. It re-declares the regir read
// helpers' math inline-free by depending on `regir` for `regir_cell_target` /
// `regir_cell_centroid` / `regir_survivor_base` — but those reference the
// READ-ONLY `lightTree` binding, which the build pass binds read_write under a
// different name. To keep one source of truth for the math WITHOUT a
// conflicting binding, the build kernel recomputes the survivor base offset
// locally and writes via its own `regirGridRW` binding.
// ─────────────────────────────────────────────────────────────────────────────

export const REGIR_BUILD_WGSL = /* wgsl */ `// ============================================================
// ReGIR grid-build kernel. One workgroup-invocation per (cell × survivor):
// dispatched as a 1-D grid of size numCells × K. Each invocation runs one WRS
// sub-reservoir over the light tree seeded at the cell centroid and writes its
// survivor (emitterIndex + pSel) into the grid region of the combined buffer.
// ============================================================

const REGIR_FLOATS_PER_SURVIVOR: u32 = 2u;

// The combined light-tree + grid buffer, bound READ_WRITE here (the build pass
// reads the tree region for sampleLightTree, writes the grid region). RIS binds
// the SAME buffer read-only as lightTree — different access, same GPUBuffer.
@group(0) @binding(0) var<storage, read_write> regirGridRW: array<f32>;
@group(0) @binding(1) var<storage, read>       emittersRW:  array<EmitterTri>;
@group(0) @binding(2) var<uniform>             ubo:         WalkaroundUBO;

const REGIR_BUILD_STRIDE: u32 = 16u; // light-tree node stride (must match LIGHT_TREE_STRIDE; B8 12→16)

fn rb_dist2ToAabb(p: vec3f, bmin: vec3f, bmax: vec3f) -> f32 {
  let d = max(max(bmin - p, vec3f(0.0)), p - bmax);
  return dot(d, d);
}

// B8 — orientation-cone factor, mirroring lt_coneFactor (lightTree.wgsl) so the
// ReGIR grid-build descent culls oriented emitters identically to the RIS
// read-path descent. Full-sphere node (axis length 0) ⇒ 1 (no culling).
fn rb_coneFactor(axis: vec3f, cosThetaO: f32, cosThetaOE: f32, p: vec3f, c: vec3f) -> f32 {
  let al = dot(axis, axis);
  if (al < 1e-12) { return 1.0; }
  let dv = p - c;
  let dl2 = dot(dv, dv);
  if (dl2 < 1e-12) { return 1.0; }
  let d = dv * inverseSqrt(dl2);
  let a = axis * inverseSqrt(al);
  let cosTheta = dot(a, d);
  if (cosTheta < cosThetaOE) { return 0.0; }
  if (cosTheta >= cosThetaO) { return 1.0; }
  let sinTheta  = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  let sinThetaO = sqrt(max(0.0, 1.0 - cosThetaO * cosThetaO));
  return max(0.0, cosTheta * cosThetaO + sinTheta * sinThetaO);
}

fn rb_importance(base: u32, p: vec3f, dist2Floor: f32) -> f32 {
  let power = regirGridRW[base + 1u];
  if (power <= 0.0) { return 0.0; }
  let bmin = vec3f(regirGridRW[base + 4u], regirGridRW[base + 5u], regirGridRW[base + 6u]);
  let bmax = vec3f(regirGridRW[base + 7u], regirGridRW[base + 8u], regirGridRW[base + 9u]);
  let d2 = max(rb_dist2ToAabb(p, bmin, bmax), dist2Floor);
  let axis = vec3f(regirGridRW[base + 10u], regirGridRW[base + 11u], regirGridRW[base + 12u]);
  let cosThetaO  = regirGridRW[base + 13u];
  let cosThetaOE = regirGridRW[base + 14u];
  let center = 0.5 * (bmin + bmax);
  return (power / d2) * rb_coneFactor(axis, cosThetaO, cosThetaOE, p, center);
}

// Light-tree descent reading the tree region of the combined buffer. Mirrors
// sampleLightTree (lightTree.wgsl) byte-for-byte, but reads from regirGridRW.
struct RBTreeSample { emitterIndex: i32, pdf: f32 };
fn rb_sample_tree(p: vec3f, dist2Floor: f32, nodeCount: u32, rng: ptr<function, u32>) -> RBTreeSample {
  var nodeIdx: u32 = 0u;
  var pdf: f32 = 1.0;
  for (var guard: u32 = 0u; guard < nodeCount + 1u; guard = guard + 1u) {
    let base = nodeIdx * REGIR_BUILD_STRIDE;
    let leftChild  = i32(regirGridRW[base + 2u]);
    let rightChild = i32(regirGridRW[base + 3u]);
    if (leftChild < 0 || rightChild < 0) {
      var s: RBTreeSample;
      s.emitterIndex = i32(regirGridRW[base + 0u]);
      s.pdf = pdf;
      return s;
    }
    let lBase = u32(leftChild) * REGIR_BUILD_STRIDE;
    let rBase = u32(rightChild) * REGIR_BUILD_STRIDE;
    let impL = rb_importance(lBase, p, dist2Floor);
    let impR = rb_importance(rBase, p, dist2Floor);
    let sum = impL + impR;
    let pL = select(0.5, impL / sum, sum > 0.0);
    if (rand_f32(rng) < pL) {
      pdf = pdf * pL;
      nodeIdx = u32(leftChild);
    } else {
      pdf = pdf * (1.0 - pL);
      nodeIdx = u32(rightChild);
    }
  }
  var s: RBTreeSample;
  s.emitterIndex = i32(regirGridRW[nodeIdx * REGIR_BUILD_STRIDE + 0u]);
  s.pdf = pdf;
  return s;
}

// Cell target q̂_c(e) — SAME as regir_cell_target (read module), but using the
// build pass's own emitter binding.
fn rb_cell_target(lid: u32, xc: vec3f, dist2Floor: f32) -> f32 {
  let e = emittersRW[lid];
  let power = luminance(e.Le) * e.area;
  if (power <= 0.0) { return 0.0; }
  let centroid = (e.vA + e.vB + e.vC) / 3.0;
  let toC = centroid - xc;
  let d2 = max(dot(toC, toC), dist2Floor);
  return power / d2;
}

fn rb_cell_centroid(cellIdx: u32) -> vec3f {
  let dimsXY = ubo.regirDims.x * ubo.regirDims.y;
  let cz = cellIdx / dimsXY;
  let rem = cellIdx % dimsXY;
  let cy = rem / ubo.regirDims.x;
  let cx = rem % ubo.regirDims.x;
  let cellSize = 1.0 / ubo.regirInvCellSize;
  return ubo.regirOrigin + (vec3f(f32(cx), f32(cy), f32(cz)) + vec3f(0.5)) * cellSize;
}

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

  // WRS over M tree draws. Source pdf = tree selection pmf at xc; target = q̂_c.
  var wSum: f32 = 0.0;
  var chosen: i32 = -1;
  var chosenQHat: f32 = 0.0;
  for (var i: u32 = 0u; i < M; i = i + 1u) {
    let draw = rb_sample_tree(xc, dist2Floor, ubo.lightTreeNodeCount, &rng);
    if (draw.emitterIndex < 0 || draw.pdf <= 0.0) { continue; }
    let qHat = rb_cell_target(u32(draw.emitterIndex), xc, dist2Floor);
    if (qHat <= 0.0) { continue; }
    let w = qHat / draw.pdf;
    wSum = wSum + w;
    if (rand_f32(&rng) * wSum < w) {
      chosen = draw.emitterIndex;
      chosenQHat = qHat;
    }
  }

  // Effective selection pmf = q̂(e*) / Ŝ, Ŝ = wSum / M (unbiased S_c estimate).
  let base = ubo.regirGridFloatOffset
           + (cellIdx * k + survivorJ) * REGIR_FLOATS_PER_SURVIVOR;
  if (chosen < 0 || wSum <= 0.0) {
    regirGridRW[base + 0u] = -1.0; // empty slot
    regirGridRW[base + 1u] = 0.0;
    return;
  }
  let pSel = (chosenQHat * f32(M)) / wSum;
  regirGridRW[base + 0u] = f32(chosen);
  regirGridRW[base + 1u] = pSel;
}

`;

/**
 * Grid-build include-graph entry. Requires `common` for `WalkaroundUBO`,
 * `EmitterTri`, `luminance`, `pcgInit`/`rand_f32`. Declares its OWN @group(0)
 * bindings (combined buffer read_write + emitters + ubo), so it does NOT
 * require the `lightTree` / `regir` read modules (which would re-declare a
 * conflicting read-only `lightTree` binding).
 */
export const REGIR_BUILD_MODULE: WgslModule = {
  name: 'regirBuild',
  source: REGIR_BUILD_WGSL,
  requires: ['common'],
};
