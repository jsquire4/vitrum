/**
 * PPG sample shader — Sprint 11.
 *
 * WGSL fragment for sampling a direction from the learned PPG PDF at an
 * indirect-bounce shading point.
 *
 * Included into shade.wgsl (deferred — Sprint 11 integration spec defines
 * the bind-group additions). In Sprint 11 the shader is authored and tested
 * but NOT dispatched; dispatch wiring is the Sprint 11 integration task.
 *
 * Algorithm:
 *   1. Nearest spatial cell: kd-tree NN traversal (O(log N) typical) over
 *      `ppgKdNodes`; falls back to linear scan when the buffer holds the
 *      disabled sentinel (see `encodePpgKdDisabledRoot` on the host).
 *   2. Read the cell's directional leaf (16 bins stored as vec2f pairs).
 *   3. Build CDF over the 16 bins using the accumulated radianceSum.
 *   4. Sample a bin using the provided uniform [0,1) random value u2.
 *   5. Decode the bin's octahedral direction, perturb within the bin's
 *      solid angle using u1, and return a world-space unit vector.
 *   6. Fall back to cosine-weighted hemisphere sampling if the cell is empty
 *      (all bins have sampleCount == 0).
 *
 * `ppgPDF` computes the probability density for a sampled direction in the
 * same cell. Required for MIS weighting against BSDF / NEE PDFs.
 *
 * References:
 *   - Müller et al. 2017, "Practical Path Guiding for Efficient Light-Transport
 *     Simulation", Computer Graphics Forum 36(4).
 *   - Clarberg et al. 2005, "Wavelet Importance Sampling: Efficiently Evaluating
 *     Products of Complex Functions" — octahedral bin mapping.
 *
 * @since Sprint 11, 2026-05-09
 */

import { PPG_COMMON_WGSL } from './ppgCommon.wgsl.js';

export const PPG_SAMPLE_WGSL = /* wgsl */`

// ============================================================
// PPG data structures (matching resourceManager.ts layout)
// ============================================================

// PPGSpatialCell — 32 bytes per cell.
//   bytes  0-11: position xyz (vec3f)
//   bytes 12-15: _pad (f32)
//   bytes 16-19: leafIndex (u32)
//   bytes 20-31: _pad2 (vec3u alignment)
struct PPGSpatialCell {
  position:  vec3f,
  _pad:      f32,
  leafIndex: u32,
  _pad2x:    u32,
  _pad2y:    u32,
  _pad2z:    u32,
};

// PPGDirectionalLeaf — 256 bytes per leaf (128 used, 128 reserved).
//   bins: array<vec2f, 16>  — x=radianceSum, y=sampleCount per bin
//   _reserved: array<vec2f, 16> — reserved for future split-tracking
struct PPGDirectionalLeaf {
  bins:      array<vec2f, 16>,  // x=radianceSum, y=sampleCount
  _reserved: array<vec2f, 16>,  // reserved — do not read
};

// 16-byte kd-tree node (matches buildPpgKdTree.ts / types.ts).
// Internal: meta = axis (0..2), child0/child1 = child node indices, split = plane.
// Leaf: meta has high bit set; low bits = cell index into ppgCells.
struct PPGKdNode {
  child0: u32,
  child1: u32,
  meta:   u32,
  split:  f32,
};

// ============================================================
// PPG bind group (group injected by pipelineCompiler — deferred)
// ============================================================
//
// These bindings are declared here for completeness. The host bind group
// number is TBD until the Sprint 11 integration wires them in. When
// integrated, replace @group(N) with the actual group assigned by
// pipelineCompiler.ts.
//
// For now, use group(2) as a placeholder (consistent with Sprint 9 resolve
// shader convention — lowest unused group in shade.wgsl).

@group(2) @binding(0) var<storage, read> ppgCells:   array<PPGSpatialCell>;
@group(2) @binding(1) var<storage, read> ppgLeaves: array<PPGDirectionalLeaf>;
@group(2) @binding(2) var<storage, read> ppgKdNodes: array<PPGKdNode>;

// ============================================================
// Internal helpers
// ============================================================

// Decode the bin index (0..15) into an octahedral direction.
// Bins are arranged in a 4×4 grid over the upper hemisphere in
// octahedral [0,1]² space.
// Grid cell (row, col): row = bin / 4, col = bin % 4.
fn ppgBinToOctahedral(binIdx: u32) -> vec2f {
  let row = f32(binIdx / 4u);
  let col = f32(binIdx % 4u);
  // Centre of cell in [0,1]²
  let u = (col + 0.5) / 4.0;
  let v = (row + 0.5) / 4.0;
  return vec2f(u, v);
}

// Decode an octahedral [0,1]² coordinate to a unit hemisphere direction
// (z >= 0 — upper hemisphere only; PPG handles indirect bounces above surface).
// Per Cigolle et al. 2014, "Survey of Efficient Representations for Independent
// Unit Vectors".
fn ppgOctahedralToDir(oct: vec2f) -> vec3f {
  // Map [0,1]² → [−1,1]²
  let f = oct * 2.0 - vec2f(1.0);
  // Decode
  var n = vec3f(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
  // Fold negative hemisphere back to positive (upper hemisphere only)
  let t = max(-n.z, 0.0);
  n.x += select(t, -t, n.x >= 0.0);
  n.y += select(t, -t, n.y >= 0.0);
  return normalize(n);
}

// ppgAxisComp / ppgBruteFindCell / ppgKdFindCellShared come from PPG_COMMON_WGSL
// (injected immediately after this line via the template literal).
${PPG_COMMON_WGSL}

fn ppgFindCellIndex(worldPos: vec3f) -> u32 {
  return ppgKdFindCellShared(worldPos, arrayLength(&ppgCells));
}

// Build CDF (prefix-sum) over the 16 bins.
// Returns the sum of all radianceSums (used for PDF normalisation).
fn ppgBuildCDF(leafIdx: u32, cdf: ptr<function, array<f32, 16>>) -> f32 {
  var total = 0.0;
  for (var b = 0u; b < 16u; b++) {
    total += ppgLeaves[leafIdx].bins[b].x; // radianceSum
    (*cdf)[b] = total;
  }
  return total;
}

// Check whether all bins in a leaf are empty (no path samples yet).
fn ppgLeafIsEmpty(leafIdx: u32) -> bool {
  for (var b = 0u; b < 16u; b++) {
    if (ppgLeaves[leafIdx].bins[b].y > 0.0) { return false; }
  }
  return true;
}

// ============================================================
// Public API
// ============================================================

// Sample a world-space direction from the PPG learned PDF at worldPos.
//
// u1: uniform [0,1) — used to jitter within the sampled bin's solid angle.
// u2: uniform [0,1) — used to select a bin via CDF inversion.
// n:  surface normal at the shading point (for cosine-fallback orientation).
//
// Falls back to cosine-weighted hemisphere sampling when the nearest cell
// has no accumulated path data (cold-start or newly allocated cell).
fn ppgSampleDirection(worldPos: vec3f, n: vec3f, u1: f32, u2: f32,
                      rng: ptr<function, u32>) -> vec3f {
  let cellIdx = ppgFindCellIndex(worldPos);
  let leafIdx = ppgCells[cellIdx].leafIndex;

  // Cold-start fallback: cosine-weighted hemisphere sampling.
  if (ppgLeafIsEmpty(leafIdx)) {
    return sampleCosineHemisphere(n, rng);
  }

  // Build CDF over 16 bins.
  var cdf: array<f32, 16>;
  let total = ppgBuildCDF(leafIdx, &cdf);
  if (total <= 0.0) {
    return sampleCosineHemisphere(n, rng);
  }

  // CDF inversion: binary search for bin.
  let target = u2 * total;
  var selectedBin = 15u;
  for (var b = 0u; b < 16u; b++) {
    if (cdf[b] >= target) {
      selectedBin = b;
      break;
    }
  }

  // Decode bin centre to octahedral UV, then jitter within bin cell.
  let row   = f32(selectedBin / 4u);
  let col   = f32(selectedBin % 4u);
  // Independent jitter in U and V: u1 drives U, a fresh RNG draw drives V.
  // Previously both axes were derived from u1 (different frequencies), which
  // produced correlated samples that visibly clustered inside each bin.
  let jU      = u1;
  let jV      = rand_f32(rng);
  let jitterU = (col + jU) / 4.0;
  let jitterV = (row + jV) / 4.0;
  let octUV   = vec2f(clamp(jitterU, 0.0, 1.0), clamp(jitterV, 0.0, 1.0));

  // Decode to world-space direction (upper hemisphere).
  let localDir = ppgOctahedralToDir(octUV);

  // Rotate from hemisphere-up (0,0,1) to surface normal frame.
  var T: vec3f; var B: vec3f;
  buildONB(n, &T, &B);
  return normalize(localDir.x * T + localDir.y * B + localDir.z * n);
}

// Compute the PDF (solid-angle measure) of direction dir under the PPG
// learned distribution at worldPos.
//
// Returns the PDF in sr⁻¹. Falls back to cosineHemispherePdf when the
// cell is empty (matching the ppgSampleDirection fallback path so that
// MIS weights balance correctly).
fn ppgPDF(worldPos: vec3f, n: vec3f, dir: vec3f) -> f32 {
  let cellIdx = ppgFindCellIndex(worldPos);
  let leafIdx = ppgCells[cellIdx].leafIndex;

  if (ppgLeafIsEmpty(leafIdx)) {
    return cosineHemispherePdf(n, dir);
  }

  // Encode dir back to octahedral UV to find its bin.
  // Project dir onto the normal frame to get local coords.
  var T: vec3f; var B: vec3f;
  buildONB(n, &T, &B);
  let localDir = vec3f(dot(dir, T), dot(dir, B), dot(dir, n));

  // Encode local direction to octahedral [0,1]²
  // (inverse of ppgOctahedralToDir — simplified for unit hemisphere).
  let lenL1 = abs(localDir.x) + abs(localDir.y) + abs(localDir.z);
  var oct: vec2f;
  if (lenL1 > 0.0) {
    oct = localDir.xy / lenL1;
  } else {
    oct = vec2f(0.0);
  }
  // Fold negative z back (upper hemisphere)
  if (localDir.z < 0.0) {
    let tmp = oct;
    oct.x = (1.0 - abs(tmp.y)) * select(-1.0, 1.0, tmp.x >= 0.0);
    oct.y = (1.0 - abs(tmp.x)) * select(-1.0, 1.0, tmp.y >= 0.0);
  }
  // Map [−1,1]² → [0,1]²
  oct = oct * 0.5 + vec2f(0.5);
  oct = clamp(oct, vec2f(0.0), vec2f(1.0));

  // Find bin index.
  let col = u32(oct.x * 4.0);
  let row = u32(oct.y * 4.0);
  let binIdx = clamp(row * 4u + col, 0u, 15u);

  // PDF = (binRadiance / totalRadiance) / binSolidAngle.
  // Each bin covers 1/16 of the upper hemisphere solid angle = 2π/16.
  var cdf: array<f32, 16>;
  let total = ppgBuildCDF(leafIdx, &cdf);
  if (total <= 0.0) {
    return cosineHemispherePdf(n, dir);
  }

  let binRadiance = ppgLeaves[leafIdx].bins[binIdx].x;
  let binProb     = binRadiance / total;
  let binSolidAngle = 2.0 * 3.14159265358979 / 16.0; // 2π / 16 bins
  return binProb / binSolidAngle;
}

`;
