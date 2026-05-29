// nrcEncoding.wgsl.ts — WGSL forward of the NRC input encoding.
//
// GPU mirror of the CPU oracle `../nrcEncoding.ts`. Emits the multiresolution
// hash-grid positional encode (Müller et al. 2022 Instant-NGP §3), the one-blob
// direction encode (Müller et al. 2019 §4.3), and the raw-feature concat — the
// exact arithmetic the host-side oracle computes, so the two are pinned to ~1e-6
// (f32) by `__tests__/nrcEncoding.test.ts`.
//
// This file emits the FORWARD encode (cache query + the forward half of
// training) AND the trainable hash-grid BACKWARD (`nrcEncodeBackwardWgsl`,
// gradient scatter into hashed table rows). STATUS (verified 2026-05-29):
//   • FORWARD — WIRED. `nrcEncodeHelpersWgsl` is composed into the dispatched
//     gi-ris NRC variant (`buildRisGiNrcModule`); the query runs + gathers
//     self-training records, and `NrcSubsystem.trainFromRecords` runs one MLP
//     `trainStep` per frame (host-owns-cadence) when `nrcEnabled`.
//   • hash-grid BACKWARD — NOT wired. `nrcEncodeBackwardWgsl` is emitted +
//     unit-pinned but never dispatched, so the hash-grid feature tables stay
//     frozen at their random init (`NrcSubsystem._tablesBuf` is write-once).
//     The MLP learns over a fixed positional embedding; the multiresolution
//     encoding itself does not yet learn. Wiring this scatter into the train
//     step is the remaining NRC work.
//
// Hash + interpolation conventions MUST match nrcEncoding.ts exactly:
//   * spatialHash3D: (ix·1 ^ iy·0x9E3779B1 ^ iz·0x30034BB7) mod tableSize, u32 wrap.
//   * trilinear: 8 corners, weight = ∏ axis(frac | 1-frac), Σ weights = 1.
//   * one-blob: k Gaussian bins centred at (i+0.5)/k, L1-normalised.
//
// COMPOSITION NOTE: the helper signatures take `array<f32, NRC_MAX_LF>`
// / `array<f32, NRC_MAX_BLOB>` scratch by pointer. The pass composing these
// modules emits the matching `const NRC_MAX_LF : u32 = <L·F>;` and
// `const NRC_MAX_BLOB : u32 = <k>;` (and `let F : u32` from the config) ahead of
// the helpers — these emitters are deliberately config-agnostic on those sizes
// so the pass picks them from the live encoding config. The FORWARD helpers are
// composed into the dispatched gi-ris NRC variant today; only the hash-grid
// BACKWARD scatter (above) remains undispatched.

export interface NrcEncodeWgslOptions {
  /** Hash-grid resolution levels L. */
  levels: number;
  /** Features per table entry F. */
  featuresPerEntry: number;
  /** One-blob bins k (per encoded scalar; direction → 2 scalars → 2k). */
  oneBlobBins: number;
}

// Shared hash + trilinear + one-blob helpers. Kept as a separate emitter so the
// forward and backward modules both include them without duplication.
export function nrcEncodeHelpersWgsl(): string {
  return /* wgsl */`
// Instant-NGP spatial hash (Müller 2022 §3 Eq.4). u32 wraparound matches the
// CPU oracle's Math.imul / >>>0 emulation exactly.
fn nrcSpatialHash3D(ix: u32, iy: u32, iz: u32, tableSize: u32) -> u32 {
  // x prime is 1 (no multiply), y = 0x9E3779B1, z = 0x30034BB7.
  let h: u32 = (ix * 1u) ^ (iy * 2654435761u) ^ (iz * 805459861u);
  return h % tableSize;
}

// Normalise a world position into the grid AABB → [0,1]^3 (clamped).
fn nrcNormalizeToAabb(pos: vec3f, aabbMin: vec3f, aabbMax: vec3f) -> vec3f {
  let ext = aabbMax - aabbMin;
  let safeExt = max(ext, vec3f(1e-20));
  return clamp((pos - aabbMin) / safeExt, vec3f(0.0), vec3f(1.0));
}

// One-blob encode one scalar u∈[0,1] into k Gaussian bins, L1-normalised
// (Müller 2019 §4.3). Writes to out[base + 0 .. base + k-1].
fn nrcOneBlobScalar(out: ptr<function, array<f32, NRC_MAX_BLOB>>, base: u32, u: f32, k: u32, sigma: f32) {
  let uc = clamp(u, 0.0, 1.0);
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < k; i = i + 1u) {
    let center = (f32(i) + 0.5) / f32(k);
    let d = (uc - center) / sigma;
    let a = exp(-0.5 * d * d);
    (*out)[i] = a;
    sum = sum + a;
  }
  if (sum > 1e-20) {
    for (var i: u32 = 0u; i < k; i = i + 1u) {
      (*out)[i] = (*out)[i] / sum;
    }
  }
}
`;
}

/**
 * Forward hash-grid encode kernel body. Reads the per-level feature tables
 * (concatenated, row-major) and writes the L·F level features for one query.
 * Caller supplies the normalised query + per-level resolution/tableSize/offset.
 *
 * Layout of `tables` (storage, read): all levels' feature tables concatenated
 * by `levelTableOffset[l]` (in FEATURE-scalar units), row-major [row × F].
 */
export function nrcHashGridForwardWgsl(_o: NrcEncodeWgslOptions): string {
  return /* wgsl */`
struct NrcLevelDesc {
  resolution:  u32,
  tableSize:   u32,
  tableOffset: u32,  // scalar offset of this level's table in nrcTables
  _pad:        u32,
}

// Trilinear-interpolate the F features of the 8 hashed corners at one level and
// write to out[outBase + f]. EXACT mirror of nrcEncoding.ts trilinearCorners +
// hashGridForward.
fn nrcHashLevelForward(
  nrm: vec3f,            // query normalised to [0,1]^3
  desc: NrcLevelDesc,
  F: u32,
  tables: ptr<storage, array<f32>, read>,
  out: ptr<function, array<f32, NRC_MAX_LF>>,
  outBase: u32,
) {
  let N = f32(desc.resolution);
  let p = nrm * N;
  let i0 = vec3u(u32(floor(p.x)), u32(floor(p.y)), u32(floor(p.z)));
  let frac = p - floor(p);
  for (var f: u32 = 0u; f < F; f = f + 1u) { (*out)[outBase + f] = 0.0; }
  for (var c: u32 = 0u; c < 8u; c = c + 1u) {
    let cx = (c & 1u);
    let cy = (c >> 1u) & 1u;
    let cz = (c >> 2u) & 1u;
    let wx = select(1.0 - frac.x, frac.x, cx == 1u);
    let wy = select(1.0 - frac.y, frac.y, cy == 1u);
    let wz = select(1.0 - frac.z, frac.z, cz == 1u);
    let weight = wx * wy * wz;
    let row = nrcSpatialHash3D(i0.x + cx, i0.y + cy, i0.z + cz, desc.tableSize);
    let rb = desc.tableOffset + row * F;
    for (var f: u32 = 0u; f < F; f = f + 1u) {
      (*out)[outBase + f] = (*out)[outBase + f] + weight * (*tables)[rb + f];
    }
  }
}
`;
}

/**
 * Backward hash-grid: scatter dL/dfeature into the per-level gradient tables via
 * atomic fixed-point add (the same i32-atomic discipline the fused MLP kernel
 * uses, since WGSL has no f32 atomics — collisions onto the same row ACCUMULATE,
 * Instant-NGP §4). EXACT mirror of nrcEncoding.ts hashGridBackward.
 *
 * `gradTablesFx` holds one atomic<i32> per table feature scalar (same layout as
 * the forward `nrcTables`); the host divides out NRC_GRAD_FP after the pass.
 */
export function nrcHashGridBackwardWgsl(_o: NrcEncodeWgslOptions): string {
  return /* wgsl */`
const NRC_GRAD_FP: f32 = 1048576.0;  // 2^20 fixed-point (matches fusedMlp grad atomics)

fn nrcHashLevelBackward(
  nrm: vec3f,
  desc: NrcLevelDesc,
  F: u32,
  dOut: ptr<function, array<f32, NRC_MAX_LF>>,
  outBase: u32,
  gradTablesFx: ptr<storage, array<atomic<i32>>, read_write>,
) {
  let N = f32(desc.resolution);
  let p = nrm * N;
  let i0 = vec3u(u32(floor(p.x)), u32(floor(p.y)), u32(floor(p.z)));
  let frac = p - floor(p);
  for (var c: u32 = 0u; c < 8u; c = c + 1u) {
    let cx = (c & 1u);
    let cy = (c >> 1u) & 1u;
    let cz = (c >> 2u) & 1u;
    let wx = select(1.0 - frac.x, frac.x, cx == 1u);
    let wy = select(1.0 - frac.y, frac.y, cy == 1u);
    let wz = select(1.0 - frac.z, frac.z, cz == 1u);
    let weight = wx * wy * wz;
    let row = nrcSpatialHash3D(i0.x + cx, i0.y + cy, i0.z + cz, desc.tableSize);
    let rb = desc.tableOffset + row * F;
    for (var f: u32 = 0u; f < F; f = f + 1u) {
      let g = weight * (*dOut)[outBase + f];
      atomicAdd(&(*gradTablesFx)[rb + f], i32(g * NRC_GRAD_FP));
    }
  }
}
`;
}
