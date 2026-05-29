// nrcEncodeBackward.wgsl.ts — STANDALONE @compute kernel that scatters the
// MLP input gradient dL/dX into the TRAINABLE multiresolution hash-grid feature
// tables (Müller, Evans, Schied, Keller 2022, "Instant Neural Graphics
// Primitives with a Multiresolution Hash Encoding", ACM TOG 41(4) §4; used as
// the cache-query positional encoding in Müller, Rousselle, Novák, Keller 2021,
// "Real-time Neural Radiance Caching for Path Tracing", ACM TOG 40(4) §4).
//
// WHY THIS KERNEL EXISTS — the missing half of "the tables actually learn"
// --------------------------------------------------------------------------
// The fused MLP backward (fusedMlp.wgsl.ts) now emits dL/dX — the gradient w.r.t.
// the RAW (padded) network input — into `gradInputF`. The first L·F entries of
// each sample's dL/dX row are exactly dL/dfeature for the hash-grid encode (the
// encoding lays the L·F hash-grid features at the FRONT of the input vector;
// see nrcEncoding.ts assembleNrcInput). This kernel takes those, recomputes the
// 8 trilinear corner weights + hashed rows from the STORED query world position,
// and scatters `weight · dL/dfeature` into the per-level gradient table via i32
// fixed-point atomics — the EXACT mirror of nrcEncoding.ts hashGridBackward.
// Collisions onto the same hashed row ACCUMULATE (Instant-NGP §4); that is the
// defining behaviour of the learned hash encoding, and the atomicAdd gives it.
//
// MODULE-SCOPE STORAGE (the WGSL-pointer trap):
//   WGSL forbids passing a `var<storage>` resource to a function pointer
//   parameter (without the non-core unrestricted_pointer_parameters feature). So
//   `gradTablesFx` is bound at MODULE scope and the trilinear scatter is INLINED
//   in the entry point — we do NOT factor it into a helper taking the storage
//   buffer by ptr (that is what nrcEncoding.wgsl.ts's nrcHashLevelBackward did,
//   and it is unusable from a real dispatch for exactly this reason). The hash +
//   trilinear arithmetic is duplicated here so it stays byte-identical to the
//   forward; both are pinned to the CPU oracle by the tests.
//
// LAYOUT CONTRACT (must match nrcSubsystem + nrcQuery):
//   posBuf      : [numSamples × 3]  query world positions (densely packed, the
//                 same sample ordering as gradInputF / the trainer batch).
//   gradInputF  : [numSamples × inW] finalized dL/dX from the MLP backward; this
//                 kernel reads only the first L·F columns of each row.
//   nrcLevels   : per-level descriptor (resolution, tableSize, scalar offset).
//   gradTablesFx: one atomic<i32> per table feature scalar, same layout as the
//                 forward feature tables; the host divides out NRC_GRAD_FP after.
//   uniforms    : aabbMin/Max (normalisation), numActive (densely-packed count),
//                 inW (dL/dX row stride), L, F.

export interface NrcEncodeBackwardWgslOptions {
  /** Hash-grid resolution levels L. */
  levels: number;
  /** Features per table entry F. */
  featuresPerEntry: number;
  /** Raw encoded MLP input width inW (= dL/dX row stride). The first L·F columns
   *  are the hash-grid features this kernel scatters. */
  inWidth: number;
}

/**
 * Emit the standalone encode-backward compute kernel. One invocation per active
 * (densely-packed) training sample.
 */
export function nrcEncodeBackwardWgsl(o: NrcEncodeBackwardWgslOptions): string {
  const L = o.levels, F = o.featuresPerEntry, IN_W = o.inWidth;
  return /* wgsl */`
// fixed-point scale for the i32 grad atomics — MUST match the host's divisor and
// the fusedMlp / nrcEncoding.wgsl scale (2^20).
const NRC_GRAD_FP : f32 = 1048576.0;
const NRC_LEVELS  : u32 = ${L}u;
const NRC_FEAT    : u32 = ${F}u;
const NRC_IN_W    : u32 = ${IN_W}u;  // dL/dX row stride

struct NrcLevelDesc {
  resolution:  u32,
  tableSize:   u32,
  tableOffset: u32,  // scalar offset of this level's table in the (grad) tables
  _pad:        u32,
}

struct EncBwdParams {
  aabbMin    : vec3f,
  numActive  : u32,    // densely-packed active sample count (only these scatter)
  aabbMax    : vec3f,
  _pad0      : u32,
}

@group(0) @binding(0) var<storage, read>       posBuf       : array<f32>;        // [numActive × 3]
@group(0) @binding(1) var<storage, read>       gradInputF   : array<f32>;        // [numSamples × inW]
@group(0) @binding(2) var<storage, read>       nrcLevels    : array<NrcLevelDesc>;
@group(0) @binding(3) var<storage, read_write> gradTablesFx : array<atomic<i32>>; // fixed-point
@group(0) @binding(4) var<uniform>             p            : EncBwdParams;

// Instant-NGP spatial hash (Müller 2022 §3 Eq.4). u32 wraparound matches the CPU
// oracle (Math.imul / >>>0) and the forward (nrcQuery.wgsl nrcSpatialHash3D) EXACTLY.
fn ebSpatialHash3D(ix: u32, iy: u32, iz: u32, tableSize: u32) -> u32 {
  let h: u32 = (ix * 1u) ^ (iy * 2654435761u) ^ (iz * 805459861u);
  return h % tableSize;
}

fn ebNormalizeToAabb(pos: vec3f, aabbMin: vec3f, aabbMax: vec3f) -> vec3f {
  let ext = aabbMax - aabbMin;
  let safeExt = max(ext, vec3f(1e-20));
  return clamp((pos - aabbMin) / safeExt, vec3f(0.0), vec3f(1.0));
}

@compute @workgroup_size(64, 1, 1)
fn nrcEncodeBackward(@builtin(global_invocation_id) gid : vec3<u32>) {
  let s = gid.x;
  if (s >= p.numActive) { return; }

  let pos = vec3f(posBuf[s * 3u + 0u], posBuf[s * 3u + 1u], posBuf[s * 3u + 2u]);
  let nrm = ebNormalizeToAabb(pos, p.aabbMin, p.aabbMax);
  let rowBase = s * NRC_IN_W;   // dL/dX row for this sample

  // For each level: recompute the 8 trilinear corners and scatter weight·dOut[f]
  // into the hashed row. dOut for level l, feature f is dL/dX[ l·F + f ] (the
  // hash-grid features occupy feat[0 .. L·F-1]). INLINED scatter (no ptr arg).
  for (var l: u32 = 0u; l < NRC_LEVELS; l = l + 1u) {
    let desc = nrcLevels[l];
    let N = f32(desc.resolution);
    let pp = nrm * N;
    let i0 = vec3u(u32(floor(pp.x)), u32(floor(pp.y)), u32(floor(pp.z)));
    let frac = pp - floor(pp);
    let outBase = l * NRC_FEAT;
    for (var c: u32 = 0u; c < 8u; c = c + 1u) {
      let cx = (c & 1u);
      let cy = (c >> 1u) & 1u;
      let cz = (c >> 2u) & 1u;
      let wx = select(1.0 - frac.x, frac.x, cx == 1u);
      let wy = select(1.0 - frac.y, frac.y, cy == 1u);
      let wz = select(1.0 - frac.z, frac.z, cz == 1u);
      let weight = wx * wy * wz;
      let row = ebSpatialHash3D(i0.x + cx, i0.y + cy, i0.z + cz, desc.tableSize);
      let rb = desc.tableOffset + row * NRC_FEAT;
      for (var f: u32 = 0u; f < NRC_FEAT; f = f + 1u) {
        let g = weight * gradInputF[rowBase + outBase + f];
        atomicAdd(&gradTablesFx[rb + f], i32(g * NRC_GRAD_FP));
      }
    }
  }
}
`;
}
