// nrcQuery.wgsl.ts — INLINE single-sample NRC cache forward (query) for the
// gi-ris reconnection-vertex suffix.
//
// Müller, Rousselle, Novák, Keller 2021, "Real-time Neural Radiance Caching for
// Path Tracing", ACM TOG 40(4). At a path's suffix vertex (where the spread
// heuristic §5 fires) the path is TERMINATED into the cache: the MLP predicts
// the outgoing radiance Lo there, replacing the unbounded path-suffix integral.
//
// WHY A SEPARATE FORWARD (vs reusing the fused trainer kernel):
// `fusedMlp.wgsl.ts` is a BATCH training kernel — one workgroup per tile, with
// the activation tensor resident in workgroup SHARED memory and ping-ponged
// across layers. That layout is unusable from inside the gi-ris compute shader,
// where each invocation must evaluate the MLP for its OWN single query vertex
// with no cross-invocation coordination. This module is the per-thread inline
// forward: it reads the SAME concatenated weight/bias buffers the trainer owns
// (so the network the trainer learns is the network the query evaluates) and
// the SAME hash-grid feature tables, runs the multiresolution hash-grid +
// one-blob encode (mirroring `nrcEncoding.ts` / `nrcEncoding.wgsl.ts`), then a
// dense ReLU MLP forward into the OUT_W=3 RGB radiance prediction.
//
// ════════════════════════════════════════════════════════════════════════════
// COMPILE-TIME STRUCTURAL GATE
// ════════════════════════════════════════════════════════════════════════════
// This module + its @group(4) bindings are composed into gi-ris ONLY when the
// host opts into `nrcEnabled` (a compile-time engine-creation flag). When OFF
// (the default), the gi-ris pipeline is byte-for-byte the pre-NRC pass — NO
// @group(4), NO NRC symbols, NO layout delta. This mirrors the GRIS
// (`restirPtReuse`) compile-time gate; a runtime-only UBO flag that bound an
// extra group on the default path is exactly what regressed the default
// walkaround render to a black frame (f8df9a4). The structure is gated at
// compile time so the default pipeline is provably untouched.
//
// LAYER PLAN (must match FusedMlpTrainer.planLayers): the raw encoded input is
// padded to W for the first GEMM, then HIDDEN W→W ReLU layers, then a W→OUT_W
// linear output. wOff/bOff/inW/outW per weight layer are emitted as constants by
// the WGSL builder so the inline loop reads the concatenated buffers at exactly
// the offsets the trainer wrote.

import type { NrcEncodeWgslOptions } from './nrcEncoding.wgsl.js';

export interface NrcQueryWgslOptions extends NrcEncodeWgslOptions {
  /** Hidden width W (Müller: 64). */
  width: number;
  /** Output width (RGB radiance: 3). */
  outWidth: number;
  /** Hidden node-layers (Müller: 6). */
  hidden: number;
}

/** Per-weight-layer offset plan, mirroring FusedMlpTrainer.planLayers exactly:
 *  raw input padded to W → HIDDEN×(W→W) → (W→OUT_W). */
interface QueryLayerPlan {
  wOff: number[]; bOff: number[]; inW: number[]; outW: number[];
  wlayers: number; totalW: number; totalB: number;
  /** Padded MLP input width (= W). */
  paddedInW: number;
}

export function nrcQueryLayerPlan(o: NrcQueryWgslOptions): QueryLayerPlan {
  const W = o.width, outW = o.outWidth, hidden = o.hidden;
  const widths: number[] = [W];
  for (let h = 0; h < hidden; h++) widths.push(W);
  widths.push(outW);
  const wlayers = widths.length - 1;
  const wOff: number[] = [], bOff: number[] = [], li: number[] = [], lo: number[] = [];
  let tw = 0, tb = 0;
  for (let l = 0; l < wlayers; l++) {
    const i = widths[l]!, ow = widths[l + 1]!;
    wOff.push(tw); bOff.push(tb); li.push(i); lo.push(ow);
    tw += ow * i; tb += ow;
  }
  return { wOff, bOff, inW: li, outW: lo, wlayers, totalW: tw, totalB: tb, paddedInW: W };
}

/**
 * Emit the inline NRC query helpers + the @group(4) NRC bindings.
 *
 * Bindings (group 4 — NRC-only, present ONLY when nrcEnabled is compile-time on):
 *   0 — nrcWeights      (read-only storage, f32) — concatenated weight matrices
 *   1 — nrcBiases       (read-only storage, f32) — concatenated biases
 *   2 — nrcTables       (read-only storage, f32) — hash-grid feature tables
 *   3 — nrcLevels       (read-only storage, NrcLevelDesc) — per-level descriptors
 *   4 — nrcRecords      (read_write storage, f32) — self-training record gather
 *   5 — nrcCfg          (uniform, NrcCfgUBO) — encoding + record params
 *
 * The encoded-input width and MLP sizes are baked as constants from the config
 * passed to the builder, so the inline loop reads the trainer's concatenated
 * buffers at exactly the offsets the trainer wrote (FusedMlpTrainer.planLayers).
 */
export function nrcQueryWgsl(o: NrcQueryWgslOptions): string {
  const plan = nrcQueryLayerPlan(o);
  const L = o.levels, F = o.featuresPerEntry, K = o.oneBlobBins, W = o.width;
  const inWidth = L * F + 2 * K + 7; // hash-grid + one-blob(u,v) + normal(3)+rough(1)+albedo(3)
  // Per-weight-layer offset constants (mirror the trainer's concatenated layout).
  const wOffArr = plan.wOff.join('u, ') + 'u';
  const bOffArr = plan.bOff.join('u, ') + 'u';
  const inWArr = plan.inW.join('u, ') + 'u';
  const outWArr = plan.outW.join('u, ') + 'u';
  // One-blob sigma default (Müller 2019 §4.3): one cell, σ = 1/bins.
  const sigma = 1 / K;
  return /* wgsl */`
// ── NRC query config + sizes (baked from the encoding config) ──
const NRC_LEVELS    : u32 = ${L}u;
const NRC_FEAT      : u32 = ${F}u;   // features per hash-grid entry
const NRC_BLOB_BINS : u32 = ${K}u;
const NRC_IN_W      : u32 = ${inWidth}u; // raw encoded input width
const NRC_MAX_LF    : u32 = ${L * F}u;
const NRC_MAX_BLOB  : u32 = ${K}u;
const NRC_W         : u32 = ${W}u;       // MLP hidden width (padded input width)
const NRC_OUT_W     : u32 = ${o.outWidth}u;
const NRC_WLAYERS   : u32 = ${plan.wlayers}u;
const NRC_BLOB_SIGMA: f32 = ${sigma};
const NRC_TWO_PI    : f32 = 6.2831853;

// Concatenated weight-layer offsets (mirror FusedMlpTrainer.planLayers).
const NRC_WOFF  = array<u32, ${plan.wlayers}>(${wOffArr});
const NRC_BOFF  = array<u32, ${plan.wlayers}>(${bOffArr});
const NRC_LINW  = array<u32, ${plan.wlayers}>(${inWArr});
const NRC_LOUTW = array<u32, ${plan.wlayers}>(${outWArr});

// Per-level hash-grid descriptor (resolution, table rows, scalar table offset).
// Defined here (not pulled from nrcHashGridForwardWgsl) because that module's
// forward helper takes a storage POINTER argument, which WGSL forbids; the
// inline forward below reads nrcTables directly. Mirrors nrcEncoding.ts level.
struct NrcLevelDesc {
  resolution:  u32,
  tableSize:   u32,
  tableOffset: u32,
  _pad:        u32,
}

struct NrcCfgUBO {
  aabbMin    : vec3f,
  spreadC    : f32,    // Müller §5 termination constant c
  aabbMax    : vec3f,
  recordCap  : u32,    // max training records this frame (record buffer capacity)
  recordStride : u32,  // f32s per record = NRC_IN_W + OUT_W
  _pad0 : u32, _pad1 : u32, _pad2 : u32,
}

@group(4) @binding(0) var<storage, read>       nrcWeights : array<f32>;
@group(4) @binding(1) var<storage, read>       nrcBiases  : array<f32>;
@group(4) @binding(2) var<storage, read>       nrcTables  : array<f32>;
@group(4) @binding(3) var<storage, read>       nrcLevels  : array<NrcLevelDesc>;
// Record-gather buffer. Layout per record: [0]=count-claim header is at index 0
// of the buffer (atomic via the dedicated counter buffer is overkill at half-res
// scale; we instead use a deterministic per-pixel slot = pixelIdxGi % recordCap,
// which is race-free because each invocation owns one pixel). Records are
// [NRC_IN_W encoded input | OUT_W radiance target]. A record with target == 0
// across all channels is treated as empty by the host gather.
@group(4) @binding(4) var<storage, read_write> nrcRecords : array<f32>;
@group(4) @binding(5) var<uniform>             nrcCfg     : NrcCfgUBO;

// ── One-blob encode of a scalar into NRC_BLOB_BINS bins, L1-normalised. Writes
// the bins into a function-local scratch (exact mirror of nrcEncoding.ts). ──
fn nrcOneBlob(u: f32, out: ptr<function, array<f32, ${K}>>) {
  let uc = clamp(u, 0.0, 1.0);
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < NRC_BLOB_BINS; i = i + 1u) {
    let center = (f32(i) + 0.5) / f32(NRC_BLOB_BINS);
    let d = (uc - center) / NRC_BLOB_SIGMA;
    let a = exp(-0.5 * d * d);
    (*out)[i] = a;
    sum = sum + a;
  }
  if (sum > 1e-20) {
    for (var i: u32 = 0u; i < NRC_BLOB_BINS; i = i + 1u) { (*out)[i] = (*out)[i] / sum; }
  }
}

// Octahedral encode of a unit direction to (u,v) ∈ [0,1]² (mirrors
// nrcEncoding.ts octEncodeDir; matches the canonical octEncode sign convention).
fn nrcOctEncodeDir(d: vec3f) -> vec2f {
  let a = abs(d);
  let s = a.x + a.y + a.z;
  let inv = select(0.0, 1.0 / s, s > 1e-20);
  var p = d.xy * inv;
  if (d.z < 0.0) {
    let sx = select(-1.0, 1.0, p.x >= 0.0);
    let sy = select(-1.0, 1.0, p.y >= 0.0);
    let ox = (1.0 - abs(p.y)) * sx;
    let oy = (1.0 - abs(p.x)) * sy;
    p = vec2f(ox, oy);
  }
  return p * 0.5 + vec2f(0.5);
}

// Trilinear hash-grid level forward, INLINED here (not a helper) because WGSL
// forbids passing a storage pointer (nrcTables) as a function argument without
// the unrestricted_pointer_parameters feature. Reads the module-scope nrcTables
// directly. EXACT mirror of nrcEncoding.ts trilinearCorners + hashGridForward /
// nrcEncoding.wgsl.ts nrcHashLevelForward.
fn nrcHashLevelForwardInline(
  nrm: vec3f, desc: NrcLevelDesc, outBase: u32,
  feat: ptr<function, array<f32, NRC_IN_W>>,
) {
  let N = f32(desc.resolution);
  let p = nrm * N;
  let i0 = vec3u(u32(floor(p.x)), u32(floor(p.y)), u32(floor(p.z)));
  let frac = p - floor(p);
  for (var f: u32 = 0u; f < NRC_FEAT; f = f + 1u) { (*feat)[outBase + f] = 0.0; }
  for (var c: u32 = 0u; c < 8u; c = c + 1u) {
    let cx = (c & 1u);
    let cy = (c >> 1u) & 1u;
    let cz = (c >> 2u) & 1u;
    let wx = select(1.0 - frac.x, frac.x, cx == 1u);
    let wy = select(1.0 - frac.y, frac.y, cy == 1u);
    let wz = select(1.0 - frac.z, frac.z, cz == 1u);
    let weight = wx * wy * wz;
    let row = nrcSpatialHash3D(i0.x + cx, i0.y + cy, i0.z + cz, desc.tableSize);
    let rb = desc.tableOffset + row * NRC_FEAT;
    for (var f: u32 = 0u; f < NRC_FEAT; f = f + 1u) {
      (*feat)[outBase + f] = (*feat)[outBase + f] + weight * nrcTables[rb + f];
    }
  }
}

// Assemble the full NRC MLP input vector into feat (length NRC_IN_W):
//   [ hash-grid(pos) | one-blob(octU) | one-blob(octV) | normal | rough | albedo ]
fn nrcAssembleInput(
  pos: vec3f, normal: vec3f, dir: vec3f, roughness: f32, albedo: vec3f,
  feat: ptr<function, array<f32, NRC_IN_W>>,
) {
  // Hash-grid positional encode → first NRC_LEVELS·NRC_FEAT entries (written
  // directly into feat[0 .. L·F-1] by the inlined per-level trilinear forward).
  let nrm = nrcNormalizeToAabb(pos, nrcCfg.aabbMin, nrcCfg.aabbMax);
  for (var l: u32 = 0u; l < NRC_LEVELS; l = l + 1u) {
    nrcHashLevelForwardInline(nrm, nrcLevels[l], l * NRC_FEAT, feat);
  }
  var o: u32 = NRC_LEVELS * NRC_FEAT;
  // One-blob of octahedral (u,v).
  let oct = nrcOctEncodeDir(dir);
  var bu: array<f32, ${K}>; var bv: array<f32, ${K}>;
  nrcOneBlob(oct.x, &bu);
  nrcOneBlob(oct.y, &bv);
  for (var i: u32 = 0u; i < NRC_BLOB_BINS; i = i + 1u) { (*feat)[o] = bu[i]; o = o + 1u; }
  for (var i: u32 = 0u; i < NRC_BLOB_BINS; i = i + 1u) { (*feat)[o] = bv[i]; o = o + 1u; }
  // Raw surface features.
  (*feat)[o] = normal.x; o = o + 1u;
  (*feat)[o] = normal.y; o = o + 1u;
  (*feat)[o] = normal.z; o = o + 1u;
  (*feat)[o] = roughness; o = o + 1u;
  (*feat)[o] = albedo.x; o = o + 1u;
  (*feat)[o] = albedo.y; o = o + 1u;
  (*feat)[o] = albedo.z; o = o + 1u;
}

// ── Inline dense MLP forward (single sample). Reads the trainer's concatenated
// weight/bias buffers; ReLU hidden, linear OUT_W output. Activations live in two
// function-local W-wide ping-pong buffers (no workgroup shared memory). ──
fn nrcMlpForward(feat: ptr<function, array<f32, NRC_IN_W>>) -> vec3f {
  var actA: array<f32, ${W}>;
  var actB: array<f32, ${W}>;
  // Node-layer 0: padded raw input (zero-pad beyond NRC_IN_W).
  for (var i: u32 = 0u; i < NRC_W; i = i + 1u) {
    actA[i] = select(0.0, (*feat)[i], i < NRC_IN_W);
  }
  for (var l: u32 = 0u; l < NRC_WLAYERS; l = l + 1u) {
    let inW = NRC_LINW[l];
    let outW = NRC_LOUTW[l];
    let wo = NRC_WOFF[l];
    let bo = NRC_BOFF[l];
    let isOut = (l == NRC_WLAYERS - 1u);
    for (var ocol: u32 = 0u; ocol < outW; ocol = ocol + 1u) {
      var acc: f32 = nrcBiases[bo + ocol];
      let wBase = wo + ocol * inW;
      if ((l & 1u) == 0u) {
        for (var i: u32 = 0u; i < inW; i = i + 1u) { acc = acc + nrcWeights[wBase + i] * actA[i]; }
      } else {
        for (var i: u32 = 0u; i < inW; i = i + 1u) { acc = acc + nrcWeights[wBase + i] * actB[i]; }
      }
      let a = select(max(0.0, acc), acc, isOut);
      if ((l & 1u) == 0u) { actB[ocol] = a; } else { actA[ocol] = a; }
    }
    // Zero the unused destination columns so a stale lane never pollutes the
    // next layer's dot product (matches the fused kernel's zeroing).
    if ((l & 1u) == 0u) {
      for (var c: u32 = outW; c < NRC_W; c = c + 1u) { actB[c] = 0.0; }
    } else {
      for (var c: u32 = outW; c < NRC_W; c = c + 1u) { actA[c] = 0.0; }
    }
  }
  // Output node-layer parity: NRC_WLAYERS writes alternate; the final write lands
  // in actB iff (NRC_WLAYERS-1) is even.
  let outInB = (((NRC_WLAYERS - 1u) & 1u) == 0u);
  if (outInB) {
    return vec3f(actB[0], actB[1], actB[2]);
  }
  return vec3f(actA[0], actA[1], actA[2]);
}

// Predict outgoing radiance at the suffix vertex (the cache QUERY). The MLP
// predicts radiance directly; we clamp to non-negative (radiance ≥ 0).
fn nrcQueryRadiance(pos: vec3f, normal: vec3f, viewDir: vec3f, roughness: f32, albedo: vec3f) -> vec3f {
  var feat: array<f32, NRC_IN_W>;
  nrcAssembleInput(pos, normal, viewDir, roughness, albedo, &feat);
  return max(nrcMlpForward(&feat), vec3f(0.0));
}

// Write one self-training record for the host gather. Deterministic per-pixel
// slot (pixelIdxGi % recordCap) — each invocation owns exactly one slot, so no
// atomics / data race. target is the radiance the path actually accumulated at
// this suffix vertex (Müller §5 self-training target). The input is re-assembled
// identically to the query so the trainer fits the SAME encoding it queried.
fn nrcWriteRecord(
  slot: u32, pos: vec3f, normal: vec3f, viewDir: vec3f, roughness: f32, albedo: vec3f,
  tgt: vec3f,
) {
  if (slot >= nrcCfg.recordCap) { return; }
  var feat: array<f32, NRC_IN_W>;
  nrcAssembleInput(pos, normal, viewDir, roughness, albedo, &feat);
  let base = slot * nrcCfg.recordStride;
  for (var i: u32 = 0u; i < NRC_IN_W; i = i + 1u) { nrcRecords[base + i] = feat[i]; }
  nrcRecords[base + NRC_IN_W + 0u] = tgt.x;
  nrcRecords[base + NRC_IN_W + 1u] = tgt.y;
  nrcRecords[base + NRC_IN_W + 2u] = tgt.z;
}
`;
}
