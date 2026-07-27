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
// (`grisReuse`) compile-time gate; a runtime-only UBO flag that bound an
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
import { NRC_DIAGNOSTIC_CONSTANTS_WGSL } from '../nrcDiagnostics.js';
import {
  NRC_INFERENCE_ARENA_MAGIC,
  NRC_INFERENCE_ARENA_SCHEMA,
  NRC_INFERENCE_ARENA_VERSION,
  NRC_INFERENCE_HEADER_FIELD,
  NRC_RUNTIME_ARENA_MAGIC,
  NRC_RUNTIME_ARENA_SCHEMA,
  NRC_RUNTIME_ARENA_VERSION,
  NRC_RUNTIME_HEADER_FIELD,
  NRC_RUNTIME_HEADER_WORD_OFFSET,
} from '../nrcArena.js';

export interface NrcQueryWgslOptions extends NrcEncodeWgslOptions {
  /** Hidden width W (Müller: 64). */
  width: number;
  /** Output width (RGB radiance: 3). */
  outWidth: number;
  /** Hidden node-layers (Müller: 6). */
  hidden: number;
  /** Bind-group index for the NRC bindings. Default 4 — the real gi-ris pipeline
   *  binds NRC as the 5th group (full-tier maxBindGroups). Isolated harnesses
   *  (e.g. nrcQueryHarness.ts) pass 0 to fit lavapipe's default maxBindGroups=4;
   *  the binding NUMBERS are unchanged. Production callers omit it, so the
   *  emitted production WGSL is byte-identical to the pre-option literal. */
  group?: number;
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
 *   6 — nrcSlotClaims   (read_write storage, atomic<u32>) — per-slot first-writer
 *                        claim flags (H27: one u32 per recordCap slot; 0=unclaimed,
 *                        1=claimed). Cleared each frame by the host. Prevents torn
 *                        records when two invocations race for the same slot.
 *
 * The encoded-input width and MLP sizes are baked as constants from the config
 * passed to the builder, so the inline loop reads the trainer's concatenated
 * buffers at exactly the offsets the trainer wrote (FusedMlpTrainer.planLayers).
 */
// MUST-MATCH (D7.7): the emitted `nrcHashLevelForwardInline` below carries an
// inline copy of the 8-corner trilinear loop (i0/frac/wx·wy·wz/hash-row) — WGSL
// forbids a shared helper taking the tables storage buffer by pointer. The same
// loop is mirrored at
//   • nrcEncoding.wgsl.ts       nrcHashLevelForward (forward, ptr-arg tables)
//   • nrcEncodeBackward.wgsl.ts inlined scatter in nrcEncodeBackward (backward)
//   • nrcEncoding.ts            trilinearCorners/hashGridForward (CPU oracle)
// Change one → change ALL FOUR; the tests pin each against the CPU oracle.
export function nrcQueryWgsl(o: NrcQueryWgslOptions): string {
  const plan = nrcQueryLayerPlan(o);
  const L = o.levels, F = o.featuresPerEntry, K = o.oneBlobBins, W = o.width;
  const G = o.group ?? 3;
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
// Defined here because the live query directly owns its module-scope table binding;
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
  spreadC    : f32,           // Müller §5 termination constant c
  aabbMax    : vec3f,
  recordCap  : u32,           // max training records this frame (record buffer capacity)
  recordStride    : u32,      // f32s per record = NRC_IN_W + OUT_W + 3 world-position floats
  // H26 — camera per-pixel solid-angle pdf (host-updated every frame).
  // Centre-pixel pinhole approximation: pdf = |fx·fy| · W · H / 4.
  // Used as the primary-edge pdf in nrcSegmentSpreadTerm so a0 accounts for
  // projection aspect and resolution instead of the old hard-coded 1.0.
  cameraPixelPdf  : f32,
  trainedSteps    : u32,      // completed host trainer windows
  warmupSteps     : u32,      // query substitution gate; records still gather below this
}

@group(${G}) @binding(7) var<storage, read> nrcInferenceArena : array<u32>;
// Record-gather buffer. The producer supplies an explicit slot; production GI-RIS
// assigns one frame-varying pixel from each disjoint slot-owned block. Records
// are [NRC_IN_W encoded input | OUT_W radiance target | 3 query WORLD pos]
// (recordStride = NRC_IN_W + OUT_W + 3). Host gather treats all-zero encoded
// input as empty; zero radiance targets are valid training samples.
@group(${G}) @binding(8) var<storage, read_write> nrcRuntimeArena : array<atomic<u32>>;
@group(${G}) @binding(9) var<uniform> nrcCfg : NrcCfgUBO;
// H27 — per-slot claim flags (atomic u32, one per recordCap slot). 0=unclaimed,
// 1=claimed. A compare-exchange in nrcWriteRecord ensures the first invocation to
// claim a slot wins; subsequent racers see 1 and skip the write (torn-record fix).
// The host clears this buffer to zero at the start of each frame window.
${NRC_DIAGNOSTIC_CONSTANTS_WGSL}
const NRC_INF_MAGIC   : u32 = ${NRC_INFERENCE_ARENA_MAGIC}u;
const NRC_INF_VERSION : u32 = ${NRC_INFERENCE_ARENA_VERSION}u;
const NRC_INF_SCHEMA  : u32 = ${NRC_INFERENCE_ARENA_SCHEMA}u;
const NRC_RT_MAGIC    : u32 = ${NRC_RUNTIME_ARENA_MAGIC}u;
const NRC_RT_VERSION  : u32 = ${NRC_RUNTIME_ARENA_VERSION}u;
const NRC_RT_SCHEMA   : u32 = ${NRC_RUNTIME_ARENA_SCHEMA}u;
const NRC_RT_HEADER   : u32 = ${NRC_RUNTIME_HEADER_WORD_OFFSET}u;
const NRC_INF_WEIGHTS_OFFSET : u32 = ${NRC_INFERENCE_HEADER_FIELD.weightsOffset}u;
const NRC_INF_BIASES_OFFSET  : u32 = ${NRC_INFERENCE_HEADER_FIELD.biasesOffset}u;
const NRC_INF_TABLES_OFFSET  : u32 = ${NRC_INFERENCE_HEADER_FIELD.tablesOffset}u;
const NRC_INF_LEVELS_OFFSET  : u32 = ${NRC_INFERENCE_HEADER_FIELD.levelsOffset}u;
const NRC_RT_CLAIMS_OFFSET   : u32 = ${NRC_RUNTIME_HEADER_FIELD.claimsOffset}u;
const NRC_RT_RECORDS_OFFSET  : u32 = ${NRC_RUNTIME_HEADER_FIELD.recordsOffset}u;
const NRC_RT_DIAGNOSTICS_OFFSET : u32 = ${NRC_RUNTIME_HEADER_FIELD.diagnosticsOffset}u;

fn nrcInferenceArenaValid() -> bool {
  return nrcInferenceArena[0] == NRC_INF_MAGIC
    && nrcInferenceArena[1] == NRC_INF_VERSION
    && nrcInferenceArena[2] != 0u
    && nrcInferenceArena[3] == NRC_INF_SCHEMA;
}
fn nrcRuntimeArenaValid() -> bool {
  return atomicLoad(&nrcRuntimeArena[NRC_RT_HEADER + 0u]) == NRC_RT_MAGIC
    && atomicLoad(&nrcRuntimeArena[NRC_RT_HEADER + 1u]) == NRC_RT_VERSION
    && atomicLoad(&nrcRuntimeArena[NRC_RT_HEADER + 2u]) != 0u
    && atomicLoad(&nrcRuntimeArena[NRC_RT_HEADER + 3u]) == NRC_RT_SCHEMA;
}
fn nrcLoadWeight(index: u32) -> f32 {
  return bitcast<f32>(nrcInferenceArena[nrcInferenceArena[NRC_INF_WEIGHTS_OFFSET] + index]);
}
fn nrcLoadBias(index: u32) -> f32 {
  return bitcast<f32>(nrcInferenceArena[nrcInferenceArena[NRC_INF_BIASES_OFFSET] + index]);
}
fn nrcLoadTable(index: u32) -> f32 {
  return bitcast<f32>(nrcInferenceArena[nrcInferenceArena[NRC_INF_TABLES_OFFSET] + index]);
}
fn nrcLoadLevel(index: u32) -> NrcLevelDesc {
  let base = nrcInferenceArena[NRC_INF_LEVELS_OFFSET] + index * 4u;
  return NrcLevelDesc(
    nrcInferenceArena[base], nrcInferenceArena[base + 1u],
    nrcInferenceArena[base + 2u], nrcInferenceArena[base + 3u],
  );
}
fn nrcRuntimeDiagnosticsBase() -> u32 {
  return atomicLoad(&nrcRuntimeArena[NRC_RT_HEADER + NRC_RT_DIAGNOSTICS_OFFSET]);
}
fn nrcRuntimeClaimsBase() -> u32 {
  return atomicLoad(&nrcRuntimeArena[NRC_RT_HEADER + NRC_RT_CLAIMS_OFFSET]);
}
fn nrcRuntimeRecordsBase() -> u32 {
  return atomicLoad(&nrcRuntimeArena[NRC_RT_HEADER + NRC_RT_RECORDS_OFFSET]);
}
fn nrcAddDiagnostic(index: u32) {
  atomicAdd(&nrcRuntimeArena[nrcRuntimeDiagnosticsBase() + index], 1u);
}
const NRC_RECORD_CAS_ATTEMPTS : u32 = 64u;
const NRC_MAX_TRAIN_TARGET    : f32 = 1024.0;

fn nrcFinite(v: f32) -> bool {
  return v == v && abs(v) <= 3.402823e38;
}
fn nrcFinite3(v: vec3f) -> bool {
  return nrcFinite(v.x) && nrcFinite(v.y) && nrcFinite(v.z);
}
fn nrcRecordInvalidPdf() {
  if (nrcRuntimeArenaValid()) {
    atomicAdd(&nrcRuntimeArena[nrcRuntimeDiagnosticsBase() + NRC_DIAG_INVALID_PDF], 1u);
  }
}
fn nrcRecordSaturatedValue() {
  if (nrcRuntimeArenaValid()) {
    atomicAdd(&nrcRuntimeArena[nrcRuntimeDiagnosticsBase() + NRC_DIAG_SATURATED], 1u);
  }
}
fn nrcRecordNonFiniteValue() {
  if (nrcRuntimeArenaValid()) {
    atomicAdd(&nrcRuntimeArena[nrcRuntimeDiagnosticsBase() + NRC_DIAG_NONFINITE], 1u);
  }
}

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
      (*feat)[outBase + f] = (*feat)[outBase + f] + weight * nrcLoadTable(rb + f);
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
    nrcHashLevelForwardInline(nrm, nrcLoadLevel(l), l * NRC_FEAT, feat);
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
  // Node-layer 0: padded raw input. WGSL select evaluates both operands, so a
  // select-based pad would still read feat[i] out of bounds when i >= NRC_IN_W.
  for (var i: u32 = 0u; i < NRC_W; i = i + 1u) { actA[i] = 0.0; }
  for (var i: u32 = 0u; i < NRC_IN_W; i = i + 1u) {
    actA[i] = (*feat)[i];
  }
  for (var l: u32 = 0u; l < NRC_WLAYERS; l = l + 1u) {
    let inW = NRC_LINW[l];
    let outW = NRC_LOUTW[l];
    let wo = NRC_WOFF[l];
    let bo = NRC_BOFF[l];
    let isOut = (l == NRC_WLAYERS - 1u);
    for (var ocol: u32 = 0u; ocol < outW; ocol = ocol + 1u) {
      var acc: f32 = nrcLoadBias(bo + ocol);
      let wBase = wo + ocol * inW;
      if ((l & 1u) == 0u) {
        for (var i: u32 = 0u; i < inW; i = i + 1u) { acc = acc + nrcLoadWeight(wBase + i) * actA[i]; }
      } else {
        for (var i: u32 = 0u; i < inW; i = i + 1u) { acc = acc + nrcLoadWeight(wBase + i) * actB[i]; }
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
  if (!nrcInferenceArenaValid() || !nrcRuntimeArenaValid()) { return vec3f(0.0); }
  var feat: array<f32, NRC_IN_W>;
  nrcAssembleInput(pos, normal, viewDir, roughness, albedo, &feat);
  let raw = nrcMlpForward(&feat);
  if (!nrcFinite3(raw)) {
    nrcAddDiagnostic(NRC_DIAG_NONFINITE);
    return vec3f(0.0);
  }
  let bounded = clamp(raw, vec3f(0.0), vec3f(65504.0));
  if (any(bounded != raw)) {
    nrcAddDiagnostic(NRC_DIAG_SATURATED);
  }
  return bounded;
}

// Write one self-training record for the host gather.
//
// H27 first-writer-wins claim (torn-record fix). The current bounded producer
// assigns at most one eligible invocation per slot, but this generic writer keeps
// the atomic gate so future producers cannot interleave a record accidentally.
// The host clears nrcSlotClaims to 0 at the start of each frame.
//
// target is a matched, independently traced four-vertex path-suffix estimate for
// the same encoded vertex/direction. It reads neither DDGI nor the cache output.
//
// Record layout (recordStride = NRC_IN_W + OUT_W + 3 f32s):
//   [ NRC_IN_W encoded input | OUT_W radiance target | 3 query WORLD pos ]
// The raw query world position is appended so the host can drive the hash-grid
// encode-backward (nrcEncodeBackward.wgsl.ts): the scatter recomputes the
// trilinear corners from this pos + the scene AABB. (Müller 2022 Instant-NGP §4.)
fn nrcWriteRecord(
  slot: u32, pos: vec3f, normal: vec3f, viewDir: vec3f, roughness: f32, albedo: vec3f,
  tgt: vec3f,
) {
  if (!nrcInferenceArenaValid() || !nrcRuntimeArenaValid()) { return; }
  if (slot >= nrcCfg.recordCap) {
    nrcAddDiagnostic(NRC_DIAG_DROPPED_RECORD);
    return;
  }
  if (!nrcFinite3(pos) || !nrcFinite3(normal) || !nrcFinite3(viewDir)
      || !nrcFinite(roughness) || !nrcFinite3(albedo) || !nrcFinite3(tgt)) {
    nrcAddDiagnostic(NRC_DIAG_NONFINITE);
    return;
  }

  // Weak CAS may fail spuriously. Retry while the observed value remains 0;
  // stop immediately if another invocation really owns the slot.
  var ownsSlot = false;
  for (var attempt: u32 = 0u; attempt < NRC_RECORD_CAS_ATTEMPTS; attempt = attempt + 1u) {
    let claimed = atomicCompareExchangeWeak(&nrcRuntimeArena[nrcRuntimeClaimsBase() + slot], 0u, 1u);
    if (claimed.exchanged) {
      ownsSlot = true;
      break;
    }
    if (claimed.old_value != 0u) { break; }
  }
  if (!ownsSlot) {
    nrcAddDiagnostic(NRC_DIAG_DROPPED_RECORD);
    return;
  }

  var feat: array<f32, NRC_IN_W>;
  nrcAssembleInput(pos, normal, viewDir, roughness, albedo, &feat);
  for (var i: u32 = 0u; i < NRC_IN_W; i = i + 1u) {
    if (!nrcFinite(feat[i])) {
      atomicStore(&nrcRuntimeArena[nrcRuntimeClaimsBase() + slot], 0u);
      nrcAddDiagnostic(NRC_DIAG_NONFINITE);
      return;
    }
  }

  let boundedTgt = clamp(tgt, vec3f(0.0), vec3f(NRC_MAX_TRAIN_TARGET));
  if (any(boundedTgt != tgt)) {
    nrcAddDiagnostic(NRC_DIAG_SATURATED);
  }
  let base = nrcRuntimeRecordsBase() + slot * nrcCfg.recordStride;
  for (var i: u32 = 0u; i < NRC_IN_W; i = i + 1u) {
    atomicStore(&nrcRuntimeArena[base + i], bitcast<u32>(feat[i]));
  }
  atomicStore(&nrcRuntimeArena[base + NRC_IN_W + 0u], bitcast<u32>(boundedTgt.x));
  atomicStore(&nrcRuntimeArena[base + NRC_IN_W + 1u], bitcast<u32>(boundedTgt.y));
  atomicStore(&nrcRuntimeArena[base + NRC_IN_W + 2u], bitcast<u32>(boundedTgt.z));
  atomicStore(&nrcRuntimeArena[base + NRC_IN_W + 3u], bitcast<u32>(pos.x));
  atomicStore(&nrcRuntimeArena[base + NRC_IN_W + 4u], bitcast<u32>(pos.y));
  atomicStore(&nrcRuntimeArena[base + NRC_IN_W + 5u], bitcast<u32>(pos.z));
}
`;
}
