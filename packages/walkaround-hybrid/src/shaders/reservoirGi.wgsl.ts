/**
 * ReSTIR-GI reservoir ADT and the `PrimarySurface` struct it builds on.
 *
 * The sole live layout appends eight u32 words to the historical 20-u32 prefix
 * without reordering it. Those words carry the
 * native one-bounce DDGI-proxy sample state needed for transformed-density
 * reuse: direction, reconnection visibility, prefix vertex count, sample kind,
 * native target, and mutation epoch. This is deliberately not a full path
 * representation and does not make a ReSTIR-PT or unbiased-rendering claim.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { RESERVOIR_GI_STRIDE_U32 } from '../gi/giLayout.js';

export function buildReservoirGiWgsl(): string {
  const strideU32 = RESERVOIR_GI_STRIDE_U32;
  return /* wgsl */ `// ============================================================
// PrimarySurface — derived from re-casting the primary ray through the BVH.
// Replaces the pre-fix placeholder G-buffer reads that returned constant
// values for all pixels. Shared by temporal and spatial passes; shade.wgsl
// reads the same fields inline.
// ============================================================
struct PrimarySurface {
  hit:    bool,
  pos:    vec3f,
  normal: vec3f,
  geoNormal: vec3f,
  clearcoatNormal: vec3f,
  wo:     vec3f,
  albedo: vec3f,
  rough:  f32,
  metal:  f32,
  transmission: f32,
  isGlass: bool,
  specular: vec4f,
  anisotropy: vec2f,
  anisotropyTangent: vec3f,
  anisotropyBitangent: vec3f,
  iridescence: vec4f,
  clearcoat: vec2f,
  sheen: vec4f,
  sheenRoughness: f32,
  reflectionLayerTransmission: vec3f,
  layerTransmission: vec3f,
  volumeScattering: vec4f,
  bulkThickness: f32,
  envMapIntensity: f32,
  depth:  f32,
  // Stable correspondence keys emitted by the canonical BVH cast. In merged
  // mode instanceId is zero; in TLAS mode the pair
  // (instanceId, triangleId) identifies the exact visible primitive surface.
  triangleId: u32,
  instanceId: u32,
  // Packed scalar material payload available at the hit. Texture identity is
  // already implied by the exact triangle key; this additionally catches
  // in-place scalar material replacement when history was not yet cleared.
  materialKey: u32,
};

// ============================================================
// ReSTIR-GI / GRIS reservoir. The Sprint-16/17 fields occupy u32 [0..19]
// (UNCHANGED word offsets versus the old ReservoirGI). Generalized reuse fields
// are always appended at u32 [20..27]. ReservoirGI is kept as a type alias so the existing pass call
// sites (risGi/temporalGi/spatialGi/shade) compile unchanged.
// ============================================================
const GI_SAMPLE_SURFACE: u32 = 0u;
const GI_SAMPLE_ENVIRONMENT: u32 = 1u;
const GI_PREFIX_INVALID: u32 = 0u;
const GI_PREFIX_RECONNECTABLE: u32 = 1u;
// Camera-side dielectric throughput is baked into Lo and cannot be shifted to
// a different receiver without storing the full refractive prefix.
const GI_PREFIX_CAMERA_TRANSMISSION: u32 = 2u;
// Word-19 sample/technique flags. NRC's target depends on outgoing direction
// and source spread state that the 28-word reservoir cannot reconstruct after
// a shift, so its native technique remains local. Ordinary RIS remains a
// shiftable technique, but any native estimator that accumulated an angular
// or stochastic-alpha contribution must be admitted only at its own receiver.
// Its technique M still normalizes safe candidates from other reservoirs.
const GI_SAMPLE_FLAG_LOCAL_TECHNIQUE: u32 = 1u;
const GI_SAMPLE_FLAG_LOCAL_ESTIMATOR: u32 = 2u;
// A shifted/reused representative stores only scalar visibility in word 23.
// Its exact colored reconnection transmittance must therefore be retraced by
// final shade and applied componentwise to the selected contribution.
const GI_SAMPLE_FLAG_RECAST_TINT: u32 = 4u;
// Word 11 persists the uncapped estimator numerator in log2 space.  The
// sentinel is deliberately finite so it can be stored in the existing f32
// lane without changing the 28-word ABI.
const RESERVOIR_GI_LOG_ZERO: f32 = -3.402823466e38;
const RESERVOIR_GI_LOG2_ROUND_TO_ZERO: f32 = -150.0;
const RESERVOIR_GI_LOG2_OVERFLOW: f32 = 128.0;
const RESERVOIR_GI_MAX_FINITE_F32: f32 = 3.402823466e38;

fn reservoirGiFinite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e38;
}

fn reservoirGiValidLog(value: f32) -> bool {
  return reservoirGiFinite(value) && value > RESERVOIR_GI_LOG_ZERO;
}

fn reservoirGiLogPositive(value: f32) -> f32 {
  if (!reservoirGiFinite(value) || !(value > 0.0)) {
    return RESERVOIR_GI_LOG_ZERO;
  }
  return log2(value);
}

// Convert a log-domain value only when an API genuinely requires linear f32.
// Preserve subnormals down to the correctly-rounded half-min-subnormal cutoff;
// saturate only at exp2's actual overflow boundary. Reservoir metadata and
// estimator arithmetic remain logarithmic and do not call this helper.
fn reservoirGiRepresentPositiveLog(logValue: f32) -> f32 {
  if (!reservoirGiValidLog(logValue)
   || logValue <= RESERVOIR_GI_LOG2_ROUND_TO_ZERO) {
    return 0.0;
  }
  if (logValue >= RESERVOIR_GI_LOG2_OVERFLOW) {
    return RESERVOIR_GI_MAX_FINITE_F32;
  }
  return min(exp2(logValue), RESERVOIR_GI_MAX_FINITE_F32);
}

fn reservoirGiLogPositiveProduct(a: f32, b: f32) -> f32 {
  let logA = reservoirGiLogPositive(a);
  let logB = reservoirGiLogPositive(b);
  if (!reservoirGiValidLog(logA) || !reservoirGiValidLog(logB)) {
    return RESERVOIR_GI_LOG_ZERO;
  }
  let result = logA + logB;
  if (!reservoirGiFinite(result)) { return RESERVOIR_GI_LOG_ZERO; }
  return result;
}

// Stable log2(alpha*pGuide + (1-alpha)*pCos).  This is the proposal density
// actually represented by native RIS, even when the linear mixture would
// underflow before division.
fn reservoirGiLogProposalMixture(alpha: f32, pGuide: f32, pCos: f32) -> f32 {
  if (!reservoirGiFinite(alpha) || alpha < 0.0 || alpha > 1.0) {
    return RESERVOIR_GI_LOG_ZERO;
  }
  if (
    (alpha > 0.0 && (!reservoirGiFinite(pGuide) || pGuide < 0.0)) ||
    (alpha < 1.0 && (!reservoirGiFinite(pCos) || pCos < 0.0))
  ) {
    return RESERVOIR_GI_LOG_ZERO;
  }
  var maxTerm = RESERVOIR_GI_LOG_ZERO;
  var guideTerm = RESERVOIR_GI_LOG_ZERO;
  var cosineTerm = RESERVOIR_GI_LOG_ZERO;
  if (alpha > 0.0 && reservoirGiFinite(pGuide) && pGuide > 0.0) {
    guideTerm = log2(alpha) + log2(pGuide);
    maxTerm = guideTerm;
  }
  if (alpha < 1.0 && reservoirGiFinite(pCos) && pCos > 0.0) {
    cosineTerm = log2(1.0 - alpha) + log2(pCos);
    maxTerm = max(maxTerm, cosineTerm);
  }
  if (!reservoirGiValidLog(maxTerm)) { return RESERVOIR_GI_LOG_ZERO; }
  var scaledSum = 0.0;
  if (reservoirGiValidLog(guideTerm)) {
    scaledSum = scaledSum + exp2(guideTerm - maxTerm);
  }
  if (reservoirGiValidLog(cosineTerm)) {
    scaledSum = scaledSum + exp2(cosineTerm - maxTerm);
  }
  if (!reservoirGiFinite(scaledSum) || !(scaledSum > 0.0)) {
    return RESERVOIR_GI_LOG_ZERO;
  }
  return maxTerm + log2(scaledSum);
}

fn reservoirGiSaturatingAddU32(a: u32, b: u32) -> u32 {
  if (b > 0xffffffffu - a) { return 0xffffffffu; }
  return a + b;
}

fn reservoirGiDirectionBetween(fromPoint: vec3f, toPoint: vec3f) -> vec3f {
  if (
    !reservoirGiFinite(fromPoint.x) || !reservoirGiFinite(fromPoint.y) ||
    !reservoirGiFinite(fromPoint.z) || !reservoirGiFinite(toPoint.x) ||
    !reservoirGiFinite(toPoint.y) || !reservoirGiFinite(toPoint.z)
  ) {
    return vec3f(0.0);
  }
  let coordinateScale = max(
    max(abs(fromPoint.x), max(abs(fromPoint.y), abs(fromPoint.z))),
    max(abs(toPoint.x), max(abs(toPoint.y), abs(toPoint.z))),
  );
  if (!(coordinateScale > 0.0)) { return vec3f(0.0); }
  let scaledDelta = toPoint / coordinateScale - fromPoint / coordinateScale;
  let deltaScale = max(
    abs(scaledDelta.x),
    max(abs(scaledDelta.y), abs(scaledDelta.z)),
  );
  if (!(deltaScale > 0.0)) { return vec3f(0.0); }
  return safe_normalize(scaledDelta / deltaScale);
}

struct ReservoirPT {
  // ── Sprint-16/17 reconnection sample (u32 [0..19], layout-compatible) ──
  xv:      vec3f,   // visible point (primary hit)        idx 0..2
  // Former padding word. Scaled reservoirs use it to reject material/primitive
  // domain changes before shifting a representative sample to a full-res
  // receiver. Scale 1 deliberately ignores this key for cold-snapshot
  // compatibility with the historical zero-filled word.
  receiverMaterialKey: u32, // receiver-domain identity   idx 3
  nv:      vec3f,   // normal at xv                       idx 4..6
  logW:    f32,     // capped log2 RIS contribution weight idx 7
  xs:      vec3f,   // sample point (reconnection vertex)  idx 8..10
  H:       f32,     // log2(W_uncapped * selected pHat)     idx 11
  ns:      vec3f,   // normal at xs                       idx 12..14
  M:       u32,     // confidence (candidate count)        idx 15
  Lo:      vec3f,   // outgoing radiance at xs             idx 16..18
  sampleFlags: u32, // technique/sample behavior flags    idx 19
  // ── GRIS DDGI-proxy reuse metadata (u32 [20..27], appended) ──
  wi_recon:          vec3f, // unit incident dir xv→xs    idx 20..22
  sampleVisibility:      f32,   // reconnection visibility    idx 23
  prefixVertexCount: u32,   // reconnection-support class  idx 24
  sampleKind:           u32,   //                             idx 25
  nativeLogPHat:        f32,   // exact log2 native target    idx 26
  historyEpoch:           u32,   //                             idx 27
};

// ReservoirGI — back-compat alias. The Sprint-16/17 passes (risGi,
// temporalGi, spatialGi, shade) refer to the type by this name and only touch
// the [0..19] fields; the appended GRIS fields ride along untouched.
alias ReservoirGI = ReservoirPT;

fn emptyReservoirGI() -> ReservoirPT {
  var r: ReservoirPT;
  r.xv = vec3f(0.0); r.nv = vec3f(0,1,0);
  r.xs = vec3f(0.0); r.ns = vec3f(0,1,0);
  r.Lo = vec3f(0.0);
  r.logW = RESERVOIR_GI_LOG_ZERO;
  r.H = RESERVOIR_GI_LOG_ZERO;
  r.M = 0u;
  r.sampleFlags = 0u; r.receiverMaterialKey = 0u;
  // Generalized-reuse metadata is zero-initialised when gi-ris produces no
  // reconnectable vertex. The canonical temporal and spatial passes consume it.
  r.wi_recon = vec3f(0.0);
  r.sampleVisibility = 0.0;
  r.prefixVertexCount = 0u;
  r.sampleKind = GI_SAMPLE_SURFACE;
  r.nativeLogPHat = RESERVOIR_GI_LOG_ZERO;
  r.historyEpoch = 0u;
  return r;
}

// Sprint 16 / GRIS — ReservoirPT byte layout:
//   [0..2]   xv.xyz       [3]    receiverMaterialKey
//   [4..6]   nv.xyz       [7]    capped logW
//   [8..10]  xs.xyz       [11]   H = log2(W_uncapped * selected pHat)
//   [12..14] ns.xyz       [15]   M
//   [16..18] Lo.xyz       [19]   sampleFlags
//   ── appended GRIS DDGI-proxy metadata ──
//   [20..22] wi_recon.xyz [23]   sampleVisibility
//   [24]     prefixVertexCount
//   [25]     sampleKind   [26]   nativeLogPHat [27] historyEpoch
// Strided storage in array<u32> (4-byte elements): the live stride is 28 u32.
// NOTE: indices [0..19] preserve the pre-GRIS ReservoirGI word layout. Old
// snapshots migrate with receiverMaterialKey=0 because word 3 was padding.
const RESERVOIR_GI_STRIDE: u32 = ${strideU32}u;

fn unpackReservoirGI(words: array<u32, ${strideU32}>) -> ReservoirPT {
  var r: ReservoirPT;
  r.xv      = vec3f(bitcast<f32>(words[0u]), bitcast<f32>(words[1u]), bitcast<f32>(words[2u]));
  r.receiverMaterialKey = words[3u];
  r.nv      = vec3f(bitcast<f32>(words[4u]), bitcast<f32>(words[5u]), bitcast<f32>(words[6u]));
  r.logW    = bitcast<f32>(words[7u]);
  r.xs      = vec3f(bitcast<f32>(words[8u]), bitcast<f32>(words[9u]), bitcast<f32>(words[10u]));
  r.H       = bitcast<f32>(words[11u]);
  r.ns      = vec3f(bitcast<f32>(words[12u]), bitcast<f32>(words[13u]), bitcast<f32>(words[14u]));
  r.M       = words[15u];
  r.Lo      = vec3f(bitcast<f32>(words[16u]), bitcast<f32>(words[17u]), bitcast<f32>(words[18u]));
  r.sampleFlags = words[19u];
  // Generalized DDGI-proxy metadata.
  r.wi_recon          = vec3f(bitcast<f32>(words[20u]), bitcast<f32>(words[21u]), bitcast<f32>(words[22u]));
  r.sampleVisibility      = bitcast<f32>(words[23u]);
  r.prefixVertexCount = words[24u];
  r.sampleKind           = words[25u];
  r.nativeLogPHat    = bitcast<f32>(words[26u]);
  r.historyEpoch           = words[27u];
  return r;
}

fn packReservoirGI(r: ReservoirPT) -> array<u32, ${strideU32}> {
  var words: array<u32, ${strideU32}>;
  words[0u]  = bitcast<u32>(r.xv.x);
  words[1u]  = bitcast<u32>(r.xv.y);
  words[2u]  = bitcast<u32>(r.xv.z);
  words[3u]  = r.receiverMaterialKey;
  words[4u]  = bitcast<u32>(r.nv.x);
  words[5u]  = bitcast<u32>(r.nv.y);
  words[6u]  = bitcast<u32>(r.nv.z);
  words[7u]  = bitcast<u32>(r.logW);
  words[8u]  = bitcast<u32>(r.xs.x);
  words[9u]  = bitcast<u32>(r.xs.y);
  words[10u] = bitcast<u32>(r.xs.z);
  words[11u] = bitcast<u32>(r.H);
  words[12u] = bitcast<u32>(r.ns.x);
  words[13u] = bitcast<u32>(r.ns.y);
  words[14u] = bitcast<u32>(r.ns.z);
  words[15u] = r.M;
  words[16u] = bitcast<u32>(r.Lo.x);
  words[17u] = bitcast<u32>(r.Lo.y);
  words[18u] = bitcast<u32>(r.Lo.z);
  words[19u] = r.sampleFlags;
  // Generalized DDGI-proxy metadata (written by gi-ris; read by temporal/spatial reuse).
  words[20u] = bitcast<u32>(r.wi_recon.x);
  words[21u] = bitcast<u32>(r.wi_recon.y);
  words[22u] = bitcast<u32>(r.wi_recon.z);
  words[23u] = bitcast<u32>(r.sampleVisibility);
  words[24u] = r.prefixVertexCount;
  words[25u] = r.sampleKind;
  words[26u] = bitcast<u32>(r.nativeLogPHat);
  words[27u] = r.historyEpoch;
  return words;
}

fn foldInvalidReservoirGICandidates(
  r: ptr<function, ReservoirPT>,
  attemptCount: u32,
  sampleKind: u32,
  historyEpoch: u32,
) {
  (*r).M = reservoirGiSaturatingAddU32((*r).M, attemptCount);
  if (!reservoirGiValidLog((*r).nativeLogPHat)) {
    (*r).sampleKind = sampleKind;
    (*r).nativeLogPHat = RESERVOIR_GI_LOG_ZERO;
    (*r).sampleVisibility = 0.0;
    (*r).historyEpoch = historyEpoch;
  }
}

fn recordInvalidReservoirGICandidate(
  r: ptr<function, ReservoirPT>,
  sampleKind: u32,
  historyEpoch: u32,
) {
  foldInvalidReservoirGICandidates(r, 1u, sampleKind, historyEpoch);
}

fn updateReservoirGIWithMetadata(
  r: ptr<function, ReservoirPT>,
  wrs: ptr<function, RepresentedWrsState>,
  xs: vec3f, ns: vec3f, Lo: vec3f,
  sampleKind: u32, sampleDirection: vec3f,
  sampleFlags: u32,
  nativeLogPHat: f32, sampleVisibility: f32, historyEpoch: u32,
  logWeight: f32,
  rng: ptr<function, u32>,
) {
  (*r).M = reservoirGiSaturatingAddU32((*r).M, 1u);
  if (representedWrsUpdate(wrs, logWeight, rng)) {
    (*r).xs = xs;
    (*r).ns = ns;
    (*r).Lo = Lo;
    (*r).sampleKind = sampleKind;
    (*r).sampleFlags = sampleFlags;
    (*r).wi_recon = sampleDirection;
    (*r).nativeLogPHat = nativeLogPHat;
    (*r).sampleVisibility = sampleVisibility;
    (*r).historyEpoch = historyEpoch;
  }
}

// Refresh the GRIS DDGI-proxy reuse metadata fields on a reservoir after
// the final sample is chosen (risGi / risGiNrc producers).  Populates wi_recon,
// sampleVisibility, and prefixVertexCount from the chosen base path edge xv → xs.
// Leaves the direction zeroed for an all-null technique while preserving its
// declared shiftability class and M. Only a truly cold record (M == 0) or an
// invalid class becomes GI_PREFIX_INVALID.
// Call after the final visibility test and W update.
fn refreshGrisMetadata(r: ptr<function, ReservoirPT>) {
  let declaredPrefix = (*r).prefixVertexCount;
  if (!reservoirGiValidLog((*r).H)) {
    (*r).wi_recon = vec3f(0.0);
    if (
      (*r).M == 0u ||
      (declaredPrefix != GI_PREFIX_RECONNECTABLE &&
       declaredPrefix != GI_PREFIX_CAMERA_TRANSMISSION)
    ) {
      (*r).prefixVertexCount = GI_PREFIX_INVALID;
    }
    return;
  }
  if (declaredPrefix != GI_PREFIX_CAMERA_TRANSMISSION) {
    (*r).prefixVertexCount = GI_PREFIX_RECONNECTABLE;
  }
  if ((*r).sampleKind == GI_SAMPLE_ENVIRONMENT) {
    (*r).wi_recon = safe_normalize((*r).wi_recon);
    return;
  }
  let reconnectDirection = reservoirGiDirectionBetween((*r).xv, (*r).xs);
  if (dot(reconnectDirection, reconnectDirection) > 0.0) {
    (*r).wi_recon = reconnectDirection;
  } else {
    (*r).wi_recon = vec3f(0.0);
    (*r).prefixVertexCount = GI_PREFIX_INVALID;
  }
}

fn reservoirGiCollapseLogParts(parts: vec2f) -> f32 {
  if (!reservoirGiValidLog(parts.x) || !reservoirGiFinite(parts.y)) {
    return RESERVOIR_GI_LOG_ZERO;
  }
  let result = parts.x + parts.y;
  if (!reservoirGiFinite(result)) { return RESERVOIR_GI_LOG_ZERO; }
  return result;
}

fn reservoirGiHasEstimatorNumerator(r: ReservoirPT) -> bool {
  return
    reservoirGiValidLog(r.H) &&
    reservoirGiValidLog(r.nativeLogPHat);
}

fn reservoirGiFiniteReceiver(r: ReservoirPT) -> bool {
  return
    reservoirGiFinite(r.xv.x) && reservoirGiFinite(r.xv.y) &&
    reservoirGiFinite(r.xv.z) && reservoirGiFinite(r.nv.x) &&
    reservoirGiFinite(r.nv.y) && reservoirGiFinite(r.nv.z) &&
    dot(r.nv, r.nv) > 0.0;
}

// A domain can represent scheduled proposal attempts even when every sampled
// target was zero and no candidate survived. It then has no candidate row,
// but its M remains in every other candidate's generalized-balance
// denominator. Camera-prefix and NRC techniques stay local because this ABI
// does not contain the state required to evaluate their shifted target.
fn reservoirGiHasShiftableTechnique(r: ReservoirPT) -> bool {
  return
    r.M > 0u &&
    r.prefixVertexCount == GI_PREFIX_RECONNECTABLE &&
    (r.sampleFlags & GI_SAMPLE_FLAG_LOCAL_TECHNIQUE) == 0u &&
    reservoirGiFiniteReceiver(r);
}

fn reservoirGiHasShiftableCandidate(r: ReservoirPT) -> bool {
  return
    reservoirGiHasShiftableTechnique(r) &&
    (r.sampleFlags & GI_SAMPLE_FLAG_LOCAL_ESTIMATOR) == 0u &&
    reservoirGiHasEstimatorNumerator(r) &&
    reservoirGiFinite(r.sampleVisibility) && r.sampleVisibility > 0.0 &&
    (r.sampleKind == GI_SAMPLE_SURFACE ||
     r.sampleKind == GI_SAMPLE_ENVIRONMENT);
}

fn reservoirGiHasLocalEstimator(r: ReservoirPT) -> bool {
  return
    (r.sampleFlags & GI_SAMPLE_FLAG_LOCAL_ESTIMATOR) != 0u &&
    reservoirGiHasEstimatorNumerator(r);
}

fn reservoirGiSourceLogW(r: ReservoirPT) -> f32 {
  if (!reservoirGiHasEstimatorNumerator(r)) {
    return RESERVOIR_GI_LOG_ZERO;
  }
  let result = r.H - r.nativeLogPHat;
  if (!reservoirGiFinite(result)) { return RESERVOIR_GI_LOG_ZERO; }
  return result;
}

// Word 7 stores capped log2(W), not a linear value.  Keeping this lane in log
// space preserves contributions for which linear W would underflow even though
// Lo*W remains representable.  Eligibility and reuse always use H.
fn reservoirGiFinaliseLogWFromH(
  r: ptr<function, ReservoirPT>,
  wCap: f32,
) {
  (*r).logW = RESERVOIR_GI_LOG_ZERO;
  if (
    reservoirGiHasEstimatorNumerator(*r) &&
    reservoirGiFinite(wCap) && wCap > 0.0
  ) {
    let logW = reservoirGiSourceLogW(*r);
    if (!reservoirGiValidLog(logW)) { return; }
    let logCap = log2(wCap);
    let cappedLogW = min(logW, logCap);
    if (reservoirGiValidLog(cappedLogW)) {
      (*r).logW = cappedLogW;
    }
  }
}

// Native RIS counts every scheduled attempt in M, including null attempts.
// WRS itself sees only finite positive log weights.  Its exact represented
// selection probability supplies the correction without a linear sum:
//   H = log2(a_selected / selectionProbability / M).
// Visibility is already present exactly once in nativeLogPHat/logWeight and is
// therefore not multiplied again here.
fn finaliseGIReservoirFromNativeWrs(
  r: ptr<function, ReservoirPT>,
  wrs: RepresentedWrsState,
  wCap: f32,
) {
  (*r).H = RESERVOIR_GI_LOG_ZERO;
  (*r).logW = RESERVOIR_GI_LOG_ZERO;
  if (
    (*r).M == 0u || !wrs.hasSelection ||
    !reservoirGiValidLog((*r).nativeLogPHat)
  ) {
    return;
  }
  var correction = representedWrsSelectedLogCorrectionParts(wrs);
  correction = representedWrsAddLogTerm(
    correction.x,
    correction.y,
    -log2(f32((*r).M)),
  );
  (*r).H = reservoirGiCollapseLogParts(correction);
  reservoirGiFinaliseLogWFromH(r, wCap);
}

`;
}

export interface ReservoirGiAccessorOptions {
  readonly loadReadWriteBinding?: string;
  readonly loadReadBinding?: string;
  readonly storeReadWriteBinding?: string;
}

function assertAccessorBinding(value: string, label: string): void {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new Error(`[reservoirGi] ${label} must be a WGSL identifier; received '${value}'`);
  }
}

/** Emit naga-native pass-local storage accessors against exact bindings. */
export function reservoirGiAccessorsWgsl(options: ReservoirGiAccessorOptions): string {
  const chunks: string[] = [];
  const emitLoad = (name: string, binding: string): void => {
    assertAccessorBinding(binding, name);
    chunks.push(/* wgsl */ `
fn ${name}(pixelIdx: u32) -> ReservoirPT {
  let base = pixelIdx * RESERVOIR_GI_STRIDE;
  var words: array<u32, RESERVOIR_GI_STRIDE>;
  for (var i: u32 = 0u; i < RESERVOIR_GI_STRIDE; i = i + 1u) {
    words[i] = ${binding}[base + i];
  }
  return unpackReservoirGI(words);
}
`);
  };
  if (options.loadReadWriteBinding !== undefined) {
    emitLoad('loadReservoirGI_rw', options.loadReadWriteBinding);
  }
  if (options.loadReadBinding !== undefined) {
    emitLoad('loadReservoirGI_ro', options.loadReadBinding);
  }
  if (options.storeReadWriteBinding !== undefined) {
    const binding = options.storeReadWriteBinding;
    assertAccessorBinding(binding, 'storeReservoirGI_rw');
    chunks.push(/* wgsl */ `
fn storeReservoirGI_rw(pixelIdx: u32, r: ReservoirPT) {
  let base = pixelIdx * RESERVOIR_GI_STRIDE;
  let words = packReservoirGI(r);
  for (var i: u32 = 0u; i < RESERVOIR_GI_STRIDE; i = i + 1u) {
    ${binding}[base + i] = words[i];
  }
}
`);
  }
  return chunks.join('');
}

export const RESERVOIR_GI_WGSL = buildReservoirGiWgsl();

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export function buildReservoirGiModule(): WgslModule {
  return {
    name: "reservoirGi",
    source: buildReservoirGiWgsl(),
    requires: [],
  };
}

export const RESERVOIR_GI_MODULE: WgslModule = {
  name: "reservoirGi",
  source: RESERVOIR_GI_WGSL,
  requires: [],
};
