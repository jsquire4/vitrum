/**
 * ReSTIR-DI reservoir ADT + the emitter / G-buffer structs it co-locates.
 *
 * Split out of common.wgsl.ts (T9-stepA): the `EmitterTri` (80-byte) struct,
 * the `ReservoirDI` struct (with stored xi —
 * Bitterli 2020 §4), `emptyReservoirDI` / `updateReservoirDI`, and the
 * strided pack/unpack helpers (load/store, 8×u32 stride) shared by
 * ris/temporal/spatial.
 *
 * `updateReservoirDI` and the finalizers forward-reference the represented-WRS
 * state/functions defined in the later shared-primitives module. WGSL resolves
 * module-scope declarations regardless of source order, and `common` preserves
 * that canonical aggregate order, so the forward references are well-formed.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { RESERVOIR_DI_STRIDE_U32 } from '../restir/reservoirDiLayout.js';

export const RESERVOIR_DI_WGSL = /* wgsl */ `// ============================================================
// ReSTIR DI Reservoir (32 bytes — 8 × u32)
// ============================================================
//
// The xi field (2 × f32 stored as 2 × u32 via bitcast) captures the
// random sample params used by sampleEmitterPoint when this candidate
// won the WRS. Without it the visibility test stage couldn't reconstruct
// the original sample point and fell back to centroid — a real bias
// (visibility at centroid vs at the actual sample disagrees for any
// emitter whose extent is comparable to the occluder's). Bitterli 2020
// section 4 documents this as the canonical "store xi alongside lightId" path.
//
// Wave 4 — ENV_SAMPLE_SENTINEL: when an importance-sampled HDRI candidate wins
// the reservoir, lightId is set to 0xFFFFFFFFu (unreachable as a real emitter
// index since emitterCount < 2^32 in practice). The xi field is repurposed to
// store the sampled direction as equirect UV:
//   xi.x = theta / PI       (v-coord: 0 = up/+Y, 1 = down/-Y)
//   xi.y = phi / (2PI) + 0.5  (u-coord: 0..1, wrapping at ±PI)
// envDirFromReservoirXi() reconstructs the world-space direction.
// Passes that call restir_di_compute_phat_xi() transparently handle both
// emitter and env-sentinel reservoirs via the lid guard.
const ENV_SAMPLE_SENTINEL: u32 = 0xFFFFFFFFu;

// Encode a world-space direction into reservoir xi (equirect UV).
fn envDirToXi(dir: vec3f) -> vec2f {
  let d = safe_normalize(dir);
  let phi = atan2(d.z, d.x);          // [-PI, PI]
  let theta = acos(clamp(d.y, -1.0, 1.0));  // [0, PI]
  return vec2f(theta * INV_PI, fract(phi * INV_PI * 0.5 + 0.5));
}

// Decode reservoir xi back to a world-space direction (env sentinel only).
fn envDirFromXi(xi: vec2f) -> vec3f {
  let theta = xi.x * PI;
  let phi   = (xi.y - 0.5) * (2.0 * PI);
  let st = sin(theta);
  return safe_normalize(vec3f(cos(phi) * st, cos(theta), sin(phi) * st));
}

struct ReservoirDI {
  lightId: u32,
  M:       u32,
  // H = log2(W_uncapped * pHat_selected).  This is the represented WRS
  // estimator numerator, not a raw linear weight sum.  Keeping H in word 2
  // preserves reuse support independently of the shading endpoint.
  logEstimatorNumerator: f32,
  // Persisted log2(W_uncapped), capped only to the maximum finite *log
  // coordinate* (not log2(F32_MAX)). Shading combines this with the selected
  // contribution before exp2, so W cannot underflow or overflow prematurely.
  logW:    f32,
  xi:      vec2f,    // sampled (u, v) on the chosen emitter
  areaM:   u32,      // scheduled finite-emitter-domain proposals
  envM:    u32,      // scheduled directional-environment proposals
};

const RESERVOIR_DI_MAX_FINITE_F32: f32 = 3.402823466e38;
const RESERVOIR_DI_LOG_ZERO: f32 = -3.402823466e38;
const RESERVOIR_DI_INVALID_LOG_DENSITY: f32 = RESERVOIR_DI_LOG_ZERO;
// DI reuse keeps the exact sampled point xi on the same finite emitter.
// Both source and destination therefore use emitter-area measure and the shift
// Jacobian is exactly one (unlike a receiver-reconnection path-space shift).
const RESERVOIR_DI_EMITTER_AREA_SHIFT_JACOBIAN: f32 = 1.0;

fn reservoirDiFinite(value: f32) -> bool {
  return value >= -RESERVOIR_DI_MAX_FINITE_F32
      && value <= RESERVOIR_DI_MAX_FINITE_F32;
}

fn reservoirDiPositiveLog2(value: f32) -> f32 {
  if (!reservoirDiFinite(value) || !(value > 0.0)) {
    return RESERVOIR_DI_LOG_ZERO;
  }
  return log2(value);
}

fn reservoirDiHasEstimatorNumerator(r: ReservoirDI) -> bool {
  return r.logEstimatorNumerator > RESERVOIR_DI_LOG_ZERO &&
    reservoirDiFinite(r.logEstimatorNumerator);
}

fn reservoirDiCoarseReuseLogWeight(r: ReservoirDI) -> f32 {
  if (r.M == 0u || !reservoirDiHasEstimatorNumerator(r)) {
    return RESERVOIR_DI_LOG_ZERO;
  }
  let result = r.logEstimatorNumerator + log2(f32(r.M));
  return select(
    RESERVOIR_DI_LOG_ZERO,
    result,
    result > RESERVOIR_DI_LOG_ZERO && reservoirDiFinite(result),
  );
}

fn reservoirDiSaturatingAddU32(a: u32, b: u32) -> u32 {
  if (b > 0xffffffffu - a) { return 0xffffffffu; }
  return a + b;
}

// Stratified RIS schedules finite-emitter and environment candidates on
// disjoint measures. Generalized reuse must balance a selected candidate only
// against attempts that had support on that candidate's measure.
fn reservoirDiSupportForLight(r: ReservoirDI, lightId: u32) -> u32 {
  return select(r.areaM, r.envM, lightId == ENV_SAMPLE_SENTINEL);
}

// Compute log2(M_i * pHat_i(y)) without ever forming the potentially
// overflowing product. Talbot denominators are accumulated after subtracting
// their largest log term (log-sum-exp), preserving ratios across the full
// finite f32 density and u32 support range.
fn reservoirDiLogWeightedDensity(attempts: u32, density: f32) -> f32 {
  if (attempts == 0u || !reservoirDiFinite(density) || !(density > 0.0)) {
    return RESERVOIR_DI_INVALID_LOG_DENSITY;
  }
  return log2(f32(attempts)) + log2(density);
}

fn reservoirDiScaledDensityFromLog(
  logDensity: f32,
  maxLogDensity: f32,
) -> f32 {
  if (
    logDensity == RESERVOIR_DI_INVALID_LOG_DENSITY ||
    maxLogDensity == RESERVOIR_DI_INVALID_LOG_DENSITY ||
    !reservoirDiFinite(logDensity) ||
    !reservoirDiFinite(maxLogDensity)
  ) {
    return 0.0;
  }
  return exp2(min(0.0, logDensity - maxLogDensity));
}

fn reservoirDiLogSumExpFromMaxScale(
  maxLogDensity: f32,
  scaledTechniqueDenominator: f32,
) -> f32 {
  if (
    maxLogDensity == RESERVOIR_DI_INVALID_LOG_DENSITY ||
    !reservoirDiFinite(maxLogDensity) ||
    !reservoirDiFinite(scaledTechniqueDenominator) ||
    !(scaledTechniqueDenominator > 0.0)
  ) {
    return RESERVOIR_DI_INVALID_LOG_DENSITY;
  }
  let result = maxLogDensity + log2(scaledTechniqueDenominator);
  return select(
    RESERVOIR_DI_INVALID_LOG_DENSITY,
    result,
    reservoirDiFinite(result),
  );
}

fn reservoirDiSourceLogW(
  sourceLogEstimatorNumerator: f32,
  sourceDensity: f32,
) -> f32 {
  if (
    sourceLogEstimatorNumerator <= RESERVOIR_DI_LOG_ZERO ||
    !reservoirDiFinite(sourceLogEstimatorNumerator) ||
    !reservoirDiFinite(sourceDensity) ||
    !(sourceDensity > 0.0)
  ) {
    return RESERVOIR_DI_INVALID_LOG_DENSITY;
  }
  let logW = sourceLogEstimatorNumerator - log2(sourceDensity);
  if (!reservoirDiFinite(logW)) {
    return RESERVOIR_DI_INVALID_LOG_DENSITY;
  }
  return min(logW, RESERVOIR_DI_MAX_FINITE_F32);
}

// Initial RIS candidate weight in log space:
//   log2(pHat / (p_select * p_within) * stratifiedScale).
// logSelectionPmf may come from a represented producer such as ReGIR, so it
// remains logarithmic instead of being forced through a linear f32 endpoint.
fn reservoirDiInitialCandidateLogWeight(
  targetDensity: f32,
  logSelectionPmf: f32,
  withinEmitterPdf: f32,
  logStratifiedScale: f32,
) -> f32 {
  if (
    !reservoirDiFinite(targetDensity) ||
    !(targetDensity > 0.0) ||
    logSelectionPmf <= RESERVOIR_DI_LOG_ZERO ||
    !reservoirDiFinite(logSelectionPmf) ||
    !reservoirDiFinite(withinEmitterPdf) ||
    !(withinEmitterPdf > 0.0) ||
    !reservoirDiFinite(logStratifiedScale)
  ) {
    return RESERVOIR_DI_LOG_ZERO;
  }
  let result =
    log2(targetDensity) -
    logSelectionPmf -
    log2(withinEmitterPdf) +
    logStratifiedScale;
  return select(
    RESERVOIR_DI_LOG_ZERO,
    result,
    result > RESERVOIR_DI_LOG_ZERO && reservoirDiFinite(result),
  );
}

// Generalized Talbot MIS (Lin et al. 2022, Eq. 36 / supplemental Eq. S.7)
// after grouping the M_i represented attempts of each input reservoir:
//
//   m_i(y) = M_i pHat_i(y) / sum_j M_j pHat_j(y).
//
// The returned log-weight is Eq. 19 with the identity DI shift:
//
//   w_i = m_i(y_i) pHat_0(y_i) W_i |J_i|,  |J_i| = 1.
//
// The source UCW is reconstructed from H rather than word 3. Consequently a
// capped logW never removes a valid source from reuse. The represented
// finite-RNG WRS consumes this log-weight
// directly; no max-scaled linear candidate weight is formed.
fn reservoirDiGeneralizedReuseLogWeight(
  sourceLogDensity: f32,
  logTechniqueDenominator: f32,
  canonicalDensity: f32,
  sourceDensity: f32,
  sourceLogEstimatorNumerator: f32,
) -> f32 {
  if (
    sourceLogDensity == RESERVOIR_DI_INVALID_LOG_DENSITY ||
    !reservoirDiFinite(sourceLogDensity) ||
    logTechniqueDenominator == RESERVOIR_DI_INVALID_LOG_DENSITY ||
    !reservoirDiFinite(logTechniqueDenominator) ||
    !reservoirDiFinite(canonicalDensity) ||
    !(canonicalDensity > 0.0)
  ) {
    return RESERVOIR_DI_INVALID_LOG_DENSITY;
  }
  let sourceLogW = reservoirDiSourceLogW(
    sourceLogEstimatorNumerator,
    sourceDensity,
  );
  if (sourceLogW == RESERVOIR_DI_INVALID_LOG_DENSITY) {
    return RESERVOIR_DI_INVALID_LOG_DENSITY;
  }
  let logWeight =
    sourceLogDensity -
    logTechniqueDenominator +
    log2(canonicalDensity) +
    sourceLogW +
    log2(RESERVOIR_DI_EMITTER_AREA_SHIFT_JACOBIAN);
  return select(
    RESERVOIR_DI_INVALID_LOG_DENSITY,
    logWeight,
    reservoirDiFinite(logWeight),
  );
}

fn reservoirDiCollapseLogParts(parts: vec2f) -> f32 {
  let result = parts.x + parts.y;
  return select(
    RESERVOIR_DI_LOG_ZERO,
    result,
    result > RESERVOIR_DI_LOG_ZERO && reservoirDiFinite(result),
  );
}

fn reservoirDiFinaliseLogWFromH(
  r: ptr<function, ReservoirDI>,
  selectedCanonicalDensity: f32,
) {
  (*r).logW = RESERVOIR_DI_LOG_ZERO;
  if (
    !reservoirDiHasEstimatorNumerator(*r) ||
    !reservoirDiFinite(selectedCanonicalDensity) ||
    !(selectedCanonicalDensity > 0.0)
  ) {
    return;
  }
  let result = (*r).logEstimatorNumerator - log2(selectedCanonicalDensity);
  if (!reservoirDiFinite(result)) {
    return;
  }
  (*r).logW = min(result, RESERVOIR_DI_MAX_FINITE_F32);
}

// Native RIS and coarse reuse represent M scheduled attempts.  The finite-RNG
// WRS primitive supplies log2(a_selected / pi_selected); division by M turns
// that represented total into the estimator numerator H. Visibility is a
// target-receiver term and is deliberately deferred to final shading so a
// source receiver's occlusion cannot survive a temporal/spatial shift.
fn finaliseReservoirDIFromNativeWrs(
  r: ptr<function, ReservoirDI>,
  wrs: RepresentedWrsState,
  selectedCanonicalDensity: f32,
) {
  (*r).logEstimatorNumerator = RESERVOIR_DI_LOG_ZERO;
  (*r).logW = RESERVOIR_DI_LOG_ZERO;
  if (
    (*r).M == 0u ||
    !wrs.hasSelection ||
    !reservoirDiFinite(selectedCanonicalDensity) ||
    !(selectedCanonicalDensity > 0.0)
  ) {
    return;
  }
  let representedCorrection = representedWrsSelectedLogCorrectionParts(wrs);
  let normalized = representedWrsAddLogTerm(
    representedCorrection.x,
    representedCorrection.y,
    -log2(f32((*r).M)),
  );
  (*r).logEstimatorNumerator = reservoirDiCollapseLogParts(normalized);
  reservoirDiFinaliseLogWFromH(r, selectedCanonicalDensity);
}

// Generalized reuse already includes represented attempt multiplicity in its
// all-technique MIS denominator, so it must not divide by M again.
fn finaliseReservoirDIFromGeneralizedReuse(
  r: ptr<function, ReservoirDI>,
  wrs: RepresentedWrsState,
  selectedCanonicalDensity: f32,
) {
  (*r).logEstimatorNumerator = RESERVOIR_DI_LOG_ZERO;
  (*r).logW = RESERVOIR_DI_LOG_ZERO;
  if (
    (*r).M == 0u ||
    !wrs.hasSelection ||
    !reservoirDiFinite(selectedCanonicalDensity) ||
    !(selectedCanonicalDensity > 0.0)
  ) {
    return;
  }
  (*r).logEstimatorNumerator = reservoirDiCollapseLogParts(
    representedWrsSelectedLogCorrectionParts(wrs),
  );
  reservoirDiFinaliseLogWFromH(r, selectedCanonicalDensity);
}

fn emptyReservoirDI() -> ReservoirDI {
  return ReservoirDI(
    0u,
    0u,
    RESERVOIR_DI_LOG_ZERO,
    RESERVOIR_DI_LOG_ZERO,
    vec2f(0.0, 0.0),
    0u,
    0u,
  );
}

fn updateReservoirDI(
  r: ptr<function, ReservoirDI>,
  wrs: ptr<function, RepresentedWrsState>,
  lid: u32,
  xi: vec2f,
  logWeight: f32,
  rng: ptr<function, u32>,
) {
  if (representedWrsUpdate(wrs, logWeight, rng)) {
    (*r).lightId = lid;
    (*r).xi      = xi;
  }
}

// Reduce a reservoir's effective history while retaining the area/env proposal
// ratio. H and logW are the already-normalized estimator and remain unchanged;
// only confidence/support multiplicity is reduced. The chosen sample's domain
// keeps at least one count whenever possible, so later reductions do not erase
// the support that produced the stored sample.
fn scaleReservoirDIToM(r: ptr<function, ReservoirDI>, targetMUnclamped: u32) {
  let oldM = (*r).M;
  let targetM = min(oldM, targetMUnclamped);
  if (targetM == oldM) { return; }
  if (oldM == 0u || targetM == 0u) {
    (*r).M = 0u;
    (*r).areaM = 0u;
    (*r).envM = 0u;
    (*r).logEstimatorNumerator = RESERVOIR_DI_LOG_ZERO;
    (*r).logW = RESERVOIR_DI_LOG_ZERO;
    return;
  }

  var nextArea = min(targetM, u32(round(f32((*r).areaM) * f32(targetM) / f32(oldM))));
  var nextEnv = targetM - nextArea;
  if ((*r).lightId == ENV_SAMPLE_SENTINEL && (*r).envM > 0u && nextEnv == 0u) {
    nextEnv = 1u;
    nextArea = targetM - 1u;
  } else if ((*r).lightId != ENV_SAMPLE_SENTINEL && (*r).areaM > 0u && nextArea == 0u) {
    nextArea = 1u;
    nextEnv = targetM - 1u;
  }
  (*r).M = targetM;
  (*r).areaM = nextArea;
  (*r).envM = nextEnv;
}

// ============================================================
// ReservoirDI pack/unpack helpers — canonical, used by ris/temporal/spatial.
// lightId, M, areaM, and envM are u32; H and logW are
// bit-cast to/from u32 to preserve f32 precision through the storage buffer.
// ============================================================
// 8 u32 = 32 bytes per reservoir (6 u32 before support-count persistence).
const RESERVOIR_DI_STRIDE = ${RESERVOIR_DI_STRIDE_U32}u;

fn unpackReservoirDI(words: array<u32, ${RESERVOIR_DI_STRIDE_U32}>) -> ReservoirDI {
  return ReservoirDI(
    words[0u],
    words[1u],
    bitcast<f32>(words[2u]),
    bitcast<f32>(words[3u]),
    vec2f(bitcast<f32>(words[4u]), bitcast<f32>(words[5u])),
    words[6u],
    words[7u],
  );
}

fn packReservoirDI(r: ReservoirDI) -> array<u32, ${RESERVOIR_DI_STRIDE_U32}> {
  return array<u32, ${RESERVOIR_DI_STRIDE_U32}>(
    r.lightId,
    r.M,
    bitcast<u32>(r.logEstimatorNumerator),
    bitcast<u32>(r.logW),
    bitcast<u32>(r.xi.x),
    bitcast<u32>(r.xi.y),
    r.areaM,
    r.envM,
  );
}

`;

export interface ReservoirDiAccessorOptions {
  readonly loadReadWriteBinding?: string;
  readonly loadReadBinding?: string;
  readonly storeReadWriteBinding?: string;
}

function assertWgslIdentifier(value: string, label: string): void {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new Error(`[reservoirDi] ${label} must be a WGSL identifier; received '${value}'`);
  }
}

/** Emit naga-native pass-local storage accessors against exact bindings. */
export function reservoirDiAccessorsWgsl(options: ReservoirDiAccessorOptions): string {
  const chunks: string[] = [];
  const emitLoad = (name: string, binding: string): void => {
    assertWgslIdentifier(binding, name);
    chunks.push(/* wgsl */ `
fn ${name}(pixelIdx: u32) -> ReservoirDI {
  let base = pixelIdx * RESERVOIR_DI_STRIDE;
  var words: array<u32, ${RESERVOIR_DI_STRIDE_U32}>;
  for (var i: u32 = 0u; i < RESERVOIR_DI_STRIDE; i = i + 1u) {
    words[i] = ${binding}[base + i];
  }
  return unpackReservoirDI(words);
}
`);
  };
  if (options.loadReadWriteBinding !== undefined) {
    emitLoad('loadReservoirDI_rw', options.loadReadWriteBinding);
  }
  if (options.loadReadBinding !== undefined) {
    emitLoad('loadReservoirDI_ro', options.loadReadBinding);
  }
  if (options.storeReadWriteBinding !== undefined) {
    const binding = options.storeReadWriteBinding;
    assertWgslIdentifier(binding, 'storeReservoirDI_rw');
    chunks.push(/* wgsl */ `
fn storeReservoirDI_rw(pixelIdx: u32, r: ReservoirDI) {
  let base = pixelIdx * RESERVOIR_DI_STRIDE;
  let words = packReservoirDI(r);
  for (var i: u32 = 0u; i < RESERVOIR_DI_STRIDE; i = i + 1u) {
    ${binding}[base + i] = words[i];
  }
}
`);
  }
  return chunks.join('');
}

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const RESERVOIR_DI_MODULE: WgslModule = {
  name: "reservoirDi",
  source: RESERVOIR_DI_WGSL,
  requires: [],
};
