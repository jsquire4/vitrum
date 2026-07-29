/**
 * ReSTIR-DI reservoir ADT + the emitter / G-buffer structs it co-locates.
 *
 * Split out of common.wgsl.ts (T9-stepA): the `EmitterTri` (80-byte) struct,
 * the `ReservoirDI` struct (with stored xi —
 * Bitterli 2020 §4), `emptyReservoirDI` / `updateReservoirDI`, and the
 * strided pack/unpack helpers (load/store, 8×u32 stride) shared by
 * ris/temporal/spatial.
 *
 * `updateReservoirDI` forward-references `rand_f32` (defined in the shared
 * primitives module). WGSL resolves module-scope functions regardless of
 * declaration order, and `common` aggregates the modules in the original
 * source order, so the forward reference is well-formed.
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
  w_sum:   f32,
  W:       f32,
  xi:      vec2f,    // sampled (u, v) on the chosen emitter
  areaM:   u32,      // scheduled finite-emitter-domain proposals
  envM:    u32,      // scheduled directional-environment proposals
};

const RESERVOIR_DI_MAX_FINITE_F32: f32 = 3.402823466e38;
const RESERVOIR_DI_INVALID_LOG_DENSITY: f32 = -3.402823466e38;
// DI reuse keeps the exact sampled point xi on the same finite emitter.
// Both source and destination therefore use emitter-area measure and the shift
// Jacobian is exactly one (unlike a receiver-reconnection path-space shift).
const RESERVOIR_DI_EMITTER_AREA_SHIFT_JACOBIAN: f32 = 1.0;

fn reservoirDiFinite(value: f32) -> bool {
  return value >= -RESERVOIR_DI_MAX_FINITE_F32
      && value <= RESERVOIR_DI_MAX_FINITE_F32;
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

// Generalized Talbot MIS (Lin et al. 2022, Eq. 36 / supplemental Eq. S.7)
// after grouping the M_i represented attempts of each input reservoir:
//
//   m_i(y) = M_i pHat_i(y) / sum_j M_j pHat_j(y).
//
// The returned log-weight is Eq. 19 with the identity DI shift:
//
//   w_i = m_i(y_i) pHat_0(y_i) W_i |J_i|,  |J_i| = 1.
//
// Callers collect every candidate log-weight first, subtract their shared
// maximum, and only then run WRS. This preserves candidate ratios across the
// complete finite f32 input range; independently capping each pHat_0 * W_i
// would collapse unequal overflowing products to the same value.
fn reservoirDiGeneralizedReuseLogWeight(
  sourceLogDensity: f32,
  maxLogDensity: f32,
  scaledTechniqueDenominator: f32,
  canonicalDensity: f32,
  sourceReservoirW: f32,
) -> f32 {
  let sourceNumerator = reservoirDiScaledDensityFromLog(
    sourceLogDensity,
    maxLogDensity,
  );
  if (
    !(sourceNumerator > 0.0) ||
    !reservoirDiFinite(scaledTechniqueDenominator) ||
    !(scaledTechniqueDenominator > 0.0) ||
    !reservoirDiFinite(canonicalDensity) ||
    !(canonicalDensity > 0.0) ||
    !reservoirDiFinite(sourceReservoirW) ||
    !(sourceReservoirW > 0.0)
  ) {
    return RESERVOIR_DI_INVALID_LOG_DENSITY;
  }
  let misWeight = sourceNumerator / scaledTechniqueDenominator;
  if (!reservoirDiFinite(misWeight) || !(misWeight > 0.0)) {
    return RESERVOIR_DI_INVALID_LOG_DENSITY;
  }
  let logWeight =
    log2(misWeight) +
    log2(canonicalDensity) +
    log2(sourceReservoirW) +
    log2(RESERVOIR_DI_EMITTER_AREA_SHIFT_JACOBIAN);
  return select(
    RESERVOIR_DI_INVALID_LOG_DENSITY,
    logWeight,
    reservoirDiFinite(logWeight),
  );
}

// Generalized reuse already includes represented attempt multiplicity in the
// all-technique MIS denominator, so it must not divide by M again. The stored
// WRS sum is max-log-scaled; recover the unscaled UCW directly in log space:
// W = 2^maxLogWeight * scaledWeightSum / pHat_0(z). M/areaM/envM remain
// confidence/support counts.
fn finaliseReservoirDIFromGeneralizedReuse(
  r: ptr<function, ReservoirDI>,
  maxLogWeight: f32,
  selectedCanonicalDensity: f32,
) {
  (*r).W = 0.0;
  if (
    (*r).M == 0u ||
    maxLogWeight == RESERVOIR_DI_INVALID_LOG_DENSITY ||
    !reservoirDiFinite(maxLogWeight) ||
    !reservoirDiFinite((*r).w_sum) ||
    !((*r).w_sum > 0.0) ||
    !reservoirDiFinite(selectedCanonicalDensity) ||
    !(selectedCanonicalDensity > 0.0)
  ) {
    return;
  }
  let logW =
    maxLogWeight +
    log2((*r).w_sum) -
    log2(selectedCanonicalDensity);
  if (!reservoirDiFinite(logW)) {
    return;
  }
  if (logW >= log2(RESERVOIR_DI_MAX_FINITE_F32)) {
    (*r).W = RESERVOIR_DI_MAX_FINITE_F32;
    return;
  }
  let rawW = exp2(logW);
  if (reservoirDiFinite(rawW) && rawW > 0.0) {
    (*r).W = rawW;
  }
}

fn emptyReservoirDI() -> ReservoirDI {
  return ReservoirDI(0u, 0u, 0.0, 0.0, vec2f(0.0, 0.0), 0u, 0u);
}

fn updateReservoirDI(r: ptr<function, ReservoirDI>, lid: u32, xi: vec2f, w: f32, rng: ptr<function, u32>) {
  if (!reservoirDiFinite(w) || !(w > 0.0)) { return; }
  let nextWeightSum = (*r).w_sum + w;
  if (!reservoirDiFinite(nextWeightSum) || !(nextWeightSum > 0.0)) { return; }
  (*r).w_sum = nextWeightSum;
  if (rand_f32(rng) * nextWeightSum < w) {
    (*r).lightId = lid;
    (*r).xi      = xi;
  }
}

// Reduce a reservoir's effective history while retaining the area/env proposal
// ratio and the exact W = w_sum / (M * pHat) identity. The chosen sample's
// domain keeps at least one count whenever possible, so later reductions do not
// erase the support that produced the stored sample.
fn scaleReservoirDIToM(r: ptr<function, ReservoirDI>, targetMUnclamped: u32) {
  let oldM = (*r).M;
  let targetM = min(oldM, targetMUnclamped);
  if (targetM == oldM) { return; }
  if (oldM == 0u || targetM == 0u) {
    (*r).M = 0u;
    (*r).areaM = 0u;
    (*r).envM = 0u;
    (*r).w_sum = 0.0;
    (*r).W = 0.0;
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
  (*r).w_sum = (*r).w_sum * f32(targetM) / f32(oldM);
  (*r).M = targetM;
  (*r).areaM = nextArea;
  (*r).envM = nextEnv;
}

// ============================================================
// ReservoirDI pack/unpack helpers — canonical, used by ris/temporal/spatial.
// lightId, M, areaM, and envM are u32; w_sum and W are
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
    bitcast<u32>(r.w_sum),
    bitcast<u32>(r.W),
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
