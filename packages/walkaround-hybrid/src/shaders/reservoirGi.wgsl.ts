/**
 * ReSTIR-GI reservoir ADT and the `PrimarySurface` struct it builds on.
 *
 * The compact layout is the original 20-u32 reservoir. The opt-in GRIS layout
 * appends eight u32 words without reordering that prefix. Those words carry the
 * native one-bounce DDGI-proxy sample state needed for transformed-density
 * reuse: direction, reconnection visibility, prefix vertex count, sample kind,
 * native target, and mutation epoch. This is deliberately not a full path
 * representation and does not make a ReSTIR-PT or unbiased-rendering claim.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import {
  RESERVOIR_GI_BASE_STRIDE_U32,
  RESERVOIR_GI_GRIS_STRIDE_U32,
} from '../gi/giLayout.js';

export interface ReservoirGiWgslOptions {
  /** Include the appended GRIS reconnection-shift cache fields on the GPU buffer layout. */
  readonly grisCache?: boolean;
}

export function buildReservoirGiWgsl(options?: ReservoirGiWgslOptions): string {
  const grisCache = options?.grisCache !== false;
  const strideU32 = grisCache ? RESERVOIR_GI_GRIS_STRIDE_U32 : RESERVOIR_GI_BASE_STRIDE_U32;
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
  clearcoatNormal: vec3f,
  wo:     vec3f,
  albedo: vec3f,
  rough:  f32,
  metal:  f32,
  isGlass: bool,
  specular: vec4f,
  anisotropy: vec2f,
  anisotropyTangent: vec3f,
  anisotropyBitangent: vec3f,
  iridescence: vec4f,
  clearcoat: vec2f,
  sheen: vec4f,
  sheenRoughness: f32,
  layerTransmission: vec3f,
  volumeScattering: vec4f,
  bulkThickness: f32,
  envMapIntensity: f32,
  depth:  f32,
};

// ============================================================
// ReSTIR-GI / GRIS reservoir. The Sprint-16/17 fields occupy u32 [0..19]
// (UNCHANGED — byte-identical to the old ReservoirGI). GRIS fields are appended
// at u32 [20..27] only in the widened grisReuse variant; the default
// generated shader stores the compact 20-u32 prefix and zeroes appended struct
// fields on load. ReservoirGI is kept as a type alias so the existing pass call
// sites (risGi/temporalGi/spatialGi/shade) compile unchanged.
// ============================================================
const GI_SAMPLE_SURFACE: u32 = 0u;
const GI_SAMPLE_ENVIRONMENT: u32 = 1u;

fn reservoirGiFinite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e38;
}

fn reservoirGiSaturatingAddU32(a: u32, b: u32) -> u32 {
  if (b > 0xffffffffu - a) { return 0xffffffffu; }
  return a + b;
}

struct ReservoirPT {
  // ── Sprint-16/17 reconnection sample (u32 [0..19], byte-identical) ──
  xv:      vec3f,   // visible point (primary hit)        idx 0..2
  _pad0:   f32,     //                                    idx 3
  nv:      vec3f,   // normal at xv                       idx 4..6
  W:       f32,     // RIS contribution weight    idx 7
  xs:      vec3f,   // sample point (reconnection vertex)  idx 8..10
  w_sum:   f32,     // running RIS weight sum              idx 11
  ns:      vec3f,   // normal at xs                       idx 12..14
  M:       u32,     // confidence (candidate count)        idx 15
  Lo:      vec3f,   // outgoing radiance at xs             idx 16..18
  lightId: u32,     //                                    idx 19
  // ── GRIS DDGI-proxy reuse metadata (u32 [20..27], appended) ──
  wi_recon:          vec3f, // unit incident dir xv→xs    idx 20..22
  sampleVisibility:      f32,   // reconnection visibility    idx 23
  prefixVertexCount: u32,   // path-prefix vertex count    idx 24
  sampleKind:           u32,   //                             idx 25
  nativePHat:           f32,   //                             idx 26
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
  r.Lo = vec3f(0.0); r.W = 0.0; r.w_sum = 0.0; r.M = 0u;
  r.lightId = 0u; r._pad0 = 0.0;
  // GRIS reconnection-shift cache — zero-initialised when risGi produces no
  // reconnection vertex; read by the GRIS variants of spatialGi + temporalGi
  // (spatialGi.wgsl.ts lines 298–303, 379–383; temporalGi.wgsl.ts lines 311–317,
  // 333, 383–387). comment-only update 2026-06-10.
  r.wi_recon = vec3f(0.0);
  r.sampleVisibility = 0.0;
  r.prefixVertexCount = 0u;
  r.sampleKind = GI_SAMPLE_SURFACE; r.nativePHat = 0.0; r.historyEpoch = 0u;
  return r;
}

// Sprint 16 / GRIS — ReservoirPT byte layout:
//   [0..2]   xv.xyz       [3]    _pad0
//   [4..6]   nv.xyz       [7]    W
//   [8..10]  xs.xyz       [11]   w_sum
//   [12..14] ns.xyz       [15]   M
//   [16..18] Lo.xyz       [19]   lightId
//   ── appended GRIS DDGI-proxy metadata ──
//   [20..22] wi_recon.xyz [23]   sampleVisibility
//   [24]     prefixVertexCount
//   [25]     sampleKind   [26]   nativePHat   [27] historyEpoch
// Strided storage in array<u32> (4-byte elements): default stride = 20 u32,
// GRIS/grisReuse stride = 28 u32.
// NOTE: indices [0..19] are byte-identical to the pre-GRIS ReservoirGI layout,
// so all existing temporal/spatial/shade reads are provably unaffected.
const RESERVOIR_GI_STRIDE: u32 = ${strideU32}u;

fn unpackReservoirGI(words: array<u32, ${strideU32}>) -> ReservoirPT {
  var r: ReservoirPT;
  r.xv      = vec3f(bitcast<f32>(words[0u]), bitcast<f32>(words[1u]), bitcast<f32>(words[2u]));
  r._pad0   = bitcast<f32>(words[3u]);
  r.nv      = vec3f(bitcast<f32>(words[4u]), bitcast<f32>(words[5u]), bitcast<f32>(words[6u]));
  r.W       = bitcast<f32>(words[7u]);
  r.xs      = vec3f(bitcast<f32>(words[8u]), bitcast<f32>(words[9u]), bitcast<f32>(words[10u]));
  r.w_sum   = bitcast<f32>(words[11u]);
  r.ns      = vec3f(bitcast<f32>(words[12u]), bitcast<f32>(words[13u]), bitcast<f32>(words[14u]));
  r.M       = words[15u];
  r.Lo      = vec3f(bitcast<f32>(words[16u]), bitcast<f32>(words[17u]), bitcast<f32>(words[18u]));
  r.lightId = words[19u];
  ${grisCache ? /* wgsl */ `// GRIS DDGI-proxy metadata.
  r.wi_recon          = vec3f(bitcast<f32>(words[20u]), bitcast<f32>(words[21u]), bitcast<f32>(words[22u]));
  r.sampleVisibility      = bitcast<f32>(words[23u]);
  r.prefixVertexCount = words[24u];
  r.sampleKind           = words[25u];
  r.nativePHat       = bitcast<f32>(words[26u]);
  r.historyEpoch           = words[27u];` : /* wgsl */ `// Compact default layout: appended GRIS cache is not stored on GPU.
  r.wi_recon = vec3f(0.0);
  r.sampleVisibility = 0.0;
  r.prefixVertexCount = 0u;
  r.sampleKind = GI_SAMPLE_SURFACE; r.nativePHat = 0.0; r.historyEpoch = 0u;`}
  return r;
}

fn packReservoirGI(r: ReservoirPT) -> array<u32, ${strideU32}> {
  var words: array<u32, ${strideU32}>;
  words[0u]  = bitcast<u32>(r.xv.x);
  words[1u]  = bitcast<u32>(r.xv.y);
  words[2u]  = bitcast<u32>(r.xv.z);
  words[3u]  = bitcast<u32>(r._pad0);
  words[4u]  = bitcast<u32>(r.nv.x);
  words[5u]  = bitcast<u32>(r.nv.y);
  words[6u]  = bitcast<u32>(r.nv.z);
  words[7u]  = bitcast<u32>(r.W);
  words[8u]  = bitcast<u32>(r.xs.x);
  words[9u]  = bitcast<u32>(r.xs.y);
  words[10u] = bitcast<u32>(r.xs.z);
  words[11u] = bitcast<u32>(r.w_sum);
  words[12u] = bitcast<u32>(r.ns.x);
  words[13u] = bitcast<u32>(r.ns.y);
  words[14u] = bitcast<u32>(r.ns.z);
  words[15u] = r.M;
  words[16u] = bitcast<u32>(r.Lo.x);
  words[17u] = bitcast<u32>(r.Lo.y);
  words[18u] = bitcast<u32>(r.Lo.z);
  words[19u] = r.lightId;
  ${grisCache ? /* wgsl */ `// GRIS DDGI-proxy metadata (written by gi-ris; read by GRIS temporal/spatial reuse).
  words[20u] = bitcast<u32>(r.wi_recon.x);
  words[21u] = bitcast<u32>(r.wi_recon.y);
  words[22u] = bitcast<u32>(r.wi_recon.z);
  words[23u] = bitcast<u32>(r.sampleVisibility);
  words[24u] = r.prefixVertexCount;
  words[25u] = r.sampleKind;
  words[26u] = bitcast<u32>(r.nativePHat);
  words[27u] = r.historyEpoch;` : /* wgsl */ `// Compact default layout: no appended GRIS cache stores.`}
  return words;
}

fn updateReservoirGI(
  r: ptr<function, ReservoirPT>,
  xs: vec3f, ns: vec3f, Lo: vec3f,
  w: f32,
  rng: ptr<function, u32>,
) {
  (*r).M = reservoirGiSaturatingAddU32((*r).M, 1u);
  (*r).w_sum = (*r).w_sum + w;
  if (rand_f32(rng) * (*r).w_sum < w) {
    (*r).xs = xs;
    (*r).ns = ns;
    (*r).Lo = Lo;
  }
}


fn foldInvalidReservoirGICandidates(
  r: ptr<function, ReservoirPT>,
  attemptCount: u32,
  sampleKind: u32,
  historyEpoch: u32,
) {
  (*r).M = reservoirGiSaturatingAddU32((*r).M, attemptCount);
  if ((*r).w_sum <= 0.0) {
    (*r).sampleKind = sampleKind;
    (*r).nativePHat = 0.0;
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
  xs: vec3f, ns: vec3f, Lo: vec3f,
  sampleKind: u32, sampleDirection: vec3f,
  nativePHat: f32, sampleVisibility: f32, historyEpoch: u32,
  w: f32,
  rng: ptr<function, u32>,
) {
  (*r).M = reservoirGiSaturatingAddU32((*r).M, 1u);
  (*r).w_sum = (*r).w_sum + w;
  if (rand_f32(rng) * (*r).w_sum < w) {
    (*r).xs = xs;
    (*r).ns = ns;
    (*r).Lo = Lo;
    (*r).sampleKind = sampleKind;
    (*r).wi_recon = sampleDirection;
    (*r).nativePHat = nativePHat;
    (*r).sampleVisibility = sampleVisibility;
    (*r).historyEpoch = historyEpoch;
  }
}

// Refresh the GRIS DDGI-proxy reuse metadata fields on a reservoir after
// the final sample is chosen (risGi / risGiNrc producers).  Populates wi_recon,
// sampleVisibility, and prefixVertexCount from the chosen base path edge xv → xs.
// Leaves the direction zeroed and prefixVertexCount = 0
// when the reservoir is empty (M == 0) or degenerate (‖xv − xs‖ ≤ 1e-6).
// Call after the final visibility test and W update.
fn refreshGrisMetadata(r: ptr<function, ReservoirPT>) {
  if ((*r).M == 0u) {
    (*r).wi_recon = vec3f(0.0);
    (*r).prefixVertexCount = 0u;
    return;
  }
  (*r).prefixVertexCount = 1u;
  if ((*r).sampleKind == GI_SAMPLE_ENVIRONMENT) {
    (*r).wi_recon = safe_normalize((*r).wi_recon);
    return;
  }
  let toRecon = (*r).xs - (*r).xv;
  let dRecon = length(toRecon);
  if (dRecon > 1e-6) {
    let wiR = toRecon / dRecon;
    (*r).wi_recon = wiR;
  } else {
    (*r).wi_recon = vec3f(0.0);
    (*r).prefixVertexCount = 0u;
  }
}

// Finalise the stored ReSTIR-GI reservoir contribution weight.
//
// Compact reuse divides by the raw attempt count M. The GRIS DDGI-proxy
// variant has already folded each domain's exact attempt multiplicity into its
// bounded all-technique denominator, so dividing by M again would double-count
// that normalization. Neither path removes the configured finite wCap.
fn finaliseGIReservoirWFromPHat(
  r: ptr<function, ReservoirPT>,
  wCap: f32,
  gris: bool,
  pHatF: f32,
) {
  (*r).W = 0.0;
  if ((*r).M == 0u) { return; }

  // Compact mode normalises the raw attempt count here. GRIS mode carries
  // attempt multiplicity in the technique matrix. Positive subnormal pHat
  // values remain valid: no epsilon cutoff is applied. A mathematically huge
  // finite ratio that overflows f32 is saturated by the configured W cap.
  let normaliser = select(f32((*r).M), 1.0, gris);
  if (
    reservoirGiFinite(pHatF) && pHatF > 0.0 &&
    reservoirGiFinite((*r).w_sum) && (*r).w_sum >= 0.0 &&
    reservoirGiFinite(wCap) && wCap >= 0.0
  ) {
    let denominator = normaliser * pHatF;
    if (reservoirGiFinite(denominator) && denominator > 0.0) {
      let W_raw = (*r).w_sum / denominator;
      if (reservoirGiFinite(W_raw)) {
        (*r).W = min(max(W_raw, 0.0), wCap);
      } else if (W_raw > 0.0) {
        (*r).W = wCap;
      }
    }
  }
}

fn finaliseGIReservoirW(r: ptr<function, ReservoirPT>, wCap: f32, gris: bool) {
  if ((*r).M > 0u) {
    let toSf = (*r).xs - (*r).xv;
    let distSf = length(toSf);
    if (distSf > 1e-4) {
      let wiF = toSf / distSf;
      let cosThetaF = max(0.0, dot((*r).nv, wiF));
      let pHatF = luminance((*r).Lo) * cosThetaF * INV_PI;
      finaliseGIReservoirWFromPHat(r, wCap, gris, pHatF);
    } else {
      (*r).W = 0.0;
    }
  }
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

export const RESERVOIR_GI_WGSL = buildReservoirGiWgsl({ grisCache: true });

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export function buildReservoirGiModule(options?: ReservoirGiWgslOptions): WgslModule {
  return {
    name: "reservoirGi",
    source: buildReservoirGiWgsl(options),
    requires: [],
  };
}

export const RESERVOIR_GI_MODULE: WgslModule = {
  name: "reservoirGi",
  source: RESERVOIR_GI_WGSL,
  requires: [],
};
