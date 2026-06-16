/**
 * ReSTIR-GI reservoir ADT + the `PrimarySurface` struct it builds on.
 *
 * Split out of common.wgsl.ts (T9-stepA): the `PrimarySurface` struct
 * (re-cast primary-ray receiver, shared by temporal/spatial and read inline
 * by shade), the GI reservoir struct, its empty constructor, strided
 * load/store helpers, and `updateReservoirGI` (Sprint 16 ReSTIR-GI).
 * `updateReservoirGI` forward-references `rand_f32` (shared primitives) —
 * see reservoirDi.wgsl.ts header note on ordering.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * GRIS / ReSTIR-PT EVOLUTION — PHASE 0 (foundation; NO render-output change)
 * ════════════════════════════════════════════════════════════════════════════
 * Lin, Kettunen, Bitterli, Pantaleoni, Jakob, Nowrouzezahrai — "Generalized
 * Resampled Importance Sampling: Foundations of ReSTIR", SIGGRAPH 2022.
 *
 * The Sprint-16/17 reservoir stored a single-bounce *reconnection* sample
 * (xv/nv visible vertex, xs/ns reconnection vertex, Lo, plus W/w_sum/M/lightId)
 * occupying u32 indices [0..19] / 80 bytes. GRIS reuse across pixels requires a
 * *shift map* T that re-maps the base path's reconnection vertex onto an offset
 * path rooted at the neighbour pixel, together with the change-of-variables
 * Jacobian |∂T/∂·| that re-weights the contribution under the integral.
 *
 * Phase 0 (this struct widening + the CPU `reconnectionShift.ts` oracle in
 * @vitrum/shared-samplers) APPENDS — never reorders — the per-reservoir data
 * the future GPU shift + Jacobian will read. The existing [0..19] layout is
 * byte-identical, so every current load/store/read in temporalGi, spatialGi,
 * risGi and shade is provably unaffected. The appended fields [20..29] are
 * WRITTEN by risGi (initialised via `emptyReservoirGI` when no reconnection
 * vertex is produced) and READ by the GRIS variants of spatialGi and temporalGi
 * (spatialGi.wgsl.ts lines 298–303, 379–383; temporalGi.wgsl.ts lines 311–317,
 * 333, 383–387). comment-only update 2026-06-10.
 *
 * Phases 1-2 (dispatched separately) will:
 *   • Phase 1 — implement the GPU reconnection-shift application in spatialGi
 *     (and temporalGi): given a neighbour reservoir, re-root its reconnection
 *     vertex onto THIS pixel's visible vertex (xv/nv) and recompute the
 *     reconnection-edge geometry, mirroring `reconnectionShift` in the oracle.
 *   • Phase 2 — replace the current scalar RIS reuse with the GRIS pairwise /
 *     generalized-balance-heuristic MIS, weighting each shifted contribution by
 *     |∂T/∂·| (mirroring `reconnectionJacobian`).
 *
 * Appended fields and the shift-Jacobian role of each (see oracle for the
 * exact measure conversion, Lin 2022 §5 "reconnection shift"):
 *   • wi_recon  (vec3f, idx 20..22) — unit INCIDENT direction into the
 *       reconnection vertex along the BASE path (xv → xs). The shift discards
 *       this and re-derives the offset direction xv' → xs; caching it lets the
 *       reverse shift T⁻¹ and the BSDF-at-xv ratio be evaluated without a
 *       re-trace.
 *   • pdfReconBsdf (f32, idx 23) — the BSDF-sample solid-angle pdf at the
 *       visible vertex that GENERATED wi_recon (cosine-hemisphere pdf in the
 *       Phase-0 producer; the learned dTree mixture pdf when PPG is live).
 *       Cached headroom: the single-bounce reconnection-shift reuse weight does
 *       NOT read it (a reservoir's W already bakes in its source pdf, so reuse
 *       is m·p̂·W·J with no /p_src — see the spatialGi/temporalGi GRIS notes).
 *       Retained for a future multi-bounce shift's reverse-pdf evaluation.
 *   • distRecon (f32, idx 24) — ‖xv − xs‖ along the base path. The dist² term
 *       of the base reconnection-edge geometry G = cosθ_out / dist²; one of the
 *       two halves of the Jacobian ratio. Cached so the GPU need not recompute
 *       length(xs − xv) (and to validate the offset re-trace landed on the same
 *       reconnection vertex within tolerance).
 *   • cosReconOut (f32, idx 25) — |cos θ_out| at the reconnection vertex xs
 *       between its normal ns and the BASE incident direction (−wi_recon). The
 *       numerator of the base geometry term G. distRecon + cosReconOut together
 *       give the base half-G the Jacobian divides by.
 *   • prefixVertexCount (u32, idx 26) — number of path-prefix vertices before
 *       the reconnection vertex (1 for the single-bounce reconnection sample
 *       today; >1 once multi-bounce prefixes are reconnected). GRIS only shifts
 *       paths whose prefix length matches; a mismatch forces the random-replay
 *       fallback shift instead of the reconnection shift.
 *   • _padPT (u32×3, idx 27..29) — pad to a 16-byte (4-u32) multiple so the
 *       30-u32 / 120-byte stride keeps vec4-aligned strided indexing simple and
 *       leaves headroom for the Phase-1 reverse-shift cache without another
 *       stride bump. Zeroed; never read.
 *
 * References: Lin et al. 2022 (GRIS), Eq. 12 (shift Jacobian) and §5
 * (reconnection shift); Bitterli et al. 2020 (ReSTIR DI/GI base reservoir).
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import {
  RESERVOIR_GI_BASE_STRIDE_U32,
  RESERVOIR_GI_GRIS_STRIDE_U32,
} from '../restir/reservoirGiLayout.js';

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
  specular: vec4f,
  anisotropy: vec2f,
  iridescence: vec4f,
  clearcoat: vec2f,
  sheen: vec4f,
  sheenRoughness: f32,
  envMapIntensity: f32,
  depth:  f32,
};

// ============================================================
// ReSTIR-GI / GRIS reservoir. The Sprint-16/17 fields occupy u32 [0..19]
// (UNCHANGED — byte-identical to the old ReservoirGI). GRIS fields are appended
// at u32 [20..29] only in the widened restirPtReuse variant; the default
// generated shader stores the compact 20-u32 prefix and zeroes appended struct
// fields on load. ReservoirGI is kept as a type alias so the existing pass call
// sites (risGi/temporalGi/spatialGi/shade) compile unchanged.
// ============================================================
struct ReservoirPT {
  // ── Sprint-16/17 reconnection sample (u32 [0..19], byte-identical) ──
  xv:      vec3f,   // visible point (primary hit)        idx 0..2
  _pad0:   f32,     //                                    idx 3
  nv:      vec3f,   // normal at xv                       idx 4..6
  W:       f32,     // RIS unbiased contribution weight    idx 7
  xs:      vec3f,   // sample point (reconnection vertex)  idx 8..10
  w_sum:   f32,     // running RIS weight sum              idx 11
  ns:      vec3f,   // normal at xs                       idx 12..14
  M:       u32,     // confidence (candidate count)        idx 15
  Lo:      vec3f,   // outgoing radiance at xs             idx 16..18
  lightId: u32,     //                                    idx 19
  // ── GRIS Phase-0 reconnection-shift cache (u32 [20..29], appended) ──
  wi_recon:          vec3f, // unit incident dir xv→xs    idx 20..22
  pdfReconBsdf:      f32,   // BSDF-sample SA pdf at xv    idx 23
  distRecon:         f32,   // ‖xv − xs‖                   idx 24
  cosReconOut:       f32,   // |cos θ_out| at xs           idx 25
  prefixVertexCount: u32,   // path-prefix vertex count    idx 26
  _padPT0:           u32,   //                             idx 27
  _padPT1:           u32,   //                             idx 28
  _padPT2:           u32,   //                             idx 29
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
  r.pdfReconBsdf = 0.0;
  r.distRecon = 0.0;
  r.cosReconOut = 0.0;
  r.prefixVertexCount = 0u;
  r._padPT0 = 0u; r._padPT1 = 0u; r._padPT2 = 0u;
  return r;
}

// Sprint 16 / GRIS — ReservoirPT byte layout:
//   [0..2]   xv.xyz       [3]    _pad0
//   [4..6]   nv.xyz       [7]    W
//   [8..10]  xs.xyz       [11]   w_sum
//   [12..14] ns.xyz       [15]   M
//   [16..18] Lo.xyz       [19]   lightId
//   ── appended GRIS reconnection-shift cache (zeroed in Phase 0) ──
//   [20..22] wi_recon.xyz [23]   pdfReconBsdf
//   [24]     distRecon    [25]   cosReconOut
//   [26]     prefixVertexCount
//   [27..29] _padPT0..2
// Strided storage in array<u32> (4-byte elements): default stride = 20 u32,
// GRIS/restirPtReuse stride = 30 u32.
// NOTE: indices [0..19] are byte-identical to the pre-GRIS ReservoirGI layout,
// so all existing temporal/spatial/shade reads are provably unaffected.
const RESERVOIR_GI_STRIDE: u32 = ${strideU32}u;

fn loadReservoirGI_rw(buf: ptr<storage, array<u32>, read_write>, pixelIdx: u32) -> ReservoirPT {
  let b = pixelIdx * RESERVOIR_GI_STRIDE;
  var r: ReservoirPT;
  r.xv      = vec3f(bitcast<f32>(buf[b + 0u]), bitcast<f32>(buf[b + 1u]), bitcast<f32>(buf[b + 2u]));
  r._pad0   = bitcast<f32>(buf[b + 3u]);
  r.nv      = vec3f(bitcast<f32>(buf[b + 4u]), bitcast<f32>(buf[b + 5u]), bitcast<f32>(buf[b + 6u]));
  r.W       = bitcast<f32>(buf[b + 7u]);
  r.xs      = vec3f(bitcast<f32>(buf[b + 8u]), bitcast<f32>(buf[b + 9u]), bitcast<f32>(buf[b + 10u]));
  r.w_sum   = bitcast<f32>(buf[b + 11u]);
  r.ns      = vec3f(bitcast<f32>(buf[b + 12u]), bitcast<f32>(buf[b + 13u]), bitcast<f32>(buf[b + 14u]));
  r.M       = buf[b + 15u];
  r.Lo      = vec3f(bitcast<f32>(buf[b + 16u]), bitcast<f32>(buf[b + 17u]), bitcast<f32>(buf[b + 18u]));
  r.lightId = buf[b + 19u];
  ${grisCache ? /* wgsl */ `// GRIS Phase-0 cache.
  r.wi_recon          = vec3f(bitcast<f32>(buf[b + 20u]), bitcast<f32>(buf[b + 21u]), bitcast<f32>(buf[b + 22u]));
  r.pdfReconBsdf      = bitcast<f32>(buf[b + 23u]);
  r.distRecon         = bitcast<f32>(buf[b + 24u]);
  r.cosReconOut       = bitcast<f32>(buf[b + 25u]);
  r.prefixVertexCount = buf[b + 26u];
  r._padPT0           = buf[b + 27u];
  r._padPT1           = buf[b + 28u];
  r._padPT2           = buf[b + 29u];` : /* wgsl */ `// Compact default layout: appended GRIS cache is not stored on GPU.
  r.wi_recon = vec3f(0.0);
  r.pdfReconBsdf = 0.0;
  r.distRecon = 0.0;
  r.cosReconOut = 0.0;
  r.prefixVertexCount = 0u;
  r._padPT0 = 0u; r._padPT1 = 0u; r._padPT2 = 0u;`}
  return r;
}

fn loadReservoirGI_ro(buf: ptr<storage, array<u32>, read>, pixelIdx: u32) -> ReservoirPT {
  let b = pixelIdx * RESERVOIR_GI_STRIDE;
  var r: ReservoirPT;
  r.xv      = vec3f(bitcast<f32>(buf[b + 0u]), bitcast<f32>(buf[b + 1u]), bitcast<f32>(buf[b + 2u]));
  r._pad0   = bitcast<f32>(buf[b + 3u]);
  r.nv      = vec3f(bitcast<f32>(buf[b + 4u]), bitcast<f32>(buf[b + 5u]), bitcast<f32>(buf[b + 6u]));
  r.W       = bitcast<f32>(buf[b + 7u]);
  r.xs      = vec3f(bitcast<f32>(buf[b + 8u]), bitcast<f32>(buf[b + 9u]), bitcast<f32>(buf[b + 10u]));
  r.w_sum   = bitcast<f32>(buf[b + 11u]);
  r.ns      = vec3f(bitcast<f32>(buf[b + 12u]), bitcast<f32>(buf[b + 13u]), bitcast<f32>(buf[b + 14u]));
  r.M       = buf[b + 15u];
  r.Lo      = vec3f(bitcast<f32>(buf[b + 16u]), bitcast<f32>(buf[b + 17u]), bitcast<f32>(buf[b + 18u]));
  r.lightId = buf[b + 19u];
  ${grisCache ? /* wgsl */ `// GRIS Phase-0 cache.
  r.wi_recon          = vec3f(bitcast<f32>(buf[b + 20u]), bitcast<f32>(buf[b + 21u]), bitcast<f32>(buf[b + 22u]));
  r.pdfReconBsdf      = bitcast<f32>(buf[b + 23u]);
  r.distRecon         = bitcast<f32>(buf[b + 24u]);
  r.cosReconOut       = bitcast<f32>(buf[b + 25u]);
  r.prefixVertexCount = buf[b + 26u];
  r._padPT0           = buf[b + 27u];
  r._padPT1           = buf[b + 28u];
  r._padPT2           = buf[b + 29u];` : /* wgsl */ `// Compact default layout: appended GRIS cache is not stored on GPU.
  r.wi_recon = vec3f(0.0);
  r.pdfReconBsdf = 0.0;
  r.distRecon = 0.0;
  r.cosReconOut = 0.0;
  r.prefixVertexCount = 0u;
  r._padPT0 = 0u; r._padPT1 = 0u; r._padPT2 = 0u;`}
  return r;
}

fn storeReservoirGI_rw(buf: ptr<storage, array<u32>, read_write>, pixelIdx: u32, r: ReservoirPT) {
  let b = pixelIdx * RESERVOIR_GI_STRIDE;
  buf[b + 0u]  = bitcast<u32>(r.xv.x);
  buf[b + 1u]  = bitcast<u32>(r.xv.y);
  buf[b + 2u]  = bitcast<u32>(r.xv.z);
  buf[b + 3u]  = bitcast<u32>(r._pad0);
  buf[b + 4u]  = bitcast<u32>(r.nv.x);
  buf[b + 5u]  = bitcast<u32>(r.nv.y);
  buf[b + 6u]  = bitcast<u32>(r.nv.z);
  buf[b + 7u]  = bitcast<u32>(r.W);
  buf[b + 8u]  = bitcast<u32>(r.xs.x);
  buf[b + 9u]  = bitcast<u32>(r.xs.y);
  buf[b + 10u] = bitcast<u32>(r.xs.z);
  buf[b + 11u] = bitcast<u32>(r.w_sum);
  buf[b + 12u] = bitcast<u32>(r.ns.x);
  buf[b + 13u] = bitcast<u32>(r.ns.y);
  buf[b + 14u] = bitcast<u32>(r.ns.z);
  buf[b + 15u] = r.M;
  buf[b + 16u] = bitcast<u32>(r.Lo.x);
  buf[b + 17u] = bitcast<u32>(r.Lo.y);
  buf[b + 18u] = bitcast<u32>(r.Lo.z);
  buf[b + 19u] = r.lightId;
  ${grisCache ? /* wgsl */ `// GRIS Phase-0 cache (written by gi-ris; read by GRIS temporal/spatial reuse).
  buf[b + 20u] = bitcast<u32>(r.wi_recon.x);
  buf[b + 21u] = bitcast<u32>(r.wi_recon.y);
  buf[b + 22u] = bitcast<u32>(r.wi_recon.z);
  buf[b + 23u] = bitcast<u32>(r.pdfReconBsdf);
  buf[b + 24u] = bitcast<u32>(r.distRecon);
  buf[b + 25u] = bitcast<u32>(r.cosReconOut);
  buf[b + 26u] = r.prefixVertexCount;
  buf[b + 27u] = r._padPT0;
  buf[b + 28u] = r._padPT1;
  buf[b + 29u] = r._padPT2;` : /* wgsl */ `// Compact default layout: no appended GRIS cache stores.`}
}

fn updateReservoirGI(
  r: ptr<function, ReservoirPT>,
  xs: vec3f, ns: vec3f, Lo: vec3f,
  w: f32,
  rng: ptr<function, u32>,
) {
  (*r).M = (*r).M + 1u;
  (*r).w_sum = (*r).w_sum + w;
  if (rand_f32(rng) * (*r).w_sum < w) {
    (*r).xs = xs;
    (*r).ns = ns;
    (*r).Lo = Lo;
  }
}

// Refresh the GRIS Phase-0 reconnection-shift cache fields on a reservoir after
// the final sample is chosen (risGi / risGiNrc producers).  Populates wi_recon,
// distRecon, cosReconOut, pdfReconBsdf, and prefixVertexCount from the chosen
// base path edge xv → xs.  Leaves the cache zeroed and prefixVertexCount = 0
// when the reservoir is empty (M == 0) or degenerate (‖xv − xs‖ ≤ 1e-6).
// Call after the final visibility test and W update.
fn refreshPhase0Cache(r: ptr<function, ReservoirPT>) {
  let toRecon = (*r).xs - (*r).xv;
  let dRecon = length(toRecon);
  if (dRecon > 1e-6 && (*r).M > 0u) {
    let wiR = toRecon / dRecon;
    (*r).wi_recon    = wiR;
    (*r).distRecon   = dRecon;
    (*r).cosReconOut = abs(dot((*r).ns, -wiR));
    (*r).pdfReconBsdf = max(0.0, dot((*r).nv, wiR)) * INV_PI;
  } else {
    (*r).wi_recon    = vec3f(0.0);
    (*r).distRecon   = 0.0;
    (*r).cosReconOut = 0.0;
    (*r).pdfReconBsdf = 0.0;
  }
  (*r).prefixVertexCount = select(0u, 1u, (*r).M > 0u);
}

// Finalise the RIS unbiased-contribution weight W for a ReSTIR-GI reservoir.
// D5.3 — unified from the former finaliseGIReservoirW (gris=false) and
// finaliseGIReservoirWGris (gris=true) pair.
//
// Non-GRIS (gris=false, standard ReSTIR-GI):
//   W = w_sum / (M × p̂)  — Talbot 2005 + ReSTIR DI/GI 2020.
//   The M candidates each contributed weight w_i; the M-count normalises back.
//
// GRIS (gris=true, Lin et al. 2022 §generalised RIS):
//   W = w_sum / p̂  (denominator = 1, NOT M).
//   GRIS folds each sample with a pairwise MIS weight m_i where Σ m_i = 1,
//   so the M-count does NOT normalise the sum — dividing by M again would
//   under-energise the estimate.
//
// "gris" is a literal argument at every call site (never a UBO read) so the
// unified function does NOT introduce any cross-module UBO reference — safe to
// emit into any includer (d0ef37b regression class avoided).
//
// "wCap" must be passed as a parameter (see d0ef37b lesson — this is the
// exact function that caused that regression when wCap leaked as a UBO ref).
//
// Call AFTER the final updateReservoirGI / M update.
fn finaliseGIReservoirWFromPHat(r: ptr<function, ReservoirPT>, wCap: f32, gris: bool, pHatF: f32) {
  if ((*r).M > 0u) {
    // gris=false: divide by M (standard MIS-weight-1 RIS normalisation).
    // gris=true:  divide by 1 (GRIS pairwise MIS weights already sum to 1).
    let denom = select(f32((*r).M), 1.0, gris);
    let W_raw = select(0.0, (*r).w_sum / (denom * pHatF), pHatF > 1e-9);
    (*r).W = min(W_raw, wCap);
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
