// spreadTermination.wgsl.ts — WGSL of the NRC path-spread cache-termination
// predicate. GPU mirror of the CPU oracle `../spreadTermination.ts`
// (Müller et al. 2021 §5). Pinned to ~1e-6 (f32) by the tests.
//
// The predicate is pure arithmetic (sqrt + running sum + a comparison), no
// kinks, so the WGSL ↔ CPU match is exact to f32 epsilon. This is the GPU code
// the cache-query pass inlines to decide, per path vertex, whether to terminate
// the GI suffix into the NRC cache (query the MLP for outgoing radiance) instead
// of continuing to trace.
//
// LIVE when nrcEnabled: composed into the `risGiNrc` gi-ris variant
// (buildRisGiNrcModule). At the reconnection vertex (where Lo is otherwise the
// DDGI-atlas estimate) the cache query replaces that suffix estimate when the
// spread heuristic fires. See HARDWARE-VALIDATION-NEEDS.md V20.

export function nrcSpreadTerminationWgsl(): string {
  return /* wgsl */`
// sqrt(max-finite-f32), derived so both the segment term and its square remain
// representable. We use a log-domain fallback only when pdf*|cos| itself leaves
// f32 range; positive subnormal PDFs are not cut off. A value is saturated only
// at this format boundary, and every such event increments diagnostics.
const NRC_SPREAD_MAX_ROOT_F32: f32 = 1.844674297e19;
const NRC_SPREAD_MAX_ROOT_LOG2: f32 = 64.0;

fn nrcSpreadFinite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e38;
}

fn nrcSegmentSpreadTerm(dist: f32, pdf: f32, cosTheta: f32) -> f32 {
  if (!nrcSpreadFinite(dist) || !nrcSpreadFinite(pdf) || !nrcSpreadFinite(cosTheta)) {
    nrcRecordNonFiniteValue();
    return NRC_SPREAD_MAX_ROOT_F32;
  }
  let absCos = abs(cosTheta);
  if (!(pdf > 0.0) || !(absCos > 0.0)) {
    nrcRecordInvalidPdf();
    return NRC_SPREAD_MAX_ROOT_F32;
  }
  let absDist = abs(dist);
  if (absDist == 0.0) { return 0.0; }

  let density = pdf * absCos;
  var term: f32;
  if (nrcSpreadFinite(density) && density > 0.0) {
    term = absDist / sqrt(density);
  } else {
    // Multiplication overflow/underflow: evaluate the identical ratio in log2
    // space, which preserves finite-positive inputs without an epsilon cutoff.
    let logTerm = log2(absDist) - 0.5 * (log2(pdf) + log2(absCos));
    if (!nrcSpreadFinite(logTerm) || logTerm >= NRC_SPREAD_MAX_ROOT_LOG2) {
      nrcRecordSaturatedValue();
      return NRC_SPREAD_MAX_ROOT_F32;
    }
    term = exp2(logTerm);
  }

  if (!nrcSpreadFinite(term) || term > NRC_SPREAD_MAX_ROOT_F32) {
    nrcRecordSaturatedValue();
    return NRC_SPREAD_MAX_ROOT_F32;
  }
  if (term == 0.0) { nrcRecordSaturatedValue(); }
  return term;
}

// Accumulate one more segment into the running spread state. The state carries
// the running SUM of per-segment terms; a(x) is its square (Müller 2021 §5).
// Call once per traced edge starting at the primary edge.
//   runningSum_io : in/out running Σ of per-segment terms
//   returns        : a(x) = runningSum² after adding this segment
fn nrcAccumulateSpread(runningSum_io: ptr<function, f32>, dist: f32, pdf: f32, cosTheta: f32) -> f32 {
  let next = *runningSum_io + nrcSegmentSpreadTerm(dist, pdf, cosTheta);
  if (!nrcSpreadFinite(next) || next > NRC_SPREAD_MAX_ROOT_F32) {
    nrcRecordSaturatedValue();
    *runningSum_io = NRC_SPREAD_MAX_ROOT_F32;
  } else {
    *runningSum_io = max(next, 0.0);
  }
  return (*runningSum_io) * (*runningSum_io);
}

// The cache-termination test at a vertex with accumulated spread a(x), given the
// primary footprint a0 and the tunable constant c (Müller 2021 §5):
//     a(x) > c · a0
// The caller must ensure this is never evaluated at the primary vertex itself
// (k ≥ 1) — the primary hit is never a cache-query target.
fn nrcShouldTerminateIntoCache(aX: f32, a0: f32, c: f32) -> bool {
  return aX > c * a0;
}
`;
}
