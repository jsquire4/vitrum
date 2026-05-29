// spreadTermination.wgsl.ts — WGSL of the NRC path-spread cache-termination
// predicate. GPU mirror of the CPU oracle `../spreadTermination.ts`
// (Müller et al. 2021 §5). Pinned to ~1e-6 (f32) by the tests.
//
// The predicate is pure arithmetic (sqrt + running sum + a comparison), no
// kinks, so the WGSL ↔ CPU match is exact to f32 epsilon. This is the GPU code
// the future cache-query pass will inline to decide, per path vertex, whether to
// terminate the GI suffix into the NRC cache (query the MLP for outgoing
// radiance) instead of continuing to trace.
//
// Gated-OFF-inert: emitted + unit-pinned, NOT yet registered as a pipeline pass.
// The integration site is risGi.wgsl's reconnection-vertex loop (where Lo is
// computed today by DDGI-atlas sampling); the cache query replaces that suffix
// estimate when nrcEnabled && the spread heuristic fires. See the V-item.

export function nrcSpreadTerminationWgsl(): string {
  return /* wgsl */`
// Per-segment spread term sqrt( d² / (p·|cosθ|) ), with the same 1e-12 denom
// clamp as the CPU oracle (a degenerate segment → huge spread → early
// termination, the conservative behaviour).
fn nrcSegmentSpreadTerm(dist: f32, pdf: f32, cosTheta: f32) -> f32 {
  let denom = max(pdf * abs(cosTheta), 1e-12);
  return sqrt((dist * dist) / denom);
}

// Accumulate one more segment into the running spread state. The state carries
// the running SUM of per-segment terms; a(x) is its square (Müller 2021 §5).
// Call once per traced edge starting at the primary edge.
//   runningSum_io : in/out running Σ of per-segment terms
//   returns        : a(x) = runningSum² after adding this segment
fn nrcAccumulateSpread(runningSum_io: ptr<function, f32>, dist: f32, pdf: f32, cosTheta: f32) -> f32 {
  *runningSum_io = *runningSum_io + nrcSegmentSpreadTerm(dist, pdf, cosTheta);
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
