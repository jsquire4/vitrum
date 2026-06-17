/**
 * restirPtTemporal.wgsl.ts — the ReSTIR-PT TEMPORAL reuse pass (GRIS, Lin 2022).
 *
 * A SEPARATE `@compute` entry point (`restirPtTemporal`) — a DIRECT PORT of the
 * SHIPPING walkaround-hybrid `TEMPORAL_GI_GRIS_WGSL`
 * (`@vitrum/walkaround-hybrid/src/shaders/temporalGi.wgsl.ts`, the ON / opt-in
 * GRIS variant), generalized to pt-webgpu's full-res hero reservoir.
 *
 * Per full-res pixel it:
 *   1. reprojects the current visible vertex xv through the PREVIOUS-frame camera
 *      (params.prevViewProj) to find the same world point in the previous
 *      reservoir,
 *   2. geometric-rejects on depth (relative) + normal (dot) — material swap /
 *      occlusion,
 *   3. M-clamps the previous history (rptParams.mClamp),
 *   4. forms the TWO-DOMAIN (current/previous) pairwise generalized-balance MIS,
 *   5. shifts the previous reservoir's reconnection sample onto THIS pixel's
 *      visible vertex via the reconnection shift + its Jacobian, gated by a
 *      reconnection-visibility ray (traceAny along xv → xs),
 *   6. folds both samples into a fresh GRIS reservoir and finalises W = w_sum/p̂
 *      (NO /M — the MIS weights already sum to 1).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * THE LOAD-BEARING LESSON (mirrored from temporalGi.wgsl.ts:358-367)
 * ════════════════════════════════════════════════════════════════════════════
 * The reused (previous) reservoir's resampling weight is
 *     w_prev = m_prev · p̂_cur(T z_prev) · W_prev · J          (J = shift Jacobian)
 * with NO division by a source pdf. rPrev is a RESERVOIR: its W_prev already
 * bakes in the source pdf (the producer finalised W = w_sum/p̂). Reservoir reuse
 * re-weights by m·p̂·W·J only — the Jacobian alone carries the reconnection-edge
 * measure conversion. This is the temporal FEEDBACK loop (this pass's output
 * becomes next frame's rPrev). An extra /p_src would multiply the carried weight
 * by ≈π…∞ every frame, driving the recursion gain above 1 → W pins at wCap and
 * the GI mean climbs without bound (the V19 grison-divergence). REPLICATED HERE
 * EXACTLY — see the w_prev line; there is NO /p_src.
 *
 * For prefix length 1 the source/target pre-reconnection vertices ARE the visible
 * vertices (xPre == xv), so the live shift Jacobian is the hybrid form
 *     J_geom(rPrev.xv→rCur.xv) · p_replay(prev domain) / p_replay(cur domain).
 *
 * ── Bind groups ─────────────────────────────────────────────────────────────
 * Composes the SHARED pt-webgpu modules (for traceAny / projectToNdc); the
 * ReSTIR-PT reservoirs + params live in @group(4):
 *   @binding(1) rpt_reservoirCur  (read_write) — this frame's producer output, fused in place
 *   @binding(2) rpt_reservoirPrev (read)        — last frame's temporal output
 *   @binding(4) rptParams         (uniform)
 * The reconnection-visibility ray traverses the inherited @group(0..2) scene
 * (traceAny). (maxBindGroups ≥ 5 — see the compose-module note.)
 */

export const RESTIR_PT_TEMPORAL_WGSL = /* wgsl */ `
@group(4) @binding(1) var<storage, read_write> rpt_resCurrent: array<u32>;
@group(4) @binding(2) var<storage, read>       rpt_resPrev:    array<u32>;
@group(4) @binding(4) var<uniform>             rptParams:      RestirPtParams;

const RPT_DEPTH_REL_TOL: f32 = 0.1;   // |Δdepth|/depth reject (GI DEPTH_REL_TOL)
const RPT_NORMAL_DOT_MIN: f32 = 0.906; // cos(25°) normal reject (GI NORMAL_DOT_MIN)
const RPT_RECON_NORMAL_BIAS: f32 = 1e-3;

// Reproject a world point through the PREVIOUS-frame camera to its full-res
// pixel. Mirrors the GI projectToPrevHalfPx, but full-res and via the VP matrix
// pt-webgpu carries (params.prevViewProj) rather than separate view/proj. Returns
// the integer pixel coords, or a negative sentinel when behind the prev camera.
fn rptProjectToPrevPx(worldPos: vec3f) -> vec2i {
  let clip = params.prevViewProj * vec4f(worldPos, 1.0);
  if (clip.w <= 1e-6) { return vec2i(-1, -1); }
  let ndc = clip.xy / clip.w;
  let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  if (uv.x < 0.0 || uv.x >= 1.0 || uv.y < 0.0 || uv.y >= 1.0) { return vec2i(-1, -1); }
  let px = i32(uv.x * f32(params.width));
  let py = i32(uv.y * f32(params.height));
  return vec2i(px, py);
}

// GRIS reconnection visibility — clear edge xv → xs through the scene BVH/TLAS
// (traceAny over the inherited @group(0..2)). Mirrors GI tgiReconnectionVisible.
fn rptReconnectionVisible(xv: vec3f, nv: vec3f, xs: vec3f) -> bool {
  let toS = xs - xv;
  let dist = length(toS);
  if (dist < 1e-4) { return false; }
  let wi = toS / dist;
  let orig = xv + nv * RPT_RECON_NORMAL_BIAS;
  return !traceAny(Ray(orig, wi), 1e-4, max(dist - 2e-3, 1e-3));
}

@compute @workgroup_size(8, 8, 1)
fn restirPtTemporal(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  let pixelIdx = gid.y * params.width + gid.x;
  var rCur = loadReservoirPTHero_rw(&rpt_resCurrent, pixelIdx);

  // Need a visible-surface point to reproject. If the producer wrote an empty
  // reservoir (primary miss / specular / transmissive xv), nothing to fuse.
  if (rCur.M == 0u) {
    storeReservoirPTHero_rw(&rpt_resCurrent, pixelIdx, rCur);
    return;
  }

  // Reproject the current visible vertex through the prev camera — the SAME world
  // point in both frames (camera moves, scene static for this increment).
  let prevPx = rptProjectToPrevPx(rCur.xv);
  if (prevPx.x < 0 || prevPx.y < 0
   || u32(prevPx.x) >= params.width || u32(prevPx.y) >= params.height) {
    storeReservoirPTHero_rw(&rpt_resCurrent, pixelIdx, rCur);
    return;
  }
  let prevIdx = u32(prevPx.y) * params.width + u32(prevPx.x);
  let rPrev = loadReservoirPTHero_ro(&rpt_resPrev, prevIdx);

  if (rPrev.M == 0u || rPrev.W <= 0.0) {
    storeReservoirPTHero_rw(&rpt_resCurrent, pixelIdx, rCur);
    return;
  }

  // Geometric-consistency test: depth (relative to the camera) + normal.
  let dDepth = abs(length(rCur.xv - params.cameraPos.xyz) - length(rPrev.xv - params.cameraPos.xyz));
  let depthRef = max(1e-3, length(rCur.xv - params.cameraPos.xyz));
  if (dDepth / depthRef > RPT_DEPTH_REL_TOL) {
    storeReservoirPTHero_rw(&rpt_resCurrent, pixelIdx, rCur);
    return;
  }
  if (dot(rCur.nv, rPrev.nv) < RPT_NORMAL_DOT_MIN) {
    storeReservoirPTHero_rw(&rpt_resCurrent, pixelIdx, rCur);
    return;
  }

  // M-clamp: bound prev history before contributing.
  let prevM = min(rPrev.M, rptParams.mClamp);

  var rng = pcgInit(
    gid.x ^ (params.frameSeed * 0x71E5u),
    gid.y ^ (params.frameSeed * 0xE571u),
    params.frameSeed ^ 0x9B7Fu,
  );

  // ── GRIS reconnection-shift temporal reuse (Lin 2022 §5 + Eq. 12 + MIS) ──
  // Two-domain pairwise MIS: the CURRENT (canonical) sample and the reprojected
  // PREVIOUS sample, both folded into a FRESH reservoir with their own pairwise
  // weights (the canonical is a resampling technique, not an un-weighted base).
  let cCur = f32(rCur.M);
  let cPrev = f32(prevM);
  // Eye directions for the integrand-matching target (the BRDF needs wo at each
  // domain's visible vertex). The scene is static this increment; the prev camera
  // ≈ current for small motion, so params.cameraPos serves both domains — an
  // unbiased approximation (p̂ only sets resampling variance, not the mean).
  let woCur  = restirpt_safe_normalize(params.cameraPos.xyz - rCur.xv);
  let woPrev = restirpt_safe_normalize(params.cameraPos.xyz - rPrev.xv);
  let pHatCur_native = restirPtTargetForDomain(rCur, woCur, rCur.xs, rCur.Lo);

  // Decide whether prev is a VALID reconnection-shift candidate (prefix match,
  // non-degenerate base half-G, positive Jacobian, non-zero shifted+native
  // targets, AND reconnection-visible). For prefix length 1 the pre-reconnection
  // vertices are the visible vertices: J = hybrid half-G × BSDF replay-pdf ratio.
  var prevValid = (rPrev.prefixVertexCount == rCur.prefixVertexCount)
               && (rPrev.prefixVertexCount != 0u);
  var J: f32 = 0.0;
  var pHatPrev_atCur: f32 = 0.0;
  var pHatPrev_native: f32 = 0.0;
  if (prevValid) {
    J = restirPtHybridShiftJacobianForPair(rPrev, rCur, woPrev, woCur);
    pHatPrev_atCur  = restirPtTargetForDomain(rCur, woCur, rPrev.xs, rPrev.Lo);
    pHatPrev_native = restirPtTargetForDomain(rPrev, woPrev, rPrev.xs, rPrev.Lo);
    prevValid = (J > 0.0)
             && (pHatPrev_atCur >= 1e-9) && (pHatPrev_native >= 1e-9)
             && rptReconnectionVisible(rCur.xv, rCur.nv, rPrev.xs);
  }

  var rGris = emptyReservoirPTHero();
  copyReservoirPTVisibleDomain(&rGris, rCur);

  // Canonical (current) sample, MIS-weighted against the prev pair.
  if (rCur.M > 0u && pHatCur_native > 1e-9) {
    var m_cur: f32 = 1.0;
    if (prevValid) {
      // prev's sample re-rooted onto the CURRENT domain is p̂_cur(T z_prev); the
      // canonical's own sample re-rooted onto prev is p̂_prev(T⁻¹ z_cur).
      let pHatPrev_atCurSample = restirPtTargetForDomain(rPrev, woPrev, rCur.xs, rCur.Lo);
      let denomCur = restirPtPairwiseDenomCanonical(cCur, pHatCur_native, cPrev, pHatPrev_atCurSample);
      m_cur = select(1.0, (cCur * pHatCur_native) / denomCur, denomCur > 1e-12);
    }
    // Canonical: no shift (J = 1), already at this pixel (no visibility re-test).
    let w_cur = m_cur * pHatCur_native * rCur.W;
    let oldM = rGris.M;
    let curReplayPdf = restirPtVisibleReplayPdfForDomain(rCur, woCur, rCur.xs);
    updateReservoirPTWithHybrid(&rGris, rCur.xs, rCur.ns, rCur.Lo, rCur.pdfSrc, 1.0, curReplayPdf, rCur.rngSeed, w_cur, &rng);
    rGris.M = oldM + rCur.M;
  }

  // Previous (reprojected) sample, reconnection-shifted + MIS-weighted.
  if (prevValid) {
    let denomPrev = restirPtPairwiseDenomNeighbor(cCur, pHatPrev_atCur, cPrev, pHatPrev_native);
    let m_prev = select(0.0, (cPrev * pHatPrev_native) / denomPrev, denomPrev > 1e-12);
    // GRIS resampling weight for a REUSED reservoir sample (Lin 2022, Alg. 3 /
    // Eq. 9):  w_prev = m_prev · p̂_cur(T z_prev) · W_prev · |∂T/∂·|.
    // NO /p_src — rPrev is a reservoir; W_prev already bakes its source pdf in.
    // (See the load-bearing lesson in the file header; an extra /p_src diverges
    // the temporal feedback loop — V19 grison.)
    let w_prev = m_prev * pHatPrev_atCur * rPrev.W * J;
    let oldM = rGris.M;
    // Carry the prev sample's source pdf alongside it (resolve uses the chosen
    // sample's pdfSrc; the prev domain's reconnection edge differs but its p_src
    // is the stored producer value — the reconnection-shift reuse does not
    // re-derive it, matching the GI reservoir's pass-through of the cached pdf).
    let prevReplayPdfAtCur = restirPtVisibleReplayPdfForDomain(rCur, woCur, rPrev.xs);
    updateReservoirPTWithHybrid(&rGris, rPrev.xs, rPrev.ns, rPrev.Lo, rPrev.pdfSrc, J, prevReplayPdfAtCur, rPrev.rngSeed, w_prev, &rng);
    rGris.M = oldM + prevM;
  }

  // GRIS finalise: W = w_sum / p̂ (the MIS weights already sum to 1 — no /M).
  finaliseReservoirPTWGris(&rGris, rptParams.wCap, params.cameraPos.xyz);
  // Refresh the reconnection-shift cache so downstream reuse (and next frame's
  // temporal step, since rGris becomes rPrev) sees a base edge rooted at THIS
  // pixel's visible vertex.
  refreshReconnectionCachePT(&rGris, params.cameraPos.xyz, params.frameSeed ^ pixelIdx);

  storeReservoirPTHero_rw(&rpt_resCurrent, pixelIdx, rGris);
}
`;
