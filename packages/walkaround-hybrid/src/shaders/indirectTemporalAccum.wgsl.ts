/**
 * Sprint 18 follow-up — indirect-channel pre-atrous temporal accumulator.
 *
 * The ReSTIR-GI reservoir re-rolls samples every frame (different M=8 RIS
 * candidates in risGi, different K=5 spatial neighbours in spatialGi), so
 * shade's per-pixel `Lo_indirect` reading from the reservoir varies
 * substantially frame-to-frame.  The atrous chain that follows is spatial-
 * only and preserves bright outliers (its chromaticity edge stop fires on
 * fireflies, suppressing the neighbour-smoothing that would otherwise
 * dilute them).  The main post-temporalAccum at α=0.01 admits each frame's
 * pattern at 1% — but new fireflies appear at *different* pixels each
 * frame, so the eye reads the moving pattern as "dancing" noise that
 * never settles.
 *
 * This pass solves both symptoms (red-region fireflies + shadow splotches)
 * by inserting a dedicated temporal accumulator *before* atrous-indirect:
 *
 *   1. Compute the current frame's 3×3 neighbourhood [min, max] in HDR
 *      linear space.  This is the local plausible range.
 *   2. TCBB-clip the previous-frame accumulator output to that range
 *      (anti-ghost: when lighting legitimately changes, history is pulled
 *      toward the new range).
 *   3. Clip the current frame's centre pixel to nmax * SPIKE_BOUND_MULT to
 *      reject fireflies (a centre pixel cannot exceed its brightest
 *      neighbour by more than this multiplier — kills isolated spikes
 *      without flattening edge transitions).
 *   4. α-blend the clipped current into the clipped history.
 *
 * α = 0.1 (vs the main post-atrous accumulator's 0.01) gives ~10-frame
 * effective history at this stage; the main accumulator stacks another
 * ~100-frame history on top, for ~1000-frame effective convergence on
 * the static-camera steady state.  Camera-move resets are handled by the
 * main accumulator's existing motion-reset path — when α=1 there, this
 * pre-accumulator's history is naturally flushed via end-of-frame copy.
 *
 * Bindings:
 *   @group(0) @binding(0) currentRaw  hdrIndirectTexture (sampled, full-res)
 *   @group(0) @binding(1) prevAccum   indirectAccum*PrevTexture (sampled)
 *   @group(0) @binding(2) outAccum    indirectAccum*Texture (storage write)
 */

export const INDIRECT_TEMPORAL_ACCUM_WGSL = /* wgsl */ `

@group(0) @binding(0) var ita_currentRaw: texture_2d<f32>;
@group(0) @binding(1) var ita_prevAccum:  texture_2d<f32>;
@group(0) @binding(2) var ita_outAccum:   texture_storage_2d<rgba16float, write>;

// Per-frame admit rate.  Lower = more history weight = smoother but slower
// to converge after a real lighting change.  0.1 is a balance: 90 % history
// per frame, so the temporal pattern from a single frame's reservoir choice
// gets averaged out over ~10 frames before atrous-indirect even reads it.
const ITA_ALPHA: f32 = 0.1;

// TCBB box expansion factor — slightly inflate the 3×3 [min,max] range so
// the clip doesn't bite into legitimate small variations.
const ITA_BBOX_EXPAND: f32 = 1.2;

// Firefly bound: the centre pixel cannot exceed its brightest neighbour by
// more than this multiplier.  Aggressive enough to kill isolated 5–10×
// spikes without flattening real bright edges (which have monotonic
// neighbours, so the bound moves with the edge).
const ITA_SPIKE_BOUND_MULT: f32 = 1.5;

@compute @workgroup_size(16, 16, 1)
fn indirectTemporalAccumMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(ita_outAccum);
  if (any(gid.xy >= dims)) { return; }

  let cur = textureLoad(ita_currentRaw, gid.xy, 0).rgb;

  // 3×3 neighbourhood [min, max] excluding the centre — so a firefly centre
  // doesn't enlarge its own bounding box.  If centre exceeds neighbours'
  // max by ITA_SPIKE_BOUND_MULT, it gets capped to that bound.
  var nmin = vec3f(1e10);
  var nmax = vec3f(-1e10);
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      if (dx == 0 && dy == 0) { continue; }
      let qx = i32(gid.x) + dx;
      let qy = i32(gid.y) + dy;
      if (qx < 0 || qy < 0 || u32(qx) >= dims.x || u32(qy) >= dims.y) { continue; }
      let c = textureLoad(ita_currentRaw, vec2u(u32(qx), u32(qy)), 0).rgb;
      nmin = min(nmin, c);
      nmax = max(nmax, c);
    }
  }

  // Firefly cap on the current centre.
  let cur_capped = min(cur, nmax * ITA_SPIKE_BOUND_MULT);

  // Anti-ghost clip on history: clamp prev to the (slightly expanded)
  // 3×3 neighbourhood range so a stale value can't haunt a region whose
  // lighting changed.
  let centre = (nmin + nmax) * 0.5;
  let half   = (nmax - nmin) * 0.5 * ITA_BBOX_EXPAND;
  let lo = max(vec3f(0.0), centre - half);
  let hi = centre + half;
  let prev = textureLoad(ita_prevAccum, gid.xy, 0).rgb;
  let prev_clipped = clamp(prev, lo, hi);

  let result = mix(prev_clipped, cur_capped, ITA_ALPHA);
  textureStore(ita_outAccum, gid.xy, vec4f(result, 1.0));
}
`;
