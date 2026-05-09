/**
 * Temporal accumulation pass — runs once per frame between atrous and
 * composite. Blends the current frame's atrous output with the previous
 * frame's accumulated HDR via:
 *
 *   accum = current * alpha + clamped_prev * (1 - alpha)
 *
 * **Variance-clamped history** (added 2026-05-08 for the user-reported
 * "sparkle around came/solder" + "rays fading and respawning" symptom).
 * The naive EMA blends the previous frame BLINDLY with the current,
 * which means at silhouette edges (came-glass boundary) the prev-frame
 * value can carry a color very different from what's currently visible
 * — the noise pattern flickers between "what was here last frame" and
 * "what's here now." Variance clamping fixes this by computing the local
 * color statistics (mean ± k·std_dev) over a 3×3 current-frame
 * neighborhood and clamping the prev-frame value into that range BEFORE
 * blending. Pixels at edges retain their crispness because:
 *   - Glass-side pixels have neighborhoods of similar bright-colored
 *     samples → prev-frame clamped to bright color
 *   - Came-side pixels have neighborhoods of dark metallic samples →
 *     prev-frame clamped to dark
 * The boundary stops accumulating cross-talk between the two regions.
 *
 * On big camera motion the alpha is set to 1.0 by the caller, so
 * variance clamping is bypassed (no history blend).
 *
 * Reference: Karis, "High-Quality Temporal Supersampling" (Unreal 4),
 * SIGGRAPH 2014. The variance-clamping technique is now standard in
 * temporal denoisers (Lumen, NRD, etc.).
 */

export const TEMPORAL_ACCUM_WGSL = /* wgsl */`

struct AccumUBO {
  alpha: f32,        // [0, 1] blend weight on the current frame
  varianceK: f32,    // std-dev multiplier for the clamp box (typical 1.0-2.0)
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var currentAtrous : texture_2d<f32>;
@group(0) @binding(1) var prevAccum     : texture_2d<f32>;
@group(0) @binding(2) var accumOut      : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> ubo  : AccumUBO;

@compute @workgroup_size(16, 16, 1)
fn temporalAccumMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(currentAtrous);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let px = vec2u(gid.x, gid.y);
  let curr = textureLoad(currentAtrous, px, 0).rgb;

  // Fast path on big motion — alpha ≈ 1 means history weight is ~0,
  // so the variance-clamp work would be wasted. Threshold conservatively.
  if (ubo.alpha > 0.95) {
    textureStore(accumOut, px, vec4f(curr, 1.0));
    return;
  }

  // ── 3×3 neighborhood AABB on the current frame ──────────────────────
  // Was 5×5; reduced to 3×3 to shrink the rim-brightening artifact at
  // caustic boundaries. With 5×5, an edge pixel's neighborhood spans 2px
  // into the bright caustic interior, pulling prev-frame value up to
  // bright max. The temporal blend then over-weights the rim, reading
  // as 3D-shaded volume on what should be flat caustic patches.
  // 3×3 keeps the clamp's role (suppress slow-motion ghosting without
  // motion vectors) but limits the rim radius to ≤1px.
  var aabbMin = vec3f(1e10);
  var aabbMax = vec3f(-1e10);
  let dimsI = vec2i(dims);
  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      let p = vec2i(gid.xy) + vec2i(dx, dy);
      if (p.x < 0 || p.y < 0 || p.x >= dimsI.x || p.y >= dimsI.y) { continue; }
      let c = textureLoad(currentAtrous, vec2u(p), 0).rgb;
      aabbMin = min(aabbMin, c);
      aabbMax = max(aabbMax, c);
    }
  }

  // Read the previous-frame accumulator and clamp it to the local
  // neighborhood AABB. Stale history outside the box (e.g. cross-region
  // bleed) gets snapped back; values inside the box pass through
  // unchanged so the EMA can integrate noise normally.
  let prevRaw = textureLoad(prevAccum, px, 0).rgb;
  let prev = clamp(prevRaw, aabbMin, aabbMax);

  let blended = curr * ubo.alpha + prev * (1.0 - ubo.alpha);
  textureStore(accumOut, px, vec4f(blended, 1.0));
}
`;
