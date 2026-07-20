/**
 * temporalGiCommon.wgsl.ts — helpers shared verbatim by BOTH temporal-GI
 * bodies (the OFF/default `TEMPORAL_GI_WGSL` and the GRIS opt-in
 * `TEMPORAL_GI_GRIS_WGSL`).
 *
 * Holds the geometric-rejection constants (`DEPTH_REL_TOL`, `NORMAL_DOT_MIN`)
 * and the previous-frame reprojection helper `projectToPrevHalfPx`. Both bodies
 * source this once instead of inlining two byte-identical copies.
 *
 * NOTE — the formerly-co-located `worldFromHalfPx_temporal` helper was DEAD
 * (defined in both copies, called by neither — the pass reprojects via
 * `rCur.xv`, not a depth-reconstructed world point) and has been DELETED. It is
 * intentionally NOT reproduced here. This is the single behavioural delta of
 * the temporalGiCommon dedup: an unused function disappears from the composed
 * string; every executed instruction is unchanged.
 *
 * The fragment begins exactly where the inline copy began (the
 * "Geometric-rejection thresholds" comment) and ends at the closing brace of
 * `projectToPrevHalfPx`, so each body interpolates it where the inline block
 * used to live with no surrounding-byte change.
 */
// D8-7 (complexity-sweep 2026-07-20, T4-4): the 11-line temporal-GI M-clamp
// rationale comment is byte-identical between the OFF and GRIS bodies. Shared
// here (raw string, no bindings — it's a comment) and interpolated verbatim at
// each body's pre-`${TEMPORAL_GI_COMMON_WGSL}` position so both composed shaders
// stay byte-identical. Emitted with NO trailing newline; the call site controls
// the surrounding whitespace.
export const TEMPORAL_GI_MCLAMP_COMMENT_WGSL = /* wgsl */ `// The temporal-GI M clamp (ubo.restirGiMClamp, Cornell default 50)
// controls how strongly the previous-frame reservoir dominates temporal
// reuse.  Higher = the chosen sample changes less often per-pixel → less
// per-frame pattern jitter (the temporal accumulator's per-frame
// contribution looks stabler).  Bitterli 2020 uses M=20 for ReSTIR-DI;
// Majercik 2021 §4.5 suggests ~30–100 for GI since the indirect signal
// varies less per pixel than DI light-source swaps.  Empirically 50 cuts
// visible pattern dance on Cornell static frames in half compared to 20
// without introducing motion lag (the camera-move reset path forces α=1
// and discards prev independently). Library consumers override via
// HybridEngineOptions.restirGiMClamp.`;

export const TEMPORAL_GI_COMMON_WGSL = /* wgsl */ `// Geometric-rejection thresholds for "is this the same world surface".
// 0.1 × current depth = 10 % depth tolerance — generous enough for sub-pixel
// jitter and 1-frame camera motion, tight enough to reject occlusion changes.
const DEPTH_REL_TOL: f32 = 0.1;
const NORMAL_DOT_MIN: f32 = 0.906; // cos(25°)

fn projectToPrevHalfPx(worldPos: vec3f, halfDims: vec2u, fullDims: vec2u) -> vec2i {
  let prevClip = ubo.prevViewProjMatrix * vec4f(worldPos, 1.0);
  if (prevClip.w <= 1e-6) { return vec2i(-1, -1); }
  let ndc = prevClip.xyz / prevClip.w;
  if (any(abs(ndc.xy) > vec2f(1.0))) { return vec2i(-1, -1); }
  let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let fullPxF = uv * vec2f(f32(fullDims.x), f32(fullDims.y));
  let halfPxF = fullPxF * 0.5;
  return vec2i(floor(halfPxF));
}`;
