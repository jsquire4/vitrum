/**
 * screenCoordHelpers.wgsl — Screen-coordinate clamping helper.
 *
 * `clampCoord(c, w, h)` clamps a signed pixel coordinate to the valid
 * [0, w-1] × [0, h-1] texture range. Used by cbPrefill and resolve for
 * temporal reprojection — both passes need to clamp the motion-vector-offset
 * coordinate before a textureLoad call.
 *
 * Shared module (D5.4 dedup, 2026-06-10).
 *
 * @version 1 (2026-06-10 — D5.4 complexity-sweep dedup)
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const SCREEN_COORD_HELPERS_WGSL = /* wgsl */ `

// Clamp a signed pixel coordinate to the valid texture bounds [0, w-1] × [0, h-1].
fn clampCoord(c: vec2<i32>, w: u32, h: u32) -> vec2<i32> {
  let cx = clamp(c.x, 0, i32(w) - 1);
  let cy = clamp(c.y, 0, i32(h) - 1);
  return vec2<i32>(cx, cy);
}

`;

/**
 * Screen-coordinate clamping helper. Self-contained; no UBO or binding references.
 * Required by cbPrefill and resolve for motion-vector reprojection coordinate clamping.
 */
export const SCREEN_COORD_HELPERS_MODULE: WgslModule = {
  name: 'screenCoordHelpers',
  source: SCREEN_COORD_HELPERS_WGSL,
  requires: [],
};
