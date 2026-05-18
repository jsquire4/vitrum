/**
 * Host-app stained-glass procedural surface-texture pattern functions.
 *
 * W7-H6 — extracted from the former `surfaceTextures.wgsl.ts`. This module
 * owns the eight authored stained-glass patterns (smooth/hammered/ripple/
 * granite/baroque/waterglass/catspaw/flemish) plus the `surfaceTextureMod`
 * dispatch switch over `texId` 0..7.
 *
 * Layering rationale (matches the in-flight library-vs-host split):
 *   - The library-general per-channel BVH shadow walker lives in
 *     `../glassVisibility.wgsl.ts` (any glass+sunlight engine needs it).
 *   - This file is the HOST app's authored pattern catalogue — the only
 *     consumer outside the stained-glass demo is the visibility walker,
 *     which calls `surfaceTextureMod` by name; the walker therefore takes
 *     a hard dependency on this module being composed in first.
 *
 * Eventual home: when the new `@vitrum/stained-glass-extensions` package
 * lands (W3-D2+D3), this file moves there as-is. It stays inside
 * walkaround-hybrid for THIS round under a clearly-labeled sub-module so
 * the cross-package move is the only follow-up.
 */

import type { WgslModule } from '../../pipeline/wgslComposer.js';

export const STAINED_GLASS_SURFACE_MODS_WGSL = /* wgsl */ `

// ============================================================
// Procedural surface-texture pattern functions
// ============================================================
//
// One function per authored surface-texture name. Each takes the hit's
// interpolated UV (already in [0,1]) and returns a scalar modulation
// factor — multiplied into the cell's emission to produce visible
// per-pixel colour variation.
//
// These functions are faithful WGSL re-implementations of the GLSL
// surface bakers — the same trig + noise math, evaluated per-shader-
// invocation instead of per-texel-during-bake.  They DO NOT hardcode
// colours; they only produce the *pattern*, which the shade pass
// multiplies into the per-cell baseColor it decoded from bvhIndex.

fn _hash21(p: vec2f) -> f32 {
  // Quick deterministic 2D-to-1D hash, range ~[0,1).
  let h = sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453;
  return fract(h);
}

fn _vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = _hash21(i);
  let b = _hash21(i + vec2f(1.0, 0.0));
  let c = _hash21(i + vec2f(0.0, 1.0));
  let d = _hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn _fbm(p: vec2f) -> f32 {
  // 4-octave fractal Brownian motion.
  var v = 0.0;
  var a = 0.5;
  var pp = p;
  for (var k = 0; k < 4; k = k + 1) {
    v = v + a * _vnoise(pp);
    pp = pp * 2.03;
    a  = a  * 0.5;
  }
  return v;
}

fn _waterglassMod(uv: vec2f) -> f32 {
  let k = 12.0;
  let w1 = sin(uv.x * k * 4.0 + uv.y * 2.0) * 0.4;
  let w2 = sin(uv.y * k * 5.0 - uv.x * 1.5) * 0.3;
  let w3 = sin((uv.x + uv.y) * k * 3.0) * 0.2;
  let w4 = sin((uv.x - uv.y) * k * 6.0) * 0.1;
  return 0.95 + (w1 + w2 + w3 + w4) * 0.45;
}

fn _rippleMod(uv: vec2f) -> f32 {
  let p = uv - vec2f(0.5);
  let r = length(p) * 22.0;
  let s = sin(r);
  return 0.95 + s * 0.4;
}

fn _hammeredMod(uv: vec2f) -> f32 {
  let n = _vnoise(uv * 14.0);
  let n2 = _vnoise(uv * 28.0 + vec2f(11.0, 7.0));
  return 0.85 + (n - 0.5) * 0.7 + (n2 - 0.5) * 0.25;
}

fn _graniteMod(uv: vec2f) -> f32 {
  let n = _vnoise(uv * 32.0);
  let n2 = _vnoise(uv * 64.0 + vec2f(3.0, 9.0));
  return 0.85 + (n - 0.5) * 0.45 + (n2 - 0.5) * 0.2;
}

fn _baroqueMod(uv: vec2f) -> f32 {
  let warp = vec2f(_fbm(uv * 2.0), _fbm(uv * 2.0 + vec2f(7.3, 1.7)));
  let v = _fbm(uv * 3.5 + warp * 1.5);
  return 0.7 + v * 0.7;
}

fn _catspawMod(uv: vec2f) -> f32 {
  let lattice = sin(uv.x * 18.0) * sin(uv.y * 18.0);
  let n = _vnoise(uv * 8.0);
  return 0.95 + lattice * 0.25 + (n - 0.5) * 0.3;
}

fn _flemishMod(uv: vec2f) -> f32 {
  let stripes = sin(uv.y * 28.0 + sin(uv.x * 6.0) * 1.2);
  let n = _vnoise(uv * vec2f(20.0, 4.0));
  return 0.9 + stripes * 0.3 + (n - 0.5) * 0.25;
}

/**
 * Procedural surface modulation factor for a glass cell.
 * Returns a single scalar that the shade pass multiplies into the
 * cell's emitted radiance, producing visible per-pixel patterns.
 *
 * Result is clamped to [0.2, 1.8] so cells stay distinguishably
 * coloured (no full black-out, no over-bright NaN-prone values).
 */
fn surfaceTextureMod(uv: vec2f, texId: u32) -> f32 {
  var m: f32 = 1.0;
  switch (texId) {
    case 0u: { m = 1.0; }                   // smooth — flat
    case 1u: { m = _hammeredMod(uv); }
    case 2u: { m = _rippleMod(uv); }
    case 3u: { m = _graniteMod(uv); }
    case 4u: { m = _baroqueMod(uv); }
    case 5u: { m = _waterglassMod(uv); }
    case 6u: { m = _catspawMod(uv); }
    case 7u: { m = _flemishMod(uv); }
    default: { m = 1.0; }
  }
  // Wider clamp range [0.2, 1.8] for more dramatic per-pixel texture modulation.
  return clamp(m, 0.2, 1.8);
}
`;

/** W1-R6 — declarative include-graph entry. */
export const STAINED_GLASS_SURFACE_MODS_MODULE: WgslModule = {
  name: 'surfaceMods',
  source: STAINED_GLASS_SURFACE_MODS_WGSL,
  requires: ['common'],
};
