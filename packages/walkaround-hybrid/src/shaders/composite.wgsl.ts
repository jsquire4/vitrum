/**
 * Composite render pass — blits the denoised ReSTIR HDR output to the swap-chain.
 *
 * Runs as a WebGPU render pass (not compute) because the WebGPU spec does not
 * allow compute shaders to write directly to the swap-chain texture.
 *
 * The fullscreen triangle covers the entire viewport via:
 *   NDC vertex 0: (-1, -1)
 *   NDC vertex 1: (3, -1)
 *   NDC vertex 2: (-1, 3)
 * This is the standard "clip-space large triangle" trick that avoids a geometry
 * buffer; the rasterizer clips to [-1,1] automatically.
 *
 * No vertex buffer required — vertex positions are computed from vertex_index
 * in the vertex shader.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const COMPOSITE_VERT_WGSL = /* wgsl */ `
@vertex
fn vertMain(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  // Large triangle covering the entire clip space
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  return vec4f(pos[idx], 0.0, 1.0);
}
`;

export const COMPOSITE_FRAG_WGSL = /* wgsl */ `
@group(0) @binding(0) var denoisedTex: texture_2d<f32>;

// Linear → sRGB conversion (IEC 61966-2-1, the standard gamma curve).
// Applies to the denoised linear HDR output before writing to the swap-chain.
// Without this the display shows linear values interpreted by the monitor as
// gamma-encoded, making mid-tones ~5× too dark (0.5 linear = 0.22 display).
fn linearToSRGB(c: vec3f) -> vec3f {
  // Piece-wise sRGB transfer function:
  //   c ≤ 0.0031308 → 12.92 × c
  //   c > 0.0031308 → 1.055 × c^(1/2.4) − 0.055
  let cutoff = vec3f(0.0031308);
  let lo = c * 12.92;
  let hi = pow(max(c, cutoff), vec3f(1.0 / 2.4)) * 1.055 - 0.055;
  return select(hi, lo, c <= cutoff);
}

// ACES filmic tone mapping — Narkowicz 2015 RRT+ODT fit, applied per-channel.
// Matches three.js's ACESFilmicToneMapping (the R3F default that PT and raster
// inherit). Same curve as src/.../tone_mapping/aces.glsl.js in three.js.
fn acesFilm(rgb: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fragMain(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
  // Use textureLoad (nearest, no sampler) because rgba16float textures
  // require unfilterable-float sample type in WebGPU, which disallows
  // textureSample(). Nearest-pixel composite is correct at 1:1 pixel mapping.
  let px = vec2u(u32(fragCoord.x), u32(fragCoord.y));
  let hdr = textureLoad(denoisedTex, px, 0).rgb;

  // Per-channel ACES at exposure 1.0 — matches three.js's R3F default
  // toneMapping=ACESFilmic + toneMappingExposure=1.0 that PT and raster use.
  let tonemapped = acesFilm(max(vec3f(0.0), hdr));

  // Apply sRGB gamma encoding before writing to the 8-bit swap-chain.
  // The swap-chain format is bgra8unorm (NOT bgra8unorm-srgb), so the GPU
  // does NOT auto-convert on store — we must apply the curve manually.
  let srgb = linearToSRGB(tonemapped);
  return vec4f(srgb, 1.0);
}
`;

/** W1-R6 — declarative include-graph entries. Vert and frag are
 *  independent shader modules; neither prepends COMMON_WGSL. */
export const COMPOSITE_VERT_MODULE: WgslModule = {
  name: 'compositeVert',
  source: COMPOSITE_VERT_WGSL,
  requires: [],
};
export const COMPOSITE_FRAG_MODULE: WgslModule = {
  name: 'compositeFrag',
  source: COMPOSITE_FRAG_WGSL,
  requires: [],
};
