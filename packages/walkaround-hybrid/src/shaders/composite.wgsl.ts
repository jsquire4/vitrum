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

// The vertex shader emits a [0,1]² screen UV (location 0) alongside the
// clip-space position. The UV is the load-bearing value for the resolution-
// factor blit (see fragMain): because it is a vertex-shader output interpolated
// across the rasterized region, it spans 0→1 over the WHOLE swap-chain
// regardless of the swap-chain pixel count. `fragCoord` is swap-chain-pixel-
// sized (and therefore useless for indexing an internal-res texture under
// resolutionFactor < 1), but `uv` is resolution-independent.
export const COMPOSITE_VERT_WGSL = /* wgsl */ `
struct CompositeVaryings {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vertMain(@builtin(vertex_index) idx: u32) -> CompositeVaryings {
  // Large triangle covering the entire clip space.
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  let p = pos[idx];
  var out: CompositeVaryings;
  out.clip = vec4f(p, 0.0, 1.0);
  // Map clip-space → top-left-origin [0,1]² UV. The visible [-1,1]² region
  // maps to uv [0,1]²; uv(0,0) is the top-left texel (row 0). y is flipped
  // because clip-space +y is up but texture row 0 is the top.
  out.uv = vec2f(p.x * 0.5 + 0.5, p.y * -0.5 + 0.5);
  return out;
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
fn fragMain(in: CompositeVaryings) -> @location(0) vec4f {
  // Use textureLoad (nearest, no sampler) because rgba16float textures
  // require unfilterable-float sample type in WebGPU, which disallows
  // textureSample().
  //
  // RESOLUTION-FACTOR BLIT: denoisedTex is sized at the INTERNAL render
  // resolution (swap × resolutionFactor), which is SMALLER than the swap
  // chain when factor < 1. Indexing it with swap-chain fragCoord would
  // read out of bounds (->0) for every pixel outside the top-left internal
  // region, leaving the rest of the canvas black (the pre-fix bug).
  //
  // Instead we index by the resolution-independent screen UV: uv in [0,1]^2
  // spans the whole swap chain, so uv * internalDims lands inside
  // denoisedTex for any factor. With no filtering available this is a
  // nearest-neighbour upscale (factor < 1) or downscale (factor > 1) —
  // acceptable + intentional for the composite blit. At factor == 1 the
  // mapping is bit-identical to the old 1:1 index (uv*dims floors back to
  // the same texel).
  let dims = vec2f(textureDimensions(denoisedTex));
  // clamp to dims-1 so uv==1.0 (the bottom/right triangle edge) does not
  // round up to an out-of-range texel.
  let px = vec2u(min(in.uv * dims, dims - vec2f(1.0)));
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
