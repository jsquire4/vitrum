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
 *
 * Tonemap operators via `vitrumTonemap` from `@vitrum/shared-samplers`:
 *   mode 0 = aces (default — historical Narkowicz 2015 filmic curve)
 *   mode 1 = agx (Wrensch minimal log2 sigmoid)
 *   mode 2 = reinhard (x / (1+x) per channel)
 *   mode 3 = linear (exposure + clamp only)
 *   mode 4 = none (raw HDR, no operator)
 * Wired 2026-06-10: FrameQualitySettings.tonemap / .exposure / .outputColorSpace live.
 */

import { tonemapWgsl } from '@vitrum/shared-samplers';
import type { WgslModule } from '../pipeline/wgslComposer.js';

const COMPOSITE_VARYINGS_WGSL = /* wgsl */ `
struct CompositeVaryings {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
}
`;

// The vertex shader emits a [0,1]² screen UV (location 0) alongside the
// clip-space position. The UV is the load-bearing value for the resolution-
// factor blit (see fragMain): because it is a vertex-shader output interpolated
// across the rasterized region, it spans 0→1 over the WHOLE swap-chain
// regardless of the swap-chain pixel count. `fragCoord` is swap-chain-pixel-
// sized (and therefore useless for indexing an internal-res texture under
// resolutionFactor < 1), but `uv` is resolution-independent.
export const COMPOSITE_VERT_WGSL = /* wgsl */ `
${COMPOSITE_VARYINGS_WGSL}

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

function buildCompositFragWgsl(): string {
  return /* wgsl */ `
${COMPOSITE_VARYINGS_WGSL}

@group(0) @binding(0) var denoisedTex: texture_2d<f32>;
@group(0) @binding(1) var _compositeSampler: sampler;

// CompositeUniforms — per-frame tonemap/exposure/outputColorSpace dials.
// tonemapMode: 0=aces(default) 1=agx 2=reinhard 3=linear(clamped) 4=none.
// outputColorSpace: 0=srgb(default, apply OETF) 1=linear(skip OETF).
struct CompositeUniforms {
  tonemapMode:      u32,
  exposure:         f32,
  outputColorSpace: u32,
  _pad:             u32,
}
@group(0) @binding(2) var<uniform> compositeParams: CompositeUniforms;

${tonemapWgsl()}

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

  // Apply exposure + tonemap operator selected per-frame by
  // FrameQualitySettings.tonemap (default: aces, mode=0) and .exposure
  // (default: 1.0). vitrumTonemap is the GPU twin of applyTonemap() in
  // @vitrum/shared-samplers (kept in lockstep by shared-samplers tests).
  let tonemapped = vitrumTonemap(max(vec3f(0.0), hdr), compositeParams.tonemapMode, compositeParams.exposure);

  // outputColorSpace: 0=srgb (default) → apply the IEC 61966-2-1 OETF before
  // writing to the 8-bit swap-chain (bgra8unorm is NOT auto-converted by the
  // GPU). outputColorSpace: 1=linear → skip the OETF, write raw tonemapped
  // linear values (useful for HDR/linear pipeline hosts or screenshot capture).
  // The sRGB OETF (vt_linearToSrgb) is defined in the vitrumTonemap block above.
  if (compositeParams.outputColorSpace == 0u) {
    return vec4f(vt_linearToSrgb(tonemapped), 1.0);
  }
  return vec4f(tonemapped, 1.0);
}
`;
}

/** Lazily-built fragment source (built once; the tonemap WGSL block is
 *  a fixed string so this is stable across the lifetime of the module). */
export const COMPOSITE_FRAG_WGSL: string = buildCompositFragWgsl();

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
