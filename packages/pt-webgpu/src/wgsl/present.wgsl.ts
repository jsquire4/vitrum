/**
 * present.wgsl.ts — per-frame tonemap / exposure / outputColorSpace present pass
 * for @vitrum/pt-webgpu.
 *
 * A compute pass that reads the raw linear-HDR accumulation texture
 * (`accumTex`, texture_2d<f32>, rgba16float, carries the running mean) and
 * writes a tonemapped / OETF-encoded result to `presentTex`
 * (texture_storage_2d<rgba16float, write>).
 *
 * Wired 2026-06-10: FrameQualitySettings.tonemap / .exposure / .outputColorSpace
 * contract fields live on pt-webgpu.  Defaults: aces(0), 1.0, srgb(0).
 *
 * CHANGE vs prior behavior: previously pt-webgpu returned the raw linear HDR
 * accumulation texture as primaryRadiance (no tonemap applied). The contract
 * default aces@1.0@srgb applies the Narkowicz ACES filmic curve + IEC 61966-2-1
 * OETF, changing rendered appearance vs the prior raw output. Hosts that
 * previously applied their own tonemap should set tonemap:'none' +
 * outputColorSpace:'linear' to recover raw linear values.
 *
 * Readback / adjoint paths (inverse session, OIDN inputs) continue consuming the
 * raw accumTexture directly — this pass does NOT touch accumTexture.
 *
 * PresentParams UBO layout (16 bytes, matches GpuResources.PRESENT_PARAMS_BYTES):
 *   u32 tonemapMode      — 0=aces(default) 1=agx 2=reinhard 3=linear 4=none
 *   f32 exposure         — linear exposure multiplier, default 1.0
 *   u32 outputColorSpace — 0=srgb(default, apply OETF) 1=linear(skip OETF)
 *   u32 _pad
 *
 * Tonemap operators match TONEMAP_MODE_INDEX from @vitrum/shared-samplers
 * (kept in lockstep by shared-samplers/__tests__/tonemap.test.ts).
 *
 * Bind group 0:
 *   binding 0 — PresentParams (uniform)
 *   binding 1 — accumTex      (texture_2d<f32>, rgba16float — the running mean)
 *   binding 2 — presentTex    (texture_storage_2d<rgba16float, write> — output)
 */

import { tonemapWgsl } from '@vitrum/shared-samplers';

export function buildPresentWgsl(): string {
  return /* wgsl */ `
struct PresentParams {
  tonemapMode:      u32,
  exposure:         f32,
  outputColorSpace: u32,
  _pad:             u32,
}

@group(0) @binding(0) var<uniform> presentParams: PresentParams;
@group(0) @binding(1) var          accumTex:      texture_2d<f32>;
@group(0) @binding(2) var          presentTex:    texture_storage_2d<rgba16float, write>;

${tonemapWgsl()}

@compute @workgroup_size(8, 8, 1)
fn presentMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(accumTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let px = vec2i(gid.xy);
  // accumTex stores the running Welford mean (written by accumulateFrame in the
  // kernel: display = accum.xyz / max(accum.w, 1.0)). The .rgb carries the per-pixel
  // HDR mean; .a is always 1 (the display-normalized write in the kernel).
  let hdr = textureLoad(accumTex, px, 0).rgb;
  let tonemapped = vitrumTonemap(max(vec3f(0.0), hdr), presentParams.tonemapMode, presentParams.exposure);
  var outColor: vec4f;
  if (presentParams.outputColorSpace == 0u) {
    // sRGB (default): apply IEC 61966-2-1 OETF before writing to the output texture.
    outColor = vec4f(vt_linearToSrgb(tonemapped), 1.0);
  } else {
    // linear: skip the OETF — write raw tonemapped linear values.
    outColor = vec4f(tonemapped, 1.0);
  }
  textureStore(presentTex, px, outColor);
}
`;
}

/** Lazily-built present WGSL (built once; the tonemap block is a fixed string). */
export const PT_WEBGPU_PRESENT_WGSL: string = buildPresentWgsl();
