/**
 * bilinearUpsample.wgsl.ts — Bilinear 2× upsampling compute shader.
 *
 * Doubles both spatial dimensions (H → 2H, W → 2W) using bilinear interpolation.
 * Used in the UNet decoder path before the transposed convolution, or as an
 * alternative to transposed convolution for simpler scaling.
 *
 * For each output pixel (oy, ox), the source coordinate in the input image is:
 *
 *   sx = (ox + 0.5) / scale - 0.5
 *   sy = (oy + 0.5) / scale - 0.5
 *
 * This aligns pixel centers. The four nearest input pixels are sampled and
 * blended bilinearly with weights derived from the fractional coordinate offsets.
 *
 * Buffer layout (channels-last):
 *   input:  [inputH × inputW × channels]  f32
 *   output: [outputH × outputW × channels]  f32
 *   outputH = inputH × scale
 *   outputW = inputW × scale
 *
 * The scale factor is always 2 (2× upsample) per the UNet architecture. A
 * UpsampleParams uniform carries the dimensions and channel count.
 *
 * Workgroup: 8×8 × 1 threads tile (ox, oy, 0); z not used (channels handled
 * in the inner loop per thread for cache coherence).
 *
 * References:
 *   Keys "Cubic convolution interpolation for digital image processing"
 *   IEEE ASSP 1981. (Bilinear is the order-1 case.)
 *
 *   Odena, Dumoulin, Olah "Deconvolution and Checkerboard Artifacts"
 *   Distill 2016. https://distill.pub/2016/deconv-checkerboard/
 *   (Bilinear upsample + conv avoids checkerboard; preferred over pure transposed conv.)
 *
 * @since Sprint 13, 2026-05-09
 */

export const BILINEAR_UPSAMPLE_WGSL = /* wgsl */ `

// ============================================================
// Upsample parameter uniform (std140-compatible; 16 bytes)
// ============================================================
struct UpsampleParams {
  inputH:   u32,   // bytes  0-3
  inputW:   u32,   // bytes  4-7
  channels: u32,   // bytes  8-11
  scale:    u32,   // bytes 12-15  — must be 2 for the UNet decoder path
};

// ============================================================
// Bindings
// ============================================================
@group(0) @binding(0) var<storage, read>       ups_input:  array<f32>;  // inputH×inputW×channels
@group(0) @binding(1) var<storage, read_write> ups_output: array<f32>;  // (inputH×scale)×(inputW×scale)×channels
@group(0) @binding(2) var<uniform>             ups_params: UpsampleParams;

// ============================================================
// Read one input pixel, clamping OOB to border.
// ============================================================
fn ups_read(iy: i32, ix: i32, ch: u32) -> f32 {
  let H  = i32(ups_params.inputH);
  let W  = i32(ups_params.inputW);
  let cy = clamp(iy, 0, H - 1);
  let cx = clamp(ix, 0, W - 1);
  let idx = u32(cy) * ups_params.inputW * ups_params.channels
          + u32(cx) * ups_params.channels
          + ch;
  return ups_input[idx];
}

// ============================================================
// Entry point — one thread computes all channels at (oy, ox).
// ============================================================
@compute @workgroup_size(8, 8, 1)
fn bilinearUpsampleKernel(@builtin(global_invocation_id) gid: vec3u) {
  let s       = ups_params.scale;
  let outputH = ups_params.inputH * s;
  let outputW = ups_params.inputW * s;

  let oy = gid.y;
  let ox = gid.x;
  if (oy >= outputH || ox >= outputW) { return; }

  // Map output pixel center to input space.
  // Center-aligned: src = (out + 0.5) / scale - 0.5
  let sf  = f32(s);
  let sx  = (f32(ox) + 0.5) / sf - 0.5;
  let sy  = (f32(oy) + 0.5) / sf - 0.5;

  // Integer parts and fractional weights.
  let ix0 = i32(floor(sx));
  let iy0 = i32(floor(sy));
  let ix1 = ix0 + 1;
  let iy1 = iy0 + 1;
  let fx  = sx - f32(ix0);
  let fy  = sy - f32(iy0);

  // Bilinear weights for the four corners.
  let w00 = (1.0 - fx) * (1.0 - fy);
  let w10 = fx          * (1.0 - fy);
  let w01 = (1.0 - fx) * fy;
  let w11 = fx          * fy;

  let C = ups_params.channels;
  for (var ch = 0u; ch < C; ch++) {
    let v = ups_read(iy0, ix0, ch) * w00
          + ups_read(iy0, ix1, ch) * w10
          + ups_read(iy1, ix0, ch) * w01
          + ups_read(iy1, ix1, ch) * w11;

    let outIdx = oy * outputW * C + ox * C + ch;
    ups_output[outIdx] = v;
  }
}
`;
