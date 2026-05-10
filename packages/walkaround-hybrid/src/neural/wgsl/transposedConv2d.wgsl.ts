/**
 * transposedConv2d.wgsl.ts — Transposed (deconvolutional) 2D convolution compute shader.
 *
 * Used in the UNet decoder path to upsample spatial resolution. Each decoder level
 * doubles H and W via a transposed convolution (stride=2, kernelH=kernelW=2 by default).
 *
 * Mathematical equivalence: transposed convolution with stride s is equivalent to
 * inserting (s-1) zeros between input elements (fractionally-strided convolution),
 * then applying a regular convolution with SAME padding.  We implement it via
 * the "scatter" (output-centric) formulation: each output pixel accumulates
 * contributions from all input pixels whose receptive field overlaps it.
 *
 * Buffer layout (all channels-last):
 *   input:   [inputH × inputW × inputC]  f32
 *   weights: [inputC × outputC × kernelH × kernelW]  f32  (IOKK layout)
 *   bias:    [outputC]  f32
 *   output:  [outputH × outputW × outputC]  f32
 *
 * Output dimensions (transposed convolution, stride s):
 *   outputH = inputH × s
 *   outputW = inputW × s
 *
 * The kernel uses the "gather" equivalent: for each output pixel, identify which
 * input pixels and weight kernel positions contribute, then sum.
 *
 * Workgroup: 8×8 threads tile (ox, oy); z indexes outputC.
 *
 * References:
 *   Dumoulin, Visin "A guide to convolution arithmetic for deep learning"
 *   arXiv:1603.07285 §4. https://arxiv.org/abs/1603.07285
 *
 *   Zeiler et al. "Deconvolutional Networks" CVPR 2010.
 *   https://doi.org/10.1109/CVPR.2010.5539957
 *
 * @since Sprint 13, 2026-05-09
 */

export const TRANSPOSED_CONV2D_WGSL = /* wgsl */ `

// ============================================================
// TransposedConv2D parameter uniform (std140-compatible; 32 bytes)
// ============================================================
struct TransposedConv2DParams {
  inputH:   u32,   // bytes  0-3
  inputW:   u32,   // bytes  4-7
  inputC:   u32,   // bytes  8-11
  kernelH:  u32,   // bytes 12-15
  kernelW:  u32,   // bytes 16-19
  outputC:  u32,   // bytes 20-23
  stride:   u32,   // bytes 24-27
  _pad:     u32,   // bytes 28-31  (pad to 32 bytes)
};

// ============================================================
// Bindings
// ============================================================
@group(0) @binding(0) var<storage, read>       tconv_input:   array<f32>;  // inputH×inputW×inputC
@group(0) @binding(1) var<storage, read>       tconv_weights: array<f32>;  // inputC×outputC×kH×kW (IOKK)
@group(0) @binding(2) var<storage, read>       tconv_bias:    array<f32>;  // outputC
@group(0) @binding(3) var<storage, read_write> tconv_output:  array<f32>;  // outputH×outputW×outputC
@group(0) @binding(4) var<uniform>             tconv_params:  TransposedConv2DParams;

// ============================================================
// Index helpers
// ============================================================

// Read input value; returns 0.0 for OOB (shouldn't happen in gather path).
fn tconv_input_val(iy: u32, ix: u32, ic: u32) -> f32 {
  if (iy >= tconv_params.inputH || ix >= tconv_params.inputW) { return 0.0; }
  let C   = tconv_params.inputC;
  let W   = tconv_params.inputW;
  let idx = iy * W * C + ix * C + ic;
  return tconv_input[idx];
}

// Weight: IOKK layout — weight[ic, oc, ky, kx].
fn tconv_weight_val(ic: u32, oc: u32, ky: u32, kx: u32) -> f32 {
  let OC  = tconv_params.outputC;
  let kH  = tconv_params.kernelH;
  let kW  = tconv_params.kernelW;
  let idx = ic * (OC * kH * kW) + oc * (kH * kW) + ky * kW + kx;
  return tconv_weights[idx];
}

// ============================================================
// Entry point — gather formulation
//
// For output pixel (oy, ox, oc):
//   acc = Σ_{ic, ky, kx} such that:
//     (oy + padH - ky) mod stride == 0
//     (ox + padW - kx) mod stride == 0
//   of  input[(oy+padH-ky)/stride, (ox+padW-kx)/stride, ic]
//       * weight[ic, oc, ky, kx]
//
// SAME padding (padH = kH/2, padW = kW/2) — output size = input × stride.
// ============================================================
@compute @workgroup_size(8, 8, 1)
fn transposedConv2dKernel(@builtin(global_invocation_id) gid: vec3u) {
  let oc = gid.z;
  if (oc >= tconv_params.outputC) { return; }

  let s       = tconv_params.stride;
  let outputH = tconv_params.inputH * s;
  let outputW = tconv_params.inputW * s;

  let oy = gid.y;
  let ox = gid.x;
  if (oy >= outputH || ox >= outputW) { return; }

  let kH   = tconv_params.kernelH;
  let kW   = tconv_params.kernelW;
  let padH = kH / 2u;
  let padW = kW / 2u;

  var acc: f32 = 0.0;

  for (var ky = 0u; ky < kH; ky++) {
    // Is this kernel row compatible with this output row?
    let oyPadKy = oy + padH;
    if (oyPadKy < ky) { continue; }
    let rem_y = (oyPadKy - ky) % s;
    if (rem_y != 0u) { continue; }
    let iy = (oyPadKy - ky) / s;
    if (iy >= tconv_params.inputH) { continue; }

    for (var kx = 0u; kx < kW; kx++) {
      let oxPadKx = ox + padW;
      if (oxPadKx < kx) { continue; }
      let rem_x = (oxPadKx - kx) % s;
      if (rem_x != 0u) { continue; }
      let ix = (oxPadKx - kx) / s;
      if (ix >= tconv_params.inputW) { continue; }

      for (var ic = 0u; ic < tconv_params.inputC; ic++) {
        acc += tconv_input_val(iy, ix, ic) * tconv_weight_val(ic, oc, ky, kx);
      }
    }
  }

  acc += tconv_bias[oc];

  let OC     = tconv_params.outputC;
  let outIdx = oy * outputW * OC + ox * OC + oc;
  tconv_output[outIdx] = acc;
}
`;
