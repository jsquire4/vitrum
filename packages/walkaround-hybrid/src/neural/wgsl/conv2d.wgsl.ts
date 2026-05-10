/**
 * conv2d.wgsl.ts — 2D convolution compute shader fragment.
 *
 * Implements a standard 2D convolution with SAME (zero-) padding:
 *
 *   output[y, x, oc] = Σ_{ky, kx, ic}
 *     input[y*stride + ky*dilation - padding, x*stride + kx*dilation - padding, ic]
 *     * weights[oc, ic, ky, kx]
 *     + bias[oc]
 *
 * Padding is automatically set to SAME when padding = floor(kernelH/2) for stride=1.
 * Dilation > 1 produces atrous (dilated) convolution — useful for multi-scale
 * receptive fields without additional parameters.
 *
 * Buffer layout (all row-major, channels-last for input/output):
 *   input:   [inputH × inputW × inputC]  f32
 *   weights: [outputC × inputC × kernelH × kernelW]  f32  (OIKK layout)
 *   bias:    [outputC]  f32
 *   output:  [outputH × outputW × outputC]  f32
 *
 * Output dimensions (SAME padding, stride s, dilation d):
 *   outputH = ceil(inputH / s)
 *   outputW = ceil(inputW / s)
 *
 * Workgroup: 8×8 threads tile the (x, y) output space; z indexes outputC.
 * Each thread computes one output element.
 *
 * References:
 *   Dumoulin, Visin "A guide to convolution arithmetic for deep learning"
 *   arXiv:1603.07285. https://arxiv.org/abs/1603.07285
 *
 *   WebGPU compute shader best practices:
 *   https://www.w3.org/TR/webgpu/#compute-pipeline
 *
 * @since Sprint 13, 2026-05-09
 */

export const CONV2D_WGSL = /* wgsl */ `

// ============================================================
// Conv2D parameter uniform (std140-compatible; 32 bytes)
// ============================================================
struct Conv2DParams {
  inputH:   u32,   // bytes  0-3
  inputW:   u32,   // bytes  4-7
  inputC:   u32,   // bytes  8-11
  kernelH:  u32,   // bytes 12-15
  kernelW:  u32,   // bytes 16-19
  outputC:  u32,   // bytes 20-23
  stride:   u32,   // bytes 24-27
  dilation: u32,   // bytes 28-31
};
// Total: 32 bytes — fits one 32-byte std140 block; no padding needed.

// ============================================================
// Bindings
// ============================================================
@group(0) @binding(0) var<storage, read>       conv_input:   array<f32>;  // HxWxC_in
@group(0) @binding(1) var<storage, read>       conv_weights: array<f32>;  // C_out x C_in x kH x kW (OIKK)
@group(0) @binding(2) var<storage, read>       conv_bias:    array<f32>;  // C_out
@group(0) @binding(3) var<storage, read_write> conv_output:  array<f32>;  // HxWxC_out
@group(0) @binding(4) var<uniform>             conv_params:  Conv2DParams;

// ============================================================
// Index helpers
// ============================================================

// Input: channels-last layout, SAME padding applied with zero-fill.
// Returns 0.0 if the (iy, ix) is out of bounds (zero-padding).
fn conv_input_val(iy: i32, ix: i32, ic: u32) -> f32 {
  let H = i32(conv_params.inputH);
  let W = i32(conv_params.inputW);
  if (iy < 0 || iy >= H || ix < 0 || ix >= W) { return 0.0; }
  let C = conv_params.inputC;
  let idx = (u32(iy) * u32(W) * C) + (u32(ix) * C) + ic;
  return conv_input[idx];
}

// Weight: OIKK layout — weight[oc, ic, ky, kx].
fn conv_weight_val(oc: u32, ic: u32, ky: u32, kx: u32) -> f32 {
  let C  = conv_params.inputC;
  let kH = conv_params.kernelH;
  let kW = conv_params.kernelW;
  let idx = oc * (C * kH * kW) + ic * (kH * kW) + ky * kW + kx;
  return conv_weights[idx];
}

// ============================================================
// Entry point
// ============================================================
@compute @workgroup_size(8, 8, 1)
fn conv2dKernel(@builtin(global_invocation_id) gid: vec3u) {
  // gid.x → output column (x / width dimension)
  // gid.y → output row    (y / height dimension)
  // gid.z → output channel

  let oc = gid.z;
  if (oc >= conv_params.outputC) { return; }

  // Output spatial dimensions (SAME padding with the given stride).
  let outputH = (conv_params.inputH + conv_params.stride - 1u) / conv_params.stride;
  let outputW = (conv_params.inputW + conv_params.stride - 1u) / conv_params.stride;

  let oy = gid.y;
  let ox = gid.x;
  if (oy >= outputH || ox >= outputW) { return; }

  // SAME padding offsets (half-kernel, integer division).
  let padH = conv_params.kernelH / 2u;
  let padW = conv_params.kernelW / 2u;

  // Accumulate: Σ_{ky, kx, ic} input[...] * weight[...]
  var acc: f32 = 0.0;
  for (var ky = 0u; ky < conv_params.kernelH; ky++) {
    for (var kx = 0u; kx < conv_params.kernelW; kx++) {
      // Input pixel position with dilation and stride.
      let iy = i32(oy * conv_params.stride + ky * conv_params.dilation) - i32(padH * conv_params.dilation);
      let ix = i32(ox * conv_params.stride + kx * conv_params.dilation) - i32(padW * conv_params.dilation);
      for (var ic = 0u; ic < conv_params.inputC; ic++) {
        acc += conv_input_val(iy, ix, ic) * conv_weight_val(oc, ic, ky, kx);
      }
    }
  }

  // Add bias for this output channel.
  acc += conv_bias[oc];

  // Write output: channels-last layout.
  let outIdx = (oy * outputW * conv_params.outputC) + (ox * conv_params.outputC) + oc;
  conv_output[outIdx] = acc;
}
`;
