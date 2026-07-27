import { NEURAL_FINITE_WGSL } from '../preprocessing.js';

/**
 * transposedConv2d.wgsl.ts — 2D transposed convolution (deconvolution) kernel.
 *
 * Matches PyTorch ConvTranspose2d:
 *   outputH = (inputH - 1) * stride - 2 * padding
 *           + dilation * (kH - 1) + outputPadding + 1
 *   outputW = (inputW - 1) * stride - 2 * padding
 *           + dilation * (kW - 1) + outputPadding + 1
 *
 * Weight layout: IOKW — inputC × outputC × kH × kW (standard PyTorch ConvTranspose2d).
 * Each weight at [i, o, kh, kw] → index: i*outC*kH*kW + o*kH*kW + kh*kW + kw
 *
 * Canonical binding layout (same as conv2d):
 *   @group(0) @binding(0)  input   : array<f32>   — input tensor [H × W × inC]
 *   @group(0) @binding(1)  weights : array<f32>   — IOKW layout [inC × outC × kH × kW]
 *   @group(0) @binding(2)  biases  : array<f32>   — [outC]
 *   @group(0) @binding(3)  output  : array<f32>   — output tensor [H*stride × W*stride × outC]
 *   @group(0) @binding(4)  params  : TConv2DParams — shape uniform
 *
 * Workgroup: 8×8×1 over (outH, outW, outC).
 */

export const TRANSPOSED_CONV2D_WGSL = /* wgsl */`${NEURAL_FINITE_WGSL}
struct TConv2DParams {
  inputH  : u32,
  inputW  : u32,
  inC     : u32,
  outC    : u32,
  kH      : u32,   // kernel height (vitrum U-Net uses kH=2)
  kW      : u32,   // kernel width  (vitrum U-Net uses kW=2)
  stride  : u32,   // (vitrum U-Net uses stride=2)
  padding : u32,   // (vitrum U-Net uses padding=0)
  dilation: u32,   // kernel spacing (vitrum U-Net uses dilation=1)
  outputPadding: u32, // output-shape adjustment (vitrum U-Net uses 0)
  _reserved0: u32,
  _reserved1: u32,
}

// Canonical binding layout:
//   0 = input, 1 = weights, 2 = biases, 3 = output, 4 = params
@group(0) @binding(0) var<storage, read>       input   : array<f32>;
@group(0) @binding(1) var<storage, read>       weights : array<f32>;
@group(0) @binding(2) var<storage, read>       biases  : array<f32>;
@group(0) @binding(3) var<storage, read_write> output  : array<f32>;
@group(0) @binding(4) var<uniform>             params  : TConv2DParams;

@compute @workgroup_size(8, 8, 1)
fn transposedConv2dMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let effectiveKH = params.dilation * (params.kH - 1u) + 1u;
  let effectiveKW = params.dilation * (params.kW - 1u) + 1u;
  // Add all positive terms before subtracting padding so valid shapes cannot
  // transiently underflow u32 arithmetic.
  let outH = (params.inputH - 1u) * params.stride
           + effectiveKH + params.outputPadding - 2u * params.padding;
  let outW = (params.inputW - 1u) * params.stride
           + effectiveKW + params.outputPadding - 2u * params.padding;

  let oy = gid.x;  // output row
  let ox = gid.y;  // output col
  let oc = gid.z;  // output channel

  if (oy >= outH || ox >= outW || oc >= params.outC) { return; }

  var acc: f32 = neuralSanitizeSigned(biases[oc]);

  // Transposed convolution: output(oy, ox) accumulates from input pixels
  // whose strided conv would have contributed to (oy, ox).
  // Equivalent: for each input (iy, ix), add input[iy,ix,ic] * weight[ic,oc,oy-iy*s, ox-ix*s]
  // when oy-iy*s ∈ [0, kH) and ox-ix*s ∈ [0, kW).
  for (var kh: u32 = 0u; kh < params.kH; kh++) {
    let oyPadded = oy + params.padding;
    let khOffset = kh * params.dilation;
    if (oyPadded < khOffset) { continue; }
    let iy_r = oyPadded - khOffset;
    if (iy_r % params.stride != 0u) { continue; }
    let iy = iy_r / params.stride;
    if (iy >= params.inputH) { continue; }

    for (var kw: u32 = 0u; kw < params.kW; kw++) {
      let oxPadded = ox + params.padding;
      let kwOffset = kw * params.dilation;
      if (oxPadded < kwOffset) { continue; }
      let ix_r = oxPadded - kwOffset;
      if (ix_r % params.stride != 0u) { continue; }
      let ix = ix_r / params.stride;
      if (ix >= params.inputW) { continue; }

      for (var ic: u32 = 0u; ic < params.inC; ic++) {
        // Input index: [iy, ix, ic]
        let inIdx = iy * params.inputW * params.inC + ix * params.inC + ic;
        // Weight index: IOKW → [ic, oc, kh, kw]
        let wIdx = ic * params.outC * params.kH * params.kW
                 + oc * params.kH * params.kW
                 + kh * params.kW
                 + kw;
        acc = neuralSanitizeSigned(
          acc + neuralSanitizeSigned(input[inIdx]) * neuralSanitizeSigned(weights[wIdx]),
        );
      }
    }
  }

  // Output index: [oy, ox, oc]
  let outIdx = oy * outW * params.outC + ox * params.outC + oc;
  output[outIdx] = neuralSanitizeSigned(acc);
}
`;
