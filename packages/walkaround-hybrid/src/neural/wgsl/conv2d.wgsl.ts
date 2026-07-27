import { NEURAL_FINITE_WGSL } from '../preprocessing.js';

/**
 * conv2d.wgsl.ts — 2D convolution kernel for the vitrum neural denoiser.
 *
 * Canonical binding layout (matches host dispatch order exactly):
 *   @group(0) @binding(0)  input   : array<f32>   — input tensor [H × W × inC]
 *   @group(0) @binding(1)  weights : array<f32>   — OIKW layout [outC × inC × kH × kW]
 *   @group(0) @binding(2)  biases  : array<f32>   — [outC]
 *   @group(0) @binding(3)  output  : array<f32>   — output tensor [outH × outW × outC]
 *   @group(0) @binding(4)  params  : Conv2DParams — shape uniform
 *
 * Weight layout: OIKW — outputC × inputC × kH × kW (standard PyTorch Conv2d layout).
 * Each weight at [o, i, kh, kw] → index: o*inC*kH*kW + i*kH*kW + kh*kW + kw
 *
 * Workgroup: 8×8×1 threads over (outH, outW, outC). Each thread computes one
 * output pixel for one output channel.
 */

export const CONV2D_WGSL = /* wgsl */`${NEURAL_FINITE_WGSL}
// ── Conv2DParams uniform (written by host before dispatch) ────────────────────
struct Conv2DParams {
  inputH  : u32,
  inputW  : u32,
  inC     : u32,
  outC    : u32,
  kH      : u32,
  kW      : u32,
  stride  : u32,
  padding : u32,
}

// Canonical binding layout:
//   0 = input, 1 = weights, 2 = biases, 3 = output, 4 = params
@group(0) @binding(0) var<storage, read>       input   : array<f32>;
@group(0) @binding(1) var<storage, read>       weights : array<f32>;
@group(0) @binding(2) var<storage, read>       biases  : array<f32>;
@group(0) @binding(3) var<storage, read_write> output  : array<f32>;
@group(0) @binding(4) var<uniform>             params  : Conv2DParams;

@compute @workgroup_size(8, 8, 1)
fn conv2dMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outH = (params.inputH + 2u * params.padding - params.kH) / params.stride + 1u;
  let outW = (params.inputW + 2u * params.padding - params.kW) / params.stride + 1u;

  let oy = gid.x;  // output row
  let ox = gid.y;  // output col
  let oc = gid.z;  // output channel

  if (oy >= outH || ox >= outW || oc >= params.outC) { return; }

  var acc: f32 = neuralSanitizeSigned(biases[oc]);

  let iy_base = i32(oy * params.stride) - i32(params.padding);
  let ix_base = i32(ox * params.stride) - i32(params.padding);

  for (var kh: u32 = 0u; kh < params.kH; kh++) {
    let iy = iy_base + i32(kh);
    if (iy < 0 || iy >= i32(params.inputH)) { continue; }

    for (var kw: u32 = 0u; kw < params.kW; kw++) {
      let ix = ix_base + i32(kw);
      if (ix < 0 || ix >= i32(params.inputW)) { continue; }

      for (var ic: u32 = 0u; ic < params.inC; ic++) {
        // Input index: [iy, ix, ic]
        let inIdx = u32(iy) * params.inputW * params.inC + u32(ix) * params.inC + ic;
        // Weight index: OIKW → [oc, ic, kh, kw]
        let wIdx = oc * params.inC * params.kH * params.kW
                 + ic * params.kH * params.kW
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
