/**
 * transposedConv2d.wgsl.ts — 2D transposed convolution (deconvolution) kernel.
 *
 * Matches PyTorch ConvTranspose2d(kernel_size=2, stride=2, padding=0):
 *   outputH = inputH * stride   (when padding=0, kH=2, stride=2)
 *   outputW = inputW * stride
 *
 * Bug 5 fix (prior scaffold): The prior kernel used kH=3 with output_padding issues.
 * This implementation uses kH=2, padding=0, output = input × stride exactly.
 *
 * Weight layout: IOKW — inputC × outputC × kH × kW (standard PyTorch ConvTranspose2d).
 * Each weight at [i, o, kh, kw] → index: i*outC*kH*kW + o*kH*kW + kh*kW + kw
 *
 * Canonical binding layout (Bug 3 fix — same as conv2d):
 *   @group(0) @binding(0)  input   : array<f32>   — input tensor [H × W × inC]
 *   @group(0) @binding(1)  weights : array<f32>   — IOKW layout [inC × outC × kH × kW]
 *   @group(0) @binding(2)  biases  : array<f32>   — [outC]
 *   @group(0) @binding(3)  output  : array<f32>   — output tensor [H*stride × W*stride × outC]
 *   @group(0) @binding(4)  params  : TConv2DParams — shape uniform
 *
 * Workgroup: 8×8×1 over (outH, outW, outC).
 */

export const TRANSPOSED_CONV2D_WGSL = /* wgsl */ `
struct TConv2DParams {
  inputH  : u32,
  inputW  : u32,
  inC     : u32,
  outC    : u32,
  kH      : u32,   // must be 2 for vitrum U-Net
  kW      : u32,   // must be 2 for vitrum U-Net
  stride  : u32,   // must be 2 for vitrum U-Net
  padding : u32,   // must be 0 for vitrum U-Net
}

// Canonical binding layout (Bug 3):
//   0 = input, 1 = weights, 2 = biases, 3 = output, 4 = params
@group(0) @binding(0) var<storage, read>       input   : array<f32>;
@group(0) @binding(1) var<storage, read>       weights : array<f32>;
@group(0) @binding(2) var<storage, read>       biases  : array<f32>;
@group(0) @binding(3) var<storage, read_write> output  : array<f32>;
@group(0) @binding(4) var<uniform>             params  : TConv2DParams;

@compute @workgroup_size(8, 8, 1)
fn transposedConv2dMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Output size: inputH*stride × inputW*stride (for kH=2, stride=2, padding=0)
  let outH = params.inputH * params.stride;
  let outW = params.inputW * params.stride;

  let oy = gid.x;  // output row
  let ox = gid.y;  // output col
  let oc = gid.z;  // output channel

  if (oy >= outH || ox >= outW || oc >= params.outC) { return; }

  var acc: f32 = biases[oc];

  // Transposed convolution: output(oy, ox) accumulates from input pixels
  // whose strided conv would have contributed to (oy, ox).
  // Equivalent: for each input (iy, ix), add input[iy,ix,ic] * weight[ic,oc,oy-iy*s, ox-ix*s]
  // when oy-iy*s ∈ [0, kH) and ox-ix*s ∈ [0, kW).
  for (var kh: u32 = 0u; kh < params.kH; kh++) {
    if (oy < kh) { continue; }
    let iy_r = oy - kh;
    if (iy_r % params.stride != 0u) { continue; }
    let iy = iy_r / params.stride;
    if (iy >= params.inputH) { continue; }

    for (var kw: u32 = 0u; kw < params.kW; kw++) {
      if (ox < kw) { continue; }
      let ix_r = ox - kw;
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
        acc += input[inIdx] * weights[wIdx];
      }
    }
  }

  // Output index: [oy, ox, oc]
  let outIdx = oy * outW * params.outC + ox * params.outC + oc;
  output[outIdx] = acc;
}
`;
