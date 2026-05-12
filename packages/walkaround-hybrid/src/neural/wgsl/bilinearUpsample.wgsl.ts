/**
 * bilinearUpsample.wgsl.ts — 2× bilinear upsample kernel.
 *
 * Upsamples an input tensor from H×W×C to 2H×2W×C using bilinear interpolation.
 * This is an alternative to transposed conv when no learned upsampling is needed.
 * In the vitrum U-Net it is used as a building block within the decoder.
 *
 * Canonical binding layout (Bug 3 fix):
 *   @group(0) @binding(0)  input  : array<f32>       — input [H × W × C]
 *   @group(0) @binding(3)  output : array<f32>       — output [2H × 2W × C]
 *   @group(0) @binding(4)  params : UpsampleParams   — shape
 *
 * Bindings 1 and 2 not used (host passes placeholder 4-byte buffers).
 */

export const BILINEAR_UPSAMPLE_WGSL = /* wgsl */`
struct UpsampleParams {
  inputH  : u32,
  inputW  : u32,
  channels: u32,
  _pad0   : u32,
}

@group(0) @binding(0) var<storage, read>       inputBuf  : array<f32>;
@group(0) @binding(3) var<storage, read_write> outputBuf : array<f32>;
@group(0) @binding(4) var<uniform>             params    : UpsampleParams;

fn sampleInput(iy: u32, ix: u32, ch: u32) -> f32 {
  let clampedY = min(iy, params.inputH - 1u);
  let clampedX = min(ix, params.inputW - 1u);
  return inputBuf[clampedY * params.inputW * params.channels + clampedX * params.channels + ch];
}

@compute @workgroup_size(8, 8, 1)
fn bilinearUpsampleMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outH = params.inputH * 2u;
  let outW = params.inputW * 2u;

  let oy = gid.x;
  let ox = gid.y;
  let ch = gid.z;

  if (oy >= outH || ox >= outW || ch >= params.channels) { return; }

  // Map output pixel to fractional input coordinates.
  // Align corners=false: output pixel (oy, ox) maps to input coord
  // ((oy + 0.5) / 2 - 0.5) = (oy - 0.5) / 2
  let fy = (f32(oy) + 0.5) / 2.0 - 0.5;
  let fx = (f32(ox) + 0.5) / 2.0 - 0.5;

  let iy0 = u32(max(0.0, floor(fy)));
  let ix0 = u32(max(0.0, floor(fx)));
  let iy1 = iy0 + 1u;
  let ix1 = ix0 + 1u;

  let ty = fy - floor(fy);
  let tx = fx - floor(fx);

  let v00 = sampleInput(iy0, ix0, ch);
  let v01 = sampleInput(iy0, ix1, ch);
  let v10 = sampleInput(iy1, ix0, ch);
  let v11 = sampleInput(iy1, ix1, ch);

  let val = v00 * (1.0 - ty) * (1.0 - tx)
          + v01 * (1.0 - ty) * tx
          + v10 * ty * (1.0 - tx)
          + v11 * ty * tx;

  let outIdx = oy * outW * params.channels + ox * params.channels + ch;
  outputBuf[outIdx] = val;
}
`;
