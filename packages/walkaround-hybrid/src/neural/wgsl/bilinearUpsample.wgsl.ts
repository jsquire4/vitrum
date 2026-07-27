import { NEURAL_FINITE_WGSL } from '../preprocessing.js';

/**
 * bilinearUpsample.wgsl.ts — 2× bilinear upsample kernel.
 *
 * Upsamples an input tensor from H×W×C to 2H×2W×C using bilinear interpolation.
 * This is an alternative to transposed conv when no learned upsampling is needed.
 *
 * EXTENSION POINT (Task 4.5 D2): this kernel is fully plumbed (entry point, dim
 * solver, dispatch layout in `layerResourceAllocator.ts` / `tensorDimSolver.ts`),
 * but the canonical `buildUNetSpec()` decoder currently upsamples via
 * `transposedConv2d`, so NO shipped spec emits a `bilinearUpsample` layer today.
 * It is retained as the alternative decoder upsampler — a custom UNetSpec can
 * emit this layer to swap learned transposed-conv upsampling for parameter-free
 * bilinear. Deliberately kept; not dead code.
 *
 * Canonical binding layout:
 *   @group(0) @binding(0)  input  : array<f32>       — input [H × W × C]
 *   @group(0) @binding(3)  output : array<f32>       — output [2H × 2W × C]
 *   @group(0) @binding(4)  params : UpsampleParams   — shape
 *
 * Bindings 1 and 2 are undeclared and omitted by the host.
 */

export const BILINEAR_UPSAMPLE_WGSL = /* wgsl */`${NEURAL_FINITE_WGSL}
struct UpsampleParams {
  inputH  : u32,
  inputW  : u32,
  channels: u32,
  _pad0   : u32,
}

@group(0) @binding(0) var<storage, read>       inputBuf  : array<f32>;
@group(0) @binding(3) var<storage, read_write> outputBuf : array<f32>;
@group(0) @binding(4) var<uniform>             params    : UpsampleParams;

fn sampleInput(iy: i32, ix: i32, ch: u32) -> f32 {
  let clampedY = u32(clamp(iy, 0, i32(params.inputH) - 1));
  let clampedX = u32(clamp(ix, 0, i32(params.inputW) - 1));
  return neuralSanitizeSigned(inputBuf[clampedY * params.inputW * params.channels + clampedX * params.channels + ch]);
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

  // Keep the signed, unclamped footprint. Clamping each sample independently
  // reproduces edge-value padding: at -0.25 both taps address row/column zero.
  // Clamping the base before deriving the second tap incorrectly blends the
  // first output row/column with the interior.
  let iy0 = i32(floor(fy));
  let ix0 = i32(floor(fx));
  let iy1 = iy0 + 1;
  let ix1 = ix0 + 1;

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
  outputBuf[outIdx] = neuralSanitizeSigned(val);
}
`;
