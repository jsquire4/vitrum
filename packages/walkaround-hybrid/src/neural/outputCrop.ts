/** Exact padded-U-Net RGB crop for arbitrary logical render extents. */
import {
  NEURAL_F32_TENSOR_STORAGE,
  neuralTensorWgslPreamble,
  neuralTensorWgslType,
  type NeuralTensorStorageContract,
} from './tensorPrecision.js';

export function buildOutputCropWgsl(
  storage: NeuralTensorStorageContract = NEURAL_F32_TENSOR_STORAGE,
): string {
  const tensor = neuralTensorWgslType(storage);
  return /* wgsl */`
${neuralTensorWgslPreamble(storage)}
struct CropParams {
  logicalWidth : u32,
  logicalHeight : u32,
  inferenceWidth : u32,
  _pad0 : u32,
}
@group(0) @binding(0) var<storage, read> paddedInput : array<${tensor}>;
@group(0) @binding(1) var<storage, read_write> logicalOutput : array<${tensor}>;
@group(0) @binding(2) var<uniform> params : CropParams;

@compute @workgroup_size(256, 1, 1)
fn outputCropMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  let logicalPixelCount = params.logicalWidth * params.logicalHeight;
  if (p >= logicalPixelCount) { return; }
  let x = p % params.logicalWidth;
  let y = p / params.logicalWidth;
  let src = (y * params.inferenceWidth + x) * 3u;
  let dst = p * 3u;
  logicalOutput[dst + 0u] = paddedInput[src + 0u];
  logicalOutput[dst + 1u] = paddedInput[src + 1u];
  logicalOutput[dst + 2u] = paddedInput[src + 2u];
}
`;
}

export const OUTPUT_CROP_ENTRY = 'outputCropMain';
