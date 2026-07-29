/**
 * GPU packer for the per-pixel 9-channel neural input. The shader is generated
 * from checkpoint preprocessing metadata so training and inference cannot drift.
 */
import {
  NEURAL_PREPROCESSING_CONTRACT,
  neuralPreprocessingWgsl,
  type NeuralPreprocessingContract,
} from './preprocessing.js';
import {
  NEURAL_F32_TENSOR_STORAGE,
  neuralTensorWgslPreamble,
  neuralTensorWgslType,
  type NeuralTensorStorageContract,
} from './tensorPrecision.js';

export function buildInputPackerWgsl(
  contract: NeuralPreprocessingContract = NEURAL_PREPROCESSING_CONTRACT,
  storage: NeuralTensorStorageContract = NEURAL_F32_TENSOR_STORAGE,
): string {
  const tensor = neuralTensorWgslType(storage);
  const load = (value: string): string => storage.precision === 'f16' ? `f32(${value})` : value;
  const store = (value: string): string => storage.precision === 'f16' ? `f16(${value})` : value;
  return /* wgsl */`
${neuralTensorWgslPreamble(storage)}
${neuralPreprocessingWgsl(contract)}
struct PackParams {
  logicalWidth : u32,
  logicalHeight : u32,
  inferenceWidth : u32,
  inferenceHeight : u32,
}
@group(0) @binding(0) var<storage, read> noisyColor : array<${tensor}>;
@group(0) @binding(1) var<storage, read> albedo : array<${tensor}>;
@group(0) @binding(2) var<storage, read> normals : array<${tensor}>;
@group(0) @binding(3) var<storage, read_write> encInput : array<${tensor}>;
@group(0) @binding(4) var<uniform> params : PackParams;

@compute @workgroup_size(256, 1, 1)
fn inputPackMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  let inferencePixelCount = params.inferenceWidth * params.inferenceHeight;
  if (p >= inferencePixelCount) { return; }
  let x = p % params.inferenceWidth;
  let y = p / params.inferenceWidth;
  let outBase = p * 9u;
  if (x >= params.logicalWidth || y >= params.logicalHeight) {
    encInput[outBase + 0u] = ${store('0.0')};
    encInput[outBase + 1u] = ${store('0.0')};
    encInput[outBase + 2u] = ${store('0.0')};
    encInput[outBase + 3u] = ${store('0.0')};
    encInput[outBase + 4u] = ${store('0.0')};
    encInput[outBase + 5u] = ${store('0.0')};
    encInput[outBase + 6u] = ${store('0.0')};
    encInput[outBase + 7u] = ${store('0.0')};
    encInput[outBase + 8u] = ${store('0.0')};
    return;
  }
  let inBase = (y * params.logicalWidth + x) * 3u;
  encInput[outBase + 0u] = ${store(`neuralPreprocessRadiance(${load('noisyColor[inBase + 0u]')})`)};
  encInput[outBase + 1u] = ${store(`neuralPreprocessRadiance(${load('noisyColor[inBase + 1u]')})`)};
  encInput[outBase + 2u] = ${store(`neuralPreprocessRadiance(${load('noisyColor[inBase + 2u]')})`)};
  encInput[outBase + 3u] = ${store(`neuralSanitizeAlbedo(${load('albedo[inBase + 0u]')})`)};
  encInput[outBase + 4u] = ${store(`neuralSanitizeAlbedo(${load('albedo[inBase + 1u]')})`)};
  encInput[outBase + 5u] = ${store(`neuralSanitizeAlbedo(${load('albedo[inBase + 2u]')})`)};
  let normal = neuralSanitizeNormal(vec3f(
    ${load('normals[inBase + 0u]')},
    ${load('normals[inBase + 1u]')},
    ${load('normals[inBase + 2u]')},
  ));
  encInput[outBase + 6u] = ${store('normal.x')};
  encInput[outBase + 7u] = ${store('normal.y')};
  encInput[outBase + 8u] = ${store('normal.z')};
}
`;
}

export const INPUT_PACKER_ENTRY = 'inputPackMain';
