/** Neural output buffer-to-texture decode with finite sanitization. */
import type { WgslModule } from '../pipeline/wgslComposer.js';
import {
  NEURAL_PREPROCESSING_CONTRACT,
  neuralPreprocessingWgsl,
  type NeuralPreprocessingContract,
} from '../neural/preprocessing.js';
import {
  NEURAL_F32_TENSOR_STORAGE,
  neuralTensorWgslPreamble,
  neuralTensorWgslType,
  type NeuralTensorStorageContract,
} from '../neural/tensorPrecision.js';

export function buildNeuralUnpackWgsl(
  contract: NeuralPreprocessingContract = NEURAL_PREPROCESSING_CONTRACT,
  storage: NeuralTensorStorageContract = NEURAL_F32_TENSOR_STORAGE,
): string {
  const tensor = neuralTensorWgslType(storage);
  const load = (value: string): string => storage.precision === 'f16' ? `f32(${value})` : value;
  return /* wgsl */`
${neuralTensorWgslPreamble(storage)}
${neuralPreprocessingWgsl(contract)}
struct UnpackParams {
  width: u32,
  height: u32,
  pixelCount: u32,
  _pad0: u32,
}
@group(0) @binding(0) var<storage, read> denoisedIn: array<${tensor}>;
@group(0) @binding(1) var denoisedOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: UnpackParams;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= params.pixelCount) { return; }
  let x = p % params.width;
  let y = p / params.width;
  let base = p * 3u;
  let color = vec3f(
    neuralPostprocessRadiance(${load('denoisedIn[base]')}),
    neuralPostprocessRadiance(${load('denoisedIn[base + 1u]')}),
    neuralPostprocessRadiance(${load('denoisedIn[base + 2u]')}),
  );
  textureStore(denoisedOut, vec2u(x, y), vec4f(color, 1.0));
}
`;
}

export const NEURAL_UNPACK_WGSL = buildNeuralUnpackWgsl();
export const NEURAL_UNPACK_MODULE: WgslModule = {
  name: 'neuralUnpack',
  source: NEURAL_UNPACK_WGSL,
  requires: [],
};
