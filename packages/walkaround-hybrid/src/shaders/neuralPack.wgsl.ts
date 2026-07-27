/** Neural input texture-to-buffer staging with finite sanitization. */
import type { WgslModule } from '../pipeline/wgslComposer.js';
import { NORMAL_DEPTH_DECODE_WGSL } from '@vitrum/shared-denoisers';
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

export function buildNeuralPackWgsl(
  contract: NeuralPreprocessingContract = NEURAL_PREPROCESSING_CONTRACT,
  storage: NeuralTensorStorageContract = NEURAL_F32_TENSOR_STORAGE,
): string {
  const tensor = neuralTensorWgslType(storage);
  const store = (value: string): string => storage.precision === 'f16' ? `f16(${value})` : value;
  return /* wgsl */`
${neuralTensorWgslPreamble(storage)}
${neuralPreprocessingWgsl(contract)}
${NORMAL_DEPTH_DECODE_WGSL}
struct PackParams {
  width: u32,
  height: u32,
  pixelCount: u32,
  _pad0: u32,
}
@group(0) @binding(0) var noisyTex: texture_2d<f32>;
@group(0) @binding(1) var albedoTex: texture_2d<f32>;
@group(0) @binding(2) var normalDepthTex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> noisyOut: array<${tensor}>;
@group(0) @binding(4) var<storage, read_write> albedoOut: array<${tensor}>;
@group(0) @binding(5) var<storage, read_write> normalsOut: array<${tensor}>;
@group(0) @binding(6) var<uniform> params: PackParams;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= params.pixelCount) { return; }
  let xy = vec2i(i32(p % params.width), i32(p / params.width));
  let noisy = textureLoad(noisyTex, xy, 0).rgb;
  let albedo = textureLoad(albedoTex, xy, 0).rgb;
  let normal = neuralSanitizeNormal(
    decodeNormalDepthWorldNormal(textureLoad(normalDepthTex, xy, 0).xyz),
  );
  let base = p * 3u;
  noisyOut[base] = ${store('neuralSanitizeSigned(noisy.r)')};
  noisyOut[base + 1u] = ${store('neuralSanitizeSigned(noisy.g)')};
  noisyOut[base + 2u] = ${store('neuralSanitizeSigned(noisy.b)')};
  albedoOut[base] = ${store('neuralSanitizeAlbedo(albedo.r)')};
  albedoOut[base + 1u] = ${store('neuralSanitizeAlbedo(albedo.g)')};
  albedoOut[base + 2u] = ${store('neuralSanitizeAlbedo(albedo.b)')};
  normalsOut[base] = ${store('normal.x')};
  normalsOut[base + 1u] = ${store('normal.y')};
  normalsOut[base + 2u] = ${store('normal.z')};
}
`;
}

export const NEURAL_PACK_WGSL = buildNeuralPackWgsl();
export const NEURAL_PACK_MODULE: WgslModule = {
  name: 'neuralPack',
  source: NEURAL_PACK_WGSL,
  requires: [],
};
