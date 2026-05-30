/**
 * inputPacker.ts — GPU compute packer for the 9-channel U-Net input tensor.
 *
 * Provides the GPU packing pass that assembles enc_input on-device.
 *
 * Packs three H×W×3 buffers (noisyColor, albedo, normals) into a single
 * H×W×9 buffer (enc_input) with per-pixel interleaving:
 *   enc_input[p*9+0..2] = noisyColor[p*3+0..2]
 *   enc_input[p*9+3..5] = albedo[p*3+0..2]
 *   enc_input[p*9+6..8] = normals[p*3+0..2]
 *
 * References:
 *   Chaitanya et al. 2017 §4: "We augment the noisy color with albedo
 *   and normal auxiliary features, concatenated per-pixel, as the network input."
 */

export const INPUT_PACKER_WGSL = /* wgsl */`
struct PackParams {
  pixelCount : u32,   // H × W
  _pad0      : u32,
  _pad1      : u32,
  _pad2      : u32,
}

// Canonical binding layout: 0=input (noisy), 1=input2 (albedo), 2=input3 (normals), 3=output, 4=params
@group(0) @binding(0) var<storage, read>       noisyColor : array<f32>;  // H×W×3
@group(0) @binding(1) var<storage, read>       albedo     : array<f32>;  // H×W×3
@group(0) @binding(2) var<storage, read>       normals    : array<f32>;  // H×W×3
@group(0) @binding(3) var<storage, read_write> encInput   : array<f32>;  // H×W×9
@group(0) @binding(4) var<uniform>             params     : PackParams;

@compute @workgroup_size(256, 1, 1)
fn inputPackMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= params.pixelCount) { return; }

  let inBase  = p * 3u;
  let outBase = p * 9u;

  // noisyColor channels 0-2
  encInput[outBase + 0u] = noisyColor[inBase + 0u];
  encInput[outBase + 1u] = noisyColor[inBase + 1u];
  encInput[outBase + 2u] = noisyColor[inBase + 2u];
  // albedo channels 3-5
  encInput[outBase + 3u] = albedo[inBase + 0u];
  encInput[outBase + 4u] = albedo[inBase + 1u];
  encInput[outBase + 5u] = albedo[inBase + 2u];
  // normals channels 6-8
  encInput[outBase + 6u] = normals[inBase + 0u];
  encInput[outBase + 7u] = normals[inBase + 1u];
  encInput[outBase + 8u] = normals[inBase + 2u];
}
`;

/** Compute shader entry point name for the input packer. */
export const INPUT_PACKER_ENTRY = 'inputPackMain';
