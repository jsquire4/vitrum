/**
 * Neural denoiser input-pack shader.
 *
 * Reads noisy/albedo/normalDepth textures and packs each pixel's RGB
 * channels into three interleaved f32 storage buffers (noisyOut, albedoOut,
 * normalsOut).  The normal channel is remapped from [0,1] to [-1,1] and
 * normalised before storage so the inference graph receives unit-length
 * surface normals.
 *
 * Binding layout (group 0):
 *   0 — noisyTex        (texture_2d<f32>)
 *   1 — albedoTex       (texture_2d<f32>)
 *   2 — normalDepthTex  (texture_2d<f32>)
 *   3 — noisyOut        (storage, read_write  f32[])
 *   4 — albedoOut       (storage, read_write  f32[])
 *   5 — normalsOut      (storage, read_write  f32[])
 *   6 — params          (uniform PackParams)
 *
 * Extracted from NeuralDenoiser.initialize inline literal (Issue 2 /
 * complexity-sweep 2026-06-02) — character-identical to the original
 * embedded string.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const NEURAL_PACK_WGSL = /* wgsl */`
struct PackParams {
  width: u32,
  height: u32,
  pixelCount: u32,
  _pad0: u32,
}
@group(0) @binding(0) var noisyTex: texture_2d<f32>;
@group(0) @binding(1) var albedoTex: texture_2d<f32>;
@group(0) @binding(2) var normalDepthTex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> noisyOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> albedoOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> normalsOut: array<f32>;
@group(0) @binding(6) var<uniform> params: PackParams;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= params.pixelCount) { return; }
  let x = p % params.width;
  let y = p / params.width;
  let n = textureLoad(noisyTex, vec2i(i32(x), i32(y)), 0).rgb;
  let a = textureLoad(albedoTex, vec2i(i32(x), i32(y)), 0).rgb;
  let nd = textureLoad(normalDepthTex, vec2i(i32(x), i32(y)), 0).xyz;
  let nrm = normalize(nd * 2.0 - 1.0);
  let base = p * 3u;
  noisyOut[base + 0u] = n.r;
  noisyOut[base + 1u] = n.g;
  noisyOut[base + 2u] = n.b;
  albedoOut[base + 0u] = a.r;
  albedoOut[base + 1u] = a.g;
  albedoOut[base + 2u] = a.b;
  normalsOut[base + 0u] = nrm.r;
  normalsOut[base + 1u] = nrm.g;
  normalsOut[base + 2u] = nrm.b;
}
`;

/** Neural denoiser input-pack compute shader module. Self-contained; no deps. */
export const NEURAL_PACK_MODULE: WgslModule = {
  name: 'neuralPack',
  source: NEURAL_PACK_WGSL,
  requires: [],
};
