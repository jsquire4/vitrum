/**
 * Neural denoiser output-unpack shader.
 *
 * Reads the denoised f32 storage buffer produced by InferenceGraph and
 * writes it back to an rgba16float output texture, clamping each channel
 * to [0, ∞) to eliminate any spurious negative artefacts from the network.
 *
 * Binding layout (group 0):
 *   0 — denoisedIn   (storage, read  f32[])
 *   1 — denoisedOut  (texture_storage_2d<rgba16float, write>)
 *   2 — params       (uniform UnpackParams)
 *
 * Extracted from NeuralDenoiser.initialize inline literal (Issue 2 /
 * complexity-sweep 2026-06-02) — character-identical to the original
 * embedded string.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const NEURAL_UNPACK_WGSL = /* wgsl */`
struct UnpackParams {
  width: u32,
  height: u32,
  pixelCount: u32,
  _pad0: u32,
}
@group(0) @binding(0) var<storage, read> denoisedIn: array<f32>;
@group(0) @binding(1) var denoisedOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: UnpackParams;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= params.pixelCount) { return; }
  let x = p % params.width;
  let y = p / params.width;
  let base = p * 3u;
  let c = vec3f(
    max(0.0, denoisedIn[base + 0u]),
    max(0.0, denoisedIn[base + 1u]),
    max(0.0, denoisedIn[base + 2u]),
  );
  textureStore(denoisedOut, vec2u(x, y), vec4f(c, 1.0));
}
`;

/** Neural denoiser output-unpack compute shader module. Self-contained; no deps. */
export const NEURAL_UNPACK_MODULE: WgslModule = {
  name: 'neuralUnpack',
  source: NEURAL_UNPACK_WGSL,
  requires: [],
};
