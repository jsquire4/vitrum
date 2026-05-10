/**
 * relu.wgsl.ts — ReLU activation compute shader fragment.
 *
 * Applies the rectified linear unit activation elementwise:
 *
 *   output[i] = max(input[i], 0.0)
 *
 * Used between every convolutional layer in the UNet encoder and decoder.
 * Each thread processes one element; dispatch over (totalElements, 1, 1).
 *
 * Buffer layout: flat f32 arrays. The element count is passed as the
 * lone field of ReLUParams.
 *
 * References:
 *   Nair, Hinton "Rectified Linear Units Improve Restricted Boltzmann Machines"
 *   ICML 2010. https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf
 *
 * @since Sprint 13, 2026-05-09
 */

export const RELU_WGSL = /* wgsl */ `

// ============================================================
// ReLU parameter uniform (std140-compatible; 16 bytes)
// ============================================================
struct ReLUParams {
  totalElements: u32,   // bytes  0-3  — flat element count
  _pad0:         u32,   // bytes  4-7
  _pad1:         u32,   // bytes  8-11
  _pad2:         u32,   // bytes 12-15
};

// ============================================================
// Bindings
// ============================================================
@group(0) @binding(0) var<storage, read>       relu_input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> relu_output: array<f32>;
@group(0) @binding(2) var<uniform>             relu_params: ReLUParams;

// ============================================================
// Entry point
// ============================================================
@compute @workgroup_size(256, 1, 1)
fn reluKernel(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= relu_params.totalElements) { return; }
  relu_output[i] = max(relu_input[i], 0.0);
}
`;
