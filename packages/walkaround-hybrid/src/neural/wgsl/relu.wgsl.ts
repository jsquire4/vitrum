/**
 * relu.wgsl.ts — element-wise ReLU activation kernel.
 *
 * Canonical binding layout (Bug 3 fix — no weights/biases, but same group/binding
 * structure for buffers that exist):
 *   @group(0) @binding(0)  input  : array<f32>  — input tensor (any shape)
 *   @group(0) @binding(3)  output : array<f32>  — output tensor (same shape, read_write)
 *   @group(0) @binding(4)  params : ReluParams  — element count
 *
 * Bindings 1 and 2 (weights/biases) are not used by relu but the host
 * creates zero-byte placeholder buffers to keep the bind-group layout consistent.
 */

export const RELU_WGSL = /* wgsl */ `
struct ReluParams {
  count : u32,  // total number of elements
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
}

@group(0) @binding(0) var<storage, read>       input  : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;
@group(0) @binding(4) var<uniform>             params : ReluParams;

@compute @workgroup_size(256, 1, 1)
fn reluMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.count) { return; }
  output[idx] = max(0.0, input[idx]);
}
`;
