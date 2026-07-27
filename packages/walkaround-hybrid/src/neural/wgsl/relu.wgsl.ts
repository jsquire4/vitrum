import { NEURAL_FINITE_WGSL } from '../preprocessing.js';

/**
 * relu.wgsl.ts — element-wise ReLU activation kernel.
 *
 * Canonical binding layout:
 *   @group(0) @binding(0)  input  : array<f32>  — input tensor (any shape)
 *   @group(0) @binding(3)  output : array<f32>  — output tensor (same shape, read_write)
 *   @group(0) @binding(4)  params : ReluParams  — element count
 *
 * The host omits undeclared bindings 1 and 2 from this kernel's bind group.
 */

export const RELU_WGSL = /* wgsl */`${NEURAL_FINITE_WGSL}
struct ReluParams {
  count : u32,  // total number of elements
  groupsX : u32,
  _pad1 : u32,
  _pad2 : u32,
}

@group(0) @binding(0) var<storage, read>       input  : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;
@group(0) @binding(4) var<uniform>             params : ReluParams;

@compute @workgroup_size(256, 1, 1)
fn reluMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.y * params.groupsX * 256u + gid.x;
  if (idx >= params.count) { return; }
  output[idx] = neuralSanitizeSigned(max(0.0, neuralSanitizeSigned(input[idx])));
}
`;
