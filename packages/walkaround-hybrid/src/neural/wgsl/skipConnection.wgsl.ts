/**
 * skipConnection.wgsl.ts — element-wise skip-add kernel for the U-Net decoder.
 *
 * Adds two tensors element-wise (skip connection). Both tensors must have the
 * same total element count (same H, W, C). This enforces the Bug 1 fix:
 * if the spatial or channel dimensions don't match, the dispatch will produce
 * wrong results — the test suite validates shape matching before dispatch.
 *
 * Canonical binding layout (Bug 3 fix):
 *   @group(0) @binding(0)  inputA : array<f32>  — first input  (decoder upsample output)
 *   @group(0) @binding(1)  inputB : array<f32>  — second input (encoder skip source)
 *   @group(0) @binding(3)  output : array<f32>  — sum: inputA + inputB
 *   @group(0) @binding(4)  params : SkipParams  — element count
 *
 * Note: binding 2 (biases) not used; host passes a 4-byte placeholder.
 * inputB occupies binding 1 (normally weights) — this is intentional for skip layers.
 */

export const SKIP_CONNECTION_WGSL = /* wgsl */`
struct SkipParams {
  count : u32,   // total number of elements (H × W × C)
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
}

@group(0) @binding(0) var<storage, read>       inputA : array<f32>;
@group(0) @binding(1) var<storage, read>       inputB : array<f32>;  // skip source
@group(0) @binding(3) var<storage, read_write> output : array<f32>;
@group(0) @binding(4) var<uniform>             params : SkipParams;

@compute @workgroup_size(256, 1, 1)
fn skipConnectionMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.count) { return; }
  output[idx] = inputA[idx] + inputB[idx];
}
`;
