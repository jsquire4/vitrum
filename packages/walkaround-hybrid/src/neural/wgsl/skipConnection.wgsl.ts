/**
 * skipConnection.wgsl.ts — UNet skip-connection (elementwise add) compute shader.
 *
 * Implements the residual / skip connection used in UNet decoders:
 *
 *   output[i] = inputA[i] + inputB[i]
 *
 * At each decoder level, the upsampled feature map (inputA) is summed with the
 * corresponding encoder feature map (inputB) that bypassed the bottleneck.
 * Both tensors must have identical shape (H × W × C); the element count is
 * provided via SkipParams.
 *
 * Alternative: channel concatenation is another common skip strategy. This
 * implementation uses addition, which keeps output channels the same as input.
 * Concatenation would double channels and require a follow-up 1×1 conv to
 * project back — addition is cheaper and sufficient for denoising UNets.
 *
 * References:
 *   Ronneberger, Fischer, Brox "U-Net: Convolutional Networks for Biomedical
 *   Image Segmentation" MICCAI 2015. https://arxiv.org/abs/1505.04597
 *   (skip connections — Figure 1, Section 2)
 *
 *   He et al. "Deep Residual Learning for Image Recognition" CVPR 2016.
 *   https://arxiv.org/abs/1512.03385 (residual addition)
 *
 * @since Sprint 13, 2026-05-09
 */

export const SKIP_CONNECTION_WGSL = /* wgsl */ `

// ============================================================
// Skip-connection parameter uniform (std140-compatible; 16 bytes)
// ============================================================
struct SkipParams {
  totalElements: u32,   // bytes  0-3  — flat element count (must match for A and B)
  _pad0:         u32,   // bytes  4-7
  _pad1:         u32,   // bytes  8-11
  _pad2:         u32,   // bytes 12-15
};

// ============================================================
// Bindings
// ============================================================
@group(0) @binding(0) var<storage, read>       skip_inputA:  array<f32>;  // upsampled decoder features
@group(0) @binding(1) var<storage, read>       skip_inputB:  array<f32>;  // encoder features (skip path)
@group(0) @binding(2) var<storage, read_write> skip_output:  array<f32>;  // summed features
@group(0) @binding(3) var<uniform>             skip_params:  SkipParams;

// ============================================================
// Entry point
// ============================================================
@compute @workgroup_size(256, 1, 1)
fn skipConnectionKernel(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= skip_params.totalElements) { return; }
  skip_output[i] = skip_inputA[i] + skip_inputB[i];
}
`;
