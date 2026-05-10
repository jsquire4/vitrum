/**
 * unetArchitecture.ts — Canonical UNet architecture spec for the walkaround neural denoiser.
 *
 * Defines a compact, browser-feasible UNet topology that fits within the 1–3 MB weight
 * budget and targets <50 ms inference at 1080p on a mid-range discrete GPU.
 *
 * Architecture summary:
 *
 *   Input: 9 channels — noisy RGB (3) + denoised albedo (3) + world-space normals (3).
 *   These auxiliary buffers come from the SVGF G-buffer pipeline (Sprint 10a).
 *
 *   Encoder (3 levels, downsampling 2× per level via stride-2 conv):
 *     Level 1: conv2d(9→24,  3×3, stride 2) + ReLU  → feature map H/2  × W/2  × 24
 *     Level 2: conv2d(24→48, 3×3, stride 2) + ReLU  → feature map H/4  × W/4  × 48
 *     Level 3: conv2d(48→96, 3×3, stride 2) + ReLU  → feature map H/8  × W/8  × 96
 *
 *   Bottleneck (no spatial change):
 *     conv2d(96→192, 3×3, stride 1) + ReLU           → feature map H/8  × W/8  × 192
 *
 *   Decoder (3 levels, upsampling 2× per level via transposed conv):
 *     Level 3: transposedConv2d(192→96, 2×2, stride 2) + skip-add(enc3 output) + ReLU
 *              + conv2d(96→96, 3×3) + ReLU             → H/4 × W/4 × 96
 *     Level 2: transposedConv2d(96→48, 2×2, stride 2)  + skip-add(enc2 output) + ReLU
 *              + conv2d(48→48, 3×3) + ReLU             → H/2 × W/2 × 48
 *     Level 1: transposedConv2d(48→24, 2×2, stride 2)  + skip-add(enc1 output) + ReLU
 *              + conv2d(24→24, 3×3) + ReLU             → H   × W   × 24
 *
 *   Output projection:
 *     conv2d(24→3, 1×1, stride 1)  → denoised RGB, same spatial resolution as input.
 *
 * Parameter count (verified):
 *   enc1:        9 × 24 × 9 + 24  =   1,968
 *   enc2:       24 × 48 × 9 + 48  =  10,416
 *   enc3:       48 × 96 × 9 + 96  =  41,568
 *   bottleneck: 96 × 192 × 9 + 192 = 166,080
 *   dec3_tconv: 192 × 96 × 4 + 96 =  73,824   (2×2 kernel)
 *   dec3_conv:   96 × 96 × 9 + 96 =  83,040
 *   dec2_tconv:  96 × 48 × 4 + 48 =  18,480
 *   dec2_conv:   48 × 48 × 9 + 48 =  20,784
 *   dec1_tconv:  48 × 24 × 4 + 24 =   4,632
 *   dec1_conv:   24 × 24 × 9 + 24 =   5,208
 *   proj:        24 ×  3 × 1 +  3 =      75
 *   ─────────────────────────────────────────
 *   TOTAL:                           426,075 parameters
 *   BYTES (f32):                   1,704,300 bytes  ≈ 1.63 MB
 *
 * DoD target: 1–3 MB. This design lands at 1.63 MB — well within target.
 *
 * Inference performance expectation:
 *   - 1080p input: H=1080, W=1920 → ~8M output pixels × 426K multiply-adds ≈ 3.4B ops.
 *   - At 10 TFLOP/s (mid-range GPU WebGPU throughput): ~0.34 ms theoretical.
 *   - With WebGPU dispatch overhead and memory bandwidth at 1080p (~16 MB intermediate
 *     tensors): expected 10–50 ms practical range on current hardware.
 *
 * Memory (GPU, 1080p, f32):
 *   Weights: 1.63 MB (CPU + GPU copy).
 *   Intermediate tensors at 1080p:
 *     enc1 output: (540 × 960 × 24)  × 4 = ~49.8 MB
 *     enc2 output: (270 × 480 × 48)  × 4 = ~24.9 MB
 *     enc3 output: (135 × 240 × 96)  × 4 = ~12.4 MB
 *     bottleneck:  (135 × 240 × 192) × 4 = ~24.9 MB
 *     dec3 output: (270 × 480 × 96)  × 4 = ~49.8 MB
 *     dec2 output: (540 × 960 × 48)  × 4 = ~99.5 MB
 *     dec1 output: (1080 × 1920 × 24)× 4 = ~199 MB
 *   Total intermediate: ~460 MB at 1080p.
 *   Production note: reduce via fp16 (halve) or channel-pruning to fit 8 GB VRAM budget.
 *   See plan/sprint-13-walkaround-integration.md §Memory for mitigation options.
 *
 * Training reference:
 *   Architecture defined here; training pipeline documented in
 *   tools/neural-denoiser-training/. Model weights are NOT shipped with vitrum —
 *   users train their own from stained-glass scene render pairs.
 *
 * References:
 *   Ronneberger, Fischer, Brox "U-Net: Convolutional Networks for Biomedical
 *   Image Segmentation" MICCAI 2015. https://arxiv.org/abs/1505.04597
 *
 *   Chaitanya et al. "Interactive Reconstruction of Monte Carlo Image Sequences
 *   using a Recurrent Denoising Autoencoder" SIGGRAPH 2017.
 *   https://doi.org/10.1145/3072959.3073601
 *   (UNet variant for real-time path-traced denoising — direct precedent.)
 *
 * @since Sprint 13, 2026-05-09
 */

import type { InferenceGraphSpec } from './InferenceGraph.js';

// ─────────────────────────────────────────────────────────────────────────────
// Architecture constants (exported for introspection + test assertions)
// ─────────────────────────────────────────────────────────────────────────────

/** Input channel count: noisy RGB (3) + albedo (3) + normals (3). */
export const UNET_INPUT_CHANNELS = 9 as const;

/** Output channel count: denoised RGB. */
export const UNET_OUTPUT_CHANNELS = 3 as const;

/** Channel widths at each encoder level: [enc1, enc2, enc3, bottleneck]. */
export const UNET_ENCODER_CHANNELS = [24, 48, 96, 192] as const;

/** Channel widths at each decoder level (mirrors encoder): [dec3, dec2, dec1]. */
export const UNET_DECODER_CHANNELS = [96, 48, 24] as const;

/** Total learnable parameter count (weights + biases combined). */
export const UNET_TOTAL_PARAMETERS = 426_075 as const;

/** Approximate weight storage size in bytes (f32). */
export const UNET_WEIGHT_BYTES = 1_704_300 as const;  // UNET_TOTAL_PARAMETERS * 4

/** Named input tensors the InferenceGraph expects in run() inputs map. */
export const UNET_INPUT_TENSOR_NAMES = [
  'noisyColor',   // RGBA noisy path-traced color (RGB channels used; A ignored)
  'albedo',       // Denoised surface albedo from G-buffer
  'normals',      // World-space normals from G-buffer
] as const;

/** Named output tensor the InferenceGraph writes to in run() outputs map. */
export const UNET_OUTPUT_TENSOR_NAMES = ['denoisedColor'] as const;

// ─────────────────────────────────────────────────────────────────────────────
// Canonical UNet inference graph spec
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Default UNet architecture for the walkaround neural denoiser.
 *
 * Encoder: 3 levels of stride-2 conv2d + ReLU (downsampling 2× per level).
 * Bottleneck: 1 stride-1 conv2d + ReLU block.
 * Decoder: 3 levels of transposed_conv2d + skip-connection (add) + conv2d + ReLU.
 * Output: 1×1 conv2d projecting to 3-channel denoised RGB.
 *
 * Channel widths: 24 → 48 → 96 → 192 (bottleneck) → 96 → 48 → 24 → 3.
 * Total parameters: 426,075. Storage: ~1.63 MB (f32). DoD target: 1–3 MB. ✓
 *
 * Dispatch params in each layer's `params` field:
 *   dispatchX / dispatchY / dispatchZ — workgroup counts for the layer dispatch.
 *   These are NOT inferred from tensor shapes at spec-definition time; the host
 *   must compute and set them based on the actual render resolution before calling
 *   InferenceGraph.run(). Defaults shown assume 1080p (1920×1080). Update for other
 *   resolutions. See plan/sprint-13-walkaround-integration.md §Dispatch Sizing.
 */
export const WALKAROUND_DENOISER_UNET_SPEC: InferenceGraphSpec = {
  inputTensors:  [...UNET_INPUT_TENSOR_NAMES],
  outputTensors: [...UNET_OUTPUT_TENSOR_NAMES],

  layers: [
    // ── Encoder Level 1: 9 → 24 channels, stride 2 ───────────────────────
    // Input: concatenated (noisyColor + albedo + normals) = 9 channels.
    // The host is responsible for packing the three inputs into a single
    // HxWx9 buffer named 'enc_input'; see the integration spec.
    {
      kind:   'conv2d',
      inputs: ['enc_input'],         // HxWx9 packed input
      output: 'enc1_conv',
      params: {
        inputH: 1080, inputW: 1920, inputC:   9,
        kernelH: 3, kernelW: 3, outputC: 24,
        stride: 2, dilation: 1,
        // Dispatch: ceil(W/2 / 8) × ceil(H/2 / 8) × outputC
        dispatchX: Math.ceil(960 / 8), dispatchY: Math.ceil(540 / 8), dispatchZ: 24,
      },
    },
    {
      kind:   'relu',
      inputs: ['enc1_conv'],
      output: 'enc1',                // H/2 × W/2 × 24
      params: {
        totalElements: 540 * 960 * 24,
        dispatchX: Math.ceil(540 * 960 * 24 / 256), dispatchY: 1, dispatchZ: 1,
      },
    },

    // ── Encoder Level 2: 24 → 48 channels, stride 2 ──────────────────────
    {
      kind:   'conv2d',
      inputs: ['enc1'],
      output: 'enc2_conv',
      params: {
        inputH: 540, inputW: 960, inputC: 24,
        kernelH: 3, kernelW: 3, outputC: 48,
        stride: 2, dilation: 1,
        dispatchX: Math.ceil(480 / 8), dispatchY: Math.ceil(270 / 8), dispatchZ: 48,
      },
    },
    {
      kind:   'relu',
      inputs: ['enc2_conv'],
      output: 'enc2',                // H/4 × W/4 × 48
      params: {
        totalElements: 270 * 480 * 48,
        dispatchX: Math.ceil(270 * 480 * 48 / 256), dispatchY: 1, dispatchZ: 1,
      },
    },

    // ── Encoder Level 3: 48 → 96 channels, stride 2 ──────────────────────
    {
      kind:   'conv2d',
      inputs: ['enc2'],
      output: 'enc3_conv',
      params: {
        inputH: 270, inputW: 480, inputC: 48,
        kernelH: 3, kernelW: 3, outputC: 96,
        stride: 2, dilation: 1,
        dispatchX: Math.ceil(240 / 8), dispatchY: Math.ceil(135 / 8), dispatchZ: 96,
      },
    },
    {
      kind:   'relu',
      inputs: ['enc3_conv'],
      output: 'enc3',                // H/8 × W/8 × 96
      params: {
        totalElements: 135 * 240 * 96,
        dispatchX: Math.ceil(135 * 240 * 96 / 256), dispatchY: 1, dispatchZ: 1,
      },
    },

    // ── Bottleneck: 96 → 192 channels, stride 1 ──────────────────────────
    {
      kind:   'conv2d',
      inputs: ['enc3'],
      output: 'btn_conv',
      params: {
        inputH: 135, inputW: 240, inputC: 96,
        kernelH: 3, kernelW: 3, outputC: 192,
        stride: 1, dilation: 1,
        dispatchX: Math.ceil(240 / 8), dispatchY: Math.ceil(135 / 8), dispatchZ: 192,
      },
    },
    {
      kind:   'relu',
      inputs: ['btn_conv'],
      output: 'btn',                 // H/8 × W/8 × 192
      params: {
        totalElements: 135 * 240 * 192,
        dispatchX: Math.ceil(135 * 240 * 192 / 256), dispatchY: 1, dispatchZ: 1,
      },
    },

    // ── Decoder Level 3: 192 → 96 channels, upsample 2× ─────────────────
    {
      kind:   'transposed_conv2d',
      inputs: ['btn'],
      output: 'dec3_up',             // H/4 × W/4 × 96
      params: {
        inputH: 135, inputW: 240, inputC: 192,
        kernelH: 2, kernelW: 2, outputC: 96,
        stride: 2,
        dispatchX: Math.ceil(480 / 8), dispatchY: Math.ceil(270 / 8), dispatchZ: 96,
      },
    },
    {
      kind:   'skip',
      inputs: ['dec3_up', 'enc3'],   // upsampled + encoder skip
      output: 'dec3_skip',
      params: {
        totalElements: 270 * 480 * 96,
        dispatchX: Math.ceil(270 * 480 * 96 / 256), dispatchY: 1, dispatchZ: 1,
      },
    },
    {
      kind:   'relu',
      inputs: ['dec3_skip'],
      output: 'dec3_relu1',
      params: {
        totalElements: 270 * 480 * 96,
        dispatchX: Math.ceil(270 * 480 * 96 / 256), dispatchY: 1, dispatchZ: 1,
      },
    },
    {
      kind:   'conv2d',
      inputs: ['dec3_relu1'],
      output: 'dec3_conv',
      params: {
        inputH: 270, inputW: 480, inputC: 96,
        kernelH: 3, kernelW: 3, outputC: 96,
        stride: 1, dilation: 1,
        dispatchX: Math.ceil(480 / 8), dispatchY: Math.ceil(270 / 8), dispatchZ: 96,
      },
    },
    {
      kind:   'relu',
      inputs: ['dec3_conv'],
      output: 'dec3',                // H/4 × W/4 × 96
      params: {
        totalElements: 270 * 480 * 96,
        dispatchX: Math.ceil(270 * 480 * 96 / 256), dispatchY: 1, dispatchZ: 1,
      },
    },

    // ── Decoder Level 2: 96 → 48 channels, upsample 2× ──────────────────
    {
      kind:   'transposed_conv2d',
      inputs: ['dec3'],
      output: 'dec2_up',             // H/2 × W/2 × 48
      params: {
        inputH: 270, inputW: 480, inputC: 96,
        kernelH: 2, kernelW: 2, outputC: 48,
        stride: 2,
        dispatchX: Math.ceil(960 / 8), dispatchY: Math.ceil(540 / 8), dispatchZ: 48,
      },
    },
    {
      kind:   'skip',
      inputs: ['dec2_up', 'enc2'],
      output: 'dec2_skip',
      params: {
        totalElements: 540 * 960 * 48,
        dispatchX: Math.ceil(540 * 960 * 48 / 256), dispatchY: 1, dispatchZ: 1,
      },
    },
    {
      kind:   'relu',
      inputs: ['dec2_skip'],
      output: 'dec2_relu1',
      params: {
        totalElements: 540 * 960 * 48,
        dispatchX: Math.ceil(540 * 960 * 48 / 256), dispatchY: 1, dispatchZ: 1,
      },
    },
    {
      kind:   'conv2d',
      inputs: ['dec2_relu1'],
      output: 'dec2_conv',
      params: {
        inputH: 540, inputW: 960, inputC: 48,
        kernelH: 3, kernelW: 3, outputC: 48,
        stride: 1, dilation: 1,
        dispatchX: Math.ceil(960 / 8), dispatchY: Math.ceil(540 / 8), dispatchZ: 48,
      },
    },
    {
      kind:   'relu',
      inputs: ['dec2_conv'],
      output: 'dec2',                // H/2 × W/2 × 48
      params: {
        totalElements: 540 * 960 * 48,
        dispatchX: Math.ceil(540 * 960 * 48 / 256), dispatchY: 1, dispatchZ: 1,
      },
    },

    // ── Decoder Level 1: 48 → 24 channels, upsample 2× ──────────────────
    {
      kind:   'transposed_conv2d',
      inputs: ['dec2'],
      output: 'dec1_up',             // H × W × 24
      params: {
        inputH: 540, inputW: 960, inputC: 48,
        kernelH: 2, kernelW: 2, outputC: 24,
        stride: 2,
        dispatchX: Math.ceil(1920 / 8), dispatchY: Math.ceil(1080 / 8), dispatchZ: 24,
      },
    },
    {
      kind:   'skip',
      inputs: ['dec1_up', 'enc1'],
      output: 'dec1_skip',
      params: {
        totalElements: 1080 * 1920 * 24,
        dispatchX: Math.ceil(1080 * 1920 * 24 / 256), dispatchY: 1, dispatchZ: 1,
      },
    },
    {
      kind:   'relu',
      inputs: ['dec1_skip'],
      output: 'dec1_relu1',
      params: {
        totalElements: 1080 * 1920 * 24,
        dispatchX: Math.ceil(1080 * 1920 * 24 / 256), dispatchY: 1, dispatchZ: 1,
      },
    },
    {
      kind:   'conv2d',
      inputs: ['dec1_relu1'],
      output: 'dec1_conv',
      params: {
        inputH: 1080, inputW: 1920, inputC: 24,
        kernelH: 3, kernelW: 3, outputC: 24,
        stride: 1, dilation: 1,
        dispatchX: Math.ceil(1920 / 8), dispatchY: Math.ceil(1080 / 8), dispatchZ: 24,
      },
    },
    {
      kind:   'relu',
      inputs: ['dec1_conv'],
      output: 'dec1',                // H × W × 24
      params: {
        totalElements: 1080 * 1920 * 24,
        dispatchX: Math.ceil(1080 * 1920 * 24 / 256), dispatchY: 1, dispatchZ: 1,
      },
    },

    // ── Output projection: 24 → 3 channels (denoised RGB) ────────────────
    {
      kind:   'conv2d',
      inputs: ['dec1'],
      output: 'denoisedColor',
      params: {
        inputH: 1080, inputW: 1920, inputC: 24,
        kernelH: 1, kernelW: 1, outputC: 3,
        stride: 1, dilation: 1,
        dispatchX: Math.ceil(1920 / 8), dispatchY: Math.ceil(1080 / 8), dispatchZ: 3,
      },
    },
  ],
};
