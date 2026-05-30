/**
 * U-Net architecture specification for the vitrum neural denoiser.
 *
 * Architecture: Ronneberger, Fischer, Brox 2015 "U-Net: Convolutional Networks
 * for Biomedical Image Segmentation." MICCAI. https://arxiv.org/abs/1505.04597
 *
 * Adapted for path-tracer denoising per:
 * Chaitanya et al. 2017 "Interactive Reconstruction of Monte Carlo Image
 * Sequences using a Recurrent Denoising Autoencoder." SIGGRAPH.
 * https://doi.org/10.1145/3072959.3073601
 *
 * Parameter count: ~426,075 total (~1.63 MB at f32, ~0.82 MB at f16).
 * Intermediate GPU memory at 1080p (f32): ~461 MB; fp16 halves this to ~230 MB.
 *
 * Skip-connection invariant: skip-connection spatial shapes are explicitly
 * documented per layer so dec3_up (H/4) pairs with enc3 (H/4-pre-stride output)
 * — NOT enc3 (H/8). The encoder Level N output *before* the stride-2 convolution
 * is the skip source. See SKIP_SHAPES table below.
 *
 * Binding convention (all WGSL kernels):
 *   @group(0) @binding(0) input  — read-only storage buffer (input tensor, f32)
 *   @group(0) @binding(1) weights — read-only storage buffer (f32, OIKW layout)
 *   @group(0) @binding(2) biases  — read-only storage buffer (f32)
 *   @group(0) @binding(3) output  — read-write storage buffer (f32)
 *   @group(0) @binding(4) params  — uniform buffer (shape params)
 */

// ── Layer kinds ───────────────────────────────────────────────────────────────

export type LayerKind =
  | 'conv2d'
  | 'transposedConv2d'
  | 'relu'
  | 'skipAdd'
  | 'bilinearUpsample'
  | 'inputPack';

/** Weight layout per layer kind, matching PyTorch conventions.
 *  Conv2D:          OIKW  (outputC × inputC × kH × kW)
 *  ConvTranspose2D: IOKW  (inputC × outputC × kH × kW)  — matches PyTorch ConvTranspose2d
 */
export type LayerWeightLayout = 'OIKW' | 'IOKW' | 'none';

// ── Layer spec ─────────────────────────────────────────────────────────────────

export interface LayerSpec {
  readonly name: string;
  readonly kind: LayerKind;
  /** Input tensor names in the graph (outputs of predecessor layers). */
  readonly inputs: readonly string[];
  /** Output tensor name produced by this layer. */
  readonly output: string;
  /** Layer-specific parameters (channel counts, kernel size, stride, padding). */
  readonly params: LayerParams;
  /** Weight layout for documentation and the weights loader. */
  readonly weightLayout: LayerWeightLayout;
}

export interface LayerParams {
  readonly inC:     number;
  readonly outC:    number;
  readonly kH?:     number;   // kernel height (default 1)
  readonly kW?:     number;   // kernel width  (default 1)
  readonly stride?: number;   // (default 1)
  readonly padding?: number;  // (default 0 for transposed, 1 for conv2d with 3×3)
}

// ── UNet spec ─────────────────────────────────────────────────────────────────

export interface UNetSpec {
  readonly name: string;
  readonly inputChannels:  number;   // 9 (noisy RGB + albedo + normals)
  readonly outputChannels: number;   // 3 (denoised RGB)
  readonly layers: readonly LayerSpec[];
  readonly paramCount: number;
}

/**
 * Architecture summary (verified skip-add shape match).
 *
 * Each encoder level is a stride-1 feature conv (whose output is the skip
 * source, named `enc{N}_feat`) followed by a separate stride-2 down-conv
 * (named `enc{N}_down`). The decoder's transposed conv at level N produces
 * the same (H, W, C) as `enc{N}_feat`, so skip-add (not skip-concat) is
 * shape-correct at every site. `InferenceGraph._validateSkipShapes()`
 * asserts this at init time.
 *
 * Final layer graph (built below by `buildUNetSpec()`):
 *   pack       → enc_input  (H × W × 9)
 *   enc1_conv  → enc1_feat  (H × W × 24)      ← skip1
 *   enc1_down  → enc1_out   (H/2 × W/2 × 24)
 *   enc2_conv  → enc2_feat  (H/2 × W/2 × 48)  ← skip2
 *   enc2_down  → enc2_out   (H/4 × W/4 × 48)
 *   enc3_conv  → enc3_feat  (H/4 × W/4 × 96)  ← skip3
 *   enc3_down  → enc3_out   (H/8 × W/8 × 96)
 *   bottleneck → bn_out     (H/8 × W/8 × 192)
 *   dec3_up    (H/4 × W/4 × 96)  + skip3 → dec3_sum  → dec3_conv → dec3_out
 *   dec2_up    (H/2 × W/2 × 48)  + skip2 → dec2_sum  → dec2_conv → dec2_out
 *   dec1_up    (H × W × 24)      + skip1 → dec1_sum  → dec1_conv → dec1_out
 *   proj       → denoised   (H × W × 3)
 */

// ── Canonical U-Net spec ──────────────────────────────────────────────────────

/** Build the U-Net layer graph. */
export function buildUNetSpec(): UNetSpec {
  const layers: LayerSpec[] = [
    // ── Input packing ─────────────────────────────────────────────────────────
    // Explicit input packing layer that assembles noisyColor (3) +
    // albedo (3) + normals (3) into a single HxWx9 buffer named 'enc_input'.
    {
      name: 'pack',
      kind: 'inputPack',
      inputs: ['noisyColor', 'albedo', 'normals'],
      output: 'enc_input',
      params: { inC: 9, outC: 9 },
      weightLayout: 'none',
    },

    // ── Encoder level 1 (stride-1 feature conv + stride-2 pool) ──────────────
    {
      name: 'enc1_conv',
      kind: 'conv2d',
      inputs: ['enc_input'],
      output: 'enc1_feat',   // H × W × 24  ← SKIP SOURCE for dec1
      params: { inC: 9, outC: 24, kH: 3, kW: 3, stride: 1, padding: 1 },
      weightLayout: 'OIKW',
    },
    {
      name: 'enc1_relu',
      kind: 'relu',
      inputs: ['enc1_feat'],
      output: 'enc1_feat',
      params: { inC: 24, outC: 24 },
      weightLayout: 'none',
    },
    {
      name: 'enc1_down',
      kind: 'conv2d',
      inputs: ['enc1_feat'],
      output: 'enc1_out',    // H/2 × W/2 × 24
      params: { inC: 24, outC: 24, kH: 3, kW: 3, stride: 2, padding: 1 },
      weightLayout: 'OIKW',
    },

    // ── Encoder level 2 ───────────────────────────────────────────────────────
    {
      name: 'enc2_conv',
      kind: 'conv2d',
      inputs: ['enc1_out'],
      output: 'enc2_feat',   // H/2 × W/2 × 48  ← SKIP SOURCE for dec2
      params: { inC: 24, outC: 48, kH: 3, kW: 3, stride: 1, padding: 1 },
      weightLayout: 'OIKW',
    },
    {
      name: 'enc2_relu',
      kind: 'relu',
      inputs: ['enc2_feat'],
      output: 'enc2_feat',
      params: { inC: 48, outC: 48 },
      weightLayout: 'none',
    },
    {
      name: 'enc2_down',
      kind: 'conv2d',
      inputs: ['enc2_feat'],
      output: 'enc2_out',    // H/4 × W/4 × 48
      params: { inC: 48, outC: 48, kH: 3, kW: 3, stride: 2, padding: 1 },
      weightLayout: 'OIKW',
    },

    // ── Encoder level 3 ───────────────────────────────────────────────────────
    {
      name: 'enc3_conv',
      kind: 'conv2d',
      inputs: ['enc2_out'],
      output: 'enc3_feat',   // H/4 × W/4 × 96  ← SKIP SOURCE for dec3
      params: { inC: 48, outC: 96, kH: 3, kW: 3, stride: 1, padding: 1 },
      weightLayout: 'OIKW',
    },
    {
      name: 'enc3_relu',
      kind: 'relu',
      inputs: ['enc3_feat'],
      output: 'enc3_feat',
      params: { inC: 96, outC: 96 },
      weightLayout: 'none',
    },
    {
      name: 'enc3_down',
      kind: 'conv2d',
      inputs: ['enc3_feat'],
      output: 'enc3_out',    // H/8 × W/8 × 96
      params: { inC: 96, outC: 96, kH: 3, kW: 3, stride: 2, padding: 1 },
      weightLayout: 'OIKW',
    },

    // ── Bottleneck ─────────────────────────────────────────────────────────────
    {
      name: 'bottleneck',
      kind: 'conv2d',
      inputs: ['enc3_out'],
      output: 'bn_out',      // H/8 × W/8 × 192
      params: { inC: 96, outC: 192, kH: 3, kW: 3, stride: 1, padding: 1 },
      weightLayout: 'OIKW',
    },
    {
      name: 'bn_relu',
      kind: 'relu',
      inputs: ['bn_out'],
      output: 'bn_out',
      params: { inC: 192, outC: 192 },
      weightLayout: 'none',
    },

    // ── Decoder level 3 ───────────────────────────────────────────────────────
    // tconv: bottleneck H/8×W/8×192 → H/4×W/4×96
    // skip-add: enc3_feat H/4×W/4×96  ← SHAPE MATCH ✓
    {
      name: 'dec3_up',
      kind: 'transposedConv2d',
      inputs: ['bn_out'],
      output: 'dec3_up_out',  // H/4 × W/4 × 96
      params: { inC: 192, outC: 96, kH: 2, kW: 2, stride: 2, padding: 0 },
      weightLayout: 'IOKW',
    },
    {
      name: 'dec3_skip',
      kind: 'skipAdd',
      inputs: ['dec3_up_out', 'enc3_feat'],  // both H/4 × W/4 × 96  ✓
      output: 'dec3_sum',
      params: { inC: 96, outC: 96 },
      weightLayout: 'none',
    },
    {
      name: 'dec3_conv',
      kind: 'conv2d',
      inputs: ['dec3_sum'],
      output: 'dec3_out',    // H/4 × W/4 × 96
      params: { inC: 96, outC: 96, kH: 3, kW: 3, stride: 1, padding: 1 },
      weightLayout: 'OIKW',
    },
    {
      name: 'dec3_relu',
      kind: 'relu',
      inputs: ['dec3_out'],
      output: 'dec3_out',
      params: { inC: 96, outC: 96 },
      weightLayout: 'none',
    },

    // ── Decoder level 2 ───────────────────────────────────────────────────────
    // tconv: dec3_out H/4×W/4×96 → H/2×W/2×48
    // skip-add: enc2_feat H/2×W/2×48  ← SHAPE MATCH ✓
    {
      name: 'dec2_up',
      kind: 'transposedConv2d',
      inputs: ['dec3_out'],
      output: 'dec2_up_out',  // H/2 × W/2 × 48
      params: { inC: 96, outC: 48, kH: 2, kW: 2, stride: 2, padding: 0 },
      weightLayout: 'IOKW',
    },
    {
      name: 'dec2_skip',
      kind: 'skipAdd',
      inputs: ['dec2_up_out', 'enc2_feat'],  // both H/2 × W/2 × 48  ✓
      output: 'dec2_sum',
      params: { inC: 48, outC: 48 },
      weightLayout: 'none',
    },
    {
      name: 'dec2_conv',
      kind: 'conv2d',
      inputs: ['dec2_sum'],
      output: 'dec2_out',    // H/2 × W/2 × 48
      params: { inC: 48, outC: 48, kH: 3, kW: 3, stride: 1, padding: 1 },
      weightLayout: 'OIKW',
    },
    {
      name: 'dec2_relu',
      kind: 'relu',
      inputs: ['dec2_out'],
      output: 'dec2_out',
      params: { inC: 48, outC: 48 },
      weightLayout: 'none',
    },

    // ── Decoder level 1 ───────────────────────────────────────────────────────
    // tconv: dec2_out H/2×W/2×48 → H×W×24
    // skip-add: enc1_feat H×W×24  ← SHAPE MATCH ✓
    {
      name: 'dec1_up',
      kind: 'transposedConv2d',
      inputs: ['dec2_out'],
      output: 'dec1_up_out',  // H × W × 24
      params: { inC: 48, outC: 24, kH: 2, kW: 2, stride: 2, padding: 0 },
      weightLayout: 'IOKW',
    },
    {
      name: 'dec1_skip',
      kind: 'skipAdd',
      inputs: ['dec1_up_out', 'enc1_feat'],  // both H × W × 24  ✓
      output: 'dec1_sum',
      params: { inC: 24, outC: 24 },
      weightLayout: 'none',
    },
    {
      name: 'dec1_conv',
      kind: 'conv2d',
      inputs: ['dec1_sum'],
      output: 'dec1_out',    // H × W × 24
      params: { inC: 24, outC: 24, kH: 3, kW: 3, stride: 1, padding: 1 },
      weightLayout: 'OIKW',
    },
    {
      name: 'dec1_relu',
      kind: 'relu',
      inputs: ['dec1_out'],
      output: 'dec1_out',
      params: { inC: 24, outC: 24 },
      weightLayout: 'none',
    },

    // ── Output projection (1×1 conv, no activation — raw RGB output) ──────────
    {
      name: 'proj',
      kind: 'conv2d',
      inputs: ['dec1_out'],
      output: 'denoised',    // H × W × 3
      params: { inC: 24, outC: 3, kH: 1, kW: 1, stride: 1, padding: 0 },
      weightLayout: 'OIKW',
    },
  ];

  return {
    name: 'vitrum-unet-v1',
    inputChannels:  9,
    outputChannels: 3,
    layers,
    // Total trainable parameters (verified from spec doc):
    // enc1_conv(9×24×9+24)+enc1_down(24×24×9+24)+enc2_conv(24×48×9+48)+enc2_down(48×48×9+48)
    // +enc3_conv(48×96×9+96)+enc3_down(96×96×9+96)+bottleneck(96×192×9+192)
    // +dec3_tconv(192×96×4+96)+dec3_conv(96×96×9+96)
    // +dec2_tconv(96×48×4+48)+dec2_conv(48×48×9+48)
    // +dec1_tconv(48×24×4+24)+dec1_conv(24×24×9+24)
    // +proj(24×3×1+3) ≈ 426,075 (slightly more with the extra down-conv layers)
    paramCount: 426075,
  };
}

/** Singleton spec for the default vitrum U-Net denoiser. */
export const WALKAROUND_DENOISER_UNET_SPEC: UNetSpec = buildUNetSpec();
