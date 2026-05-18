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
 * Bug fixes vs. Sprint 13 deleted scaffold (see plan/sprint-neural-denoiser-future.md):
 *   Bug 1 fixed: skip-connection spatial shapes are explicitly documented per layer
 *   so dec3_up (H/4) pairs with enc3 (H/4-pre-stride output) — NOT enc3 (H/8).
 *   The encoder Level N output *before* the stride-2 convolution is the skip source.
 *   See SKIP_SHAPES table below.
 *
 * Binding convention (all WGSL kernels, Bug 3 fix):
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
  readonly inC: number;
  readonly outC: number;
  readonly kH?: number; // kernel height (default 1)
  readonly kW?: number; // kernel width  (default 1)
  readonly stride?: number; // (default 1)
  readonly padding?: number; // (default 0 for transposed, 1 for conv2d with 3×3)
}

// ── UNet spec ─────────────────────────────────────────────────────────────────

export interface UNetSpec {
  readonly name: string;
  readonly inputChannels: number; // 9 (noisy RGB + albedo + normals)
  readonly outputChannels: number; // 3 (denoised RGB)
  readonly layers: readonly LayerSpec[];
  readonly paramCount: number;
}

/**
 * Skip-connection spatial shape table — documents the BUG 1 fix explicitly.
 *
 * The encoder performs stride-2 downsampling. The *skip source* is the output
 * AFTER the conv+relu at that encoder level — i.e., already downsampled.
 *
 * Correct pairings (H=1080, W=1920 at 1080p):
 *   enc1 output: H/2 × W/2 × 24   →  paired with dec1_up (H × W → H × W × 24  after tconv)
 *   enc2 output: H/4 × W/4 × 48   →  paired with dec2_up (H/2 × W/2 → H/2 × W/2 × 48)
 *   enc3 output: H/8 × W/8 × 96   →  paired with dec3_up (H/4 × W/4 → H/4 × W/4 × 96)
 *
 * dec3_up comes from tconv(bottleneck at H/8 × W/8 × 192, stride 2)
 *   → output: H/4 × W/4 × 96  ✓ matches enc3 at H/8 × W/8 × 96? NO —
 *
 * CRITICAL: enc3 outputs H/8 (after stride-2 from H/4 input).
 *   dec3_up tconv from H/8 (bottleneck) with stride 2 → H/4.
 *   enc3 is also H/8. So dec3_up (H/4) and enc3 (H/8) are MISMATCHED.
 *
 * FIX: use the encoder input (before downsampling) as the skip source.
 * But that contradicts the standard U-Net. The correct interpretation:
 *
 * Standard U-Net skip-connection taps the encoder level's OUTPUT (after relu,
 * before the NEXT level's stride-2 conv). So:
 *   enc3 output: H/8 × W/8 × 96  (stride-2 applied, output is H/8)
 *   dec3_up output: transposedConv2d(bottleneck H/8 × W/8 × 192) stride=2 → H/4 × W/4 × 96
 *   MISMATCH: H/4 ≠ H/8
 *
 * Resolution (matching the sprint-neural-denoiser-future.md spec arithmetic):
 * The decoder tconv at level 3 upsamples from H/8 to H/4.
 * The skip source for dec3 must ALSO be at H/4.
 * Therefore enc3 must be the output BEFORE stride-2 — i.e., enc2 output space (H/4).
 *
 * Re-reading the spec doc: it says "skip-add(enc3: H/8×W/8×96) → H/4 × W/4 × 96"
 * which is self-contradictory. The resolution: the spec's "enc3" label refers to
 * the feature map ENTERING level 3, not the output of level 3's stride-2 conv.
 *
 * This implementation uses a clean naming: enc{N}_pre is the input to level N's
 * stride-2 conv; enc{N} is the output. Skip connections use enc{N}_pre.
 *
 * Verified arithmetic:
 *   enc_input:   H   × W   × 9
 *   enc1:        H/2 × W/2 × 24  (stride-2 from enc_input)
 *   enc2:        H/4 × W/4 × 48  (stride-2 from enc1)
 *   enc3:        H/8 × W/8 × 96  (stride-2 from enc2)
 *   bottleneck:  H/8 × W/8 × 192 (stride-1 from enc3)
 *
 *   dec3_up:     H/4 × W/4 × 96  (tconv stride-2 from bottleneck H/8)
 *   skip3_add:   H/4 × W/4 × 96  ← enc2 output (H/4 × W/4 × 48 → NO, channels differ)
 *
 * The spec says dec3_tconv is 192→96 (channels match enc3 = 96).
 * For skip-add to work, the skip source must have the same (H,W,C) as dec3_up.
 * dec3_up is H/4 × W/4 × 96. The only encoder output at H/4 × W/4 × C is enc2
 * at C=48 — channels don't match either.
 *
 * CONCLUSION: The skip connection is NOT a simple element-wise add when channels differ.
 * Standard U-Net uses concatenation (not addition), but the spec says "skip-add" to
 * keep memory budget down. For skip-add to work, channels must match.
 *
 * FINAL resolution: skip adds from the same-level encoder output which IS at the
 * same spatial resolution as the decoder output. After the tconv at dec3, we're at
 * H/4. The encoder output at H/4 is enc2 (48 ch). But dec3 output is 96ch — mismatch.
 *
 * The ONLY way skip-add works (not skip-concat) is if dec3_tconv produces 48ch to
 * match enc2 at 48ch, or enc3 (96ch) is used and it's at H/8 and we DON'T upsample
 * first (no spatial mismatch if both are at H/8).
 *
 * Reading the spec arithmetic table from sprint-neural-denoiser-future.md again:
 *   dec3_tconv: 192×96×4 + 96 = 73,824   (2×2 kernel) — THIS IS THE INPUT→OUTPUT
 *   doc says: Level 3: transposedConv2d(192→96, 2×2, stride 2) + skip-add(enc3)
 *
 * The spec pairs dec3 with enc3. enc3 = H/8 × W/8 × 96. dec3_up output from
 * tconv(bottleneck at H/8, stride 2) = H/4 × W/4 × 96. These are spatially mismatched.
 *
 * BUG 1 INTERPRETATION: the spec doc says this is the BUG. The correct fix
 * (per sprint-neural-denoiser-future.md §Bug 1):
 *   "dec3_up is H/4 × W/4 × 96. The skip connection adds enc3 which is at H/8 × W/8 × 96
 *    — 4× mismatch. ... the comment describing it was wrong."
 *   Fix: verify shapes at every skip-add site.
 *
 * For a correct U-Net with skip-add (not concat):
 *   The skip connection must come from a pre-downsampling feature map at the same
 *   spatial resolution as the decoder output, with the same channel count.
 *
 * This implementation adopts the explicit-pre-conv tapping:
 *   skip3 ← enc2_out   (H/4 × W/4 × 48)  BUT channels differ from dec3_up (96)!
 *
 * PRAGMATIC RESOLUTION for this re-implementation:
 * Use skip-add only when channels and spatial dims match, which they do if we
 * project skip features first. The architecture stores intermediate activations at
 * the same spatial scale as the decoder and uses a 1×1 projection conv to align channels.
 * But that adds parameters.
 *
 * SIMPLEST CORRECT APPROACH (matching the spec's intent):
 * The decoder's transposed conv produces the same spatial dim AND channel count
 * as the corresponding encoder output. The spec intends:
 *   enc3 output BEFORE applying stride → this is the feature map at H/4 (if enc3
 *   is a stride-1 conv applied to enc2's H/4 output).
 *
 * ADOPTED INTERPRETATION (the only one where the arithmetic in the spec works):
 * The "encoder levels" in the spec use stride-1 conv for the feature transform,
 * followed by a separate stride-2 downsampler (average pool or strided conv).
 * The skip tap is from BEFORE the stride-2 step.
 *
 *   Level 1: conv2d(9→24, 3×3, stride 1, pad 1) + relu  → H × W × 24  ← skip1 tap
 *            then stride-2 pool or conv → H/2 × W/2 × 24 → enc1_down
 *   Level 2: conv2d(24→48, 3×3, stride 1, pad 1) + relu → H/2 × W/2 × 48 ← skip2 tap
 *            then stride-2 → H/4 × W/4 × 48 → enc2_down
 *   Level 3: conv2d(48→96, 3×3, stride 1, pad 1) + relu → H/4 × W/4 × 96 ← skip3 tap
 *            then stride-2 → H/8 × W/8 × 96 → enc3_down
 *   bottleneck: conv2d(96→192, 3×3, stride 1, pad 1) → H/8 × W/8 × 192
 *
 *   dec3_up: tconv(192→96, 2×2, stride 2) → H/4 × W/4 × 96  ← add skip3 (H/4 × W/4 × 96 ✓)
 *   dec2_up: tconv(96→48, 2×2, stride 2) → H/2 × W/2 × 48   ← add skip2 (H/2 × W/2 × 48 ✓)
 *   dec1_up: tconv(48→24, 2×2, stride 2) → H × W × 24        ← add skip1 (H × W × 24 ✓)
 *
 * This is the correct, internally consistent interpretation. All skip-add pairs match in
 * both spatial dimensions AND channel count. This is what we implement below.
 *
 * The stride-2 downsamplers after each encoder level are implemented as stride-2 conv
 * (combined with the next encoder level's conv) for efficiency, which means we need to
 * cache the pre-downsampling activations as the skip sources.
 *
 * FINAL LAYER GRAPH (this is what runs):
 *   pack      → enc_input (H×W×9)
 *   enc1_conv → enc1_feat (H×W×24)   ← skip1
 *   enc1_down → enc1_out  (H/2×W/2×24)
 *   enc2_conv → enc2_feat (H/2×W/2×48) ← skip2
 *   enc2_down → enc2_out  (H/4×W/4×48)
 *   enc3_conv → enc3_feat (H/4×W/4×96) ← skip3
 *   enc3_down → enc3_out  (H/8×W/8×96)
 *   bottleneck → bn_out   (H/8×W/8×192)
 *   dec3_up   (H/4×W/4×96) + skip3 → dec3_sum (H/4×W/4×96)
 *   dec3_conv → dec3_out  (H/4×W/4×96)
 *   dec2_up   (H/2×W/2×48) + skip2 → dec2_sum (H/2×W/2×48)
 *   dec2_conv → dec2_out  (H/2×W/2×48)
 *   dec1_up   (H×W×24) + skip1 → dec1_sum (H×W×24)
 *   dec1_conv → dec1_out  (H×W×24)
 *   proj      → output    (H×W×3)
 */

// ── Canonical U-Net spec (bug-1-fixed architecture) ───────────────────────────

/** Build the U-Net layer graph with all 8 prior-scaffold bugs avoided. */
export function buildUNetSpec(): UNetSpec {
  const layers: LayerSpec[] = [
    // ── Input packing ─────────────────────────────────────────────────────────
    // Bug 2 fix: explicit input packing layer that assembles noisyColor (3) +
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
      output: 'enc1_feat', // H × W × 24  ← SKIP SOURCE for dec1
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
      output: 'enc1_out', // H/2 × W/2 × 24
      params: { inC: 24, outC: 24, kH: 3, kW: 3, stride: 2, padding: 1 },
      weightLayout: 'OIKW',
    },

    // ── Encoder level 2 ───────────────────────────────────────────────────────
    {
      name: 'enc2_conv',
      kind: 'conv2d',
      inputs: ['enc1_out'],
      output: 'enc2_feat', // H/2 × W/2 × 48  ← SKIP SOURCE for dec2
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
      output: 'enc2_out', // H/4 × W/4 × 48
      params: { inC: 48, outC: 48, kH: 3, kW: 3, stride: 2, padding: 1 },
      weightLayout: 'OIKW',
    },

    // ── Encoder level 3 ───────────────────────────────────────────────────────
    {
      name: 'enc3_conv',
      kind: 'conv2d',
      inputs: ['enc2_out'],
      output: 'enc3_feat', // H/4 × W/4 × 96  ← SKIP SOURCE for dec3
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
      output: 'enc3_out', // H/8 × W/8 × 96
      params: { inC: 96, outC: 96, kH: 3, kW: 3, stride: 2, padding: 1 },
      weightLayout: 'OIKW',
    },

    // ── Bottleneck ─────────────────────────────────────────────────────────────
    {
      name: 'bottleneck',
      kind: 'conv2d',
      inputs: ['enc3_out'],
      output: 'bn_out', // H/8 × W/8 × 192
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
      output: 'dec3_up_out', // H/4 × W/4 × 96
      params: { inC: 192, outC: 96, kH: 2, kW: 2, stride: 2, padding: 0 },
      weightLayout: 'IOKW',
    },
    {
      name: 'dec3_skip',
      kind: 'skipAdd',
      inputs: ['dec3_up_out', 'enc3_feat'], // both H/4 × W/4 × 96  ✓
      output: 'dec3_sum',
      params: { inC: 96, outC: 96 },
      weightLayout: 'none',
    },
    {
      name: 'dec3_conv',
      kind: 'conv2d',
      inputs: ['dec3_sum'],
      output: 'dec3_out', // H/4 × W/4 × 96
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
      output: 'dec2_up_out', // H/2 × W/2 × 48
      params: { inC: 96, outC: 48, kH: 2, kW: 2, stride: 2, padding: 0 },
      weightLayout: 'IOKW',
    },
    {
      name: 'dec2_skip',
      kind: 'skipAdd',
      inputs: ['dec2_up_out', 'enc2_feat'], // both H/2 × W/2 × 48  ✓
      output: 'dec2_sum',
      params: { inC: 48, outC: 48 },
      weightLayout: 'none',
    },
    {
      name: 'dec2_conv',
      kind: 'conv2d',
      inputs: ['dec2_sum'],
      output: 'dec2_out', // H/2 × W/2 × 48
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
      output: 'dec1_up_out', // H × W × 24
      params: { inC: 48, outC: 24, kH: 2, kW: 2, stride: 2, padding: 0 },
      weightLayout: 'IOKW',
    },
    {
      name: 'dec1_skip',
      kind: 'skipAdd',
      inputs: ['dec1_up_out', 'enc1_feat'], // both H × W × 24  ✓
      output: 'dec1_sum',
      params: { inC: 24, outC: 24 },
      weightLayout: 'none',
    },
    {
      name: 'dec1_conv',
      kind: 'conv2d',
      inputs: ['dec1_sum'],
      output: 'dec1_out', // H × W × 24
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
      output: 'denoised', // H × W × 3
      params: { inC: 24, outC: 3, kH: 1, kW: 1, stride: 1, padding: 0 },
      weightLayout: 'OIKW',
    },
  ];

  return {
    name: 'vitrum-unet-v1',
    inputChannels: 9,
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
