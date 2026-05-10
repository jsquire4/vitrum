/**
 * svgfBindings.ts — TypeScript helpers for wiring up the SVGF denoiser.
 *
 * Provides typed descriptors for SVGF's bind group layouts and uniform
 * structs.  No GPU objects are created here; this is a pure-TypeScript
 * shape/packer layer consumed by host code that builds the actual WebGPU
 * pipeline.
 *
 * Two passes require two distinct bind group layouts:
 *   1. SVGFVarianceUBO  — variance estimation pass (svgfVarianceMain)
 *   2. SVGFAtrousUBO    — à-trous wavelet pass     (svgfAtrousMain)
 *
 * std140 packing notes (WebGPU uniform buffer layout rules):
 *   - Each f32/u32 scalar is 4 bytes, aligned to 4 bytes.
 *   - A struct containing only scalars requires no inter-field padding
 *     when all fields are the same primitive size.
 *   - The buffer itself must be a multiple of 16 bytes (WebGPU min binding
 *     size for uniform buffers). Explicit _pad fields keep structs 16-byte
 *     aligned for driver compatibility.
 *
 * References:
 *   Schied et al. "Spatiotemporal Variance-Guided Filtering" HPG 2017.
 *   Sprint 10a spec: plan/phase-6-roadmap.md §Sprint 10a.
 */

// ============================================================
// Variance estimation pass uniforms
// ============================================================

/**
 * Uniforms for the SVGF variance estimation pass (svgfVarianceMain).
 *
 * frameCount drives the switch between spatial-neighborhood variance
 * (when temporal history is sparse) and Welford temporal variance
 * (when the accumulation buffer has settled).
 *
 * The threshold is 4 frames (hard-coded in the shader). Hosts should
 * reset frameCount to 0 on camera move / scene change and increment
 * it each frame thereafter.
 */
export interface SVGFVarianceUniforms {
  /** Cumulative frames since the last camera reset. 0 = first frame. */
  readonly frameCount: number;
}

/**
 * Byte size of the SVGFVarianceUBO std140 struct.
 *
 * Layout:
 *   offset 0  — frameCount : u32  (4 bytes)
 *   offset 4  — _pad0      : u32  (4 bytes, alignment padding)
 *   offset 8  — _pad1      : u32  (4 bytes, alignment padding)
 *   offset 12 — _pad2      : u32  (4 bytes, alignment padding)
 * Total: 16 bytes (meets 16-byte uniform buffer alignment requirement).
 */
export const SVGF_VARIANCE_UNIFORMS_SIZE_BYTES = 16 as const;

/**
 * Pack SVGFVarianceUniforms into an ArrayBuffer at the given byte offset.
 *
 * @param u       - Uniform values to pack.
 * @param target  - Destination ArrayBuffer (must be ≥ offset + SVGF_VARIANCE_UNIFORMS_SIZE_BYTES).
 * @param offset  - Byte offset into target (default: 0).
 */
export function packSVGFVarianceUniforms(
  u: SVGFVarianceUniforms,
  target: ArrayBuffer,
  offset = 0,
): void {
  const view = new DataView(target, offset, SVGF_VARIANCE_UNIFORMS_SIZE_BYTES);
  // frameCount at offset 0, little-endian u32
  view.setUint32(0, u.frameCount >>> 0, true);
  // _pad0, _pad1, _pad2 — zero-filled for determinism
  view.setUint32(4,  0, true);
  view.setUint32(8,  0, true);
  view.setUint32(12, 0, true);
}

// ============================================================
// À-trous iteration uniforms
// ============================================================

/**
 * Uniforms for one SVGF à-trous wavelet iteration (svgfAtrousMain).
 *
 * The host dispatches svgfAtrousMain 5 times, incrementing `iteration`
 * from 0 to 4 and ping-ponging the color texture between passes.
 *
 * Default σ values follow Schied 2017 Table 1:
 *   sigmaColor  = 10.0  — normalized luminance tolerance per unit variance
 *   sigmaNormal = 128.0 — high exponent → preserves sharp surface boundaries
 *   sigmaDepth  = 1.0   — world-unit depth tolerance
 *
 * Hosts may tune σ values per scene. Recommend starting with defaults and
 * reducing sigmaColor when caustic edges over-blur on glass surfaces.
 */
export interface SVGFUniforms {
  /** À-trous iteration index (0–4). Step width = 2^iteration pixels. */
  readonly iteration: number;
  /**
   * Color edge-stop σ. Higher values → less color sensitivity → more blur.
   * Variance-modulated in the shader: effective tolerance = sigmaColor * sqrt(variance).
   * Default: 10.0 (Schied 2017 Table 1).
   */
  readonly sigmaColor: number;
  /**
   * Normal edge-stop σ, applied as an exponent on the clamped dot product.
   * Higher values → sharper preservation of geometric edges.
   * Default: 128.0 (Schied 2017 Table 1).
   */
  readonly sigmaNormal: number;
  /**
   * Depth edge-stop σ in world units. Controls blending across depth
   * discontinuities. Tune relative to scene scale.
   * Default: 1.0.
   */
  readonly sigmaDepth: number;
}

/**
 * Byte size of the SVGFAtrousUBO std140 struct.
 *
 * Layout:
 *   offset 0  — iteration   : u32  (4 bytes)
 *   offset 4  — sigmaColor  : f32  (4 bytes)
 *   offset 8  — sigmaNormal : f32  (4 bytes)
 *   offset 12 — sigmaDepth  : f32  (4 bytes)
 * Total: 16 bytes.
 */
export const SVGF_UNIFORMS_SIZE_BYTES = 16 as const;

/**
 * Pack SVGFUniforms into an ArrayBuffer at the given byte offset.
 *
 * @param u       - Uniform values to pack.
 * @param target  - Destination ArrayBuffer (must be ≥ offset + SVGF_UNIFORMS_SIZE_BYTES).
 * @param offset  - Byte offset into target (default: 0).
 */
export function packSVGFUniforms(
  u: SVGFUniforms,
  target: ArrayBuffer,
  offset = 0,
): void {
  const view = new DataView(target, offset, SVGF_UNIFORMS_SIZE_BYTES);
  // iteration at offset 0, little-endian u32
  view.setUint32(0, u.iteration >>> 0, true);
  // sigmaColor at offset 4, little-endian f32
  view.setFloat32(4,  u.sigmaColor,  true);
  // sigmaNormal at offset 8, little-endian f32
  view.setFloat32(8,  u.sigmaNormal, true);
  // sigmaDepth at offset 12, little-endian f32
  view.setFloat32(12, u.sigmaDepth,  true);
}

// ============================================================
// Bind group layout descriptors
// ============================================================

/**
 * Documents the bind group layout for the SVGF variance estimation pass.
 * Mirrors the @group(0) bindings in svgf.wgsl.ts — svgfVarianceMain.
 *
 * All texture formats are the render pipeline's conventions:
 *   - Color / radiance textures: rgba16float
 *   - Normal G-buffer: rgba16float (.xyz = world normal, .w unused)
 *   - Depth G-buffer: rgba16float (.r = linear depth) or r32float
 *   - Motion vectors: rg32float (.xy = screen-space UV delta, [-1,1])
 *   - Welford variance buffer: rg32float (.r = mean, .g = M2)
 *   - Output variance map: rg32float (.r = estimated variance, .g = frameCount)
 */
export interface SVGFVarianceBindGroupLayout {
  /** binding 0 — noisy current-frame color.  Format: rgba16float. */
  inputColor: 'texture_2d<f32>';
  /** binding 1 — reprojected previous-frame radiance.  Format: rgba16float. */
  prevRadiance: 'texture_2d<f32>';
  /** binding 2 — G-buffer world-space normal.  Format: rgba16float, .xyz channel. */
  gbufferNormal: 'texture_2d<f32>';
  /** binding 3 — G-buffer linear depth.  Format: rgba16float (.r) or r32float. */
  gbufferDepth: 'texture_2d<f32>';
  /** binding 4 — screen-space motion vectors.  Format: rg32float. */
  motionVectors: 'texture_2d<f32>';
  /**
   * binding 5 — Welford variance buffer from Sprint 9 accumulator.
   * Format: rg32float (.r = mean luminance, .g = M2 running sum).
   * @see walkaround-hybrid/src/shaders/common.wgsl.ts — WelfordVariance @version 1
   */
  varianceIn: 'texture_2d<f32>';
  /** binding 6 — estimated variance output.  Format: rg32float (storage write). */
  varianceOut: 'texture_storage_2d<rg32float, write>';
  /** binding 7 — SVGFVarianceUBO (frameCount + padding). */
  ubo: 'uniform SVGFVarianceUBO';
}

/**
 * Documents the bind group layout for the SVGF à-trous wavelet pass.
 * Mirrors the @group(0) bindings in svgf.wgsl.ts — svgfAtrousMain.
 *
 * The host ping-pongs inputColor / outputColor between the 5 iterations:
 *   Iteration 0: inputColor = noisy input, outputColor = temp buffer A
 *   Iteration 1: inputColor = temp buffer A, outputColor = temp buffer B
 *   Iteration 2: inputColor = temp buffer B, outputColor = temp buffer A
 *   Iteration 3: inputColor = temp buffer A, outputColor = temp buffer B
 *   Iteration 4: inputColor = temp buffer B, outputColor = final output
 */
export interface SVGFAtrousBindGroupLayout {
  /** binding 0 — filtered color from previous iteration (or noisy input for iter 0). */
  inputColor: 'texture_2d<f32>';
  /** binding 1 — filtered color output for this iteration.  Format: rgba16float (storage write). */
  outputColor: 'texture_storage_2d<rgba16float, write>';
  /** binding 2 — G-buffer world-space normal.  Format: rgba16float, .xyz channel. */
  gbufferNormal: 'texture_2d<f32>';
  /** binding 3 — G-buffer linear depth.  Format: rgba16float (.r) or r32float. */
  gbufferDepth: 'texture_2d<f32>';
  /**
   * binding 4 — per-pixel variance estimate from svgfVarianceMain.
   * Format: rg32float (.r = variance scalar).
   */
  varianceMap: 'texture_2d<f32>';
  /** binding 5 — SVGFAtrousUBO (iteration, sigma values). */
  ubo: 'uniform SVGFAtrousUBO';
}

// ============================================================
// Default σ values (exported as a convenience constant)
// ============================================================

/**
 * SVGF default edge-stopping σ parameters from Schied 2017 Table 1.
 *
 * These are the suggested starting values. Hosts should tune them
 * based on scene content:
 *   - Reduce sigmaColor for scenes with fine caustic detail (glass, came edges).
 *   - Reduce sigmaNormal for architectural scenes with many planar surfaces.
 *   - Adjust sigmaDepth relative to scene scale (larger rooms → larger depth σ).
 */
export const SVGF_DEFAULT_UNIFORMS: Omit<SVGFUniforms, 'iteration'> = {
  sigmaColor:  10.0,
  sigmaNormal: 128.0,
  sigmaDepth:  1.0,
} as const;
