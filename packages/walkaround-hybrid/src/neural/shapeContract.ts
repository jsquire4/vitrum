import type { DenoiserSpatialShapeRequirement } from '@vitrum/core';

/**
 * Host-visible spatial contract of the canonical walkaround U-Net.
 *
 * The graph pads every positive logical extent to its private eight-pixel
 * inference lattice, then crops the final RGB tensor back to the exact logical
 * extent. Hosts therefore have no alignment obligation.
 */
export const WALKAROUND_NEURAL_DENOISER_SHAPE_REQUIREMENT = Object.freeze({
  minWidth: 1,
  minHeight: 1,
  widthMultiple: 1,
  heightMultiple: 1,
} satisfies DenoiserSpatialShapeRequirement);

/** Private lattice required by the canonical three-level encoder/decoder. */
export const WALKAROUND_NEURAL_INFERENCE_ALIGNMENT = 8;

export interface WalkaroundNeuralInferenceExtent {
  readonly logicalWidth: number;
  readonly logicalHeight: number;
  readonly inferenceWidth: number;
  readonly inferenceHeight: number;
}

/** Return a host-readable reason when `(width, height)` violates the contract. */
export function walkaroundNeuralDenoiserShapeError(
  width: number,
  height: number,
): string | null {
  const req = WALKAROUND_NEURAL_DENOISER_SHAPE_REQUIREMENT;
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height)) {
    return 'width and height must be safe integers';
  }
  if (width < req.minWidth || height < req.minHeight) {
    return `width must be >= ${req.minWidth} and height must be >= ${req.minHeight}`;
  }
  return null;
}

/** Resolve the exact logical extent and its private, zero-padded U-Net extent. */
export function walkaroundNeuralInferenceExtent(
  width: number,
  height: number,
): WalkaroundNeuralInferenceExtent {
  assertWalkaroundNeuralDenoiserShape(width, height);
  const alignment = WALKAROUND_NEURAL_INFERENCE_ALIGNMENT;
  const inferenceWidth = Math.ceil(width / alignment) * alignment;
  const inferenceHeight = Math.ceil(height / alignment) * alignment;
  if (!Number.isSafeInteger(inferenceWidth)
      || !Number.isSafeInteger(inferenceHeight)) {
    throw new RangeError(
      `[NeuralDenoiser] padded inference extent is not a safe integer: ${inferenceWidth}x${inferenceHeight}`,
    );
  }
  return { logicalWidth: width, logicalHeight: height, inferenceWidth, inferenceHeight };
}

/**
 * Reject a non-positive or non-integral host extent synchronously and before
 * any neural GPU resource allocation.
 */
export function assertWalkaroundNeuralDenoiserShape(width: number, height: number): void {
  const reason = walkaroundNeuralDenoiserShapeError(width, height);
  if (reason == null) return;
  throw new RangeError(
    `[NeuralDenoiser] unsupported internal render size ${width}x${height}: ${reason}.`,
  );
}
