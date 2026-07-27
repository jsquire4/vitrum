import { describe, expect, it } from 'vitest';

import {
  WALKAROUND_NEURAL_DENOISER_SHAPE_REQUIREMENT,
  assertWalkaroundNeuralDenoiserShape,
  walkaroundNeuralInferenceExtent,
  walkaroundNeuralDenoiserShapeError,
} from '../shapeContract.js';
import { preflightTensorDims } from '../tensorDimSolver.js';
import { buildUNetSpec } from '../unetArchitecture.js';

describe('walkaround neural denoiser shape contract', () => {
  it('accepts every positive logical extent and resolves the private padded lattice', () => {
    expect(WALKAROUND_NEURAL_DENOISER_SHAPE_REQUIREMENT).toEqual({
      minWidth: 1,
      minHeight: 1,
      widthMultiple: 1,
      heightMultiple: 1,
    });
    expect(walkaroundNeuralDenoiserShapeError(1, 1)).toBeNull();
    expect(walkaroundNeuralDenoiserShapeError(9, 8)).toBeNull();
    expect(() => assertWalkaroundNeuralDenoiserShape(17, 9)).not.toThrow();
    expect(walkaroundNeuralInferenceExtent(1, 1)).toMatchObject({
      inferenceWidth: 8,
      inferenceHeight: 8,
    });
    expect(walkaroundNeuralInferenceExtent(9, 8)).toMatchObject({
      inferenceWidth: 16,
      inferenceHeight: 8,
    });

    const extent = walkaroundNeuralInferenceExtent(17, 9);
    const dims = preflightTensorDims(
      buildUNetSpec(),
      extent.inferenceWidth,
      extent.inferenceHeight,
    );
    expect(dims.get('denoised')).toEqual({ H: 16, W: 24, C: 3 });
  });

  it.each([
    [0, 8, 'zero width'],
    [8, -1, 'negative height'],
    [8.5, 8, 'fractional width'],
    [8, Number.NaN, 'non-finite height'],
  ])('rejects %s x %s (%s)', (width, height) => {
    expect(walkaroundNeuralDenoiserShapeError(width, height)).not.toBeNull();
    expect(() => assertWalkaroundNeuralDenoiserShape(width, height)).toThrow(
      /unsupported internal render size/,
    );
  });

  it('detects canonical skip mismatches in pure preflight', () => {
    expect(() => preflightTensorDims(buildUNetSpec(), 9, 8)).toThrow(
      /skipAdd layer .* shape mismatch/,
    );
  });
});
