import {
  buildRepresentedPmfF32,
  luminance,
} from '@vitrum/shared-samplers';
import { DIRECTIONAL_LIGHT_FLOAT_STRIDE } from './emitterPacking.js';

/**
 * Materialize the exact categorical proposal consumed by medium NEE.
 *
 * Directional record row 1 keeps radiance in `.rgb`; its `.w` lane publishes
 * the represented PMF. The optional environment PMF has no storage-buffer
 * record, so the caller publishes the returned scalar through FrameParams.
 * Both are exact multiples of 2^-24, matching the shader's canonical random
 * domain and preserving every positive physical source.
 */
export function applyDistantDirectProposalPmf(
  directionalLightsData: Float32Array,
  directionalLightCount: number,
  environmentLightTreePower: number,
  hasEnvironmentMap: boolean,
): number {
  if (!Number.isSafeInteger(directionalLightCount) || directionalLightCount < 0) {
    throw new RangeError(
      'pt-webgpu distant-direct directionalLightCount must be a non-negative safe integer.',
    );
  }
  const expectedLength = directionalLightCount * DIRECTIONAL_LIGHT_FLOAT_STRIDE;
  if (directionalLightsData.length !== expectedLength) {
    throw new Error(
      `pt-webgpu distant-direct directional data has ${directionalLightsData.length} floats; ` +
        `expected ${expectedLength}.`,
    );
  }
  if (!Number.isFinite(environmentLightTreePower) || environmentLightTreePower < 0) {
    throw new RangeError(
      'pt-webgpu distant-direct environment power must be finite and non-negative.',
    );
  }

  const weights = new Float64Array(
    directionalLightCount + (hasEnvironmentMap ? 1 : 0),
  );
  for (let i = 0; i < directionalLightCount; i += 1) {
    const base = i * DIRECTIONAL_LIGHT_FLOAT_STRIDE;
    weights[i] = Math.max(
      0,
      luminance(
        directionalLightsData[base + 4] ?? 0,
        directionalLightsData[base + 5] ?? 0,
        directionalLightsData[base + 6] ?? 0,
      ),
    );
  }
  if (hasEnvironmentMap) {
    weights[directionalLightCount] = environmentLightTreePower;
  }

  const representedPmf = buildRepresentedPmfF32(weights);
  for (let i = 0; i < directionalLightCount; i += 1) {
    directionalLightsData[i * DIRECTIONAL_LIGHT_FLOAT_STRIDE + 7] =
      representedPmf[i] ?? 0;
  }
  return hasEnvironmentMap
    ? (representedPmf[directionalLightCount] ?? 0)
    : 0;
}
