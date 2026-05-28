/**
 * Neural U-Net placeholder bundle (W4a — resourceManager split).
 *
 * W10 populated inference graph buffers lazily; frame bundle stays empty at init.
 */

import type { NeuralFrameResources } from '../resourceManager.js';

export function createNeuralFrameResources(): NeuralFrameResources {
  return Object.freeze({});
}
