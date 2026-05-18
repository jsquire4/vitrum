import type { Engine, EngineFactory } from '@vitrum/core';
import { summarizeScene, type SceneSummary } from './scene/flattenScene.js';
import {
  PTEngineWebGPU,
  PROTOTYPE_MAX_BOUNCES,
  makeStateSlot,
  type PTEngineWebGPUOptions,
} from './engine/PTEngineWebGPU.js';
import { PT_WEBGPU_COMMON_WGSL } from './wgsl/common.wgsl.js';
import {
  HAMMERSLEY_WGSL,
  OCTAHEDRAL_CORE_WGSL,
} from '@vitrum/shared-samplers';

export { PT_WEBGPU_COMMON_WGSL, HAMMERSLEY_WGSL, OCTAHEDRAL_CORE_WGSL };
export { summarizeScene };
export type { SceneSummary };
export type { PTEngineWebGPUOptions };

export const createPTEngine_WebGPU: EngineFactory<PTEngineWebGPUOptions> = async (
  opts: PTEngineWebGPUOptions,
): Promise<Engine> => {
  if (opts.device == null || typeof (opts.device as GPUDevice).createCommandEncoder !== 'function') {
    throw new TypeError(
      'createPTEngine_WebGPU: device must be a GPUDevice instance',
    );
  }
  const maxBounces = opts.maxBounces;
  if (maxBounces !== undefined && maxBounces < 1) {
    throw new RangeError(
      `createPTEngine_WebGPU: maxBounces structural cap must be >= 1 (got ${maxBounces})`,
    );
  }
  const maxSpp = opts.maxSamplesPerPixel;
  if (maxSpp !== undefined && maxSpp < 1) {
    throw new RangeError(
      `createPTEngine_WebGPU: maxSamplesPerPixel structural cap must be >= 1 (got ${maxSpp})`,
    );
  }
  if (maxBounces !== undefined && maxBounces > PROTOTYPE_MAX_BOUNCES) {
    console.warn(
      `[vitrum/pt-webgpu] maxBounces=${maxBounces} requested, clamping to prototype limit ${PROTOTYPE_MAX_BOUNCES}.`,
    );
  }
  if (opts.denoiser != null && opts.denoiser !== 'none') {
    console.warn(
      `[vitrum/pt-webgpu] denoiser="${opts.denoiser}" requested, but prototype backend has no denoiser integration yet.`,
    );
  }
  const slot = makeStateSlot();
  const engine = new PTEngineWebGPU(opts, slot);
  slot.set('ready');
  return engine;
};
