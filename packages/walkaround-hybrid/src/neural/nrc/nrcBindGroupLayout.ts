/** NRC-specific group(3) layout: hybrid layers plus two packed arenas. */

import type { BGLCache } from '../../bglTypes.js';

/**
 * The NRC query must remain within the WebGPU guaranteed four bind groups.
 * Bindings 0..6 are byte-for-byte the ordinary hybrid-layers layout. NRC adds:
 *   7 — immutable versioned inference arena (weights/biases/tables/levels)
 *   8 — mutable runtime arena (records/claims/diagnostics)
 *   9 — query config UBO
 */
export function getNrcHybridLayersBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.hybridLayersNrc) return cache.hybridLayersNrc;
  cache.hybridLayersNrc = device.createBindGroupLayout({
    label: 'hybrid-layers-nrc-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  return cache.hybridLayersNrc;
}
