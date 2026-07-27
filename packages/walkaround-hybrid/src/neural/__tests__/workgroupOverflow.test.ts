import { describe, expect, it } from 'vitest';

import { neuralDeviceLimitFailures } from '../tensorMemoryPlanner.js';
import type { TensorDims } from '../tensorDimSolver.js';
import type { UNetSpec } from '../unetArchitecture.js';

describe('neural 2D dispatch limit preflight', () => {
  it('rejects before encoding when ceil(totalGroups/maxX) exceeds maxY', () => {
    const spec: UNetSpec = {
      name: 'elementwise-overflow',
      inputChannels: 1,
      outputChannels: 1,
      layers: [{
        name: 'relu',
        kind: 'relu',
        inputs: ['input'],
        output: 'denoised',
        params: { inC: 1, outC: 1 },
        weightLayout: 'none',
      }],
      paramCount: 0,
    };
    const dims = new Map<string, TensorDims>([
      ['input', { H: 1, W: 1025, C: 1 }],
      ['denoised', { H: 1, W: 1025, C: 1 }],
    ]);
    const device = {
      limits: {
        maxBufferSize: 1_000_000,
        maxStorageBufferBindingSize: 1_000_000,
        maxUniformBufferBindingSize: 48,
        maxComputeWorkgroupsPerDimension: 2,
        maxTextureDimension2D: 2048,
        maxStorageBuffersPerShaderStage: 8,
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxComputeWorkgroupSizeY: 256,
      },
    } as unknown as GPUDevice;
    expect(neuralDeviceLimitFailures(device, spec, { layers: [] }, dims))
      .toContainEqual(expect.stringContaining("layer 'relu' workgroup axis 1=3"));
  });
});
