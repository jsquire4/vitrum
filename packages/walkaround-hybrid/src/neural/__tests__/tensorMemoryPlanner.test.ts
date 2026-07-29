import { describe, expect, it } from 'vitest';

import {
  buildTensorAllocationPlan,
  neuralDeviceLimitFailures,
} from '../tensorMemoryPlanner.js';
import { preflightTensorDims } from '../tensorDimSolver.js';
import {
  buildUNetSpec,
  deriveParamCount,
  type UNetSpec,
} from '../unetArchitecture.js';
import { NEURAL_F16_TENSOR_STORAGE } from '../tensorPrecision.js';
import { buildRandomWeightsForSpec } from '../../../__tests__/helpers/neuralWeights.js';

function inclusiveLifetimeSpec(): UNetSpec {
  const layers = [
    {
      name: 'pack',
      kind: 'inputPack',
      inputs: ['noisyColor', 'albedo', 'normals'],
      output: 'enc_input',
      params: { inC: 9, outC: 9 },
      weightLayout: 'none',
    },
    {
      name: 'relu',
      kind: 'relu',
      inputs: ['enc_input'],
      output: 'activated',
      params: { inC: 9, outC: 9 },
      weightLayout: 'none',
    },
    {
      name: 'proj',
      kind: 'conv2d',
      inputs: ['activated'],
      output: 'denoised',
      params: { inC: 9, outC: 3, kH: 1, kW: 1, stride: 1, padding: 0 },
      weightLayout: 'OIKW',
    },
  ] as const;
  return {
    name: 'inclusive-lifetime',
    inputChannels: 9,
    outputChannels: 3,
    layers,
    paramCount: deriveParamCount(layers),
  };
}

describe('neural tensor memory planner', () => {
  it('treats a final-consumer layer as inclusive and never aliases its input with its output', () => {
    const spec = inclusiveLifetimeSpec();
    const plan = buildTensorAllocationPlan(spec, preflightTensorDims(spec, 8, 8));
    const encSlot = plan.tensorToSlot.get('enc_input');
    const activatedSlot = plan.tensorToSlot.get('activated');
    const denoisedSlot = plan.tensorToSlot.get('denoised');

    expect(encSlot).toBeDefined();
    expect(activatedSlot).toBeDefined();
    expect(denoisedSlot).toBeDefined();
    expect(encSlot).not.toBe(activatedSlot);
    expect(activatedSlot).not.toBe(denoisedSlot);
    expect(denoisedSlot).toBe(encSlot);
  });

  it('never assigns a canonical layer output to any still-live input slot', () => {
    const spec = buildUNetSpec();
    const plan = buildTensorAllocationPlan(spec, preflightTensorDims(spec, 64, 64));

    for (const layer of spec.layers) {
      const outputSlot = plan.tensorToSlot.get(layer.output);
      for (const input of layer.inputs) {
        const inputSlot = plan.tensorToSlot.get(input);
        if (inputSlot !== undefined) {
          expect(
            outputSlot,
            `layer ${layer.name} aliases live input ${input}`,
          ).not.toBe(inputSlot);
        }
      }
    }
    expect(plan.reuseSavingsBytes).toBeGreaterThan(0);
    expect(plan.physicalTensorBytes).toBeLessThan(plan.logicalTensorBytes);
    expect(plan.peakLiveTensorBytes).toBeLessThanOrEqual(plan.physicalTensorBytes);
  });

  it('pins canonical 1080p storage telemetry', () => {
    const spec = buildUNetSpec();
    const plan = buildTensorAllocationPlan(spec, preflightTensorDims(spec, 1920, 1080));
    expect(plan.logicalTensorBytes).toBe(2_326_579_200);
    expect(plan.physicalTensorBytes).toBe(945_561_600);
    expect(plan.reuseSavingsBytes).toBe(1_381_017_600);
    expect(plan.peakLiveTensorBytes).toBe(622_080_000);
    expect(plan.slots).toHaveLength(8);
    expect(plan.tensorToSlot.size).toBe(25);
    expect(plan.slots.map(slot => slot.logicalTensors.length)).toEqual([5, 9, 2, 2, 2, 2, 2, 1]);
    expect(plan.slots.map(slot => slot.byteSize)).toEqual([74_649_600, 199_065_600, 199_065_600, 99_532_800, 24_883_200, 49_766_400, 99_532_800, 199_065_600]);
  });

  it('pins canonical 1080p f16 storage below a 128 MiB binding', () => {
    const spec = buildUNetSpec();
    const plan = buildTensorAllocationPlan(
      spec,
      preflightTensorDims(spec, 1920, 1080),
      NEURAL_F16_TENSOR_STORAGE,
    );
    expect(plan.logicalTensorBytes).toBe(1_163_289_600);
    expect(plan.physicalTensorBytes).toBe(472_780_800);
    expect(plan.reuseSavingsBytes).toBe(690_508_800);
    expect(plan.peakLiveTensorBytes).toBe(311_040_000);
    expect(plan.slots).toHaveLength(8);
    expect(plan.slots.map(slot => slot.byteSize)).toEqual([
      37_324_800, 99_532_800, 99_532_800, 49_766_400,
      12_441_600, 24_883_200, 49_766_400, 99_532_800,
    ]);
    expect(Math.max(...plan.slots.map(slot => slot.byteSize)))
      .toBeLessThanOrEqual(128 * 1024 * 1024);
  });

  it('fails 1080p f32 but accepts 1080p f16 on a 128 MiB storage-binding device', () => {
    const spec = buildUNetSpec();
    const dims = preflightTensorDims(spec, 1920, 1080);
    const weights = buildRandomWeightsForSpec(spec);
    const device = {
      limits: {
        maxBufferSize: 256 * 1024 * 1024,
        maxStorageBufferBindingSize: 128 * 1024 * 1024,
        maxUniformBufferBindingSize: 64 * 1024,
        maxComputeWorkgroupsPerDimension: 65_535,
        maxTextureDimension2D: 8192,
        maxStorageBuffersPerShaderStage: 8,
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxComputeWorkgroupSizeY: 256,
      },
    } as unknown as GPUDevice;
    expect(neuralDeviceLimitFailures(device, spec, weights, dims))
      .toContainEqual(expect.stringContaining('maxStorageBufferBindingSize=134217728'));
    expect(neuralDeviceLimitFailures(device, spec, weights, dims, NEURAL_F16_TENSOR_STORAGE))
      .toEqual([]);
  });
});
