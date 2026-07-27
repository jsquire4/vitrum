import { describe, expect, it, vi } from 'vitest';

import {
  DEFAULT_NRC_CONFIG,
  NrcSubsystem,
  type NrcConfig,
} from '../nrcSubsystem.js';
import {
  computeNrcResourceFootprint,
  preflightNrcResources,
  validateNrcAabb,
} from '../nrcPreflight.js';

const CONFIG: NrcConfig = {
  ...DEFAULT_NRC_CONFIG,
  levels: 1,
  featuresPerEntry: 1,
  tableSize: 16,
  nMin: 2,
  growth: 2,
  oneBlobBins: 1,
  width: 16,
  hidden: 0,
  recordCap: 4,
  tileB: 4,
  useF16: false,
};

function deviceWith(
  limits: Partial<Record<keyof GPUSupportedLimits, number>> = {},
  features: readonly GPUFeatureName[] = [],
): GPUDevice {
  return {
    limits,
    features: new Set(features),
  } as unknown as GPUDevice;
}

const MIN = [-1, -2, -3] as const;
const MAX = [1, 2, 3] as const;

describe('NRC resource preflight', () => {
  it('accounts for a complete second parameter, moment, and table generation', () => {
    const footprint = computeNrcResourceFootprint(CONFIG);

    expect(footprint.storageBindings).toMatchObject({
      candidateWeights: footprint.storageBindings.weights,
      candidateBiases: footprint.storageBindings.biases,
      candidateWeightMaster: footprint.storageBindings.weightMaster,
      candidateBiasMaster: footprint.storageBindings.biasMaster,
      candidateWeightMoment: footprint.storageBindings.weightMoment,
      candidateBiasMoment: footprint.storageBindings.biasMoment,
      candidateTables: footprint.storageBindings.tables,
      candidateTableMoment: footprint.storageBindings.tableMoment,
    });
    expect(footprint.candidateTrainableBytes).toBe(footprint.liveTrainableBytes);
    expect(footprint.peakTrainableBytes).toBe(
      footprint.liveTrainableBytes + footprint.candidateTrainableBytes,
    );
    expect(footprint.readbackBytes).toBe(footprint.recordBytes + 20);
    expect({
      live: footprint.liveTrainableBytes,
      candidate: footprint.candidateTrainableBytes,
      peak: footprint.peakTrainableBytes,
    }).toEqual({ live: 1_024, candidate: 1_024, peak: 2_048 });

    const production = computeNrcResourceFootprint(DEFAULT_NRC_CONFIG);
    expect({
      live: production.liveTrainableBytes,
      candidate: production.candidateTrainableBytes,
      peak: production.peakTrainableBytes,
    }).toEqual({
      live: 1_188_912,
      candidate: 1_188_912,
      peak: 2_377_824,
    });
  });

  it('pins all persistent buffers plus the single-readback resident peak', () => {
    const small = computeNrcResourceFootprint(CONFIG);
    expect({
      persistentCount: small.persistentBufferCount,
      persistentBytes: small.persistentBufferBytes,
      peakCount: small.peakResidentBufferCount,
      peakBytes: small.peakResidentBufferBytes,
      readbackBytes: small.readbackBytes,
    }).toEqual({
      persistentCount: 49,
      persistentBytes: 8_112,
      peakCount: 50,
      peakBytes: 8_388,
      readbackBytes: 276,
    });
    expect(Object.values(small.persistentAllocations)
      .reduce((sum, entry) => sum + entry.count, 0)).toBe(49);
    expect(Object.values(small.persistentAllocations)
      .reduce((sum, entry) => sum + entry.totalBytes, 0)).toBe(8_112);

    const production = computeNrcResourceFootprint(DEFAULT_NRC_CONFIG);
    expect({
      persistentCount: production.persistentBufferCount,
      persistentBytes: production.persistentBufferBytes,
      peakCount: production.peakResidentBufferCount,
      peakBytes: production.peakResidentBufferBytes,
      readbackBytes: production.readbackBytes,
    }).toEqual({
      persistentCount: 49,
      persistentBytes: 23_377_560,
      peakCount: 50,
      peakBytes: 24_114_860,
      readbackBytes: 737_300,
    });

    const productionF16 = computeNrcResourceFootprint({
      ...DEFAULT_NRC_CONFIG,
      useF16: true,
    });
    expect({
      persistentCount: productionF16.persistentBufferCount,
      persistentBytes: productionF16.persistentBufferBytes,
      peakCount: productionF16.peakResidentBufferCount,
      peakBytes: productionF16.peakResidentBufferBytes,
      readbackBytes: productionF16.readbackBytes,
    }).toEqual({
      persistentCount: 51,
      persistentBytes: 14_568_880,
      peakCount: 52,
      peakBytes: 15_306_180,
      readbackBytes: 737_300,
    });
  });

  it('recomputes aggregate bytes when record capacity or table sizing changes', () => {
    const records8 = computeNrcResourceFootprint({ ...CONFIG, recordCap: 8 });
    expect({
      count: records8.persistentBufferCount,
      bytes: records8.persistentBufferBytes,
      peakCount: records8.peakResidentBufferCount,
      peakBytes: records8.peakResidentBufferBytes,
      readbackBytes: records8.readbackBytes,
    }).toEqual({ count: 49, bytes: 9_968, peakCount: 50, peakBytes: 10_500, readbackBytes: 532 });

    const table32 = computeNrcResourceFootprint({ ...CONFIG, tableSize: 32 });
    expect({
      count: table32.persistentBufferCount,
      bytes: table32.persistentBufferBytes,
      peakCount: table32.peakResidentBufferCount,
      peakBytes: table32.peakResidentBufferBytes,
      readbackBytes: table32.readbackBytes,
    }).toEqual({ count: 49, bytes: 8_624, peakCount: 50, peakBytes: 8_900, readbackBytes: 276 });
  });

  it('enforces the host aggregate resident budget at the exact byte boundary', () => {
    const peak = computeNrcResourceFootprint(CONFIG).peakResidentBufferBytes;
    expect(() => preflightNrcResources(
      deviceWith(),
      { ...CONFIG, maxNrcResidentBytes: peak + 1 },
      MIN,
      MAX,
    )).not.toThrow();
    expect(() => preflightNrcResources(
      deviceWith(),
      { ...CONFIG, maxNrcResidentBytes: peak },
      MIN,
      MAX,
    )).not.toThrow();
    expect(() => preflightNrcResources(
      deviceWith(),
      { ...CONFIG, maxNrcResidentBytes: peak - 1 },
      MIN,
      MAX,
    )).toThrow(/host maxNrcResidentBytes budget/);
    for (const invalid of [0, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
      expect(() => computeNrcResourceFootprint({
        ...CONFIG,
        maxNrcResidentBytes: invalid,
      })).toThrow(/positive safe integer/);
    }
  });

  it('accepts the exact per-buffer adapter limits for both preallocated generations', () => {
    const footprint = computeNrcResourceFootprint(CONFIG);
    const maxStorage = Math.max(...Object.values(footprint.storageBindings));
    const maxUniform = Math.max(...Object.values(footprint.uniformBindings));
    const maxBuffer = Math.max(maxStorage, maxUniform, footprint.readbackBytes);
    const exactDevice = deviceWith({
      maxBufferSize: maxBuffer,
      maxStorageBufferBindingSize: maxStorage,
      maxUniformBufferBindingSize: maxUniform,
      maxComputeInvocationsPerWorkgroup: 64,
      maxComputeWorkgroupSizeX: 64,
      maxComputeWorkgroupStorageSize: footprint.workgroupStorageBytes,
      maxComputeWorkgroupsPerDimension: footprint.maxDispatchWorkgroups,
    });

    expect(preflightNrcResources(exactDevice, CONFIG, MIN, MAX)).toEqual(footprint);
  });

  it('rejects unsafe integer products before allocating a GPU resource', () => {
    expect(() => computeNrcResourceFootprint({
      ...CONFIG,
      recordCap: Number.MAX_SAFE_INTEGER,
    })).toThrow(/checked integer range/);
    expect(() => computeNrcResourceFootprint({
      ...CONFIG,
      levels: 64,
      growth: Number.MAX_VALUE,
      width: 128,
    })).toThrow(/resolution is outside u32 range/);
  });

  it('requires finite, strictly ordered scene bounds on every axis', () => {
    expect(() => validateNrcAabb([Number.NaN, 0, 0], MAX)).toThrow(/axis 0/);
    expect(() => validateNrcAabb([0, 0, 0], [0, 1, 1])).toThrow(/axis 0/);
    expect(() => validateNrcAabb([0, 2, 0], [1, 1, 1])).toThrow(/axis 1/);
    expect(() => validateNrcAabb(MIN, MAX)).not.toThrow();
  });

  it('checks every live and candidate binding against adapter buffer limits', () => {
    const footprint = computeNrcResourceFootprint(CONFIG);
    const largestBinding = Math.max(...Object.values(footprint.storageBindings));
    expect(() => preflightNrcResources(
      deviceWith({ maxStorageBufferBindingSize: largestBinding - 1 }),
      CONFIG,
      MIN,
      MAX,
    )).toThrow(/storage binding size/);
    expect(() => preflightNrcResources(
      deviceWith({ maxBufferSize: footprint.readbackBytes - 1 }),
      CONFIG,
      MIN,
      MAX,
    )).toThrow(/buffer size/);
  });

  it('checks dispatch, workgroup width, and workgroup storage limits', () => {
    const dispatchConfig = { ...CONFIG, recordCap: 256 };
    const footprint = computeNrcResourceFootprint(dispatchConfig);
    expect(() => preflightNrcResources(
      deviceWith({ maxComputeWorkgroupsPerDimension: footprint.maxDispatchWorkgroups - 1 }),
      dispatchConfig,
      MIN,
      MAX,
    )).toThrow(/dispatch workgroup count/);
    expect(() => preflightNrcResources(
      deviceWith({ maxComputeInvocationsPerWorkgroup: CONFIG.width - 1 }),
      CONFIG,
      MIN,
      MAX,
    )).toThrow(/MLP workgroup width/);
    expect(() => preflightNrcResources(
      deviceWith({ maxComputeWorkgroupStorageSize: footprint.workgroupStorageBytes - 1 }),
      CONFIG,
      MIN,
      MAX,
    )).toThrow(/workgroup storage/);
  });

  it('accepts exactly eight GI-RIS storage bindings and rejects seven', () => {
    expect(() => preflightNrcResources(
      deviceWith({
        maxBindGroups: 4,
        maxStorageBuffersPerShaderStage: 8,
        maxSampledTexturesPerShaderStage: 16,
      }), CONFIG, MIN, MAX,
    )).not.toThrow();
    expect(() => preflightNrcResources(
      deviceWith({
        maxBindGroups: 4,
        maxStorageBuffersPerShaderStage: 7,
        maxSampledTexturesPerShaderStage: 16,
      }), CONFIG, MIN, MAX,
    )).toThrow(/gi-ris storage bindings requires 8, adapter reports 7/);
  });

  it('derives the full-capacity dL/dX finalize dispatch and enforces its exact boundary', () => {
    const footprint = computeNrcResourceFootprint(DEFAULT_NRC_CONFIG);
    // 4096 records * 39 encoded lanes / 64 threads. The old incomplete
    // formula reported the 65,536-scalar table pass (1024 workgroups) as the
    // maximum and missed this larger training dispatch.
    expect(footprint.inW).toBe(39);
    expect(footprint.maxDispatchWorkgroups).toBe(2_496);

    expect(() => preflightNrcResources(
      deviceWith({ maxComputeWorkgroupsPerDimension: 2_496 }),
      DEFAULT_NRC_CONFIG,
      MIN,
      MAX,
    )).not.toThrow();
    expect(() => preflightNrcResources(
      deviceWith({ maxComputeWorkgroupsPerDimension: 2_497 }),
      DEFAULT_NRC_CONFIG,
      MIN,
      MAX,
    )).not.toThrow();
    expect(() => preflightNrcResources(
      deviceWith({ maxComputeWorkgroupsPerDimension: 2_495 }),
      DEFAULT_NRC_CONFIG,
      MIN,
      MAX,
    )).toThrow(/dispatch workgroup count requires 2496, adapter reports 2495/);
  });

  it('rejects an insufficient dispatch limit before allocating or encoding anything', async () => {
    const createBuffer = vi.fn();
    const createCommandEncoder = vi.fn();
    const writeBuffer = vi.fn();
    const device = {
      limits: { maxComputeWorkgroupsPerDimension: 2_495 },
      features: new Set<GPUFeatureName>(),
      createBuffer,
      createCommandEncoder,
      queue: { writeBuffer },
    } as unknown as GPUDevice;
    const subsystem = new NrcSubsystem(device, {});

    await expect(subsystem.initialize(MIN, MAX)).rejects.toThrow(
      /dispatch workgroup count requires 2496, adapter reports 2495/,
    );
    expect(createBuffer).not.toHaveBeenCalled();
    expect(createCommandEncoder).not.toHaveBeenCalled();
    expect(writeBuffer).not.toHaveBeenCalled();
    expect(subsystem.lifecycleState).toBe('new');
  });

  it('requires shader-f16 before accepting the mixed-precision footprint', () => {
    const f16 = { ...CONFIG, useF16: true };
    expect(() => preflightNrcResources(deviceWith(), f16, MIN, MAX)).toThrow(
      /shader-f16/,
    );
    expect(() => preflightNrcResources(
      deviceWith({}, ['shader-f16']),
      f16,
      MIN,
      MAX,
    )).not.toThrow();
  });
});
