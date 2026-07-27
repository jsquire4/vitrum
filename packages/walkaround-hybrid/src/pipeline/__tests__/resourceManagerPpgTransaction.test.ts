import { beforeAll, describe, expect, it, vi } from 'vitest';
import {
  allocatePPGResources,
  computePPGResourceFootprint,
  createFrameResources,
  destroyFrameResources,
  uploadBufferPadded,
} from '../resourceManager.js';

type MockBuffer = { destroy: ReturnType<typeof vi.fn<[], void>> };

beforeAll(() => {
  Object.assign(globalThis, {
    GPUBufferUsage: {
      COPY_SRC: 1,
      COPY_DST: 2,
      STORAGE: 4,
      UNIFORM: 8,
    },
    GPUTextureUsage: {
      COPY_SRC: 1,
      COPY_DST: 2,
      TEXTURE_BINDING: 4,
      STORAGE_BINDING: 8,
      RENDER_ATTACHMENT: 16,
    },
  });
});

describe('allocatePPGResources transaction', () => {
  it('destroys every candidate when a later allocation fails', () => {
    const created: MockBuffer[] = [];
    const device = {
      createBuffer: vi.fn(() => {
        if (created.length === 3) throw new Error('injected allocation failure');
        const buffer: MockBuffer = { destroy: vi.fn() };
        created.push(buffer);
        return buffer;
      }),
    } as unknown as GPUDevice;

    expect(() =>
      allocatePPGResources(device, 16, 16, {
        maxSpatialCells: 4,
        maxDTreeNodesPerCell: 5,
      }),
    ).toThrow('injected allocation failure');

    expect(created).toHaveLength(3);
    for (const buffer of created) expect(buffer.destroy).toHaveBeenCalledOnce();
  });

  it('preserves the allocation error and reaches later candidates when an early destroy throws', () => {
    const created: MockBuffer[] = [];
    const allocationError = new Error('injected allocation failure');
    const device = {
      createBuffer: vi.fn(() => {
        if (created.length === 3) throw allocationError;
        const index = created.length;
        const buffer: MockBuffer = {
          destroy: vi.fn(() => {
            if (index === 0) throw new Error('hostile first destroy');
          }),
        };
        created.push(buffer);
        return buffer;
      }),
    } as unknown as GPUDevice;

    expect(() => allocatePPGResources(device, 16, 16, {
      maxSpatialCells: 4,
      maxDTreeNodesPerCell: 5,
    })).toThrow(allocationError);

    expect(created).toHaveLength(3);
    for (const buffer of created) expect(buffer.destroy).toHaveBeenCalledOnce();
  });

  it('publishes one query arena plus three auxiliary buffers without destroying candidates', () => {
    const created: MockBuffer[] = [];
    const device = {
      createBuffer: vi.fn(() => {
        const buffer: MockBuffer = { destroy: vi.fn() };
        created.push(buffer);
        return buffer;
      }),
    } as unknown as GPUDevice;

    const resources = allocatePPGResources(device, 16, 16, {
      maxSpatialCells: 4,
      maxDTreeNodesPerCell: 5,
    });

    expect(Object.values(resources)).toHaveLength(6);
    expect(created).toHaveLength(4);
    for (const buffer of created) expect(buffer.destroy).not.toHaveBeenCalled();
  });
});

describe('ReGIR padded storage-buffer preflight', () => {
  it.each([
    Number.NaN,
    Number.POSITIVE_INFINITY,
    -1,
    1.5,
    Number.MAX_SAFE_INTEGER + 1,
  ])('rejects invalid padding %s before allocation', (padding) => {
    const createBuffer = vi.fn();
    const device = { createBuffer } as unknown as GPUDevice;
    expect(() =>
      uploadBufferPadded(device, new ArrayBuffer(16), padding, 0x80)
    ).toThrow(/non-negative safe integer/);
    expect(createBuffer).not.toHaveBeenCalled();
  });

  it.each([
    ['maxBufferSize', 31],
    ['maxStorageBufferBindingSize', 31],
  ] as const)('rejects an insufficient device %s before allocation', (limit, value) => {
    const createBuffer = vi.fn();
    const device = {
      limits: { [limit]: value },
      createBuffer,
    } as unknown as GPUDevice;
    expect(() =>
      uploadBufferPadded(device, new ArrayBuffer(16), 16, 0x80)
    ).toThrow(new RegExp(limit));
    expect(createBuffer).not.toHaveBeenCalled();
  });

  it('rejects a combined buffer beyond WGSL u32 element addressability before allocation', () => {
    const createBuffer = vi.fn();
    const device = { createBuffer } as unknown as GPUDevice;
    expect(() =>
      uploadBufferPadded(
        device,
        new ArrayBuffer(4),
        0x1_0000_0000 * 4,
        0x80,
      )
    ).toThrow(/u32 element-index domain/);
    expect(createBuffer).not.toHaveBeenCalled();
  });
});

describe('PPG resource arithmetic and device limits', () => {
  it('computes the exact packed-arena footprint from binary sTree and quadtree capacities', () => {
    expect(computePPGResourceFootprint(4, 5)).toEqual({
      maxSpatialCells: 4,
      maxDTreeNodesPerCell: 5,
      maxSTreeNodes: 7,
      sTreeBytes: 464,
      dTreeBytes: 704,
      dTreeOffsetsBytes: 16,
      queryArenaBytes: 1792,
      fluxAtomicsBytes: 80,
      cellSampleCountsBytes: 16,
      updateUboBytes: 16,
      totalBytes: 1904,
    });
    expect(computePPGResourceFootprint().totalBytes).toBe(12_726_544);
  });

  it.each([
    [0, 5],
    [16_385, 5],
    [4, 0],
    [4, 342],
    [4.5, 5],
  ])('rejects unsupported allocation caps (%s, %s)', (cells, nodes) => {
    expect(() => computePPGResourceFootprint(cells, nodes)).toThrow(RangeError);
  });

  it.each([
    [{ maxStorageBufferBindingSize: 463 }, /maxStorageBufferBindingSize/],
    [{ maxBufferSize: 463 }, /maxBufferSize/],
    [{ maxComputeInvocationsPerWorkgroup: 32 }, /maxComputeInvocationsPerWorkgroup/],
    [{ maxComputeWorkgroupSizeX: 32 }, /maxComputeWorkgroupSizeX/],
    [{ maxComputeWorkgroupsPerDimension: 3 }, /maxComputeWorkgroupsPerDimension/],
  ])('rejects an insufficient device limit before allocating: %o', (limits, error) => {
    const createBuffer = vi.fn();
    const device = { limits, createBuffer } as unknown as GPUDevice;
    expect(() => allocatePPGResources(device, 32, 32, {
      maxSpatialCells: 4,
      maxDTreeNodesPerCell: 5,
    })).toThrow(error);
    expect(createBuffer).not.toHaveBeenCalled();
  });

  it('allocates descriptors with exactly the reported sizes', () => {
    const descriptors: GPUBufferDescriptor[] = [];
    const device = {
      createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
        descriptors.push(descriptor);
        return { destroy: vi.fn() } as unknown as GPUBuffer;
      }),
    } as unknown as GPUDevice;
    allocatePPGResources(device, 16, 16, {
      maxSpatialCells: 4,
      maxDTreeNodesPerCell: 5,
    });
    expect(descriptors.map((descriptor) => descriptor.size)).toEqual([
      1792, 80, 16, 16,
    ]);
  });
});
describe('createFrameResources transaction', () => {
  it('rolls back every earlier buffer and texture when a later factory fails', () => {
    const created: Array<{ destroy: ReturnType<typeof vi.fn> }> = [];
    const allocate = (descriptor: { label?: string }) => {
      if (descriptor.label === 'reservoir-gi-spatial') {
        throw new Error('injected frame allocation failure');
      }
      const resource = { destroy: vi.fn() };
      created.push(resource);
      return resource;
    };
    const device = {
      createTexture: vi.fn(allocate),
      createBuffer: vi.fn(allocate),
      createSampler: vi.fn(() => ({})),
      queue: {
        writeTexture: vi.fn(),
        writeBuffer: vi.fn(),
      },
    } as unknown as GPUDevice;

    expect(() =>
      createFrameResources(device, 16, 16, {
        svgfEnabled: false,
        welfordPingPong: false,
      }),
    ).toThrow('injected frame allocation failure');

    expect(created.length).toBeGreaterThan(20);
    for (const resource of created) {
      expect(resource.destroy).toHaveBeenCalledOnce();
    }
  });

  it('destroys every frame and PPG resource despite first and middle destroy failures', () => {
    type Resource = {
      label: string;
      destroy: ReturnType<typeof vi.fn<[], void>>;
    };
    const created: Resource[] = [];
    const allocate = (descriptor: { label?: string }) => {
      const resource: Resource = {
        label: descriptor.label ?? `resource-${created.length}`,
        destroy: vi.fn(),
      };
      created.push(resource);
      return resource;
    };
    const device = {
      createTexture: vi.fn(allocate),
      createBuffer: vi.fn(allocate),
      createSampler: vi.fn(() => ({})),
      queue: {
        writeTexture: vi.fn(),
        writeBuffer: vi.fn(),
      },
    } as unknown as GPUDevice;
    const resources = createFrameResources(device, 16, 16, {
      svgfEnabled: false,
      welfordPingPong: false,
    });
    resources.ppg = allocatePPGResources(device, 16, 16, {
      maxSpatialCells: 4,
      maxDTreeNodesPerCell: 5,
    });
    const firstFrame = created[0]!;
    const firstPpg = created.find(resource => resource.label === 'ppg-query-arena')!;
    firstFrame.destroy.mockImplementation(() => { throw new Error('first frame destroy'); });
    firstPpg.destroy.mockImplementation(() => { throw new Error('first PPG destroy'); });

    expect(() => destroyFrameResources(resources)).not.toThrow();
    for (const resource of created) expect(resource.destroy).toHaveBeenCalledOnce();
  });
});
