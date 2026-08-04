import { describe, expect, it, vi } from 'vitest';
import { padTriangleIndicesToVec4 } from '../probeUpdateMaterials.js';
import {
  refitProbeTlasBuffersInPlace,
  rebuildProbeBvhFromRestir,
  rebuildProbeBvhFromScene,
  type ProbeUpdateBvhGpuBuffers,
} from '../probeUpdateBvhBuffers.js';

interface TrackedBuffer {
  readonly gpu: GPUBuffer;
  readonly destroy: ReturnType<typeof vi.fn>;
}

const bvhKeys = [
  'bvhBuf',
  'posBuf',
  'idxBuf',
  'normBuf',
  'matIdBuf',
  'tlasNodesBuf',
  'tlasInstIdxBuf',
  'tlasBlasRootsBuf',
  'tlasW2lBuf',
  'tlasL2wBuf',
  'opticalTriangleIdentityBuf',
  'opticalInstanceBoundaryIdBasePlusOneBuf',
] as const satisfies readonly (keyof ProbeUpdateBvhGpuBuffers)[];

function trackedBuffer(size = 256): TrackedBuffer {
  const destroy = vi.fn();
  return {
    gpu: { size, destroy } as unknown as GPUBuffer,
    destroy,
  };
}

function trackedCohort(size = 256): {
  readonly gpu: ProbeUpdateBvhGpuBuffers;
  readonly records: TrackedBuffer[];
} {
  const records = bvhKeys.map(() => trackedBuffer(size));
  const gpu = {} as ProbeUpdateBvhGpuBuffers;
  bvhKeys.forEach((key, index) => { gpu[key] = records[index]!.gpu; });
  return { gpu, records };
}

interface MemoryBuffer {
  readonly gpu: GPUBuffer;
  readonly bytes: Uint8Array;
  readonly destroy: ReturnType<typeof vi.fn>;
}

function memoryBuffer(size = 256): MemoryBuffer {
  const bytes = new Uint8Array(size);
  const destroy = vi.fn();
  return {
    gpu: { size, destroy } as unknown as GPUBuffer,
    bytes,
    destroy,
  };
}

function memoryRefitHarness(size = 256): {
  readonly device: GPUDevice;
  readonly gpu: ProbeUpdateBvhGpuBuffers;
  readonly records: Readonly<Record<(typeof bvhKeys)[number], MemoryBuffer>>;
  readonly recordFor: (buffer: GPUBuffer) => MemoryBuffer;
} {
  const byGpu = new Map<GPUBuffer, MemoryBuffer>();
  const records = {} as Record<(typeof bvhKeys)[number], MemoryBuffer>;
  const gpu = {} as ProbeUpdateBvhGpuBuffers;
  for (const key of bvhKeys) {
    const record = memoryBuffer(size);
    records[key] = record;
    gpu[key] = record.gpu;
    byGpu.set(record.gpu, record);
  }
  const pendingCopies: Array<{
    source: GPUBuffer;
    sourceOffset: number;
    destination: GPUBuffer;
    destinationOffset: number;
    byteLength: number;
  }> = [];
  const device = {
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      const record = memoryBuffer(descriptor.size);
      byGpu.set(record.gpu, record);
      return record.gpu;
    }),
    queue: {
      writeBuffer: vi.fn((
        destination: GPUBuffer,
        destinationOffset: number,
        source: ArrayBuffer,
        sourceOffset = 0,
        byteLength = source.byteLength - sourceOffset,
      ) => {
        byGpu.get(destination)!.bytes.set(
          new Uint8Array(source, sourceOffset, byteLength),
          destinationOffset,
        );
      }),
      submit: vi.fn(() => {
        for (const copy of pendingCopies) {
          byGpu.get(copy.destination)!.bytes.set(
            byGpu.get(copy.source)!.bytes.subarray(
              copy.sourceOffset,
              copy.sourceOffset + copy.byteLength,
            ),
            copy.destinationOffset,
          );
        }
      }),
      onSubmittedWorkDone: vi.fn(() => Promise.resolve()),
    },
    createCommandEncoder: vi.fn(() => ({
      copyBufferToBuffer: vi.fn((
        source: GPUBuffer,
        sourceOffset: number,
        destination: GPUBuffer,
        destinationOffset: number,
        byteLength: number,
      ) => {
        pendingCopies.push({
          source,
          sourceOffset,
          destination,
          destinationOffset,
          byteLength,
        });
      }),
      finish: vi.fn(() => ({} as GPUCommandBuffer)),
    })),
  } as unknown as GPUDevice;
  return {
    device,
    gpu,
    records,
    recordFor: (buffer) => byGpu.get(buffer)!,
  };
}

function restirSnapshot(): never {
  return {
    bvhNodes: new ArrayBuffer(32),
    positions: new ArrayBuffer(64),
    bvhIndex: new ArrayBuffer(32),
    normals: new ArrayBuffer(64),
    triMaterialIds: new ArrayBuffer(16),
    opticalTriangleIdentity: new ArrayBuffer(16),
    opticalInstanceBoundaryIdBasePlusOne: new ArrayBuffer(16),
  } as never;
}

function failureDevice(options: {
  createAt?: number;
  writeAt?: number;
  copyAt?: number;
  encoder?: boolean;
  finish?: boolean;
  submit?: boolean;
  completion?: boolean;
} = {}): {
  readonly device: GPUDevice;
  readonly created: TrackedBuffer[];
} {
  const created: TrackedBuffer[] = [];
  let createCount = 0;
  let writeCount = 0;
  let copyCount = 0;
  let failureAvailable = true;
  const device = {
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      createCount++;
      if (failureAvailable && createCount === options.createAt) {
        failureAvailable = false;
        throw new Error(`injected create ${createCount}`);
      }
      const record = trackedBuffer(descriptor.size);
      created.push(record);
      return record.gpu;
    }),
    queue: {
      writeBuffer: vi.fn(() => {
        writeCount++;
        if (failureAvailable && writeCount === options.writeAt) {
          failureAvailable = false;
          throw new Error(`injected write ${writeCount}`);
        }
      }),
      submit: vi.fn(() => {
        if (failureAvailable && options.submit) {
          failureAvailable = false;
          throw new Error('injected submit');
        }
      }),
      onSubmittedWorkDone: vi.fn(() => {
        if (failureAvailable && options.completion) {
          failureAvailable = false;
          throw new Error('injected completion tracker');
        }
        return Promise.resolve();
      }),
    },
    createCommandEncoder: vi.fn(() => {
      if (failureAvailable && options.encoder) {
        failureAvailable = false;
        throw new Error('injected encoder');
      }
      return {
        copyBufferToBuffer: vi.fn(() => {
          copyCount++;
          if (failureAvailable && copyCount === options.copyAt) {
            failureAvailable = false;
            throw new Error(`injected copy ${copyCount}`);
          }
        }),
        finish: vi.fn(() => {
          if (failureAvailable && options.finish) {
            failureAvailable = false;
            throw new Error('injected finish');
          }
          return {} as GPUCommandBuffer;
        }),
      } as unknown as GPUCommandEncoder;
    }),
  } as unknown as GPUDevice;
  return { device, created };
}

describe('probeUpdateBvhBuffers', () => {
  it('padTriangleIndicesToVec4 pads stride-3 indices', () => {
    const out = padTriangleIndicesToVec4(new Uint32Array([0, 1, 2, 3, 4, 5]));
    expect(Array.from(out)).toEqual([0, 1, 2, 0, 3, 4, 5, 0]);
  });

  it('rebuildProbeBvhFromScene replaces BVH buffers', () => {
    const destroyed: GPUBuffer[] = [];
    const device = {
      createBuffer: vi.fn(() => ({
        destroy: () => {
          /* tracked via destroyed length */
        },
      })),
      queue: { writeBuffer: vi.fn() },
    } as unknown as GPUDevice;
    const old = {
      destroy: () => destroyed.push(old),
    } as unknown as GPUBuffer;
    const g = {
      bvhBuf: old,
      posBuf: old,
      idxBuf: old,
      normBuf: old,
      matIdBuf: old,
      tlasNodesBuf: old,
      tlasInstIdxBuf: old,
      tlasBlasRootsBuf: old,
      tlasW2lBuf: old,
      tlasL2wBuf: old,
      opticalTriangleIdentityBuf: old,
      opticalInstanceBoundaryIdBasePlusOneBuf: old,
    };
    const buffers = {
      bvhNodes: new Float32Array(8),
      positions: new Float32Array(8),
      indices: new Uint32Array([0, 1, 2]),
      normals: new Float32Array(8),
      triMaterialId: new Uint32Array(4),
      opticalTriangleIdentity: new Uint32Array(4),
      opticalInstanceBoundaryIdBasePlusOne: new Uint32Array([1]),
    } as never;
    rebuildProbeBvhFromScene(device, g, buffers);
    expect(device.createBuffer).toHaveBeenCalled();
    expect(g.bvhBuf).not.toBe(old);
  });

  it('uploads only the visible bytes of offset-backed scene views', () => {
    const writes = vi.fn();
    const device = {
      createBuffer: vi.fn((desc: GPUBufferDescriptor) => ({
        size: desc.size,
        destroy: vi.fn(),
      } as unknown as GPUBuffer)),
      queue: { writeBuffer: writes },
    } as unknown as GPUDevice;
    const prior = trackedCohort();
    const backing = new ArrayBuffer(64);
    const nodes = new Float32Array(backing, 16, 4);
    nodes.set([11, 12, 13, 14]);
    const buffers = {
      bvhNodes: nodes,
      positions: new Float32Array(4),
      indices: new Uint32Array([0, 1, 2]),
      normals: new Float32Array(4),
      triMaterialId: new Uint32Array([0]),
      opticalTriangleIdentity: new Uint32Array([0, 1]),
      opticalInstanceBoundaryIdBasePlusOne: new Uint32Array([1]),
    } as never;

    rebuildProbeBvhFromScene(device, prior.gpu, buffers);

    expect(writes.mock.calls[0]?.[2]).toBe(backing);
    expect(writes.mock.calls[0]?.[3]).toBe(16);
    expect(writes.mock.calls[0]?.[4]).toBe(16);
  });

  it('copies the exact visible span of SharedArrayBuffer-backed scene views', () => {
    if (typeof SharedArrayBuffer !== 'function') return;
    const writes = vi.fn();
    const device = {
      createBuffer: vi.fn((desc: GPUBufferDescriptor) => ({
        size: desc.size,
        destroy: vi.fn(),
      } as unknown as GPUBuffer)),
      queue: { writeBuffer: writes },
    } as unknown as GPUDevice;
    const prior = trackedCohort();
    const shared = new SharedArrayBuffer(32);
    const nodes = new Float32Array(shared, 8, 4);
    nodes.set([21, 22, 23, 24]);
    const buffers = {
      bvhNodes: nodes,
      positions: new Float32Array(4),
      indices: new Uint32Array([0, 1, 2]),
      normals: new Float32Array(4),
      triMaterialId: new Uint32Array([0]),
      opticalTriangleIdentity: new Uint32Array([0, 1]),
      opticalInstanceBoundaryIdBasePlusOne: new Uint32Array([1]),
    } as never;

    rebuildProbeBvhFromScene(device, prior.gpu, buffers);

    const source = writes.mock.calls[0]?.[2];
    expect(source).toBeInstanceOf(ArrayBuffer);
    expect(source).not.toBe(shared);
    expect(writes.mock.calls[0]?.[3]).toBe(0);
    expect(writes.mock.calls[0]?.[4]).toBe(16);
    expect(Array.from(new Float32Array(source as ArrayBuffer, 0, 4)))
      .toEqual([21, 22, 23, 24]);
  });

  it('rebuildProbeBvhFromRestir uses a BVHNode-sized TLAS placeholder when TLAS is absent', () => {
    const created: GPUBufferDescriptor[] = [];
    const device = {
      createBuffer: vi.fn((desc: GPUBufferDescriptor) => {
        created.push(desc);
        return { size: desc.size, destroy: vi.fn() } as unknown as GPUBuffer;
      }),
      queue: { writeBuffer: vi.fn() },
    } as unknown as GPUDevice;
    const old = {
      destroy: vi.fn(),
    } as unknown as GPUBuffer;
    const g = {
      bvhBuf: old,
      posBuf: old,
      idxBuf: old,
      normBuf: old,
      matIdBuf: old,
      tlasNodesBuf: old,
      tlasInstIdxBuf: old,
      tlasBlasRootsBuf: old,
      tlasW2lBuf: old,
      tlasL2wBuf: old,
      opticalTriangleIdentityBuf: old,
      opticalInstanceBoundaryIdBasePlusOneBuf: old,
    };
    const snap = {
      bvhNodes: new ArrayBuffer(32),
      positions: new ArrayBuffer(64),
      bvhIndex: new ArrayBuffer(32),
      normals: new ArrayBuffer(64),
      triMaterialIds: new ArrayBuffer(16),
      opticalTriangleIdentity: new ArrayBuffer(16),
      opticalInstanceBoundaryIdBasePlusOne: new ArrayBuffer(16),
    } as never;

    rebuildProbeBvhFromRestir(device, g, snap);

    expect(created[5]?.size).toBe(32);
    expect(created[6]?.size).toBe(16);
    expect(created[7]?.size).toBe(16);
    expect(created[8]?.size).toBe(16);
    expect(created[9]?.size).toBe(16);
  });

  it.each([1, 6, 12])(
    'keeps the twelve-buffer BVH cohort intact when candidate create #%i fails, then retries',
    (createAt) => {
      const prior = trackedCohort();
      const previous = bvhKeys.map((key) => prior.gpu[key]);
      const stub = failureDevice({ createAt });

      expect(() => rebuildProbeBvhFromRestir(stub.device, prior.gpu, restirSnapshot()))
        .toThrow(`injected create ${createAt}`);
      expect(bvhKeys.map((key) => prior.gpu[key])).toEqual(previous);
      for (const record of prior.records) expect(record.destroy).not.toHaveBeenCalled();
      for (const record of stub.created) expect(record.destroy).toHaveBeenCalledTimes(1);

      rebuildProbeBvhFromRestir(stub.device, prior.gpu, restirSnapshot());
      for (const key of bvhKeys) expect(prior.gpu[key]).not.toBe(previous[bvhKeys.indexOf(key)]);
      for (const record of prior.records) expect(record.destroy).toHaveBeenCalledTimes(1);
    },
  );

  it('cleans every written BVH candidate and preserves live identities on a late upload failure', () => {
    const prior = trackedCohort();
    const previous = bvhKeys.map((key) => prior.gpu[key]);
    const stub = failureDevice({ writeAt: 9 });

    expect(() => rebuildProbeBvhFromRestir(stub.device, prior.gpu, restirSnapshot()))
      .toThrow('injected write 9');
    expect(bvhKeys.map((key) => prior.gpu[key])).toEqual(previous);
    expect(stub.created).toHaveLength(9);
    for (const record of stub.created) expect(record.destroy).toHaveBeenCalledTimes(1);
    for (const record of prior.records) expect(record.destroy).not.toHaveBeenCalled();

    rebuildProbeBvhFromRestir(stub.device, prior.gpu, restirSnapshot());
    for (const record of prior.records) expect(record.destroy).toHaveBeenCalledTimes(1);
  });

  it('keeps identities stable and refreshes all five TLAS streams in place', async () => {
    const harness = memoryRefitHarness();
    const previous = {
      nodes: harness.gpu.tlasNodesBuf,
      instanceIndices: harness.gpu.tlasInstIdxBuf,
      blasRoots: harness.gpu.tlasBlasRootsBuf,
      w2l: harness.gpu.tlasW2lBuf,
      l2w: harness.gpu.tlasL2wBuf,
    };
    const payloads = {
      nodes: Uint8Array.from([1, 2, 3, 4]).buffer,
      instanceIndices: Uint8Array.from([5, 6, 7, 8]).buffer,
      blasRoots: Uint8Array.from([9, 10, 11, 12]).buffer,
      worldToLocal: Uint8Array.from([13, 14, 15, 16]).buffer,
      localToWorld: Uint8Array.from([17, 18, 19, 20]).buffer,
    };

    refitProbeTlasBuffersInPlace(
      harness.device,
      harness.gpu,
      payloads,
    );

    expect(harness.gpu.tlasNodesBuf).toBe(previous.nodes);
    expect(harness.gpu.tlasInstIdxBuf).toBe(previous.instanceIndices);
    expect(harness.gpu.tlasBlasRootsBuf).toBe(previous.blasRoots);
    expect(harness.gpu.tlasW2lBuf).toBe(previous.w2l);
    expect(harness.gpu.tlasL2wBuf).toBe(previous.l2w);
    expect(Array.from(harness.records.tlasNodesBuf.bytes.subarray(0, 4)))
      .toEqual([1, 2, 3, 4]);
    expect(Array.from(harness.records.tlasInstIdxBuf.bytes.subarray(0, 4)))
      .toEqual([5, 6, 7, 8]);
    expect(Array.from(harness.records.tlasBlasRootsBuf.bytes.subarray(0, 4)))
      .toEqual([9, 10, 11, 12]);
    expect(Array.from(harness.records.tlasW2lBuf.bytes.subarray(0, 4)))
      .toEqual([13, 14, 15, 16]);
    expect(Array.from(harness.records.tlasL2wBuf.bytes.subarray(0, 4)))
      .toEqual([17, 18, 19, 20]);
    await vi.waitFor(() => {
      expect(harness.device.createBuffer).toHaveBeenCalledTimes(5);
    });
  });

  it('publishes capacity-growing replacements containing all five TLAS streams', () => {
    const harness = memoryRefitHarness(2);
    const previous = {
      nodes: harness.gpu.tlasNodesBuf,
      instanceIndices: harness.gpu.tlasInstIdxBuf,
      blasRoots: harness.gpu.tlasBlasRootsBuf,
      w2l: harness.gpu.tlasW2lBuf,
      l2w: harness.gpu.tlasL2wBuf,
    };
    const payloads = {
      nodes: Uint8Array.from([31, 32, 33, 34]).buffer,
      instanceIndices: Uint8Array.from([35, 36, 37, 38]).buffer,
      blasRoots: Uint8Array.from([39, 40, 41, 42]).buffer,
      worldToLocal: Uint8Array.from([43, 44, 45, 46]).buffer,
      localToWorld: Uint8Array.from([47, 48, 49, 50]).buffer,
    };

    refitProbeTlasBuffersInPlace(harness.device, harness.gpu, payloads);

    expect(harness.gpu.tlasNodesBuf).not.toBe(previous.nodes);
    expect(harness.gpu.tlasInstIdxBuf).not.toBe(previous.instanceIndices);
    expect(harness.gpu.tlasBlasRootsBuf).not.toBe(previous.blasRoots);
    expect(harness.gpu.tlasW2lBuf).not.toBe(previous.w2l);
    expect(harness.gpu.tlasL2wBuf).not.toBe(previous.l2w);
    expect(Array.from(
      harness.recordFor(harness.gpu.tlasNodesBuf).bytes.subarray(0, 4),
    )).toEqual([31, 32, 33, 34]);
    expect(Array.from(
      harness.recordFor(harness.gpu.tlasInstIdxBuf).bytes.subarray(0, 4),
    )).toEqual([35, 36, 37, 38]);
    expect(Array.from(
      harness.recordFor(harness.gpu.tlasBlasRootsBuf).bytes.subarray(0, 4),
    )).toEqual([39, 40, 41, 42]);
    expect(Array.from(
      harness.recordFor(harness.gpu.tlasW2lBuf).bytes.subarray(0, 4),
    )).toEqual([43, 44, 45, 46]);
    expect(Array.from(
      harness.recordFor(harness.gpu.tlasL2wBuf).bytes.subarray(0, 4),
    )).toEqual([47, 48, 49, 50]);
    for (const buffer of Object.values(previous)) {
      expect(harness.recordFor(buffer).destroy).toHaveBeenCalledTimes(1);
    }
  });

  it('preflights TLAS refit capacity and atomically replaces all five growth buffers', () => {
    const prior = trackedCohort(16);
    const previous = {
      nodes: prior.gpu.tlasNodesBuf,
      instanceIndices: prior.gpu.tlasInstIdxBuf,
      blasRoots: prior.gpu.tlasBlasRootsBuf,
      w2l: prior.gpu.tlasW2lBuf,
      l2w: prior.gpu.tlasL2wBuf,
    };
    const stub = failureDevice({ createAt: 2 });
    const tlas = {
      nodes: new ArrayBuffer(64),
      instanceIndices: new ArrayBuffer(32),
      blasRoots: new ArrayBuffer(32),
      worldToLocal: new ArrayBuffer(128),
      localToWorld: new ArrayBuffer(128),
    } as never;

    expect(() => refitProbeTlasBuffersInPlace(stub.device, prior.gpu, tlas))
      .toThrow('injected create 2');
    expect(prior.gpu.tlasNodesBuf).toBe(previous.nodes);
    expect(prior.gpu.tlasInstIdxBuf).toBe(previous.instanceIndices);
    expect(prior.gpu.tlasBlasRootsBuf).toBe(previous.blasRoots);
    expect(prior.gpu.tlasW2lBuf).toBe(previous.w2l);
    expect(prior.gpu.tlasL2wBuf).toBe(previous.l2w);
    expect(stub.created[0]!.destroy).toHaveBeenCalledTimes(1);

    refitProbeTlasBuffersInPlace(stub.device, prior.gpu, tlas);
    expect(prior.gpu.tlasNodesBuf).not.toBe(previous.nodes);
    expect(prior.gpu.tlasInstIdxBuf).not.toBe(previous.instanceIndices);
    expect(prior.gpu.tlasBlasRootsBuf).not.toBe(previous.blasRoots);
    expect(prior.gpu.tlasW2lBuf).not.toBe(previous.w2l);
    expect(prior.gpu.tlasL2wBuf).not.toBe(previous.l2w);
  });

  it.each([
    ['write 2', { writeAt: 2 }, 'injected write 2'],
    ['copy 2', { copyAt: 2 }, 'injected copy 2'],
    ['encoder', { encoder: true }, 'injected encoder'],
    ['finish', { finish: true }, 'injected finish'],
    ['submit', { submit: true }, 'injected submit'],
  ] as const)(
    'keeps same-capacity TLAS refit atomic after %s failure, then retries',
    async (_stage, options, message) => {
      const prior = trackedCohort(256);
      const previous = {
        nodes: prior.gpu.tlasNodesBuf,
        instanceIndices: prior.gpu.tlasInstIdxBuf,
        blasRoots: prior.gpu.tlasBlasRootsBuf,
        w2l: prior.gpu.tlasW2lBuf,
        l2w: prior.gpu.tlasL2wBuf,
      };
      const stub = failureDevice(options);
      const tlas = {
        nodes: new ArrayBuffer(64),
        instanceIndices: new ArrayBuffer(32),
        blasRoots: new ArrayBuffer(32),
        worldToLocal: new ArrayBuffer(128),
        localToWorld: new ArrayBuffer(128),
      } as never;

      expect(() => refitProbeTlasBuffersInPlace(stub.device, prior.gpu, tlas))
        .toThrow(message);
      expect(prior.gpu.tlasNodesBuf).toBe(previous.nodes);
      expect(prior.gpu.tlasInstIdxBuf).toBe(previous.instanceIndices);
      expect(prior.gpu.tlasBlasRootsBuf).toBe(previous.blasRoots);
      expect(prior.gpu.tlasW2lBuf).toBe(previous.w2l);
      expect(prior.gpu.tlasL2wBuf).toBe(previous.l2w);
      for (const record of prior.records) expect(record.destroy).not.toHaveBeenCalled();
      for (const record of stub.created) expect(record.destroy).toHaveBeenCalledTimes(1);

      const createdBeforeRetry = stub.created.length;
      refitProbeTlasBuffersInPlace(stub.device, prior.gpu, tlas);
      await vi.waitFor(() => {
        for (const record of stub.created.slice(createdBeforeRetry)) {
          expect(record.destroy).toHaveBeenCalledTimes(1);
        }
      });

      expect(prior.gpu.tlasNodesBuf).toBe(previous.nodes);
      expect(prior.gpu.tlasInstIdxBuf).toBe(previous.instanceIndices);
      expect(prior.gpu.tlasBlasRootsBuf).toBe(previous.blasRoots);
      expect(prior.gpu.tlasW2lBuf).toBe(previous.w2l);
      expect(prior.gpu.tlasL2wBuf).toBe(previous.l2w);
      expect(stub.created.slice(createdBeforeRetry)).toHaveLength(5);
    },
  );

  it('retires all staging after accepted submit when completion tracking throws synchronously', () => {
    const prior = trackedCohort(256);
    const previous = bvhKeys.map((key) => prior.gpu[key]);
    const stub = failureDevice({ completion: true });
    const tlas = {
      nodes: new ArrayBuffer(64),
      instanceIndices: new ArrayBuffer(32),
      blasRoots: new ArrayBuffer(32),
      worldToLocal: new ArrayBuffer(128),
      localToWorld: new ArrayBuffer(128),
    } as never;

    expect(() => refitProbeTlasBuffersInPlace(stub.device, prior.gpu, tlas))
      .not.toThrow();
    expect(bvhKeys.map((key) => prior.gpu[key])).toEqual(previous);
    expect(stub.created).toHaveLength(5);
    for (const record of stub.created) {
      expect(record.destroy).toHaveBeenCalledOnce();
    }
    for (const record of prior.records) {
      expect(record.destroy).not.toHaveBeenCalled();
    }
  });

  it('retires an aliased prior TLAS destination at most once', () => {
    const prior = trackedCohort(16);
    const shared = trackedBuffer(16);
    prior.gpu.tlasNodesBuf = shared.gpu;
    prior.gpu.tlasInstIdxBuf = shared.gpu;
    prior.gpu.tlasBlasRootsBuf = shared.gpu;
    prior.gpu.tlasW2lBuf = shared.gpu;
    prior.gpu.tlasL2wBuf = shared.gpu;
    const stub = failureDevice();
    const tlas = {
      nodes: new ArrayBuffer(64),
      instanceIndices: new ArrayBuffer(32),
      blasRoots: new ArrayBuffer(32),
      worldToLocal: new ArrayBuffer(128),
      localToWorld: new ArrayBuffer(128),
    } as never;

    refitProbeTlasBuffersInPlace(stub.device, prior.gpu, tlas);

    expect(shared.destroy).toHaveBeenCalledOnce();
    expect(new Set([
      prior.gpu.tlasNodesBuf,
      prior.gpu.tlasInstIdxBuf,
      prior.gpu.tlasBlasRootsBuf,
      prior.gpu.tlasW2lBuf,
      prior.gpu.tlasL2wBuf,
    ])).toHaveLength(5);
  });
});
