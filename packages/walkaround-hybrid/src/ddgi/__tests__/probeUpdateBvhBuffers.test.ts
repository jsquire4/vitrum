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

function restirSnapshot(): never {
  return {
    bvhNodes: new ArrayBuffer(32),
    positions: new ArrayBuffer(64),
    bvhIndex: new ArrayBuffer(32),
    normals: new ArrayBuffer(64),
    triMaterialIds: new ArrayBuffer(16),
  } as never;
}

function failureDevice(options: {
  createAt?: number;
  writeAt?: number;
  copyAt?: number;
  encoder?: boolean;
  finish?: boolean;
  submit?: boolean;
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
      onSubmittedWorkDone: vi.fn(() => Promise.resolve()),
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
    };
    const buffers = {
      bvhNodes: new Float32Array(8),
      positions: new Float32Array(8),
      indices: new Uint32Array([0, 1, 2]),
      normals: new Float32Array(8),
      triMaterialId: new Uint32Array(4),
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
    };
    const snap = {
      bvhNodes: new ArrayBuffer(32),
      positions: new ArrayBuffer(64),
      bvhIndex: new ArrayBuffer(32),
      normals: new ArrayBuffer(64),
      triMaterialIds: new ArrayBuffer(16),
    } as never;

    rebuildProbeBvhFromRestir(device, g, snap);

    expect(created[5]?.size).toBe(32);
    expect(created[6]?.size).toBe(16);
    expect(created[7]?.size).toBe(16);
    expect(created[8]?.size).toBe(16);
    expect(created[9]?.size).toBe(16);
  });

  it.each([1, 5, 10])(
    'keeps the ten-buffer BVH cohort intact when candidate create #%i fails, then retries',
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

  it('preflights TLAS refit capacity and atomically replaces all three growth buffers', () => {
    const prior = trackedCohort(16);
    const previous = {
      nodes: prior.gpu.tlasNodesBuf,
      w2l: prior.gpu.tlasW2lBuf,
      l2w: prior.gpu.tlasL2wBuf,
    };
    const stub = failureDevice({ createAt: 2 });
    const tlas = {
      nodes: new ArrayBuffer(64),
      worldToLocal: new ArrayBuffer(128),
      localToWorld: new ArrayBuffer(128),
    } as never;

    expect(() => refitProbeTlasBuffersInPlace(stub.device, prior.gpu, tlas))
      .toThrow('injected create 2');
    expect(prior.gpu.tlasNodesBuf).toBe(previous.nodes);
    expect(prior.gpu.tlasW2lBuf).toBe(previous.w2l);
    expect(prior.gpu.tlasL2wBuf).toBe(previous.l2w);
    expect(stub.created[0]!.destroy).toHaveBeenCalledTimes(1);

    refitProbeTlasBuffersInPlace(stub.device, prior.gpu, tlas);
    expect(prior.gpu.tlasNodesBuf).not.toBe(previous.nodes);
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
        w2l: prior.gpu.tlasW2lBuf,
        l2w: prior.gpu.tlasL2wBuf,
      };
      const stub = failureDevice(options);
      const tlas = {
        nodes: new ArrayBuffer(64),
        worldToLocal: new ArrayBuffer(128),
        localToWorld: new ArrayBuffer(128),
      } as never;

      expect(() => refitProbeTlasBuffersInPlace(stub.device, prior.gpu, tlas))
        .toThrow(message);
      expect(prior.gpu.tlasNodesBuf).toBe(previous.nodes);
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
      expect(prior.gpu.tlasW2lBuf).toBe(previous.w2l);
      expect(prior.gpu.tlasL2wBuf).toBe(previous.l2w);
      expect(stub.created.slice(createdBeforeRetry)).toHaveLength(3);
    },
  );
});
