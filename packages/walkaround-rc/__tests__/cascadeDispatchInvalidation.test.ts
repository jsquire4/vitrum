import { describe, expect, it, vi } from 'vitest';
import { RCDispatcher, type RCDispatchOptsRaw, type CascadeDim } from '../src/index.js';

const DIMS: CascadeDim[] = [
  { probes: [1, 1, 1], rays: 16, intervalNear: 0, intervalFar: 4 },
  { probes: [1, 1, 1], rays: 64, intervalNear: 4, intervalFar: 16 },
];

function installWebGpuConstants(): void {
  vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_DST: 2 });
  vi.stubGlobal('GPUTextureUsage', { TEXTURE_BINDING: 1, COPY_DST: 2 });
  vi.stubGlobal('GPUShaderStage', { COMPUTE: 1 });
}

function makeExternalBuffer(label: string): GPUBuffer {
  return { label, destroy: vi.fn() } as unknown as GPUBuffer;
}

function makeMockDevice() {
  const destroyBuffer = vi.fn();
  const createBindGroup = vi.fn(() => ({}));
  const pass = {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
    end: vi.fn(),
  };
  const device = {
    createShaderModule: vi.fn(() => ({})),
    createBindGroupLayout: vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    createComputePipeline: vi.fn(() => ({})),
    createBindGroup,
    createSampler: vi.fn(() => ({})),
    createTexture: vi.fn(() => ({
      createView: vi.fn(() => ({})),
      destroy: vi.fn(),
    })),
    createBuffer: vi.fn((desc: { label?: string; size?: number }) => ({
      label: desc.label,
      size: desc.size,
      getMappedRange: () => new ArrayBuffer(Math.max(desc.size ?? 16, 16)),
      unmap: vi.fn(),
      destroy: destroyBuffer,
    })),
    createCommandEncoder: vi.fn(() => ({
      beginComputePass: vi.fn(() => pass),
      finish: vi.fn(() => ({})),
    })),
    queue: {
      writeBuffer: vi.fn(),
      writeTexture: vi.fn(),
      submit: vi.fn(),
    },
  } as unknown as GPUDevice;
  return { device, createBindGroup, destroyBuffer };
}

function baseOpts(device: GPUDevice): RCDispatchOptsRaw {
  return {
    device,
    bvhNodesBuf: makeExternalBuffer('bvh-nodes'),
    bvhIndicesBuf: makeExternalBuffer('bvh-indices'),
    bvhPositionsBuf: makeExternalBuffer('bvh-positions'),
    materialsBuf: makeExternalBuffer('materials'),
    triMaterialIdBuf: makeExternalBuffer('tri-material-id'),
    cascadeBufs: [makeExternalBuffer('cascade-0'), makeExternalBuffer('cascade-1')],
    probeOriginWorld: [0, 0, 0],
    roomSize: [1, 1, 1],
    sunDirection: [0, 1, 0],
    sunColor: [1, 1, 1],
    frameSeed: 1,
  };
}

describe('RCDispatcher binding cache invalidation', () => {
  it('reuses handles for stable bindings and rebuilds on bvhMode or bounds changes', () => {
    installWebGpuConstants();
    const { device, createBindGroup, destroyBuffer } = makeMockDevice();
    const dispatcher = new RCDispatcher(DIMS);
    const merged = baseOpts(device);

    dispatcher.dispatchFrameRaw(merged);
    const bindGroupsAfterFirst = createBindGroup.mock.calls.length;
    const destroysAfterFirst = destroyBuffer.mock.calls.length;
    expect(bindGroupsAfterFirst).toBeGreaterThan(0);

    dispatcher.dispatchFrameRaw({ ...merged, frameSeed: 2 });
    expect(createBindGroup.mock.calls.length).toBe(bindGroupsAfterFirst);
    expect(destroyBuffer.mock.calls.length).toBe(destroysAfterFirst);

    dispatcher.dispatchFrameRaw({
      ...merged,
      bvhMode: 'tlas',
      tlasNodeCount: 1,
      tlasNodesBuf: makeExternalBuffer('tlas-nodes'),
      tlasInstanceIndicesBuf: makeExternalBuffer('tlas-inst'),
      tlasBlasRootsBuf: makeExternalBuffer('tlas-blas'),
      tlasInstanceWorldToLocalBuf: makeExternalBuffer('tlas-w2l'),
      tlasInstanceLocalToWorldBuf: makeExternalBuffer('tlas-l2w'),
      frameSeed: 3,
    });
    const bindGroupsAfterTlas = createBindGroup.mock.calls.length;
    const destroysAfterTlas = destroyBuffer.mock.calls.length;
    expect(bindGroupsAfterTlas).toBeGreaterThan(bindGroupsAfterFirst);
    expect(destroysAfterTlas).toBeGreaterThan(destroysAfterFirst);

    dispatcher.dispatchFrameRaw({
      ...merged,
      probeOriginWorld: [2, 0, 0],
      roomSize: [2, 1, 1],
      frameSeed: 4,
    });
    expect(createBindGroup.mock.calls.length).toBeGreaterThan(bindGroupsAfterTlas);
    expect(destroyBuffer.mock.calls.length).toBeGreaterThan(destroysAfterTlas);
  });
});
