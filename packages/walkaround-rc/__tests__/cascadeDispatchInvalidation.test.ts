import { describe, expect, it, vi } from 'vitest';
import { RCDispatcher, type RCDispatchOptsRaw, type CascadeDim } from '../src/index.js';

const DIMS: CascadeDim[] = [
  { probes: [1, 1, 1], rays: 16, intervalNear: 0, intervalFar: 4 },
  { probes: [1, 1, 1], rays: 64, intervalNear: 4, intervalFar: 16 },
];

function installWebGpuConstants(): void {
  vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8 });
  vi.stubGlobal('GPUTextureUsage', { TEXTURE_BINDING: 1, COPY_DST: 2 });
  vi.stubGlobal('GPUShaderStage', { COMPUTE: 1 });
}

function makeExternalBuffer(label: string, size = 64): GPUBuffer {
  return { label, size, usage: 5, destroy: vi.fn() } as unknown as GPUBuffer;
}

function makeExternalView(label: string): GPUTextureView {
  return { label } as unknown as GPUTextureView;
}

function makeMockDevice() {
  const destroyBuffer = vi.fn();
  const createBindGroup = vi.fn(() => ({}));
  const textureDestroyers: Array<ReturnType<typeof vi.fn>> = [];
  const copyBufferToBuffer = vi.fn();
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
    createTexture: vi.fn(() => {
      const destroy = vi.fn();
      textureDestroyers.push(destroy);
      return { createView: vi.fn(() => ({})), destroy };
    }),
    createBuffer: vi.fn((desc: { label?: string; size?: number; usage?: number }) => ({
      label: desc.label,
      size: desc.size,
      usage: desc.usage,
      getMappedRange: () => new ArrayBuffer(Math.max(desc.size ?? 16, 16)),
      unmap: vi.fn(),
      destroy: destroyBuffer,
    })),
    createCommandEncoder: vi.fn(() => ({
      copyBufferToBuffer,
      beginComputePass: vi.fn(() => pass),
      finish: vi.fn(() => ({})),
    })),
    queue: {
      writeBuffer: vi.fn(),
      writeTexture: vi.fn(),
      submit: vi.fn(),
    },
  } as unknown as GPUDevice;
  return { device, createBindGroup, destroyBuffer, copyBufferToBuffer, textureDestroyers };
}

function baseOpts(device: GPUDevice): RCDispatchOptsRaw {
  return {
    device,
    bvhNodesBuf: makeExternalBuffer('bvh-nodes'),
    bvhIndicesBuf: makeExternalBuffer('bvh-indices'),
    bvhPositionsBuf: makeExternalBuffer('bvh-positions'),
    bvhNormalsBuf: makeExternalBuffer('bvh-normals'),
    materialsBuf: makeExternalBuffer('materials'),
    triMaterialIdBuf: makeExternalBuffer('tri-material-id'),
    cascadeBufs: [makeExternalBuffer('cascade-0', 256), makeExternalBuffer('cascade-1', 1024)],
    probeOriginWorld: [0, 0, 0],
    roomSize: [1, 1, 1],
    sunDirection: [0, 1, 0],
    sunColor: [1, 1, 1],
    frameSeed: 1,
  };
}

describe('RCDispatcher binding cache invalidation', () => {
  it('packs the same resolved emitter data/alias layout that preflight validates', () => {
    installWebGpuConstants();
    const { device } = makeMockDevice();
    const dispatcher = new RCDispatcher(DIMS);
    const queue = device.queue as unknown as {
      writeBuffer: { mock: { calls: unknown[][] } };
    };
    const latestUniformWords = (): Uint32Array => {
      const write = queue.writeBuffer.mock.calls
        .filter((call) => call[2] instanceof ArrayBuffer && (call[2] as ArrayBuffer).byteLength === 160)
        .at(-1);
      if (write == null) throw new Error('missing CascadeUniforms write');
      return new Uint32Array(write[2] as ArrayBuffer);
    };

    dispatcher.dispatchFrameRaw({
      ...baseOpts(device),
      emittersBuf: makeExternalBuffer('emitter-data-and-alias', 96),
      emitterCount: 1,
    });
    expect(latestUniformWords()[32]).toBe(0);
    expect(latestUniformWords()[33]).toBe(20);

    dispatcher.dispatchFrameRaw({
      ...baseOpts(device),
      emittersBuf: makeExternalBuffer('offset-emitter-data-and-alias', 112),
      emitterDataOffset: 16,
      emitterCount: 1,
      frameSeed: 2,
    });
    expect(latestUniformWords()[32]).toBe(4);
    expect(latestUniformWords()[33]).toBe(24);
  });

  it('packs scalar-sky radiance and preserves raw custom-map inference', () => {
    installWebGpuConstants();
    const { device } = makeMockDevice();
    const dispatcher = new RCDispatcher(DIMS);
    const envTextureView = makeExternalView('directional-env');
    const envSampler = {} as GPUSampler;
    const base = {
      ...baseOpts(device),
      envTextureView,
      envSampler,
      envIntensity: 0,
      scalarSkyRadiance: [0.25, 0.5, 1] as const,
    };

    // Backward compatibility: a raw custom texture/sampler pair implies a
    // directional payload when the explicit flag is omitted.
    dispatcher.dispatchFrameRaw(base);
    const queue = device.queue as unknown as {
      writeBuffer: { mock: { calls: unknown[][] } };
    };
    const latestUniform = (): ArrayBuffer => {
      const write = queue.writeBuffer.mock.calls
        .filter((call) => call[2] instanceof ArrayBuffer && (call[2] as ArrayBuffer).byteLength === 160)
        .at(-1);
      if (write == null) throw new Error('missing CascadeUniforms write');
      return write[2] as ArrayBuffer;
    };
    expect(new Uint32Array(latestUniform())[39]).toBe(1);

    // Hybrid binds a black placeholder but explicitly selects scalar sky.
    dispatcher.dispatchFrameRaw({
      ...base,
      frameSeed: 2,
      hasDirectionalEnvironment: false,
      scalarSkyRadiance: [0.5, 1, 2],
    });
    const packed = latestUniform();
    expect(Array.from(new Float32Array(packed).slice(36, 39))).toEqual([0.5, 1, 2]);
    expect(new Uint32Array(packed)[39]).toBe(0);
  });

  it('binds canonical scene-arena geometry windows without copying them', () => {
    installWebGpuConstants();
    const { device, createBindGroup, copyBufferToBuffer } = makeMockDevice();
    const dispatcher = new RCDispatcher(DIMS);
    const arena = makeExternalBuffer('main-scene-arena', 2_048);
    dispatcher.dispatchFrameRaw({
      ...baseOpts(device),
      bvhNodesBuf: arena,
      bvhNodesOffset: 0,
      bvhNodesSize: 32,
      bvhIndicesBuf: arena,
      bvhIndicesOffset: 256,
      bvhIndicesSize: 16,
      bvhPositionsBuf: arena,
      bvhPositionsOffset: 512,
      bvhPositionsSize: 48,
      bvhNormalsBuf: arena,
      bvhNormalsOffset: 768,
      bvhNormalsSize: 48,
    });

    const bindGroupCalls = createBindGroup.mock.calls as unknown as Array<[
      { entries: Array<{ binding: number; resource: GPUBufferBinding }> },
    ]>;
    const firstEntries = bindGroupCalls[0]![0].entries;
    const resource = (binding: number) =>
      firstEntries.find((entry) => entry.binding === binding)?.resource;
    expect(resource(0)).toEqual({ buffer: arena, offset: 0, size: 32 });
    expect(resource(1)).toEqual({ buffer: arena, offset: 256, size: 16 });
    expect(resource(2)).toEqual({ buffer: arena, offset: 512, size: 48 });
    expect(resource(18)).toEqual({ buffer: arena, offset: 768, size: 48 });
    // Only RC-specific material and triangle-id records enter the compact
    // adapter arena; the four large geometry windows remain borrowed.
    expect(copyBufferToBuffer).toHaveBeenCalledTimes(2);
  });

  it('copies only dirty packed-arena slices and performs zero static-frame copies', () => {
    installWebGpuConstants();
    const { device, createBindGroup, copyBufferToBuffer } = makeMockDevice();
    const dispatcher = new RCDispatcher(DIMS);
    const merged = baseOpts(device);

    dispatcher.dispatchFrameRaw(merged);
    expect(copyBufferToBuffer).toHaveBeenCalledTimes(2);
    copyBufferToBuffer.mockClear();
    const bindGroupsAfterInitial = createBindGroup.mock.calls.length;

    dispatcher.dispatchFrameRaw({ ...merged, frameSeed: 2 });
    expect(copyBufferToBuffer).not.toHaveBeenCalled();
    expect(createBindGroup.mock.calls.length).toBe(bindGroupsAfterInitial);

    const nextMaterials = makeExternalBuffer('materials-refreshed');
    dispatcher.dispatchFrameRaw({
      ...merged,
      materialsBuf: nextMaterials,
      frameSeed: 3,
    });
    expect(copyBufferToBuffer).toHaveBeenCalledTimes(1);
    expect(copyBufferToBuffer.mock.calls[0]![0]).toBe(nextMaterials);
    expect(copyBufferToBuffer.mock.calls[0]![1]).toBe(0);
    expect(copyBufferToBuffer.mock.calls[0]![3]).toBe(64);
    expect(copyBufferToBuffer.mock.calls[0]![4]).toBe(64);
    expect(createBindGroup.mock.calls.length).toBe(bindGroupsAfterInitial);
  });

  it('copies only five TLAS slices for an in-place refit version change', () => {
    installWebGpuConstants();
    const { device, createBindGroup, copyBufferToBuffer } = makeMockDevice();
    const dispatcher = new RCDispatcher(DIMS);
    const opts: RCDispatchOptsRaw = {
      ...baseOpts(device),
      bvhMode: 'tlas',
      tlasNodeCount: 1,
      tlasNodesBuf: makeExternalBuffer('tlas-nodes'),
      tlasInstanceIndicesBuf: makeExternalBuffer('tlas-inst', 4),
      tlasBlasRootsBuf: makeExternalBuffer('tlas-blas', 4),
      tlasInstanceWorldToLocalBuf: makeExternalBuffer('tlas-w2l'),
      tlasInstanceLocalToWorldBuf: makeExternalBuffer('tlas-l2w'),
      tlasArenaVersion: 1,
    };

    dispatcher.dispatchFrameRaw(opts);
    expect(copyBufferToBuffer).toHaveBeenCalledTimes(7);
    const queue = device.queue as unknown as {
      writeBuffer: { mock: { calls: unknown[][] } };
    };
    const headerWrite = queue.writeBuffer.mock.calls.find((call) => {
      const bytes = call[2];
      return bytes instanceof ArrayBuffer && bytes.byteLength === 64;
    });
    expect(headerWrite).toBeDefined();
    expect(Array.from(new Uint32Array(headerWrite![2] as ArrayBuffer))).toEqual([
      16, 1,
      32, 16,
      48, 2,
      64, 1,
      65, 1,
      66, 4,
      82, 4,
      0, 0,
    ]);
    copyBufferToBuffer.mockClear();
    const bindGroupsAfterInitial = createBindGroup.mock.calls.length;

    dispatcher.dispatchFrameRaw({ ...opts, frameSeed: 2, tlasArenaVersion: 2 });
    expect(copyBufferToBuffer).toHaveBeenCalledTimes(5);
    expect(copyBufferToBuffer.mock.calls.map((call) => call[0])).toEqual([
      opts.tlasNodesBuf,
      opts.tlasInstanceIndicesBuf,
      opts.tlasBlasRootsBuf,
      opts.tlasInstanceWorldToLocalBuf,
      opts.tlasInstanceLocalToWorldBuf,
    ]);
    expect(copyBufferToBuffer.mock.calls.map((call) => call[1])).toEqual([0, 0, 0, 0, 0]);
    expect(copyBufferToBuffer.mock.calls.map((call) => call[3])).toEqual([
      192, 256, 260, 264, 328,
    ]);
    expect(copyBufferToBuffer.mock.calls.map((call) => call[4])).toEqual([
      64, 4, 4, 64, 64,
    ]);
    expect(createBindGroup.mock.calls.length).toBe(bindGroupsAfterInitial);
  });

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
      bvhNormalsBuf: makeExternalBuffer('bvh-normals-b'),
      frameSeed: 3,
    });
    const bindGroupsAfterNormals = createBindGroup.mock.calls.length;
    const destroysAfterNormals = destroyBuffer.mock.calls.length;
    expect(bindGroupsAfterNormals).toBeGreaterThan(bindGroupsAfterFirst);
    expect(destroysAfterNormals).toBeGreaterThan(destroysAfterFirst);

    dispatcher.dispatchFrameRaw({
      ...merged,
      bvhMode: 'tlas',
      tlasNodeCount: 1,
      tlasNodesBuf: makeExternalBuffer('tlas-nodes'),
      tlasInstanceIndicesBuf: makeExternalBuffer('tlas-inst', 4),
      tlasBlasRootsBuf: makeExternalBuffer('tlas-blas', 4),
      tlasInstanceWorldToLocalBuf: makeExternalBuffer('tlas-w2l'),
      tlasInstanceLocalToWorldBuf: makeExternalBuffer('tlas-l2w'),
      frameSeed: 4,
    });
    const bindGroupsAfterTlas = createBindGroup.mock.calls.length;
    const destroysAfterTlas = destroyBuffer.mock.calls.length;
    expect(bindGroupsAfterTlas).toBeGreaterThan(bindGroupsAfterNormals);
    expect(destroysAfterTlas).toBeGreaterThan(destroysAfterNormals);

    dispatcher.dispatchFrameRaw({
      ...merged,
      probeOriginWorld: [2, 0, 0],
      roomSize: [2, 1, 1],
      frameSeed: 5,
    });
    expect(createBindGroup.mock.calls.length).toBeGreaterThan(bindGroupsAfterTlas);
    expect(destroyBuffer.mock.calls.length).toBeGreaterThan(destroysAfterTlas);
  });

  it('binds material-atlas views and rebuilds when they change', () => {
    installWebGpuConstants();
    const { device, createBindGroup } = makeMockDevice();
    const dispatcher = new RCDispatcher(DIMS);
    const opts = {
      ...baseOpts(device),
      materialTextureAtlasView: makeExternalView('atlas-a'),
      materialMapMetaTextureView: makeExternalView('meta-a'),
      bvhTangentTextureView: makeExternalView('tangent-a'),
      bvhVertexColorTextureView: makeExternalView('vertex-color-a'),
    };

    dispatcher.dispatchFrameRaw(opts);
    const createBindGroupCalls = createBindGroup.mock.calls as unknown as Array<[
      { entries: Array<{ binding: number; resource: unknown }> },
    ]>;
    const firstEntries = createBindGroupCalls[0]![0].entries;
    expect(firstEntries.some((entry) => entry.binding === 16 && entry.resource === opts.materialTextureAtlasView)).toBe(true);
    expect(firstEntries.some((entry) => entry.binding === 17 && entry.resource === opts.materialMapMetaTextureView)).toBe(true);
    expect(firstEntries.some((entry) => {
      if (entry.binding !== 18 || typeof entry.resource !== 'object' || entry.resource == null) return false;
      return (entry.resource as { buffer?: unknown }).buffer === opts.bvhNormalsBuf;
    })).toBe(true);
    expect(firstEntries.some((entry) => entry.binding === 19 && entry.resource === opts.bvhTangentTextureView)).toBe(true);
    expect(firstEntries.some((entry) => entry.binding === 20 && entry.resource === opts.bvhVertexColorTextureView)).toBe(true);
    const bindGroupsAfterFirst = createBindGroup.mock.calls.length;

    dispatcher.dispatchFrameRaw({
      ...opts,
      frameSeed: 2,
      bvhVertexColorTextureView: makeExternalView('vertex-color-b'),
    });
    expect(createBindGroup.mock.calls.length).toBeGreaterThan(bindGroupsAfterFirst);
  });

  it('recompiles shader modules after invalidateBindings (device-change safety)', () => {
    // V1-5: shader modules are bound to the device that created them. On a device
    // change `invalidateBindings()` must null the cached cast/merge modules so the
    // next dispatch recompiles them on the fresh device — otherwise a rebuild
    // reuses an old-device module and raises a cross-device validation error.
    installWebGpuConstants();
    const { device } = makeMockDevice();
    const createShaderModule = device.createShaderModule as unknown as {
      mock: { calls: unknown[][] };
    };
    const dispatcher = new RCDispatcher(DIMS);
    const opts = baseOpts(device);

    dispatcher.dispatchFrameRaw(opts);
    const shaderModulesAfterFirst = createShaderModule.mock.calls.length;
    expect(shaderModulesAfterFirst).toBeGreaterThan(0);

    // A pure binding invalidation with no device change: subsequent dispatch reuses
    // the cached modules only because we recompile on rebuild — assert the null.
    dispatcher.invalidateBindings();
    dispatcher.dispatchFrameRaw({ ...opts, frameSeed: 2 });
    expect(createShaderModule.mock.calls.length).toBeGreaterThan(shaderModulesAfterFirst);
  });

  it('packs sunAngularRadius into the former sun-direction padding lane', () => {
    installWebGpuConstants();
    const { device } = makeMockDevice();
    const dispatcher = new RCDispatcher(DIMS);
    dispatcher.dispatchFrameRaw({
      ...baseOpts(device),
      sunAngularRadius: 0.075,
    });
    const queue = device.queue as unknown as {
      writeBuffer: { mock: { calls: unknown[][] } };
    };
    const firstUniformWrite = queue.writeBuffer.mock.calls.find((call) => {
      const buffer = call[2];
      return buffer instanceof ArrayBuffer && buffer.byteLength === 160;
    });
    expect(firstUniformWrite).toBeDefined();
    const raw = new Float32Array(firstUniformWrite![2] as ArrayBuffer);
    expect(raw[19]).toBeCloseTo(0.075, 6);
  });

  it('keeps the published handles alive and destroys every candidate owner after a late rebuild failure', () => {
    installWebGpuConstants();
    const {
      device,
      createBindGroup,
      destroyBuffer,
      textureDestroyers,
    } = makeMockDevice();
    const dispatcher = new RCDispatcher(DIMS);
    const opts = baseOpts(device);

    dispatcher.dispatchFrameRaw(opts);
    const initialBindGroupCount = createBindGroup.mock.calls.length;
    const initialTextureDestroyers = [...textureDestroyers];
    expect(initialBindGroupCount).toBe(3);

    // Two cast bind groups succeed; the merge bind group fails after all candidate
    // textures, dummies, arena, cast UBOs, and the merge UBO have been allocated.
    createBindGroup
      .mockImplementationOnce(() => ({}))
      .mockImplementationOnce(() => ({}))
      .mockImplementationOnce(() => { throw new Error('injected late bind-group failure'); });

    expect(() => dispatcher.dispatchFrameRaw({
      ...opts,
      bvhNormalsBuf: makeExternalBuffer('candidate-normals'),
      frameSeed: 2,
    })).toThrow(/injected late bind-group failure/);

    const candidateTextureDestroyers = textureDestroyers.slice(initialTextureDestroyers.length);
    expect(candidateTextureDestroyers).toHaveLength(5);
    for (const destroy of candidateTextureDestroyers) expect(destroy).toHaveBeenCalledOnce();
    for (const destroy of initialTextureDestroyers) expect(destroy).not.toHaveBeenCalled();
    expect(destroyBuffer).toHaveBeenCalledTimes(6);

    const callsAfterFailure = createBindGroup.mock.calls.length;
    dispatcher.dispatchFrameRaw({ ...opts, frameSeed: 3 });
    expect(createBindGroup).toHaveBeenCalledTimes(callsAfterFailure);
    expect(destroyBuffer).toHaveBeenCalledTimes(6);

    dispatcher.dispose();
    expect(destroyBuffer).toHaveBeenCalledTimes(12);
    for (const destroy of initialTextureDestroyers) expect(destroy).toHaveBeenCalledOnce();
  });
});
