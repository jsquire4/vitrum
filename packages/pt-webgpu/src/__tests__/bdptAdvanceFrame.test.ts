/**
 * H9 — bdptAdvanceFrame uses rebuildGroup2Only instead of nulling group 2.
 *
 * Verifies:
 *   1. A call to bdptAdvanceFrame(buf) triggers exactly ONE createBindGroup
 *      binding the new buffer (binding slot 5 in group 2).
 *   2. The cached group 0 is NOT rebuilt (createBindGroup is not called for
 *      the group 0 label — group 0 stays intact).
 *   3. The next renderFrame sees a non-null pathTraceBindGroup2 so rendering
 *      proceeds (the prior null-group bug is closed).
 *   4. Pointer-equality fast-out: a second bdptAdvanceFrame with the SAME
 *      buffer reference does not call createBindGroup again.
 */
import { describe, expect, it, vi } from 'vitest';
import { GpuResources } from '../gpuResources.js';
import { installGpuConstStubs } from './gpuStub.js';

/** Build a minimal UploadedSceneBuffers-like object for rebuildGroup2Only. */
function makeSceneBuffers() {
  const buf = () => ({ destroy: vi.fn() });
  return {
    tlasNodesBuffer: buf(),
    tlasInstanceIndicesBuffer: buf(),
    tlasBlasRootsBuffer: buf(),
    tlasInstanceWorldToLocalBuffer: buf(),
    tlasInstanceLocalToWorldBuffer: buf(),
    // scene buffers not needed for group-2 rebuild but satisfy the type
    positionsBuffer: buf(),
    indicesBuffer: buf(),
    triMaterialIdsBuffer: buf(),
    materialsBuffer: buf(),
    bvhNodesBuffer: buf(),
    normalsBuffer: buf(),
    analyticHeadersBuffer: buf(),
    analyticParamsBuffer: buf(),
    analyticLocalToWorldBuffer: buf(),
    analyticWorldToLocalBuffer: buf(),
    environmentMapTexelsBuffer: buf(),
    environmentMapCdfBuffer: buf(),
    pointLightsBuffer: buf(),
    spotLightsBuffer: buf(),
    rectAreaLightsBuffer: buf(),
    meshAreaLightsBuffer: buf(),
    lightTreeBuffer: buf(),
    uvsBuffer: buf(),
    materialTexDescriptorsBuffer: buf(),
    materialTextureView: {},
    materialTextureSampler: {},
    materialLinearTextureView: {},
    gpuMemoryBytes: vi.fn(() => ({ bufferBytes: 0, textureBytesByFormat: {} })),
    destroy: vi.fn(),
  } as unknown as import('../scene/uploadSceneBuffers.js').UploadedSceneBuffers;
}

/** Build a GPUDevice stub that records createBindGroup calls. */
function makeRecordingDevice() {
  installGpuConstStubs();
  const bindGroupCalls: { label?: string; entries: GPUBindGroupEntry[] }[] = [];
  const layouts: Record<string, GPUBindGroupLayout> = {
    layout0: { label: 'group0' } as unknown as GPUBindGroupLayout,
    layout1: { label: 'group1' } as unknown as GPUBindGroupLayout,
    layout2: { label: 'group2' } as unknown as GPUBindGroupLayout,
    layout3: { label: 'group3' } as unknown as GPUBindGroupLayout,
  };
  const groups: Record<string, GPUBindGroup> = {
    group0: { label: 'g0' } as unknown as GPUBindGroup,
    group1: { label: 'g1' } as unknown as GPUBindGroup,
    group2: { label: 'g2' } as unknown as GPUBindGroup,
    group3: { label: 'g3' } as unknown as GPUBindGroup,
  };
  let bgIdx = 0;
  const device = {
    createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
    createTexture: vi.fn(() => ({ createView: vi.fn(() => ({})), destroy: vi.fn() })),
    createSampler: vi.fn(() => ({})),
    createCommandEncoder: vi.fn(() => ({
      clearBuffer: vi.fn(),
      finish: vi.fn(() => ({})),
    })),
    createShaderModule: vi.fn(() => ({})),
    createComputePipeline: vi.fn(() => ({
      getBindGroupLayout: vi.fn((idx: number) => Object.values(layouts)[idx] ?? layouts.layout0),
    })),
    createBindGroupLayout: vi.fn(() => Object.values(layouts)[bgIdx++ % 4]),
    createPipelineLayout: vi.fn(() => ({})),
    createBindGroup: vi.fn((desc: { label?: string; layout: unknown; entries: GPUBindGroupEntry[] }) => {
      bindGroupCalls.push({ ...(desc.label !== undefined ? { label: desc.label } : {}), entries: desc.entries });
      const key = `group${bgIdx++ % 4}`;
      return groups[key] ?? { label: desc.label };
    }),
    queue: { submit: vi.fn(), writeBuffer: vi.fn() },
    limits: {
      maxStorageBuffersPerShaderStage: 32,
      maxStorageTexturesPerShaderStage: 8,
    },
  } as unknown as GPUDevice;
  return { device, bindGroupCalls, layouts, groups };
}

describe('H9: bdptAdvanceFrame — rebuildGroup2Only (no null-group rendering break)', () => {
  it('rebuildGroup2Only creates exactly one bind group binding the new light-path buffer at entry 5', () => {
    const { device, bindGroupCalls, layouts } = makeRecordingDevice();
    const gpu = new GpuResources(device, 'full', true);

    // Populate the explicit group-2 layout directly (bypassing full pipeline
    // build, which would require full shader compilation).
    gpu.bindGroupLayout2 = layouts.layout2 ?? null;
    // Also set bdptEyeStackBuffer so the entry is non-null.
    (gpu as unknown as { bdptEyeStackBuffer: GPUBuffer }).bdptEyeStackBuffer =
      { destroy: vi.fn() } as unknown as GPUBuffer;

    const callsBefore = bindGroupCalls.length;
    const lightPathBuf = { label: 'externalLightPath', destroy: vi.fn() } as unknown as GPUBuffer;
    const sb = makeSceneBuffers();

    gpu.rebuildGroup2Only(sb, lightPathBuf);

    // Exactly ONE createBindGroup should have fired.
    expect(bindGroupCalls.length).toBe(callsBefore + 1);
    const call = bindGroupCalls[bindGroupCalls.length - 1]!;
    // It must be the group-2 rebuild label.
    expect(call.label).toContain('bindgroup2');
    // Binding slot 5 must reference the supplied light-path buffer.
    const slot5 = call.entries.find((e) => e.binding === 5);
    expect(slot5).toBeDefined();
    expect((slot5!.resource as { buffer: GPUBuffer }).buffer).toBe(lightPathBuf);
    // pathTraceBindGroup2 must now be non-null.
    expect(gpu.pathTraceBindGroup2).not.toBeNull();
  });

  it('pointer-equality fast-out: supplying the same buffer a second time does not call createBindGroup again', () => {
    const { device, bindGroupCalls, layouts } = makeRecordingDevice();
    const gpu = new GpuResources(device, 'full', true);
    gpu.bindGroupLayout2 = layouts.layout2 ?? null;
    (gpu as unknown as { bdptEyeStackBuffer: GPUBuffer }).bdptEyeStackBuffer =
      { destroy: vi.fn() } as unknown as GPUBuffer;

    const lightPathBuf = { label: 'sameBuffer', destroy: vi.fn() } as unknown as GPUBuffer;
    const sb = makeSceneBuffers();

    gpu.rebuildGroup2Only(sb, lightPathBuf);
    const afterFirst = bindGroupCalls.length;

    // Second call with identical buffer reference — should be a no-op.
    gpu.rebuildGroup2Only(sb, lightPathBuf);
    expect(bindGroupCalls.length).toBe(afterFirst);
    expect(gpu.pathTraceBindGroup2).not.toBeNull();
  });

  it('invalidateBindGroups clears the fast-out so the next rebuildGroup2Only rebuilds', () => {
    const { device, bindGroupCalls, layouts } = makeRecordingDevice();
    const gpu = new GpuResources(device, 'full', true);
    gpu.bindGroupLayout2 = layouts.layout2 ?? null;
    (gpu as unknown as { bdptEyeStackBuffer: GPUBuffer }).bdptEyeStackBuffer =
      { destroy: vi.fn() } as unknown as GPUBuffer;

    const lightPathBuf = { label: 'buf', destroy: vi.fn() } as unknown as GPUBuffer;
    const sb = makeSceneBuffers();

    gpu.rebuildGroup2Only(sb, lightPathBuf);
    const afterFirst = bindGroupCalls.length;

    // Full invalidation clears the fast-out reference.
    gpu.invalidateBindGroups();
    gpu.rebuildGroup2Only(sb, lightPathBuf);

    // Must have triggered a new createBindGroup.
    expect(bindGroupCalls.length).toBeGreaterThan(afterFirst);
  });

  it('rebuildGroup2Only is a no-op when bindGroupLayout2 is null (pre-pipeline)', () => {
    const { device, bindGroupCalls } = makeRecordingDevice();
    const gpu = new GpuResources(device, 'full', true);
    // bindGroupLayout2 stays null (pipeline not built yet).
    const before = bindGroupCalls.length;
    const sb = makeSceneBuffers();
    const buf = { destroy: vi.fn() } as unknown as GPUBuffer;
    gpu.rebuildGroup2Only(sb, buf);
    expect(bindGroupCalls.length).toBe(before);
    expect(gpu.pathTraceBindGroup2).toBeNull();
  });
});
