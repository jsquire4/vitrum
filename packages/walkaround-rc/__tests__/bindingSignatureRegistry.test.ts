import { describe, expect, it, vi } from 'vitest';
import { RCDispatcher, type RCDispatchOptsRaw, type CascadeDim } from '../src/index.js';

/**
 * T2-C (D16-8) — behavioral pin for the binding-signature field registry.
 *
 * The dispatcher caches bind groups while binding-relevant inputs are stable and
 * rebuilds when any binding field changes. These tests drive the real
 * dispatcher and assert:
 *   - changing a NON-binding field (frameSeed) does NOT rebuild;
 *   - changing EACH binding field DOES rebuild.
 * If a future edit drops a field from the registry, the corresponding
 * "changing X rebuilds" case fails (the change would be silently ignored).
 */

const DIMS: CascadeDim[] = [
  { probes: [1, 1, 1], rays: 16, intervalNear: 0, intervalFar: 4 },
  { probes: [1, 1, 1], rays: 64, intervalNear: 4, intervalFar: 16 },
];

function installWebGpuConstants(): void {
  vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_DST: 2 });
  vi.stubGlobal('GPUTextureUsage', { TEXTURE_BINDING: 1, COPY_DST: 2 });
  vi.stubGlobal('GPUShaderStage', { COMPUTE: 1 });
}

function makeBuffer(label: string): GPUBuffer {
  return { label, destroy: vi.fn() } as unknown as GPUBuffer;
}
function makeView(label: string): GPUTextureView {
  return { label } as unknown as GPUTextureView;
}
function makeSampler(label: string): GPUSampler {
  return { label } as unknown as GPUSampler;
}

function makeMockDevice() {
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
    createTexture: vi.fn(() => ({ createView: vi.fn(() => ({})), destroy: vi.fn() })),
    createBuffer: vi.fn((desc: { label?: string; size?: number }) => ({
      label: desc.label,
      size: desc.size,
      getMappedRange: () => new ArrayBuffer(Math.max(desc.size ?? 16, 16)),
      unmap: vi.fn(),
      destroy: vi.fn(),
    })),
    createCommandEncoder: vi.fn(() => ({
      beginComputePass: vi.fn(() => pass),
      finish: vi.fn(() => ({})),
    })),
    queue: { writeBuffer: vi.fn(), writeTexture: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;
  return { device, createBindGroup };
}

// A fully-populated opts (all optional binding fields present) so mutating any
// one of them is a value CHANGE, not a null→value transition.
function fullOpts(device: GPUDevice): RCDispatchOptsRaw {
  return {
    device,
    bvhNodesBuf: makeBuffer('bvh-nodes'),
    bvhIndicesBuf: makeBuffer('bvh-indices'),
    bvhPositionsBuf: makeBuffer('bvh-positions'),
    bvhNormalsBuf: makeBuffer('bvh-normals'),
    materialsBuf: makeBuffer('materials'),
    triMaterialIdBuf: makeBuffer('tri-mat-id'),
    cascadeBufs: [makeBuffer('cascade-0'), makeBuffer('cascade-1')],
    probeOriginWorld: [0, 0, 0],
    roomSize: [1, 1, 1],
    sunDirection: [0, 1, 0],
    sunColor: [1, 1, 1],
    frameSeed: 1,
    bvhMode: 'tlas',
    tlasNodeCount: 1,
    tlasNodesBuf: makeBuffer('tlas-nodes'),
    tlasInstanceIndicesBuf: makeBuffer('tlas-inst'),
    tlasBlasRootsBuf: makeBuffer('tlas-blas'),
    tlasInstanceWorldToLocalBuf: makeBuffer('tlas-w2l'),
    tlasInstanceLocalToWorldBuf: makeBuffer('tlas-l2w'),
    envTextureView: makeView('env'),
    envSampler: makeSampler('env-sampler'),
    materialTextureAtlasView: makeView('atlas'),
    materialMapMetaTextureView: makeView('meta'),
    bvhTangentTextureView: makeView('tangent'),
    bvhVertexColorTextureView: makeView('vcolor'),
    emittersBuf: makeBuffer('emitters'),
    lightsBuf: makeBuffer('lights'),
  };
}

// For each binding field: a mutation that produces a DIFFERENT value.
const MUTATIONS: Array<[string, (o: RCDispatchOptsRaw) => Partial<RCDispatchOptsRaw>]> = [
  ['bvhMode', () => ({ bvhMode: 'merged' })],
  ['bvhNodesBuf', () => ({ bvhNodesBuf: makeBuffer('bvh-nodes-b') })],
  ['bvhIndicesBuf', () => ({ bvhIndicesBuf: makeBuffer('bvh-indices-b') })],
  ['bvhPositionsBuf', () => ({ bvhPositionsBuf: makeBuffer('bvh-positions-b') })],
  ['bvhNormalsBuf', () => ({ bvhNormalsBuf: makeBuffer('bvh-normals-b') })],
  ['materialsBuf', () => ({ materialsBuf: makeBuffer('materials-b') })],
  ['triMaterialIdBuf', () => ({ triMaterialIdBuf: makeBuffer('tri-mat-id-b') })],
  ['cascadeBufs', () => ({ cascadeBufs: [makeBuffer('cascade-0b'), makeBuffer('cascade-1b')] })],
  ['probeOriginWorld', () => ({ probeOriginWorld: [9, 0, 0] })],
  ['roomSize', () => ({ roomSize: [9, 1, 1] })],
  ['envTextureView', () => ({ envTextureView: makeView('env-b') })],
  ['envSampler', () => ({ envSampler: makeSampler('env-sampler-b') })],
  ['materialTextureAtlasView', () => ({ materialTextureAtlasView: makeView('atlas-b') })],
  ['materialMapMetaTextureView', () => ({ materialMapMetaTextureView: makeView('meta-b') })],
  ['bvhTangentTextureView', () => ({ bvhTangentTextureView: makeView('tangent-b') })],
  ['bvhVertexColorTextureView', () => ({ bvhVertexColorTextureView: makeView('vcolor-b') })],
  ['tlasNodesBuf', () => ({ tlasNodesBuf: makeBuffer('tlas-nodes-b') })],
  ['tlasInstanceIndicesBuf', () => ({ tlasInstanceIndicesBuf: makeBuffer('tlas-inst-b') })],
  ['tlasBlasRootsBuf', () => ({ tlasBlasRootsBuf: makeBuffer('tlas-blas-b') })],
  ['tlasInstanceWorldToLocalBuf', () => ({ tlasInstanceWorldToLocalBuf: makeBuffer('tlas-w2l-b') })],
  ['tlasInstanceLocalToWorldBuf', () => ({ tlasInstanceLocalToWorldBuf: makeBuffer('tlas-l2w-b') })],
  ['emittersBuf', () => ({ emittersBuf: makeBuffer('emitters-b') })],
  ['lightsBuf', () => ({ lightsBuf: makeBuffer('lights-b') })],
];

describe('binding-signature registry behavior', () => {
  it('does NOT rebuild when only a non-binding field (frameSeed) changes', () => {
    installWebGpuConstants();
    const { device, createBindGroup } = makeMockDevice();
    const dispatcher = new RCDispatcher(DIMS);
    const opts = fullOpts(device);
    dispatcher.dispatchFrameRaw(opts);
    const after1 = createBindGroup.mock.calls.length;
    dispatcher.dispatchFrameRaw({ ...opts, frameSeed: 99 });
    expect(createBindGroup.mock.calls.length).toBe(after1);
  });

  for (const [name, mutate] of MUTATIONS) {
    it(`rebuilds when binding field '${name}' changes`, () => {
      installWebGpuConstants();
      const { device, createBindGroup } = makeMockDevice();
      const dispatcher = new RCDispatcher(DIMS);
      const opts = fullOpts(device);
      dispatcher.dispatchFrameRaw(opts);
      const after1 = createBindGroup.mock.calls.length;
      dispatcher.dispatchFrameRaw({ ...opts, ...mutate(opts), frameSeed: 2 });
      expect(createBindGroup.mock.calls.length).toBeGreaterThan(after1);
    });
  }
});
