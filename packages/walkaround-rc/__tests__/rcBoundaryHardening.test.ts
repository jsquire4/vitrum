import { describe, expect, it, vi } from 'vitest';
import {
  RC_DEFAULT_TRANSMITTED_INTERFACE_BUDGET,
  RC_MAX_TRANSMITTED_INTERFACE_BUDGET,
  RC_MIN_TRANSMITTED_INTERFACE_BUDGET,
  RCDispatcher,
  validateCascadeDims,
  type CascadeDim,
  type RCDispatchOptsRaw,
} from '../src/index.js';
import { allocateCascades } from './support/cascadeBuffers.js';
import {
  computeOctahedralSolidAngles,
  MAX_OCTAHEDRAL_SOLID_ANGLE_GRID_SIZE,
} from './support/octahedralSolidAngles.js';
import {
  buildCascadeUniformDataInto,
  type CascadeUniformInputs,
} from '../src/cascadeDispatch.js';

const DIMS: readonly CascadeDim[] = [
  { probes: [1, 1, 1], rays: 16, intervalNear: 0, intervalFar: 4 },
];

const UNIFORMS: CascadeUniformInputs = {
  probeOriginWorld: [0, 0, 0],
  roomSize: [1, 2, 3],
  sunDir: [0, 1, 0],
  sunColor: [1, 2, 3],
  sunCastShadowDisabled: false,
  sunAngularRadius: 0,
  envIntensity: 1,
  scalarSkyRadiance: [0.25, 0.5, 0.75],
  hasDirectionalEnvironment: false,
  frameSeed: 1,
  triIntersectEpsilon: 1e-5,
  bvhMode: 0,
  tlasNodeCount: 0,
  emitterCount: 0,
  lightCount: 0,
  dims: DIMS,
};

function installWebGpuConstants(): void {
  vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8 });
  vi.stubGlobal('GPUTextureUsage', { TEXTURE_BINDING: 1, COPY_DST: 2 });
  vi.stubGlobal('GPUShaderStage', { COMPUTE: 1 });
}

function buffer(label: string, size: number, usage = 5): GPUBuffer {
  return { label, size, usage, destroy: vi.fn() } as unknown as GPUBuffer;
}

function inertDevice() {
  const mutation = vi.fn();
  const device = {
    limits: {
      maxStorageBuffersPerShaderStage: 8,
      maxSampledTexturesPerShaderStage: 16,
      maxSamplersPerShaderStage: 16,
      maxUniformBufferBindingSize: 65_536,
      maxStorageBufferBindingSize: 1 << 28,
      maxBufferSize: 1 << 28,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupsPerDimension: 65_535,
      minStorageBufferOffsetAlignment: 256,
    },
    createShaderModule: mutation,
    createBindGroupLayout: mutation,
    createPipelineLayout: mutation,
    createComputePipeline: mutation,
    createBindGroup: mutation,
    createSampler: mutation,
    createTexture: mutation,
    createBuffer: mutation,
    createCommandEncoder: mutation,
    queue: { writeBuffer: mutation, writeTexture: mutation, submit: mutation },
  } as unknown as GPUDevice;
  return { device, mutation };
}

function validDispatch(device: GPUDevice): RCDispatchOptsRaw {
  return {
    device,
    bvhNodesBuf: buffer('nodes', 32),
    bvhIndicesBuf: buffer('indices', 16),
    bvhPositionsBuf: buffer('positions', 48),
    bvhNormalsBuf: buffer('normals', 48),
    materialsBuf: buffer('materials', 64),
    triMaterialIdBuf: buffer('tri-material', 4),
    cascadeBufs: [buffer('cascade', 256)],
    probeOriginWorld: [0, 0, 0],
    roomSize: [1, 1, 1],
    sunDirection: [0, 1, 0],
    sunColor: [1, 1, 1],
    frameSeed: 1,
  };
}

describe('RC CPU allocation boundaries', () => {
  it('rejects unsafe dimensions before any typed-array allocation', () => {
    expect(() => validateCascadeDims([
      { probes: [Number.MAX_SAFE_INTEGER, 1, 1], rays: 16, intervalNear: 0, intervalFar: 1 },
    ])).toThrow(/positive integer/);
    expect(() => validateCascadeDims([
      { probes: [65_536, 65_536, 1], rays: 16, intervalNear: 0, intervalFar: 1 },
    ])).toThrow(/vec4 indexing limit/);
    expect(() => validateCascadeDims([
      { probes: [1, 1, 1], rays: 16, intervalNear: 0, intervalFar: Number.MAX_VALUE },
    ])).toThrow(/finite number/);
  });

  it('rejects invalid AABBs and over-budget CPU storage before allocating', () => {
    expect(() => allocateCascades({ min: [0, 0, 0], max: [0, 1, 1] }, DIMS)).toThrow(/positive/);
    expect(() => allocateCascades({ min: [0, 0, 0], max: [Number.NaN, 1, 1] }, DIMS)).toThrow(/finite/);
    const huge: readonly CascadeDim[] = [
      { probes: [4096, 4096, 1], rays: 16, intervalNear: 0, intervalFar: 1 },
    ];
    expect(() => allocateCascades({ min: [0, 0, 0], max: [1, 1, 1] }, huge)).toThrow(/more than/);
  });

  it('bounds octahedral integration work before its O(N² × SUB²) loop', () => {
    for (const invalid of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => computeOctahedralSolidAngles(invalid)).toThrow(/positive safe integer/);
    }
    expect(() => computeOctahedralSolidAngles(MAX_OCTAHEDRAL_SOLID_ANGLE_GRID_SIZE + 1))
      .toThrow(/validated maximum/);
  });
});

describe('CascadeUniforms strict pack boundary', () => {
  it('honors destination byteOffset and clears only the 40-word destination', () => {
    const backing = new Float32Array(48).fill(99);
    const destination = backing.subarray(4, 44);
    buildCascadeUniformDataInto(destination, 0, UNIFORMS);
    expect(Array.from(backing.subarray(0, 4))).toEqual([99, 99, 99, 99]);
    expect(Array.from(backing.subarray(44))).toEqual([99, 99, 99, 99]);
    const words = new Uint32Array(backing.buffer, destination.byteOffset, destination.length);
    const floats = new Float32Array(backing.buffer, destination.byteOffset, destination.length);
    expect(words[24]).toBe(1);
    expect(words[27]).toBe(0);
    expect(words[34]).toBe(RC_DEFAULT_TRANSMITTED_INTERFACE_BUDGET);
    expect(floats[35]).toBe(0);
    expect(Array.from(floats.slice(36, 39))).toEqual([0.25, 0.5, 0.75]);
    expect(words[39]).toBe(0);
  });

  it('packs every supported transmitted-interface budget into UBO word 34', () => {
    expect(RC_MIN_TRANSMITTED_INTERFACE_BUDGET).toBe(1);
    expect(RC_MAX_TRANSMITTED_INTERFACE_BUDGET).toBe(8);
    for (let budget = 1; budget <= 8; budget += 1) {
      const destination = new Float32Array(40);
      buildCascadeUniformDataInto(destination, 0, {
        ...UNIFORMS,
        transmittedInterfaceBudget: budget,
      });
      expect(new Uint32Array(destination.buffer)[34]).toBe(budget);
    }
  });

  it.each([
    ['short destination', () => buildCascadeUniformDataInto(new Float32Array(39), 0, UNIFORMS)],
    ['fractional index', () => buildCascadeUniformDataInto(new Float32Array(40), 0.5, UNIFORMS)],
    ['zero room extent', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, roomSize: [1, 0, 1] })],
    ['unnormalized sun', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, sunDir: [0, 2, 0] })],
    ['negative sun color', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, sunColor: [-1, 1, 1] })],
    ['negative environment intensity', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, envIntensity: -1 })],
    ['underflowing positive environment intensity', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, envIntensity: 2 ** -150 })],
    ['non-finite environment rotation', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, envRotationY: Number.NaN })],
    ['negative scalar sky radiance', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, scalarSkyRadiance: [1, -1, 1] })],
    ['non-finite scalar sky radiance', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, scalarSkyRadiance: [1, Number.POSITIVE_INFINITY, 1] })],
    ['fully underflowing positive scalar sky radiance', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, scalarSkyRadiance: [2 ** -150, 0, 0] })],
    ['non-boolean directional environment flag', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, hasDirectionalEnvironment: 1 as unknown as boolean })],
    ['fractional frame seed', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, frameSeed: 1.5 })],
    ['unknown numeric BVH mode', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, bvhMode: 2 })],
    ['light count outside u32', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, lightCount: 0x1_0000_0000 })],
    ['zero transmitted-interface budget', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, transmittedInterfaceBudget: 0 })],
    ['oversized transmitted-interface budget', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, transmittedInterfaceBudget: 9 })],
    ['fractional transmitted-interface budget', () => buildCascadeUniformDataInto(new Float32Array(40), 0, { ...UNIFORMS, transmittedInterfaceBudget: 1.5 })],
  ])('rejects %s', (_label, run) => expect(run).toThrow());
});

describe('dispatchFrameRaw preflight validation', () => {
  it('rejects malformed contracts and unsupported limits before any GPU mutation', () => {
    installWebGpuConstants();
    const { device, mutation } = inertDevice();
    const base = validDispatch(device);
    const invalid: RCDispatchOptsRaw[] = [
      { ...base, envTextureView: {} as GPUTextureView },
      { ...base, materialTextureAtlasView: {} as GPUTextureView },
      { ...base, bvhNodesOffset: 4, bvhNodesSize: 32 },
      { ...base, bvhNodesSize: 31 },
      {
        ...base,
        bvhNodesBuf: buffer('nodes-arena', 512),
        bvhNodesOffset: 256,
        bvhNodesSize: 288,
      },
      { ...base, bvhPositionsSize: 32, bvhNormalsSize: 48 },
      { ...base, cascadeBufs: [buffer('small-cascade', 240)] },
      { ...base, frameSeed: 1.5 },
      { ...base, envIntensity: -1 },
      { ...base, envIntensity: 2 ** -150 },
      { ...base, envRotationY: Number.POSITIVE_INFINITY },
      { ...base, scalarSkyRadiance: [1, -1, 1] },
      { ...base, scalarSkyRadiance: [1, Number.NaN, 1] },
      { ...base, scalarSkyRadiance: [2 ** -150, 0, 0] },
      { ...base, hasDirectionalEnvironment: 1 as unknown as boolean },
      { ...base, hasDirectionalEnvironment: true },
      { ...base, transmittedInterfaceBudget: 0 },
      { ...base, transmittedInterfaceBudget: 9 },
      { ...base, transmittedInterfaceBudget: 1.5 },
      { ...base, materialArenaVersion: -1 },
      { ...base, roomSize: [1, 0, 1] },
      { ...base, sunDirection: [0, 2, 0] },
      { ...base, emitterCount: 1 },
      { ...base, lightCount: 17 },
      { ...base, bvhMode: 'tlas', tlasNodeCount: 1 },
      { ...base, bvhMode: 'bogus' as 'merged' },
    ];
    const dispatcher = new RCDispatcher(DIMS);
    for (const opts of invalid) expect(() => dispatcher.dispatchFrameRaw(opts)).toThrow();
    expect(mutation).not.toHaveBeenCalled();
  });

  it('validates emitter ranges, light capacity, and device limits before mutation', () => {
    installWebGpuConstants();
    const { device, mutation } = inertDevice();
    const base = validDispatch(device);
    const dispatcher = new RCDispatcher(DIMS);
    expect(() => dispatcher.dispatchFrameRaw({
      ...base,
      emittersBuf: buffer('emitters', 160),
      emittersOffset: 80,
      emittersSize: 80,
      emitterCount: 1,
    })).toThrow(/minStorageBufferOffsetAlignment/);
    expect(() => dispatcher.dispatchFrameRaw({
      ...base,
      lightsBuf: buffer('lights', 1024),
      lightCount: 1,
    })).toThrow(/exact 96-byte bound range/);
    (device.limits as unknown as { maxStorageBuffersPerShaderStage: number })
      .maxStorageBuffersPerShaderStage = 7;
    expect(() => dispatcher.dispatchFrameRaw(base)).toThrow(/maxStorageBuffersPerShaderStage/);
    expect(mutation).not.toHaveBeenCalled();
  });
});
