import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { asMat4, type FrameInput, type Scene } from '@vitrum/core';
import { AdjointPass } from '../adjointPass.js';
import type { AdjointGradientRequest } from '../inverse/inverseSession.js';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';

type FailurePoint =
  | 'createBuffer'
  | 'writeBuffer'
  | 'getBindGroupLayout'
  | 'createBindGroup'
  | 'createCommandEncoder'
  | 'beginComputePass'
  | 'setPipeline'
  | 'setBindGroup'
  | 'dispatchWorkgroups'
  | 'end'
  | 'copyBufferToBuffer'
  | 'finish'
  | 'submit'
  | 'mapAsync'
  | 'getMappedRange'
  | 'sliceMappedRange'
  | 'unmap';

interface FailureConfig {
  readonly point?: FailurePoint;
  readonly at?: number;
  readonly destroyFailureIndex?: number;
}

class TrackedBuffer {
  destroyCalls = 0;
  unmapCalls = 0;
  mapped = false;

  constructor(
    readonly size: number,
    readonly ownedIndex: number | null,
    private readonly owner: FailureDevice,
  ) {}

  destroy(): void {
    this.destroyCalls += 1;
    if (this.ownedIndex === this.owner.destroyFailureIndex) {
      throw new Error('injected destroy');
    }
  }

  async mapAsync(): Promise<void> {
    this.owner.hit('mapAsync');
    this.mapped = true;
  }

  getMappedRange(): ArrayBuffer {
    this.owner.hit('getMappedRange');
    const bytes = new ArrayBuffer(this.size);
    return {
      slice: () => {
        this.owner.hit('sliceMappedRange');
        return bytes.slice(0);
      },
    } as unknown as ArrayBuffer;
  }

  unmap(): void {
    this.unmapCalls += 1;
    this.owner.hit('unmap');
    this.mapped = false;
  }
}

class FailureDevice {
  readonly ownedBuffers: TrackedBuffer[] = [];
  readonly borrowedBuffers: TrackedBuffer[] = [];
  readonly destroyFailureIndex: number | null;
  readonly device: GPUDevice;
  private readonly point: FailurePoint | undefined;
  private readonly failAt: number;
  private readonly calls = new Map<FailurePoint, number>();

  constructor(config: FailureConfig = {}) {
    this.point = config.point;
    this.failAt = config.at ?? 1;
    this.destroyFailureIndex = config.destroyFailureIndex ?? null;

    this.device = {
      limits: {
        maxBufferSize: 0xffff_ffff,
        maxStorageBufferBindingSize: 0xffff_ffff,
        maxComputeWorkgroupsPerDimension: 0xffff_ffff,
      },
      queue: {
        writeBuffer: () => {
          this.hit('writeBuffer');
        },
        submit: () => {
          this.hit('submit');
        },
      },
      createShaderModule: () => ({}),
      createComputePipeline: () => ({
        getBindGroupLayout: () => {
          this.hit('getBindGroupLayout');
          return {};
        },
      }),
      createBuffer: (descriptor: GPUBufferDescriptor) => {
        this.hit('createBuffer');
        const buffer = new TrackedBuffer(Number(descriptor.size), this.ownedBuffers.length, this);
        this.ownedBuffers.push(buffer);
        return buffer;
      },
      createBindGroup: () => {
        this.hit('createBindGroup');
        return {};
      },
      createCommandEncoder: () => {
        this.hit('createCommandEncoder');
        return {
          beginComputePass: () => {
            this.hit('beginComputePass');
            return {
              setPipeline: () => {
                this.hit('setPipeline');
              },
              setBindGroup: () => {
                this.hit('setBindGroup');
              },
              dispatchWorkgroups: () => {
                this.hit('dispatchWorkgroups');
              },
              end: () => {
                this.hit('end');
              },
            };
          },
          copyBufferToBuffer: () => {
            this.hit('copyBufferToBuffer');
          },
          finish: () => {
            this.hit('finish');
            return {};
          },
        };
      },
    } as unknown as GPUDevice;
  }

  hit(point: FailurePoint): void {
    const count = (this.calls.get(point) ?? 0) + 1;
    this.calls.set(point, count);
    if (this.point === point && count === this.failAt) {
      throw new Error(`injected ${point}`);
    }
  }

  makeBorrowedBuffer(size = 16): GPUBuffer {
    const buffer = new TrackedBuffer(size, null, this);
    this.borrowedBuffers.push(buffer);
    return buffer as unknown as GPUBuffer;
  }
}

function installWebGpuConstants(): () => void {
  const global = globalThis as unknown as {
    GPUBufferUsage?: Record<string, number>;
    GPUMapMode?: Record<string, number>;
  };
  const previousUsage = global.GPUBufferUsage;
  const previousMapMode = global.GPUMapMode;
  global.GPUBufferUsage = {
    UNIFORM: 1 << 0,
    COPY_DST: 1 << 1,
    STORAGE: 1 << 2,
    COPY_SRC: 1 << 3,
    MAP_READ: 1 << 4,
  };
  global.GPUMapMode = { READ: 1 };
  return () => {
    if (previousUsage === undefined) delete global.GPUBufferUsage;
    else global.GPUBufferUsage = previousUsage;
    if (previousMapMode === undefined) delete global.GPUMapMode;
    else global.GPUMapMode = previousMapMode;
  };
}

function frame(): FrameInput {
  const identity = asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]));
  return {
    viewMatrix: identity,
    projMatrix: identity,
    cameraPosition: [0, 0, 0],
    viewport: { width: 1, height: 1, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
  };
}

function scene(): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'emitter',
      positions: new Float32Array([
        -1, -1, -1,
        1, -1, -1,
        0, 1, -1,
      ]),
      normals: new Float32Array([
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
      ]),
      indices: new Uint32Array([0, 1, 2]),
      material: {
        baseColor: [0, 0, 0],
        roughness: 1,
        metallic: 0,
        shadingModel: 'unlit',
        emissive: [1, 1, 1],
      },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function request(): AdjointGradientRequest {
  return {
    width: 1,
    height: 1,
    channels: 3,
    samples: 1,
    dLoss_dRendered: new Float32Array([0, 0, 0]),
    params: [{
      domain: 'materials',
      id: 'emitter',
      field: 'emissive',
      offset: 0,
      length: 3,
    }],
    gradientLength: 3,
  };
}

function uploaded(stub: FailureDevice): UploadedSceneBuffers {
  const buffer = () => stub.makeBorrowedBuffer();
  return {
    triangleCount: 1,
    pointLightCount: 0,
    rectAreaLightCount: 0,
    directionalLightCount: 0,
    spotLightCount: 0,
    meshAreaLightCount: 0,
    environmentMapWidth: 0,
    environmentMapHeight: 0,
    hasEnvironmentMap: false,
    environmentHdriIntensity: 0,
    environmentHdriRotationY: 0,
    positionsBuffer: buffer(),
    indicesBuffer: buffer(),
    triMaterialIdsBuffer: buffer(),
    materialsBuffer: buffer(),
    normalsBuffer: buffer(),
    pointLightsBuffer: buffer(),
    rectAreaLightsBuffer: buffer(),
    directionalLightsBuffer: buffer(),
    spotLightsBuffer: buffer(),
    meshAreaLightsBuffer: buffer(),
    uvsBuffer: buffer(),
    materialTexDescriptorsBuffer: buffer(),
    materialTextureView: {},
    materialTextureSampler: {},
    materialLinearTextureView: {},
    colorsBuffer: buffer(),
    environmentMapTexelsBuffer: buffer(),
    environmentMapCdfBuffer: buffer(),
    tangentsBuffer: buffer(),
    materialEmissiveTextureView: {},
  } as unknown as UploadedSceneBuffers;
}

async function invoke(stub: FailureDevice): Promise<Float32Array> {
  return new AdjointPass(stub.device).computeGradient(
    request(),
    uploaded(stub),
    frame(),
    scene(),
    () => 0,
    true,
  );
}

function expectOwnedBuffersDestroyed(stub: FailureDevice): void {
  expect(stub.ownedBuffers.length).toBeGreaterThan(0);
  for (const buffer of stub.ownedBuffers) {
    expect(buffer.destroyCalls).toBe(1);
  }
  for (const buffer of stub.borrowedBuffers) {
    expect(buffer.destroyCalls).toBe(0);
  }
}

describe('AdjointPass transient lifetime', () => {
  let restoreWebGpuConstants: () => void;
  const failureCases: readonly FailureConfig[] = [
    { point: 'writeBuffer' },
    { point: 'createBuffer', at: 3 },
    { point: 'getBindGroupLayout' },
    { point: 'createBindGroup' },
    { point: 'createCommandEncoder' },
    { point: 'beginComputePass' },
    { point: 'setPipeline' },
    { point: 'setBindGroup' },
    { point: 'dispatchWorkgroups' },
    { point: 'end' },
    { point: 'copyBufferToBuffer' },
    { point: 'finish' },
    { point: 'submit' },
    { point: 'mapAsync' },
    { point: 'getMappedRange' },
    { point: 'sliceMappedRange' },
  ];

  beforeAll(() => {
    restoreWebGpuConstants = installWebGpuConstants();
  });

  afterAll(() => {
    restoreWebGpuConstants();
  });

  it.each(failureCases)(
    'destroys every owned buffer after a $point failure',
    async ({ point, at }) => {
      if (point == null) throw new Error('failure case must name a point');
      const stub = new FailureDevice(at == null ? { point } : { point, at });
      await expect(invoke(stub)).rejects.toThrow(`injected ${point}`);
      expectOwnedBuffersDestroyed(stub);

      if (point === 'getMappedRange' || point === 'sliceMappedRange') {
        expect(stub.ownedBuffers.at(-1)?.unmapCalls).toBe(1);
      }
    },
  );

  it('unmaps readback and destroys every owned buffer after successful decoding', async () => {
    const stub = new FailureDevice();
    await expect(invoke(stub)).resolves.toEqual(new Float32Array([0, 0, 0]));
    expectOwnedBuffersDestroyed(stub);
    expect(stub.ownedBuffers.at(-1)?.unmapCalls).toBe(1);
  });

  it('continues cleanup after destroy throws and reports that cleanup failure', async () => {
    const stub = new FailureDevice({ destroyFailureIndex: 0 });
    await expect(invoke(stub)).rejects.toThrow('injected destroy');
    expectOwnedBuffersDestroyed(stub);
  });

  it('preserves the primary operation error when destroy also throws', async () => {
    const stub = new FailureDevice({
      point: 'submit',
      destroyFailureIndex: 0,
    });
    await expect(invoke(stub)).rejects.toThrow('injected submit');
    expectOwnedBuffersDestroyed(stub);
  });

  it('continues destruction when best-effort unmap throws', async () => {
    const stub = new FailureDevice({ point: 'unmap' });
    await expect(invoke(stub)).rejects.toThrow('injected unmap');
    expectOwnedBuffersDestroyed(stub);
    expect(stub.ownedBuffers.at(-1)?.unmapCalls).toBe(1);
  });
});
