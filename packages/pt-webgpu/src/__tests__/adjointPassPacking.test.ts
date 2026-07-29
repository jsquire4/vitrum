import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import {
  asMat4,
  type FrameInput,
  type Scene,
  type SkinnedMeshPrimitive,
} from '@vitrum/core';
import {
  AdjointPass,
  buildAdjointWorldSpaceGeometryOverride,
} from '../adjointPass.js';
import type { AdjointGradientRequest } from '../inverse/inverseSession.js';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';
import {
  ADJOINT_FIELD_EMISSIVE,
  ADJOINT_PARAMS_UBO_BYTES,
} from '../wgsl/pathTrace/adjointPass.wgsl.js';

interface FakeBuffer {
  readonly size: number;
  readonly bytes: ArrayBuffer;
  destroyed: boolean;
  destroy(): void;
  mapAsync(): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
}

interface BufferWrite {
  readonly buffer: FakeBuffer;
  readonly bytes: Uint8Array;
}

function installWebGpuConstants(): () => void {
  const global = globalThis as unknown as {
    GPUBufferUsage?: Record<string, number>;
    GPUMapMode?: Record<string, number>;
  };
  const previousUsage = global.GPUBufferUsage;
  const previousMapMode = global.GPUMapMode;
  global.GPUBufferUsage = {
    UNIFORM: 1,
    COPY_DST: 2,
    STORAGE: 4,
    COPY_SRC: 8,
    MAP_READ: 16,
  };
  global.GPUMapMode = { READ: 1 };
  return () => {
    if (previousUsage == null) delete global.GPUBufferUsage;
    else global.GPUBufferUsage = previousUsage;
    if (previousMapMode == null) delete global.GPUMapMode;
    else global.GPUMapMode = previousMapMode;
  };
}

function fakeGpu(): {
  readonly device: GPUDevice;
  readonly writes: BufferWrite[];
  readonly bindEntries: number[][];
  readonly calls: { shaderModules: number; buffers: number };
  externalBuffer(): GPUBuffer;
} {
  const writes: BufferWrite[] = [];
  const bindEntries: number[][] = [];
  const calls = { shaderModules: 0, buffers: 0 };
  const makeBuffer = (size = 16): FakeBuffer => ({
    size,
    bytes: new ArrayBuffer(size),
    destroyed: false,
    destroy() {
      this.destroyed = true;
    },
    async mapAsync() {},
    getMappedRange() {
      return this.bytes;
    },
    unmap() {},
  });
  const device = {
    limits: {
      maxBufferSize: 0xffff_ffff,
      maxStorageBufferBindingSize: 0xffff_ffff,
      maxComputeWorkgroupsPerDimension: 0xffff_ffff,
    },
    queue: {
      writeBuffer(
        buffer: FakeBuffer,
        offset: number,
        source: ArrayBuffer,
        sourceOffset = 0,
        size?: number,
      ) {
        const byteLength = size ?? source.byteLength - sourceOffset;
        const bytes = new Uint8Array(source, sourceOffset, byteLength).slice();
        new Uint8Array(buffer.bytes, offset, byteLength).set(bytes);
        writes.push({ buffer, bytes });
      },
      submit() {},
    },
    createShaderModule() {
      calls.shaderModules += 1;
      return {};
    },
    createComputePipeline() {
      return { getBindGroupLayout: () => ({}) };
    },
    createBuffer({ size }: GPUBufferDescriptor) {
      calls.buffers += 1;
      return makeBuffer(Number(size));
    },
    createBindGroup({ entries }: GPUBindGroupDescriptor) {
      bindEntries.push(Array.from(entries, (entry) => entry.binding));
      return {};
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        copyBufferToBuffer() {},
        finish: () => ({}),
      };
    },
  } as unknown as GPUDevice;
  return {
    device,
    writes,
    bindEntries,
    calls,
    externalBuffer: () => makeBuffer() as unknown as GPUBuffer,
  };
}

function frame(): FrameInput {
  const identity = asMat4(new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]));
  return {
    viewMatrix: identity,
    projMatrix: identity,
    cameraPosition: [0, 0, 0],
    viewport: { width: 1, height: 1, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
  };
}

function unlitScene(material: Record<string, unknown> = {}): Scene {
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
        emissive: [0.25, 0.5, 1],
        emissiveIntensity: 4,
        ...material,
      },
    }],
    emitters: [],
    environment: { kind: 'none' },
  } as unknown as Scene;
}

function request(): AdjointGradientRequest {
  return {
    width: 1,
    height: 1,
    channels: 3,
    samples: 2,
    dLoss_dRendered: new Float32Array([1, -2, 3]),
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

function uploaded(externalBuffer: () => GPUBuffer): UploadedSceneBuffers {
  return {
    triangleCount: 1,
    positionsBuffer: externalBuffer(),
    indicesBuffer: externalBuffer(),
    triMaterialIdsBuffer: externalBuffer(),
    materialsBuffer: externalBuffer(),
  } as unknown as UploadedSceneBuffers;
}

describe('AdjointPass emissive-only packing and preflight', () => {
  let restoreConstants: () => void;

  beforeAll(() => {
    restoreConstants = installWebGpuConstants();
  });

  afterAll(() => {
    restoreConstants();
  });

  it('packs only the eight certified bindings and one emissive descriptor', async () => {
    const gpu = fakeGpu();
    await new AdjointPass(gpu.device).computeGradient(
      request(),
      uploaded(gpu.externalBuffer),
      frame(),
      unlitScene(),
      () => 0,
      true,
    );

    expect(gpu.bindEntries).toEqual([[0, 1, 2, 3, 4, 5, 6, 7]]);
    expect(gpu.writes[0]?.bytes.byteLength).toBe(ADJOINT_PARAMS_UBO_BYTES);
    const uniformF32 = new Float32Array(gpu.writes[0]!.bytes.buffer);
    expect(uniformF32[26]).toBeGreaterThan(0);
    const descriptor = new Uint32Array(gpu.writes[3]!.bytes.buffer);
    expect(Array.from(descriptor.slice(0, 3))).toEqual([
      0,
      ADJOINT_FIELD_EMISSIVE,
      0,
    ]);
    expect(new Float32Array(descriptor.buffer)[3]).toBe(4);
  });

  it('rejects camera-hidden primitive emission before GPU creation', async () => {
    const gpu = fakeGpu();
    await expect(new AdjointPass(gpu.device).computeGradient(
      request(),
      uploaded(gpu.externalBuffer),
      frame(),
      unlitScene(),
      () => 0,
      false,
    )).rejects.toThrow(/cameraVisibleEmitters must be true/);
    expect(gpu.calls).toEqual({ shaderModules: 0, buffers: 0 });
  });

  it('flattens transformed triangle geometry and preserves material ids', () => {
    const scene = unlitScene();
    const primitive = scene.primitives[0]!;
    if (primitive.kind !== 'mesh') throw new Error('fixture');
    const transformed: Scene = {
      ...scene,
      primitives: [{
        ...primitive,
        transform: asMat4(new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          2, 3, 4, 1,
        ])),
      }],
    };
    const override = buildAdjointWorldSpaceGeometryOverride(
      transformed,
      () => 7,
    );
    expect(override?.triangleCount).toBe(1);
    expect(Array.from(override!.positions.slice(0, 4))).toEqual([1, 2, 3, 0]);
    expect(Array.from(override!.indices)).toEqual([0, 1, 2, 0]);
    expect(Array.from(override!.triMaterialIds)).toEqual([7]);
  });

  it('solves skinning and morph targets before applying the primitive transform', () => {
    const identity = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
    const boneTranslation = new Float32Array(identity);
    boneTranslation[12] = 5;
    const primitiveTransform = asMat4(new Float32Array(identity));
    primitiveTransform[12] = 2;
    const skinWeights = new Float32Array(12);
    skinWeights[0] = 1;
    skinWeights[4] = 1;
    skinWeights[8] = 1;
    const skinned: SkinnedMeshPrimitive = {
      kind: 'skinned-mesh',
      id: 'emitter',
      positions: new Float32Array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
      ]),
      normals: new Float32Array([
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
      ]),
      indices: new Uint32Array([0, 1, 2]),
      skinIndices: new Uint32Array(12),
      skinWeights,
      bones: boneTranslation,
      boneInverses: identity,
      morphTargets: [new Float32Array([
        1, 0, 0,
        0, 0, 0,
        0, 0, 0,
      ])],
      morphWeights: new Float32Array([1]),
      material: unlitScene().primitives[0]!.material,
      transform: primitiveTransform,
    };
    const override = buildAdjointWorldSpaceGeometryOverride({
      primitives: [skinned],
      emitters: [],
      environment: { kind: 'none' },
    }, () => 0);

    // rest x=0 + morph x=1 + bone x=5 + primitive transform x=2
    expect(override?.positions[0]).toBeCloseTo(8, 6);
    expect(override?.positions[1]).toBeCloseTo(0, 6);
    expect(override?.positions[2]).toBeCloseTo(0, 6);
  });

  it.each([
    {
      name: 'lit implicit-emitter receiver',
      scene: () => unlitScene({ shadingModel: 'standard' }),
      message: /implicit emissive-mesh NEE/,
    },
    {
      name: 'transmission',
      scene: () => unlitScene({ transmission: 0.5 }),
      message: /transmission changes camera transport/,
    },
    {
      name: 'emissive map',
      scene: () => unlitScene({ emissiveMap: {} }),
      message: /emissive map/,
    },
    {
      name: 'clearcoat attenuation',
      scene: () => unlitScene({ clearcoat: 0.5 }),
      message: /clearcoat emission attenuation/,
    },
    {
      name: 'cross-primitive equal-distance tie',
      scene: () => {
        const base = unlitScene();
        return {
          ...base,
          primitives: [
            base.primitives[0]!,
            {
              ...base.primitives[0]!,
              id: 'overlapping-occluder',
            },
          ],
        };
      },
      message: /equal-distance material tie/,
    },
    {
      name: 'mesh-area emitter folding',
      scene: (): Scene => ({
        ...unlitScene(),
        emitters: [{
          kind: 'mesh-area',
          id: 'fold',
          meshId: 'emitter',
          color: [1, 1, 1],
          intensity: 1,
        }],
        }),
      message: /folds .* emission/,
    },
    {
      name: 'analytic primitive',
      scene: () => ({
        primitives: [{
          kind: 'analytic',
          id: 'emitter',
          shape: 'sphere',
          params: { center: [0, 0, -1], radius: 1 },
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
      } as unknown as Scene),
      message: /analytic primitive/,
    },
    {
      name: 'singular primitive transform',
      scene: (): Scene => {
        const scene = unlitScene();
        const primitive = scene.primitives[0]!;
        if (primitive.kind !== 'mesh') throw new Error('fixture');
        return {
          ...scene,
          primitives: [{
            ...primitive,
            transform: asMat4(new Float32Array([
              0, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 1,
            ])),
          }],
          };
      },
      message: /non-invertible replay transform/,
    },
    {
      name: 'singular instance transform',
      scene: (): Scene => {
        const scene = unlitScene();
        const primitive = scene.primitives[0]!;
        if (primitive.kind !== 'mesh') throw new Error('fixture');
        const {
          transform: _transform,
          ...mesh
        } = primitive;
        return {
          ...scene,
          primitives: [{
            ...mesh,
            kind: 'instanced-mesh',
            instances: [asMat4(new Float32Array([
              1, 0, 0, 0,
              0, 0, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 1,
            ]))],
          }],
          };
      },
      message: /non-invertible replay transform/,
    },
  ])('rejects $name before any GPU creation', async ({ scene, message }) => {
    const gpu = fakeGpu();
    await expect(new AdjointPass(gpu.device).computeGradient(
      request(),
      uploaded(gpu.externalBuffer),
      frame(),
      scene(),
      () => 0,
      true,
    )).rejects.toThrow(message);
    expect(gpu.calls).toEqual({ shaderModules: 0, buffers: 0 });
  });

  it.each([
    {
      name: 'width outside u32',
      req: () => ({ ...request(), width: 0x1_0000_0000 }),
      message: /width must be a positive u32/,
    },
    {
      name: 'sample count outside u32',
      req: () => ({ ...request(), samples: 0x1_0000_0000 }),
      message: /samples must be a positive u32/,
    },
    {
      name: 'sample count above the session limit',
      req: () => ({ ...request(), samples: 4097 }),
      message: /samples exceeds the inverse-session limit 4096/,
    },
    {
      name: 'gradient length outside u32',
      req: () => ({ ...request(), gradientLength: 0x1_0000_0000 }),
      message: /gradientLength must be a positive u32/,
    },
    {
      name: 'pixel product outside u32',
      req: () => ({ ...request(), width: 65536, height: 65536 }),
      message: /pixel count exceeds the addressable u32 resource domain/,
    },
  ])('rejects $name before any GPU creation', async ({ req, message }) => {
    const gpu = fakeGpu();
    await expect(new AdjointPass(gpu.device).computeGradient(
      req(),
      uploaded(gpu.externalBuffer),
      frame(),
      unlitScene(),
      () => 0,
      true,
    )).rejects.toThrow(message);
    expect(gpu.calls).toEqual({ shaderModules: 0, buffers: 0 });
  });

  it.each([
    {
      name: 'storage binding limit',
      limit: 'maxStorageBufferBindingSize',
      value: 8,
      message: /storage resource exceeds this device's limits/,
      req: () => request(),
    },
    {
      name: 'dispatch workgroup limit',
      limit: 'maxComputeWorkgroupsPerDimension',
      value: 1,
      message: /compute-workgroup limit/,
      req: () => ({
        ...request(),
        width: 9,
        dLoss_dRendered: new Float32Array(9 * 3),
      }),
    },
  ] as const)('rejects the device $name before GPU creation', async ({
    limit,
    value,
    message,
    req,
  }) => {
    const gpu = fakeGpu();
    (
      gpu.device.limits as unknown as Record<string, number>
    )[limit] = value;
    await expect(new AdjointPass(gpu.device).computeGradient(
      req(),
      uploaded(gpu.externalBuffer),
      frame(),
      unlitScene(),
      () => 0,
      true,
    )).rejects.toThrow(message);
    expect(gpu.calls).toEqual({ shaderModules: 0, buffers: 0 });
  });
});
