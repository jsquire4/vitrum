import { describe, expect, it } from 'vitest';
import { asMat4, type FrameInput, type Scene } from '@vitrum/core';
import { AdjointPass, buildAdjointWorldSpaceGeometryOverride } from '../adjointPass.js';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';
import type { AdjointGradientRequest } from '../inverse/inverseSession.js';
import {
  ADJOINT_EMITTER_TARGET_MESH,
  ADJOINT_FIELD_EMITTER_INTENSITY,
  ADJOINT_PARAMS_UBO_BYTES,
} from '../wgsl/pathTrace/adjointPass.wgsl.js';

interface FakeBuffer {
  readonly id: number;
  readonly size: number;
  readonly usage: number;
  destroyed: boolean;
  destroy(): void;
  mapAsync(): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
}

interface BufferWrite {
  readonly buffer: FakeBuffer;
  readonly byteLength: number;
  readonly data: ArrayBuffer;
}

interface FakeBindGroup {
  readonly entries: readonly {
    readonly binding: number;
    readonly resource: unknown;
  }[];
}

function installWebGpuConstants(): () => void {
  const g = globalThis as unknown as {
    GPUBufferUsage?: Record<string, number>;
    GPUMapMode?: Record<string, number>;
  };
  const prevUsage = g.GPUBufferUsage;
  const prevMapMode = g.GPUMapMode;
  g.GPUBufferUsage = {
    UNIFORM: 1 << 0,
    COPY_DST: 1 << 1,
    STORAGE: 1 << 2,
    COPY_SRC: 1 << 3,
    MAP_READ: 1 << 4,
  };
  g.GPUMapMode = { READ: 1 };
  return () => {
    if (prevUsage === undefined) {
      delete g.GPUBufferUsage;
    } else {
      g.GPUBufferUsage = prevUsage;
    }
    if (prevMapMode === undefined) {
      delete g.GPUMapMode;
    } else {
      g.GPUMapMode = prevMapMode;
    }
  };
}

function makeFakeDevice(): {
  readonly device: GPUDevice;
  readonly writes: BufferWrite[];
  readonly bindGroups: FakeBindGroup[];
  makeExternalBuffer(size?: number): FakeBuffer;
} {
  let nextId = 1;
  const writes: BufferWrite[] = [];
  const bindGroups: FakeBindGroup[] = [];

  const makeBuffer = (size = 16, usage = 0): FakeBuffer => ({
    id: nextId++,
    size,
    usage,
    destroyed: false,
    destroy() {
      this.destroyed = true;
    },
    async mapAsync() {
      return undefined;
    },
    getMappedRange() {
      return new ArrayBuffer(size);
    },
    unmap() {
      // no-op
    },
  });

  const device = {
    queue: {
      writeBuffer(
        buffer: FakeBuffer,
        _bufferOffset: number,
        source: ArrayBuffer,
        sourceOffset = 0,
        size?: number,
      ) {
        const byteLength = size ?? source.byteLength - sourceOffset;
        const bytes = new Uint8Array(source, sourceOffset, byteLength);
        writes.push({
          buffer,
          byteLength,
          data: bytes.slice().buffer,
        });
      },
      submit() {
        // no-op
      },
    },
    createShaderModule() {
      return {};
    },
    createComputePipeline() {
      return {
        getBindGroupLayout() {
          return {};
        },
      };
    },
    createBuffer(desc: { size: number; usage: number }) {
      return makeBuffer(desc.size, desc.usage);
    },
    createBindGroup(desc: FakeBindGroup) {
      bindGroups.push(desc);
      return desc;
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            setPipeline() {
              // no-op
            },
            setBindGroup() {
              // no-op
            },
            dispatchWorkgroups() {
              // no-op
            },
            end() {
              // no-op
            },
          };
        },
        copyBufferToBuffer() {
          // no-op
        },
        finish() {
          return {};
        },
      };
    },
  } as unknown as GPUDevice;

  return {
    device,
    writes,
    bindGroups,
    makeExternalBuffer: (size = 16) => makeBuffer(size, 0),
  };
}

function makeScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'panel',
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
        material: { baseColor: [1, 1, 1], roughness: 0.4, metallic: 0 },
      },
    ],
    emitters: [
      {
        kind: 'mesh-area',
        id: 'panel-light',
        meshId: 'panel',
        color: [0.25, 0.5, 1],
        intensity: 3,
      },
    ],
    environment: { kind: 'none' },
  };
}

function makeFrame(): FrameInput {
  const identity = asMat4(new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]));
  return {
    viewMatrix: identity,
    projMatrix: identity,
    cameraPosition: [0, 0, 4],
    viewport: { width: 1, height: 1, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
  };
}

function makeUploadedSceneBuffers(makeBuffer: (size?: number) => FakeBuffer): UploadedSceneBuffers {
  const b = (size = 16) => makeBuffer(size) as unknown as GPUBuffer;
  return {
    triangleCount: 1,
    pointLightCount: 0,
    rectAreaLightCount: 0,
    directionalLightCount: 0,
    spotLightCount: 0,
    meshAreaLightCount: 99,
    environmentMapWidth: 0,
    environmentMapHeight: 0,
    hasEnvironmentMap: false,
    environmentHdriIntensity: 0,
    environmentHdriRotationY: 0,
    positionsBuffer: b(),
    indicesBuffer: b(),
    triMaterialIdsBuffer: b(),
    materialsBuffer: b(),
    normalsBuffer: b(),
    pointLightsBuffer: b(),
    rectAreaLightsBuffer: b(),
    directionalLightsBuffer: b(),
    spotLightsBuffer: b(),
    meshAreaLightsBuffer: b(),
    uvsBuffer: b(),
    materialTexDescriptorsBuffer: b(),
    materialTextureView: {},
    materialTextureSampler: {},
    materialLinearTextureView: {},
    colorsBuffer: b(),
    environmentMapTexelsBuffer: b(),
    environmentMapCdfBuffer: b(),
    meshAreaLightSourceFactorsBuffer: b(),
    tangentsBuffer: b(),
  } as unknown as UploadedSceneBuffers;
}

describe('AdjointPass host packing', () => {
  it('builds a world-space replay override for transformed mesh geometry without remapping material slots', () => {
    const translated = asMat4(new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      2, 0, 0, 1,
    ]));
    const scene = makeScene();
    const base = scene.primitives[0]!;
    if (base.kind !== 'mesh') throw new Error('test fixture expected a mesh');
    const transformedScene: Scene = {
      ...scene,
      primitives: [{
        ...base,
        transform: translated,
        uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
        uv1: new Float32Array([0.25, 0.25, 0.75, 0.25, 0.25, 0.75]),
        tangents: new Float32Array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]),
        colors: new Float32Array([1, 0, 0, 1, 0, 1, 0, 0.5, 0, 0, 1, 1]),
      }],
    };

    const override = buildAdjointWorldSpaceGeometryOverride(
      transformedScene,
      new Set(),
      (_scene, id) => (id === 'panel' ? 7 : null),
    );

    expect(override).not.toBeNull();
    expect(override!.triangleCount).toBe(1);
    expect(Array.from(override!.positions)).toEqual([
      2, 0, 0, 0,
      3, 0, 0, 0,
      2, 1, 0, 0,
    ]);
    expect(Array.from(override!.normals)).toEqual([
      0, 0, 1, 0,
      0, 0, 1, 0,
      0, 0, 1, 0,
    ]);
    expect(Array.from(override!.indices)).toEqual([0, 1, 2, 0]);
    expect(Array.from(override!.triMaterialIds)).toEqual([7]);
    expect(Array.from(override!.uvs)).toEqual([
      0, 0, 0.25, 0.25,
      1, 0, 0.75, 0.25,
      0, 1, 0.25, 0.75,
    ]);
    expect(Array.from(override!.colors)).toEqual([
      1, 0, 0, 1,
      0, 1, 0, 0.5,
      0, 0, 1, 1,
    ]);
    expect(Array.from(override!.tangents)).toEqual([
      1, 0, 0, 1,
      1, 0, 0, 1,
      1, 0, 0, 1,
    ]);
  });

  it('builds a world-space replay override for instanced mesh geometry', () => {
    const instanceA = asMat4(new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      2, 0, 0, 1,
    ]));
    const instanceB = asMat4(new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 3, 0, 1,
    ]));
    const scene = makeScene();
    const base = scene.primitives[0]!;
    if (base.kind !== 'mesh') throw new Error('test fixture expected a mesh');
    const instancedScene: Scene = {
      ...scene,
      primitives: [{
        kind: 'instanced-mesh',
        id: base.id,
        positions: base.positions,
        normals: base.normals,
        indices: base.indices ?? new Uint32Array([0, 1, 2]),
        material: base.material,
        uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
        uv1: new Float32Array([0.25, 0.25, 0.75, 0.25, 0.25, 0.75]),
        instances: [instanceA, instanceB],
      }],
    };

    const override = buildAdjointWorldSpaceGeometryOverride(
      instancedScene,
      new Set(),
      (_scene, id) => (id === 'panel' ? 7 : null),
    );

    expect(override).not.toBeNull();
    expect(override!.triangleCount).toBe(2);
    expect(Array.from(override!.positions)).toEqual([
      2, 0, 0, 0,
      3, 0, 0, 0,
      2, 1, 0, 0,
      0, 3, 0, 0,
      1, 3, 0, 0,
      0, 4, 0, 0,
    ]);
    expect(Array.from(override!.uvs)).toEqual([
      0, 0, 0.25, 0.25,
      1, 0, 0.75, 0.25,
      0, 1, 0.25, 0.75,
      0, 0, 0.25, 0.25,
      1, 0, 0.75, 0.25,
      0, 1, 0.25, 0.75,
    ]);
    expect(Array.from(override!.triMaterialIds)).toEqual([7, 7]);
    const triangles: string[] = [];
    for (let tri = 0; tri < override!.triangleCount; tri += 1) {
      triangles.push(Array.from(override!.indices.slice(tri * 4, tri * 4 + 3)).join(','));
    }
    expect(triangles.sort()).toEqual(['0,1,2', '3,4,5']);
  });

  it('packs replay sample count and swaps mesh-area emitter adjoint replay buffers into bindings', async () => {
    const restoreWebGpuConstants = installWebGpuConstants();
    try {
      const fake = makeFakeDevice();
      const scene = makeScene();
      const sb = makeUploadedSceneBuffers(fake.makeExternalBuffer);
      const req: AdjointGradientRequest = {
        width: 1,
        height: 1,
        channels: 3,
        samples: 4,
        dLoss_dRendered: new Float32Array([0, 0, 0]),
        params: [
          {
            domain: 'emitters',
            id: 'panel-light',
            field: 'intensity',
            offset: 2,
            length: 1,
          },
        ],
        gradientLength: 4,
      };

      const grad = await new AdjointPass(fake.device).computeGradient(
        req,
        sb,
        makeFrame(),
        scene,
        new Set(),
        () => null,
      );

      expect(Array.from(grad)).toEqual([0, 0, 0, 0]);

      const paramsWrite = fake.writes.find((write) => write.byteLength === ADJOINT_PARAMS_UBO_BYTES);
      expect(paramsWrite).toBeDefined();
      const paramsU32 = new Uint32Array(paramsWrite!.data);
      expect(paramsU32[27]).toBe(4);
      expect(paramsU32[30]).toBe(1);

      const descWrite = fake.writes.find((write) => write.byteLength === 8 * Uint32Array.BYTES_PER_ELEMENT);
      expect(descWrite).toBeDefined();
      const descU32 = new Uint32Array(descWrite!.data);
      const descF32 = new Float32Array(descWrite!.data);
      expect(descU32[0]).toBe(0);
      expect(descU32[1]).toBe(ADJOINT_FIELD_EMITTER_INTENSITY);
      expect(descU32[2]).toBe(2);
      expect(descU32[3]! & 0xff).toBe(ADJOINT_EMITTER_TARGET_MESH);
      expect(descU32[3]! >>> 8).toBe(1);
      expect(descF32[4]).toBeCloseTo(0.25, 6);
      expect(descF32[5]).toBeCloseTo(0.5, 6);
      expect(descF32[6]).toBeCloseTo(1, 6);
      expect(descF32[7]).toBeCloseTo(3, 6);

      const bindGroup = fake.bindGroups.at(-1);
      expect(bindGroup).toBeDefined();
      const meshAreaLightsEntry = bindGroup!.entries.find((entry) => entry.binding === 13);
      const sourceFactorsEntry = bindGroup!.entries.find((entry) => entry.binding === 22);
      const meshAreaLightsBuffer = (meshAreaLightsEntry!.resource as { buffer: FakeBuffer }).buffer;
      const sourceFactorsBuffer = (sourceFactorsEntry!.resource as { buffer: FakeBuffer }).buffer;
      expect(meshAreaLightsBuffer).not.toBe(sb.meshAreaLightsBuffer);
      expect(sourceFactorsBuffer).not.toBe(sb.meshAreaLightSourceFactorsBuffer);
      expect(meshAreaLightsBuffer.size).toBe(16 * Float32Array.BYTES_PER_ELEMENT);
      expect(sourceFactorsBuffer.size).toBe(4 * Float32Array.BYTES_PER_ELEMENT);
    } finally {
      restoreWebGpuConstants();
    }
  });
});
