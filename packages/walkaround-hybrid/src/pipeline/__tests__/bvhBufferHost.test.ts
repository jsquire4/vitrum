import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import type { SceneBVHBuffers } from '../../restir/bvhTypes.js';
import { EMITTER_TRI_STRIDE_BYTES } from '../../restir/emitterList.js';

vi.mock('../resourceManager.js', () => ({
  uploadBuffer: vi.fn((_device, data: ArrayBuffer, usage: number) => ({
    size: data.byteLength,
    usage,
    destroy: vi.fn(),
  })),
  uploadBufferPadded: vi.fn((_device, data: ArrayBuffer, extraBytes: number, usage: number) => ({
    size: data.byteLength + extraBytes,
    usage,
    destroy: vi.fn(),
  })),
  createDummyStorageBuffer: vi.fn(() => ({
    size: 16,
    usage: 0x80,
    destroy: vi.fn(),
  })),
}));

// WS1 — beer is a texture now; mock its host helper so the test stays
// device-free (createTexture/writeTexture aren't on the mock device).
vi.mock('../bvhBeerTexture.js', () => ({
  uploadBeerTexture: vi.fn(() => ({
    texture: { createView: vi.fn(() => ({})), destroy: vi.fn() },
    width: 4096,
    height: 1,
  })),
  refreshBeerTexture: vi.fn(),
}));

// Camera-visible emitters — emissive Le is also a texture; mock its host helper
// for the same device-free reason as beer.
vi.mock('../bvhEmissiveTexture.js', () => ({
  uploadEmissiveTexture: vi.fn(() => ({
    texture: { createView: vi.fn(() => ({})), destroy: vi.fn() },
    width: 4096,
    height: 1,
  })),
  refreshEmissiveTexture: vi.fn(),
}));

vi.mock('../materialTextureAtlas.js', () => ({
  uploadMaterialTextureAtlas: vi.fn(() => ({
    atlasTexture: { destroy: vi.fn() },
    atlasTextureView: {},
    baseColorMetaTexture: { destroy: vi.fn() },
    baseColorMetaTextureView: {},
    atlasDim: 1,
    atlasLayerCount: 1,
    baseColorMetaWidth: 2,
    baseColorMetaHeight: 1,
  })),
}));

vi.mock('../bvhTangentTexture.js', () => ({
  uploadTangentTexture: vi.fn(() => ({
    texture: { createView: vi.fn(() => ({})), destroy: vi.fn() },
    width: 4096,
    height: 1,
  })),
}));

vi.mock('../bvhVertexColorTexture.js', () => ({
  uploadVertexColorTexture: vi.fn(() => ({
    texture: { createView: vi.fn(() => ({})), destroy: vi.fn() },
    width: 4096,
    height: 1,
  })),
}));

vi.mock('../analyticLightsTexture.js', () => ({
  uploadAnalyticLightsTexture: vi.fn(() => ({
    texture: { createView: vi.fn(() => ({})), destroy: vi.fn() },
    width: 4,
    height: 1,
  })),
}));

// B3 — directional IBL env resources create GPU textures/sampler/uniform; mock
// the host helper so the test stays device-free (same pattern as beer/emissive).
const mockEnv = () => ({
  map: { createView: vi.fn(() => ({})), destroy: vi.fn() },
  marginal: { createView: vi.fn(() => ({})), destroy: vi.fn() },
  conditional: { createView: vi.fn(() => ({})), destroy: vi.fn() },
  sampler: {},
  paramsBuffer: { destroy: vi.fn() },
});
vi.mock('../environmentTexture.js', () => ({
  createPlaceholderEnvironment: vi.fn(() => mockEnv()),
  uploadEnvironment: vi.fn(() => mockEnv()),
  clearEnvironment: vi.fn(() => mockEnv()),
  disposeEnvironment: vi.fn(),
}));

import { BvhBufferHost } from '../BvhBufferHost.js';
import {
  createDummyStorageBuffer,
  uploadBuffer,
  uploadBufferPadded,
} from '../resourceManager.js';
import { uploadBeerTexture } from '../bvhBeerTexture.js';
import { uploadEmissiveTexture } from '../bvhEmissiveTexture.js';
import { uploadTangentTexture } from '../bvhTangentTexture.js';
import { uploadVertexColorTexture } from '../bvhVertexColorTexture.js';
import { uploadAnalyticLightsTexture } from '../analyticLightsTexture.js';
import { uploadMaterialTextureAtlas } from '../materialTextureAtlas.js';
import {
  SCENE_STORAGE_ARENA_COMPATIBILITY_WORD,
  SCENE_STORAGE_ARENA_EPOCH_WORD,
  SCENE_STORAGE_ARENA_HEADER_WORDS,
  type SceneStorageArenaPayload,
} from '../sceneStorageArena.js';

function mockDevice(): GPUDevice {
  return {
    queue: { writeBuffer: vi.fn() },
  } as unknown as GPUDevice;
}

function storageBuffer(byteLength: number, count = 1) {
  const cpuData = new ArrayBuffer(byteLength);
  return { cpuData, byteLength, count };
}

function makeSceneBvhBuffers(emitterCount = 1): SceneBVHBuffers {
  const buf = storageBuffer(64, 1);
  const emitters = storageBuffer(EMITTER_TRI_STRIDE_BYTES * emitterCount, emitterCount);
  const emitterCdf = storageBuffer(4 * emitterCount, emitterCount);
  const emitterAlias = storageBuffer(16 * emitterCount, emitterCount);
  return {
    bvhNodes: storageBuffer(32, 1),
    bvhIndex: storageBuffer(16, 1),
    bvhBeerColors: buf,
    bvhEmissiveLe: buf,
    bvhRoughMetal: buf,
    bvhNormals: storageBuffer(16, 1),
    bvhTangents: buf,
    bvhColors: buf,
    bvhPositions: storageBuffer(16, 1),
    emitters,
    emitterCdf,
    emitterAlias,
    emitterCount,
    totalEmissivePower: 0,
    lightTree: buf,
    lightTreeNodeCount: 0,
    lightTreeEnabled: false,
    bvhMode: 'merged',
  } as SceneBVHBuffers;
}


type FaultableMock = {
  getMockImplementation(): ((...args: unknown[]) => unknown) | undefined;
  mockImplementation(implementation: (...args: unknown[]) => unknown): unknown;
  readonly mock: {
    readonly results: readonly { readonly type: string; readonly value?: unknown }[];
  };
};

const replacementAllocators: readonly FaultableMock[] = [
  uploadBuffer,
  uploadBufferPadded,
  createDummyStorageBuffer,
  uploadBeerTexture,
  uploadEmissiveTexture,
  uploadMaterialTextureAtlas,
  uploadTangentTexture,
  uploadVertexColorTexture,
  uploadAnalyticLightsTexture,
].map((mock) => mock as unknown as FaultableMock);

function destroySpiesIn(value: unknown, out: ReturnType<typeof vi.fn>[], seen = new WeakSet<object>()): void {
  if (value == null || typeof value !== 'object' || seen.has(value)) return;
  seen.add(value);
  const record = value as Record<string, unknown>;
  if (typeof record['destroy'] === 'function' && vi.isMockFunction(record['destroy'])) {
    out.push(record['destroy'] as ReturnType<typeof vi.fn>);
  }
  for (const nested of Object.values(record)) {
    if (nested != null && typeof nested === 'object') destroySpiesIn(nested, out, seen);
  }
}

function returnedSince(starts: readonly number[]): unknown[] {
  return replacementAllocators.flatMap((mock, index) =>
    mock.mock.results
      .slice(starts[index])
      .filter((result) => result.type === 'return')
      .map((result) => result.value),
  );
}

function expectSameLiveBindings(
  host: BvhBufferHost,
  before: ReturnType<BvhBufferHost['sceneBindGroupResources']>,
  emitterBefore: GPUBuffer | undefined,
  lightTreeBefore: GPUBuffer | null,
): void {
  const after = host.sceneBindGroupResources();
  for (const key of Object.keys(before) as (keyof typeof before)[]) {
    if (key === 'sceneStorageArenaBuffers') {
      expect(after.sceneStorageArenaBuffers).toStrictEqual(before.sceneStorageArenaBuffers);
    } else {
      expect(after[key]).toBe(before[key]);
    }
  }
  expect(host.emitterBufferAndCount()?.buffer).toBe(emitterBefore);
  expect(host.lightTreeBuffer()).toBe(lightTreeBefore);
}
describe('BvhBufferHost', () => {
  it('preserves ReGIR grid padding beyond the signed-32-bit range', () => {
    const host = new BvhBufferHost();
    const padding = 2_147_483_648;
    host.setRegirGridBytes(padding);
    host.uploadInitial(mockDevice(), makeSceneBvhBuffers());

    expect(vi.mocked(uploadBufferPadded)).toHaveBeenCalledWith(
      expect.anything(),
      expect.any(ArrayBuffer),
      padding,
      0x80,
    );
    host.dispose();
  });

  it.each([
    Number.NaN,
    Number.POSITIVE_INFINITY,
    -4,
    1,
    4.5,
    Number.MAX_SAFE_INTEGER + 1,
  ])('rejects invalid ReGIR grid padding %s', (padding) => {
    const host = new BvhBufferHost();
    expect(() => host.setRegirGridBytes(padding)).toThrow(
      /non-negative, 4-byte-aligned safe integer/,
    );
  });

  it('uploadInitial exposes scene bind-group resources', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    host.uploadInitial(device, makeSceneBvhBuffers());
    const r = host.sceneBindGroupResources();
    const r2 = host.sceneBindGroupResources();
    expect(r.sceneStorageArenaBuffers).toHaveLength(3);
    for (const shard of r.sceneStorageArenaBuffers) expect(shard).toBeDefined();
    expect(r2.bvhBeerTextureView).toBe(r.bvhBeerTextureView);
    expect(r2.bvhEmissiveTextureView).toBe(r.bvhEmissiveTextureView);
    expect(r2.bvhTangentTextureView).toBe(r.bvhTangentTextureView);
    expect(r2.bvhVertexColorTextureView).toBe(r.bvhVertexColorTextureView);
    expect(host.lightTreeBuffer()).toBeDefined();
    const sampling = host.emitterSamplingBufferAndCount();
    expect(sampling).not.toBeNull();
    if (sampling == null) throw new Error('expected emitter sampling buffer');
    expect((sampling.offset ?? 0) % 256).toBe(0);
    expect(sampling.emitterDataOffset).toBe(0);
    expect(sampling.emitterAliasOffset).toBe(512);
    expect(sampling.size).toBe(528);
    expect(sampling.count).toBe(1);
    expect(sampling.buffer).toBe(r.sceneStorageArenaBuffers[2]);
    const mem = host.gpuMemorySections().staticScene;
    if (mem == null) throw new Error('expected staticScene memory section');
    expect(mem['sceneStorageArenaShard0']).toMatchObject({ usage: 0x80 });
    expect(mem['sceneStorageArenaShard1']).toMatchObject({ usage: 0x80 });
    expect(mem['sceneStorageArenaShard2']).toMatchObject({ usage: 0x80 });
    expect(mem['lightTreeBuffer']).toMatchObject({ size: 64, usage: 0x80 });
    expect(mem).not.toHaveProperty('bvhNodesBuffer');
    expect(mem).not.toHaveProperty('tlasNodesBuffer');
    expect(mem).not.toHaveProperty('emitterBuffer');
    expect(mem['bvhBeerTexture']).toMatchObject({ width: 4096, height: 1, format: 'r32uint' });
    expect(mem['bvhEmissiveTexture']).toMatchObject({ width: 4096, height: 1, format: 'rgba32float' });
    expect(mem['bvhTangentTexture']).toMatchObject({ width: 4096, height: 1, format: 'rgba32float' });
    expect(mem['materialTextureAtlas']).toMatchObject({ width: 1, height: 1, depthOrArrayLayers: 1, format: 'rgba32float' });
    expect(mem['baseColorMapMetaTexture']).toMatchObject({ width: 2, height: 1, depthOrArrayLayers: 1, format: 'rgba32float' });
    host.dispose();
  });

  it('uploadInitial packs point and spot emitters into the analytic-lights texture when a scene is supplied', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    const scene: Scene = {
      primitives: [],
      emitters: [
        {
          kind: 'point',
          id: 'point-a',
          position: [1, 2, 3],
          color: [0.25, 0.5, 0.75],
          intensity: 4,
          castShadow: false,
        },
        {
          kind: 'spot',
          id: 'spot-a',
          position: [-1, 3, 2],
          direction: [0, -2, 0],
          angle: Math.PI / 3,
          penumbra: 0.25,
          color: [0.2, 0.4, 0.6],
          intensity: 5,
        },
      ],
      environment: { kind: 'none' },
    };

    const upload = vi.mocked(uploadAnalyticLightsTexture);
    upload.mockClear();

    host.uploadInitial(device, makeSceneBvhBuffers(), scene);

    expect(upload).toHaveBeenCalledTimes(1);
    const [, data, count] = upload.mock.calls[0] as [GPUDevice, Float32Array, number];
    expect(count).toBe(2);
    // Two 16-float records followed by two 4-float alias entries.
    expect(data.length).toBe(40);
    expect(Array.from(data.slice(0, 7))).toEqual([1, 2, 3, 0, 1, 2, 3]);
    expect(Array.from(data.slice(8, 13))).toEqual([0, 0, 0, 1, 0]);
    expect(data[13]).toBe(1);

    const spot = 16;
    expect(Array.from(data.slice(spot, spot + 7))).toEqual([-1, 3, 2, 0, 1, 2, 3]);
    expect(data[spot + 8]).toBeCloseTo(0, 6);
    expect(data[spot + 9]).toBeCloseTo(-1, 6);
    expect(data[spot + 10]).toBeCloseTo(0, 6);
    expect(data[spot + 11]).toBeCloseTo(Math.cos(Math.PI / 4), 6);
    expect(data[spot + 12]).toBeCloseTo(Math.cos(Math.PI / 3), 6);
    expect(data[spot + 13]).toBe(0);
    expect(data[34]! + data[38]!).toBeCloseTo(1, 6);
    expect(data[34]).toBeGreaterThan(0);
    expect(data[38]).toBeGreaterThan(0);

    host.dispose();
  });

  it('uploadInitial boots a scene-less host with an exact zero-count analytic payload', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    const upload = vi.mocked(uploadAnalyticLightsTexture);
    upload.mockClear();

    host.uploadInitial(device, makeSceneBvhBuffers());

    expect(upload).toHaveBeenCalledTimes(1);
    const [, data, count] = upload.mock.calls[0] as [GPUDevice, Float32Array, number];
    expect(count).toBe(0);
    expect(data).toHaveLength(0);

    host.dispose();
  });

  it('refreshMaterialTextureAtlas replaces only atlas texture bindings', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    const upload = vi.mocked(uploadMaterialTextureAtlas);
    upload.mockClear();
    host.uploadInitial(device, makeSceneBvhBuffers());
    expect(upload).toHaveBeenCalledTimes(1);
    const firstAtlas = upload.mock.results[0]?.value as {
      atlasTexture: { destroy: ReturnType<typeof vi.fn> };
      atlasTextureView: GPUTextureView;
      baseColorMetaTexture: { destroy: ReturnType<typeof vi.fn> };
      baseColorMetaTextureView: GPUTextureView;
    };
    const before = host.sceneBindGroupResources();

    host.refreshMaterialTextureAtlas(
      device,
      { diagnostics: [] } as unknown as SceneBVHBuffers['materialTextureAtlas'],
    );

    expect(upload).toHaveBeenCalledTimes(2);
    expect(firstAtlas.atlasTexture.destroy).toHaveBeenCalledTimes(1);
    expect(firstAtlas.baseColorMetaTexture.destroy).toHaveBeenCalledTimes(1);
    const after = host.sceneBindGroupResources();
    expect(after.sceneStorageArenaBuffers).toStrictEqual(before.sceneStorageArenaBuffers);
    expect(after.materialTextureAtlasView).not.toBe(before.materialTextureAtlasView);
    expect(after.baseColorMapMetaTextureView).not.toBe(before.baseColorMapMetaTextureView);
    host.dispose();
  });

  it('updateEmitters uses the canonical emitter payload count', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    host.uploadInitial(device, makeSceneBvhBuffers(1));

    const next = makeSceneBvhBuffers(2);
    host.updateEmitters(device, {
      emitters: next.emitters,
      emitterCdf: next.emitterCdf,
      emitterAlias: next.emitterAlias,
      lightTree: next.lightTree,
    });

    expect(host.emitterBufferAndCount()?.count).toBe(2);
    host.dispose();
  });

  it('updateEmitters rejects malformed emitter byte lengths before replacing live buffers', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    host.uploadInitial(device, makeSceneBvhBuffers(1));

    const next = makeSceneBvhBuffers(1);
    const malformed = {
      cpuData: new ArrayBuffer(EMITTER_TRI_STRIDE_BYTES + 4),
      byteLength: EMITTER_TRI_STRIDE_BYTES + 4,
      count: 1,
    };

    expect(() => host.updateEmitters(device, {
      emitters: malformed,
      emitterCdf: next.emitterCdf,
      emitterAlias: next.emitterAlias,
      lightTree: next.lightTree,
    })).toThrow(/not aligned to the 80-byte EmitterTri stride/);
    expect(host.emitterBufferAndCount()?.count).toBe(1);

    host.dispose();
  });

  it('writes one dirty segment plus generation metadata per shard', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    host.uploadInitial(device, makeSceneBvhBuffers(1));
    const buffers = host.sceneBindGroupResources().sceneStorageArenaBuffers;
    const writeBuffer = vi.mocked(device.queue.writeBuffer);
    writeBuffer.mockClear();
    const nodes = new Uint32Array(8).fill(0x5a5a_5a5a).buffer;

    host.refreshBvhNodesOnly(device, nodes);

    expect(writeBuffer).toHaveBeenCalledTimes(4);
    expect(writeBuffer.mock.calls[0]).toEqual([buffers[0], 256, nodes]);
    const epochs = new Set<number>();
    const compatibilities = new Set<number>();
    for (let shard = 0; shard < 3; shard += 1) {
      const call = writeBuffer.mock.calls[shard + 1]!;
      expect(call[0]).toBe(buffers[shard]);
      expect(call[1]).toBe(0);
      expect(call[2]).toBeInstanceOf(ArrayBuffer);
      expect((call[2] as ArrayBuffer).byteLength).toBe(
        SCENE_STORAGE_ARENA_HEADER_WORDS * 4,
      );
      const header = new Uint32Array(call[2] as ArrayBuffer);
      epochs.add(header[SCENE_STORAGE_ARENA_EPOCH_WORD]!);
      compatibilities.add(
        header[SCENE_STORAGE_ARENA_COMPATIBILITY_WORD]!,
      );
    }
    expect(epochs.size).toBe(1);
    expect(compatibilities.size).toBe(1);
    host.dispose();
  });

  it('stages exact TLAS ranges without reading retained arena sources', () => {
    const host = new BvhBufferHost();
    const base = makeSceneBvhBuffers(1);
    base.bvhMode = 'tlas';
    base.tlas = {
      nodes: storageBuffer(256, 8),
      instanceIndices: storageBuffer(32, 8),
      blasRoots: storageBuffer(32, 8),
      worldToLocal: storageBuffer(512, 8),
      localToWorld: storageBuffer(512, 8),
      nodeCount: 8,
    };
    const staged: Array<{
      readonly buffer: GPUBuffer;
      readonly mapped: ArrayBuffer;
    }> = [];
    const device = {
      queue: { writeBuffer: vi.fn() },
      createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
        const mapped = new ArrayBuffer(Number(descriptor.size));
        const buffer = {
          getMappedRange: vi.fn(() => mapped),
          unmap: vi.fn(),
          destroy: vi.fn(),
        } as unknown as GPUBuffer;
        staged.push({ buffer, mapped });
        return buffer;
      }),
    } as unknown as GPUDevice;
    const copyBufferToBuffer = vi.fn();
    const encoder = { copyBufferToBuffer } as unknown as GPUCommandEncoder;
    host.uploadInitial(device, base);
    const internals = host as unknown as {
      _sceneStorageArenaPayload: SceneStorageArenaPayload;
    };
    const payload = internals._sceneStorageArenaPayload;
    const arenaBuffers =
      host.sceneBindGroupResources().sceneStorageArenaBuffers;
    let retainedReads = 0;
    for (const source of Object.values(payload.sources)) {
      const descriptor = Object.getOwnPropertyDescriptor(source, 'data')!;
      Object.defineProperty(source, 'data', {
        configurable: true,
        get: () => {
          retainedReads += 1;
          return descriptor.get!.call(source) as ArrayBuffer;
        },
      });
    }
    const nodeA = new Uint32Array(8).fill(0x1111_1111).buffer;
    const nodeB = new Uint32Array(8).fill(0x2222_2222).buffer;
    const worldToLocal = new Float32Array(16).fill(3).buffer;
    const localToWorld = new Float32Array(16).fill(4).buffer;

    const prepared = host.prepareMutation(device, encoder, {
      tlas: {
        nodes: [
          { byteOffset: 32, data: nodeA },
          { byteOffset: 128, data: nodeB },
        ],
        worldToLocal: [{ byteOffset: 64, data: worldToLocal }],
        localToWorld: [{ byteOffset: 128, data: localToWorld }],
      },
      resetAccumulator: false,
    });

    expect(retainedReads).toBe(0);
    const expected = [
      ['tlasNodes', 32, nodeA],
      ['tlasNodes', 128, nodeB],
      ['tlasInstanceWorldToLocal', 64, worldToLocal],
      ['tlasInstanceLocalToWorld', 128, localToWorld],
    ] as const;
    expect(copyBufferToBuffer).toHaveBeenCalledTimes(expected.length + 3);
    for (let index = 0; index < expected.length; index += 1) {
      const [segment, logicalOffset, data] = expected[index]!;
      const layout = payload.segments[segment];
      expect(copyBufferToBuffer.mock.calls[index]).toEqual([
        staged[index]!.buffer,
        0,
        arenaBuffers[layout.shard],
        layout.byteOffset + logicalOffset,
        data.byteLength,
      ]);
      expect(
        new Uint8Array(staged[index]!.mapped, 0, data.byteLength),
      ).toEqual(new Uint8Array(data));
    }
    prepared.rollback();
    host.dispose();
  });

  it('replaces and publishes lighting without reading retained geometry bytes', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    host.uploadInitial(device, makeSceneBvhBuffers(1));
    const internals = host as unknown as {
      _sceneStorageArenaPayload: SceneStorageArenaPayload;
    };
    let geometryReads = 0;
    for (const name of [
      'bvhNodes',
      'bvhIndex',
      'bvhPositions',
      'bvhNormals',
      'tlasNodes',
      'tlasInstanceIndices',
      'tlasBlasRoots',
      'tlasInstanceWorldToLocal',
      'tlasInstanceLocalToWorld',
    ] as const) {
      const retained = internals._sceneStorageArenaPayload.sources[name];
      const descriptor = Object.getOwnPropertyDescriptor(retained, 'data')!;
      Object.defineProperty(retained, 'data', {
        configurable: true,
        get: () => {
          geometryReads += 1;
          return descriptor.get!.call(retained) as ArrayBuffer;
        },
      });
    }

    const next = makeSceneBvhBuffers(2);
    host.updateEmitters(device, {
      emitters: next.emitters,
      emitterCdf: next.emitterCdf,
      emitterAlias: next.emitterAlias,
      lightTree: next.lightTree,
    });

    expect(geometryReads).toBe(0);
    host.dispose();
  });

  it('rejects a stale same-size lighting candidate after geometry changes', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    host.uploadInitial(device, makeSceneBvhBuffers(1));
    const before = host.sceneBindGroupResources();
    const emitterBefore = host.emitterBufferAndCount()!.buffer;
    const mutation = host.prepareEmitterLightingReplacement(
      device,
      makeSceneBvhBuffers(2),
      { primitives: [], emitters: [], environment: { kind: 'none' } },
    );

    host.refreshBvhNodesOnly(
      device,
      new Uint32Array(8).fill(0xdead_beef).buffer,
    );

    expect(() => mutation.commit()).toThrow(
      /lighting scene-storage shard is incompatible with retained geometry\/TLAS shards/,
    );
    expect(host.sceneBindGroupResources().sceneStorageArenaBuffers[2])
      .toBe(before.sceneStorageArenaBuffers[2]);
    expect(host.emitterBufferAndCount()!.buffer).toBe(emitterBefore);
    mutation.rollback();
    host.dispose();
  });


  it('transactionally replaces BVH resources across every allocation stage', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    const initialStarts = replacementAllocators.map((mock) => mock.mock.results.length);
    host.uploadInitial(device, makeSceneBvhBuffers(1));

    const initialDestroySpies: ReturnType<typeof vi.fn>[] = [];
    for (const value of returnedSince(initialStarts)) destroySpiesIn(value, initialDestroySpies);
    const failureCases: readonly [FaultableMock, number][] = [
      [uploadBuffer as unknown as FaultableMock, 3],
      [uploadBufferPadded as unknown as FaultableMock, 1],
      [uploadBeerTexture as unknown as FaultableMock, 2],
      [uploadEmissiveTexture as unknown as FaultableMock, 1],
      [uploadMaterialTextureAtlas as unknown as FaultableMock, 1],
      [uploadTangentTexture as unknown as FaultableMock, 1],
      [uploadVertexColorTexture as unknown as FaultableMock, 1],
    ];

    for (const [allocator, invocationCount] of failureCases) {
      const original = allocator.getMockImplementation();
      if (original == null) throw new Error('expected allocator mock implementation');
      for (let failAt = 1; failAt <= invocationCount; failAt += 1) {
        const before = host.sceneBindGroupResources();
        const emitterBefore = host.emitterBufferAndCount()?.buffer;
        const lightTreeBefore = host.lightTreeBuffer();
        const starts = replacementAllocators.map((mock) => mock.mock.results.length);
        let calls = 0;
        allocator.mockImplementation((...args: unknown[]) => {
          calls += 1;
          if (calls === failAt) throw new Error(`allocation fault ${failAt}`);
          return original(...args);
        });
        try {
          expect(() => host.replaceBvhAndEmitters(device, makeSceneBvhBuffers(2)))
            .toThrow(`allocation fault ${failAt}`);
        } finally {
          allocator.mockImplementation(original);
        }

        expectSameLiveBindings(host, before, emitterBefore, lightTreeBefore);
        for (const destroy of initialDestroySpies) expect(destroy).not.toHaveBeenCalled();

        const candidateDestroySpies: ReturnType<typeof vi.fn>[] = [];
        for (const value of returnedSince(starts)) destroySpiesIn(value, candidateDestroySpies);
        for (const destroy of candidateDestroySpies) expect(destroy).toHaveBeenCalledTimes(1);
      }
    }

    const initial = host.sceneBindGroupResources();
    const initialEmitter = host.emitterBufferAndCount()?.buffer;
    host.replaceBvhAndEmitters(device, makeSceneBvhBuffers(2));
    const first = host.sceneBindGroupResources();
    expect(first.sceneStorageArenaBuffers[0]).not.toBe(initial.sceneStorageArenaBuffers[0]);
    expect(host.emitterBufferAndCount()?.buffer).not.toBe(initialEmitter);

    host.replaceBvhAndEmitters(device, makeSceneBvhBuffers(3));
    const second = host.sceneBindGroupResources();
    expect(second.sceneStorageArenaBuffers[0]).not.toBe(first.sceneStorageArenaBuffers[0]);
    expect(host.emitterBufferAndCount()?.count).toBe(3);

    expect(() => host.dispose()).not.toThrow();
    expect(() => host.dispose()).not.toThrow();
  });
  it('cleans every staged emitter resource when analytic allocation fails even if the first destroy throws', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    const initialStarts = replacementAllocators.map((mock) => mock.mock.results.length);
    host.uploadInitial(device, makeSceneBvhBuffers(1));
    const before = host.sceneBindGroupResources();
    const emitterBefore = host.emitterBufferAndCount()?.buffer;
    const lightTreeBefore = host.lightTreeBuffer();

    const initialDestroySpies: ReturnType<typeof vi.fn>[] = [];
    for (const value of returnedSince(initialStarts)) {
      destroySpiesIn(value, initialDestroySpies);
    }
    const candidateStarts = replacementAllocators.map((mock) => mock.mock.results.length);
    const emissiveAllocator = uploadEmissiveTexture as unknown as FaultableMock;
    const analyticAllocator = uploadAnalyticLightsTexture as unknown as FaultableMock;
    const originalEmissive = emissiveAllocator.getMockImplementation();
    const originalAnalytic = analyticAllocator.getMockImplementation();
    if (originalEmissive == null || originalAnalytic == null) {
      throw new Error('expected texture allocator mock implementations');
    }
    let firstCandidateDestroy: ReturnType<typeof vi.fn> | undefined;
    emissiveAllocator.mockImplementation((...args: unknown[]) => {
      const value = originalEmissive(...args) as {
        texture: { destroy: ReturnType<typeof vi.fn> };
      };
      firstCandidateDestroy = value.texture.destroy;
      firstCandidateDestroy.mockImplementationOnce(() => {
        throw new Error('candidate destroy fault');
      });
      return value;
    });
    analyticAllocator.mockImplementation(() => {
      throw new Error('analytic allocation fault');
    });

    try {
      expect(() => host.prepareEmitterLightingReplacement(
        device,
        makeSceneBvhBuffers(2),
        { primitives: [], emitters: [], environment: { kind: 'none' } },
      )).toThrow('analytic allocation fault');
    } finally {
      emissiveAllocator.mockImplementation(originalEmissive);
      analyticAllocator.mockImplementation(originalAnalytic);
    }

    expectSameLiveBindings(host, before, emitterBefore, lightTreeBefore);
    for (const destroy of initialDestroySpies) expect(destroy).not.toHaveBeenCalled();

    const candidateDestroySpies: ReturnType<typeof vi.fn>[] = [];
    for (const value of returnedSince(candidateStarts)) {
      destroySpiesIn(value, candidateDestroySpies);
    }
    expect(firstCandidateDestroy).toBeDefined();
    for (const destroy of candidateDestroySpies) {
      expect(destroy).toHaveBeenCalledTimes(1);
    }
    host.dispose();
  });

  it('restores every emitter-lighting binding after commit then rollback', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    const initialStarts = replacementAllocators.map((mock) => mock.mock.results.length);
    host.uploadInitial(device, makeSceneBvhBuffers(1));
    const before = host.sceneBindGroupResources();
    const emitterBefore = host.emitterBufferAndCount()?.buffer;
    const lightTreeBefore = host.lightTreeBuffer();
    const initialDestroySpies: ReturnType<typeof vi.fn>[] = [];
    for (const value of returnedSince(initialStarts)) {
      destroySpiesIn(value, initialDestroySpies);
    }

    const candidateStarts = replacementAllocators.map((mock) => mock.mock.results.length);
    const mutation = host.prepareEmitterLightingReplacement(
      device,
      makeSceneBvhBuffers(2),
      { primitives: [], emitters: [], environment: { kind: 'none' } },
    );
    const candidateDestroySpies: ReturnType<typeof vi.fn>[] = [];
    for (const value of returnedSince(candidateStarts)) {
      destroySpiesIn(value, candidateDestroySpies);
    }

    mutation.commit();
    expect(host.emitterBufferAndCount()?.buffer).not.toBe(emitterBefore);
    expect(host.emitterBufferAndCount()?.count).toBe(2);
    expect(host.lightTreeBuffer()).not.toBe(lightTreeBefore);
    expect(host.sceneBindGroupResources().bvhEmissiveTextureView)
      .not.toBe(before.bvhEmissiveTextureView);
    expect(host.sceneBindGroupResources().analyticLightsTextureView)
      .not.toBe(before.analyticLightsTextureView);
    for (const destroy of initialDestroySpies) expect(destroy).not.toHaveBeenCalled();

    mutation.rollback();
    expectSameLiveBindings(host, before, emitterBefore, lightTreeBefore);
    for (const destroy of candidateDestroySpies) {
      expect(destroy).toHaveBeenCalledTimes(1);
    }
    for (const destroy of initialDestroySpies) expect(destroy).not.toHaveBeenCalled();
    host.dispose();
  });

  it('retires the old emitter-lighting generation only after finalize', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    host.uploadInitial(device, makeSceneBvhBuffers(1));
    const oldEmitter = host.emitterBufferAndCount()!.buffer as GPUBuffer & {
      destroy: ReturnType<typeof vi.fn>;
    };
    const oldLightTree = host.lightTreeBuffer() as GPUBuffer & {
      destroy: ReturnType<typeof vi.fn>;
    };
    const emissiveResults = vi.mocked(uploadEmissiveTexture).mock.results;
    const oldEmissive = emissiveResults[emissiveResults.length - 1]!.value as {
      texture: { destroy: ReturnType<typeof vi.fn> };
    };
    const analyticResults = vi.mocked(uploadAnalyticLightsTexture).mock.results;
    const oldAnalytic = analyticResults[analyticResults.length - 1]!.value as {
      texture: { destroy: ReturnType<typeof vi.fn> };
    };

    const candidateStarts = replacementAllocators.map((mock) => mock.mock.results.length);
    const mutation = host.prepareEmitterLightingReplacement(
      device,
      makeSceneBvhBuffers(2),
      { primitives: [], emitters: [], environment: { kind: 'none' } },
    );
    const candidateDestroySpies: ReturnType<typeof vi.fn>[] = [];
    for (const value of returnedSince(candidateStarts)) {
      destroySpiesIn(value, candidateDestroySpies);
    }

    mutation.commit();
    expect(oldEmitter.destroy).not.toHaveBeenCalled();
    expect(oldLightTree.destroy).not.toHaveBeenCalled();
    expect(oldEmissive.texture.destroy).not.toHaveBeenCalled();
    expect(oldAnalytic.texture.destroy).not.toHaveBeenCalled();

    mutation.finalize();
    expect(oldEmitter.destroy).toHaveBeenCalledTimes(1);
    expect(oldLightTree.destroy).toHaveBeenCalledTimes(1);
    expect(oldEmissive.texture.destroy).toHaveBeenCalledTimes(1);
    expect(oldAnalytic.texture.destroy).toHaveBeenCalledTimes(1);
    for (const destroy of candidateDestroySpies) expect(destroy).not.toHaveBeenCalled();
    expect(host.emitterBufferAndCount()?.count).toBe(2);
    host.dispose();
  });

  it('sceneBindGroupResources throws before upload', () => {
    const host = new BvhBufferHost();
    expect(() => host.sceneBindGroupResources()).toThrow(/uploadInitial/);
  });
});
