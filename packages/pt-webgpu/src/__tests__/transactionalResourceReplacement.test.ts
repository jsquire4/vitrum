import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { GpuResources } from '../gpuResources.js';
import { SceneMutationRouter, type MutationHost } from '../sceneMutationRouter.js';
import {
  buildPackedScene,
  CWBVH_ROOT_PAIR_WORDS,
  isValidCwbvhRootPair,
  scenePackResultFromPacked,
  SCENE_BUFFER_REGISTRY,
  uploadPackedScene,
  uploadScenePackGeometry,
  uploadScenePackGeometryRealloc,
  uploadScenePackTlasRealloc,
  type UploadedSceneBuffers,
} from '../scene/uploadSceneBuffers.js';
import { installGpuConstStubs } from './gpuStub.js';

interface TrackedResource {
  readonly label: string;
  readonly size: number;
  readonly destroy: ReturnType<typeof vi.fn>;
  readonly createView?: (...args: unknown[]) => unknown;
  readonly width?: number;
  readonly height?: number;
  readonly depthOrArrayLayers?: number;
  readonly format?: GPUTextureFormat;
}

class FailureDevice {
  readonly resources: TrackedResource[] = [];
  allocationCount = 0;
  writeCount = 0;
  viewCount = 0;
  submitCount = 0;
  readonly clearCalls: GPUBuffer[] = [];
  failAllocationAt: number | null = null;
  failWriteAt: number | null = null;
  failViewAt: number | null = null;
  failSubmitAt: number | null = null;
  aliasBuffers = false;
  aliasTextures = false;
  aliasBufferTo: TrackedResource | null = null;
  aliasTextureTo: TrackedResource | null = null;
  private sharedBuffer: TrackedResource | null = null;
  private sharedTexture: TrackedResource | null = null;

  readonly queue = {
    writeBuffer: vi.fn(() => this.write()),
    writeTexture: vi.fn(() => this.write()),
    copyExternalImageToTexture: vi.fn(() => this.write()),
    submit: vi.fn(() => {
      this.submitCount += 1;
      if (this.submitCount === this.failSubmitAt) throw new Error('injected submit failure');
    }),
  };

  readonly device = {
    queue: this.queue,
    limits: {
      maxTextureDimension2D: 8192,
      maxTextureArrayLayers: 256,
      maxStorageBuffersPerShaderStage: 64,
      maxStorageTexturesPerShaderStage: 8,
    },
    createBuffer: vi.fn((desc: GPUBufferDescriptor) => this.allocateBuffer(desc)),
    createTexture: vi.fn((desc: GPUTextureDescriptor) => this.allocateTexture(desc)),
    createSampler: vi.fn(() => ({})),
    createCommandEncoder: vi.fn(() => ({
      clearBuffer: vi.fn((buffer: GPUBuffer) => {
        this.clearCalls.push(buffer);
      }),
      copyBufferToBuffer: vi.fn(),
      finish: vi.fn(() => ({})),
    })),
  } as unknown as GPUDevice;

  arm(kind: 'allocation' | 'write' | 'view' | 'submit', stage: number): number {
    this.failAllocationAt = null;
    this.failWriteAt = null;
    this.failViewAt = null;
    this.failSubmitAt = null;
    this.allocationCount = 0;
    this.writeCount = 0;
    this.viewCount = 0;
    this.submitCount = 0;
    this.clearCalls.length = 0;
    this[`fail${kind[0]!.toUpperCase()}${kind.slice(1)}At` as
      'failAllocationAt' | 'failWriteAt' | 'failViewAt' | 'failSubmitAt'] = stage;
    return this.resources.length;
  }

  disarm(): void {
    this.failAllocationAt = null;
    this.failWriteAt = null;
    this.failViewAt = null;
    this.failSubmitAt = null;
  }

  private write(): void {
    this.writeCount += 1;
    if (this.writeCount === this.failWriteAt) throw new Error('injected upload failure');
  }

  private beforeAllocation(): void {
    this.allocationCount += 1;
    if (this.allocationCount === this.failAllocationAt) {
      throw new Error('injected allocation failure');
    }
  }

  private allocateBuffer(desc: GPUBufferDescriptor): GPUBuffer {
    this.beforeAllocation();
    if (this.aliasBufferTo != null) {
      return this.aliasBufferTo as unknown as GPUBuffer;
    }
    if (this.aliasBuffers && this.sharedBuffer != null) {
      return this.sharedBuffer as unknown as GPUBuffer;
    }
    const resource: TrackedResource = {
      label: String(desc.label ?? ''),
      size: Number(desc.size),
      destroy: vi.fn(),
    };
    this.resources.push(resource);
    if (this.aliasBuffers) this.sharedBuffer = resource;
    return resource as unknown as GPUBuffer;
  }

  private allocateTexture(desc: GPUTextureDescriptor): GPUTexture {
    this.beforeAllocation();
    if (this.aliasTextureTo != null) {
      return this.aliasTextureTo as unknown as GPUTexture;
    }
    if (this.aliasTextures && this.sharedTexture != null) {
      return this.sharedTexture as unknown as GPUTexture;
    }
    const extent = desc.size as GPUExtent3DDict;
    const resource: TrackedResource = {
      label: String(desc.label ?? ''),
      size: 0,
      width: Number(extent.width ?? 1),
      height: Number(extent.height ?? 1),
      depthOrArrayLayers: Number(extent.depthOrArrayLayers ?? 1),
      format: desc.format,
      destroy: vi.fn(),
      createView: vi.fn(() => {
        this.viewCount += 1;
        if (this.viewCount === this.failViewAt) throw new Error('injected view failure');
        return {};
      }),
    };
    this.resources.push(resource);
    if (this.aliasTextures) this.sharedTexture = resource;
    return resource as unknown as GPUTexture;
  }
}

function scene(): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'mesh',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function geometryHandles(sb: UploadedSceneBuffers): readonly GPUBuffer[] {
  return [
    sb.positionsBuffer, sb.normalsBuffer, sb.uvsBuffer, sb.tangentsBuffer,
    sb.colorsBuffer, sb.indicesBuffer, sb.triMaterialIdsBuffer, sb.bvhNodesBuffer,
    sb.tlasNodesBuffer, sb.tlasInstanceIndicesBuffer, sb.tlasBlasRootsBuffer,
    sb.tlasInstanceWorldToLocalBuffer, sb.tlasInstanceLocalToWorldBuffer,
    sb.cwbvhNodeBoundsBuffer, sb.cwbvhChildBoundsPackedBuffer,
    sb.cwbvhChildMetaBuffer, sb.cwbvhChildCountBuffer, sb.cwbvhTlasBlasRootsBuffer,
  ];
}

function tlasHandles(sb: UploadedSceneBuffers): readonly GPUBuffer[] {
  return [
    sb.tlasNodesBuffer, sb.tlasInstanceIndicesBuffer, sb.tlasBlasRootsBuffer,
    sb.tlasInstanceWorldToLocalBuffer, sb.tlasInstanceLocalToWorldBuffer,
    sb.cwbvhTlasBlasRootsBuffer,
  ];
}

function expectRootPairsMatchTables(sb: UploadedSceneBuffers): void {
  expect(sb.cwbvhTlasBlasRoots).toHaveLength(sb.tlasBlasRoots.length * CWBVH_ROOT_PAIR_WORDS);
  for (let i = 0; i < sb.tlasBlasRoots.length; i += 1) {
    const offset = i * CWBVH_ROOT_PAIR_WORDS;
    expect(isValidCwbvhRootPair(sb.cwbvhTlasBlasRoots, offset)).toBe(true);
    expect(sb.cwbvhTlasBlasRoots[offset + 1]).toBe(sb.tlasBlasRoots[i]);
    expect(sb.cwbvhTlasBlasRoots[offset + 2]).toBeLessThan(sb.cwbvhNodeCount);
  }
}

function expectCreatedDestroyed(stub: FailureDevice, start: number): void {
  const candidates = stub.resources.slice(start);
  for (const candidate of candidates) {
    expect(candidate.destroy, candidate.label).toHaveBeenCalledOnce();
  }
}

describe('R2.1 transactional pt-webgpu resource replacement', () => {
  it('rejects the aggregate three-atlas peak before the first GPU allocation', () => {
    installGpuConstStubs();
    const packed = buildPackedScene(scene()) as ReturnType<typeof buildPackedScene> & {
      materialTextureSources: readonly unknown[];
      materialTextureLinearSources: readonly unknown[];
      materialTextureEmissiveSources: readonly unknown[];
    };
    // Each individual atlas remains below its 512 MiB peak budget at 4000²,
    // while the aggregate rgba8 + rgba8 + rgba16 upload peak exceeds it.
    const largeExternal = { width: 4000, height: 4000 };
    Object.assign(packed, {
      materialTextureSources: [largeExternal],
      materialTextureLinearSources: [largeExternal],
      materialTextureEmissiveSources: [largeExternal],
    });
    const stub = new FailureDevice();

    expect(() => uploadPackedScene(stub.device, packed)).toThrow(/aggregate material-atlas peak/);
    expect(stub.allocationCount).toBe(0);
    expect(stub.resources).toHaveLength(0);
  });

  it('cleans uploadPackedScene candidates at every allocation and upload stage', () => {
    installGpuConstStubs();
    const packed = buildPackedScene(scene());

    const allocationProbe = new FailureDevice();
    const probeScene = uploadPackedScene(allocationProbe.device, packed);
    const allocationStages = allocationProbe.allocationCount;
    probeScene.destroy();
    probeScene.destroy();
    expectCreatedDestroyed(allocationProbe, 0);
    expect(allocationStages).toBeGreaterThan(30);

    for (let stage = 1; stage <= allocationStages; stage += 1) {
      const stub = new FailureDevice();
      const start = stub.arm('allocation', stage);
      expect(() => uploadPackedScene(stub.device, buildPackedScene(scene()))).toThrow(
        'injected allocation failure',
      );
      expectCreatedDestroyed(stub, start);
    }

    const writeProbe = new FailureDevice();
    uploadPackedScene(writeProbe.device, buildPackedScene(scene())).destroy();
    const writeStages = writeProbe.writeCount;
    expect(writeStages).toBeGreaterThan(30);
    for (let stage = 1; stage <= writeStages; stage += 1) {
      const stub = new FailureDevice();
      const start = stub.arm('write', stage);
      expect(() => uploadPackedScene(stub.device, buildPackedScene(scene()))).toThrow(
        'injected upload failure',
      );
      expectCreatedDestroyed(stub, start);
    }

    const viewProbe = new FailureDevice();
    uploadPackedScene(viewProbe.device, buildPackedScene(scene())).destroy();
    const viewStages = viewProbe.viewCount;
    expect(viewStages).toBeGreaterThan(0);
    for (let stage = 1; stage <= viewStages; stage += 1) {
      const stub = new FailureDevice();
      const start = stub.arm('view', stage);
      expect(() => uploadPackedScene(stub.device, buildPackedScene(scene()))).toThrow(
        'injected view failure',
      );
      expectCreatedDestroyed(stub, start);
    }
  });

  it('preserves all live TLAS and CWBVH-root handles at each replacement stage', () => {
    installGpuConstStubs();
    const stub = new FailureDevice();
    const packed = buildPackedScene(scene());
    const sb = uploadPackedScene(stub.device, packed);
    const pack = scenePackResultFromPacked(packed);
    const previous = tlasHandles(sb);
    const previousRoots = new Uint32Array(sb.cwbvhTlasBlasRoots);
    expectRootPairsMatchTables(sb);
    // Force the sixth, CWBVH-root allocation stage as well as the five binary
    // TLAS stages; normal same-sized root updates remain in-place.
    (sb.cwbvhTlasBlasRootsBuffer as unknown as { size: number }).size = -1;

    for (let stage = 1; stage <= 6; stage += 1) {
      const start = stub.arm('allocation', stage);
      expect(() => uploadScenePackTlasRealloc(stub.device, sb, pack)).toThrow(
        'injected allocation failure',
      );
      expect(tlasHandles(sb)).toEqual(previous);
      expect(sb.cwbvhTlasBlasRoots).toEqual(previousRoots);
      expectRootPairsMatchTables(sb);
      for (const handle of previous as unknown as readonly TrackedResource[]) {
        expect(handle.destroy, handle.label).not.toHaveBeenCalled();
      }
      expectCreatedDestroyed(stub, start);
    }
    for (let stage = 1; stage <= 6; stage += 1) {
      const start = stub.arm('write', stage);
      expect(() => uploadScenePackTlasRealloc(stub.device, sb, pack)).toThrow(
        'injected upload failure',
      );
      expect(tlasHandles(sb)).toEqual(previous);
      expect(sb.cwbvhTlasBlasRoots).toEqual(previousRoots);
      expectRootPairsMatchTables(sb);
      for (const handle of previous as unknown as readonly TrackedResource[]) {
        expect(handle.destroy, handle.label).not.toHaveBeenCalled();
      }
      expectCreatedDestroyed(stub, start);
    }
  });

  it('preserves the complete BLAS/TLAS/CWBVH set at all 18 allocation stages', () => {
    installGpuConstStubs();
    const stub = new FailureDevice();
    const packed = buildPackedScene(scene());
    const sb = uploadPackedScene(stub.device, packed);
    const pack = scenePackResultFromPacked(packed);
    const previous = geometryHandles(sb);
    const previousRoots = new Uint32Array(sb.cwbvhTlasBlasRoots);
    expectRootPairsMatchTables(sb);
    // Force all five CWBVH buffers through their size-changing candidate path,
    // following the thirteen binary BLAS/TLAS allocation stages.
    for (const buffer of [
      sb.cwbvhNodeBoundsBuffer, sb.cwbvhChildBoundsPackedBuffer,
      sb.cwbvhChildMetaBuffer, sb.cwbvhChildCountBuffer,
      sb.cwbvhTlasBlasRootsBuffer,
    ]) {
      (buffer as unknown as { size: number }).size = -1;
    }

    for (let stage = 1; stage <= 18; stage += 1) {
      const start = stub.arm('allocation', stage);
      expect(() => uploadScenePackGeometryRealloc(stub.device, sb, pack)).toThrow(
        'injected allocation failure',
      );
      expect(geometryHandles(sb)).toEqual(previous);
      expect(sb.cwbvhTlasBlasRoots).toEqual(previousRoots);
      expectRootPairsMatchTables(sb);
      for (const handle of previous as unknown as readonly TrackedResource[]) {
        expect(handle.destroy, handle.label).not.toHaveBeenCalled();
      }
      expectCreatedDestroyed(stub, start);
    }
    for (let stage = 1; stage <= 18; stage += 1) {
      const start = stub.arm('write', stage);
      expect(() => uploadScenePackGeometryRealloc(stub.device, sb, pack)).toThrow(
        'injected upload failure',
      );
      expect(geometryHandles(sb)).toEqual(previous);
      expect(sb.cwbvhTlasBlasRoots).toEqual(previousRoots);
      expectRootPairsMatchTables(sb);
      for (const handle of previous as unknown as readonly TrackedResource[]) {
        expect(handle.destroy, handle.label).not.toHaveBeenCalled();
      }
      expectCreatedDestroyed(stub, start);
    }
  });

  it('rejects aliased full-upload texture/buffer candidates and destroys each unique resource once', () => {
    installGpuConstStubs();
    for (const kind of ['textures', 'buffers'] as const) {
      const stub = new FailureDevice();
      if (kind === 'textures') stub.aliasTextures = true;
      else stub.aliasBuffers = true;
      const start = stub.resources.length;

      expect(() => uploadPackedScene(stub.device, buildPackedScene(scene()))).toThrow(
        'aliased an existing',
      );
      expectCreatedDestroyed(stub, start);
    }
  });

  it('preserves a forbidden cross-generation resource returned by full upload allocation', () => {
    installGpuConstStubs();
    const stub = new FailureDevice();
    const preserved: TrackedResource = { label: 'previous-generation', size: 16, destroy: vi.fn() };
    stub.aliasBufferTo = preserved;
    const start = stub.resources.length;

    expect(() => uploadPackedScene(stub.device, buildPackedScene(scene()), [preserved])).toThrow(
      'aliased an existing GPU resource',
    );
    expect(preserved.destroy).not.toHaveBeenCalled();
    expectCreatedDestroyed(stub, start);
  });

  it('rejects a TLAS candidate that aliases a live handle without writes or retirement', () => {
    installGpuConstStubs();
    const stub = new FailureDevice();
    const packed = buildPackedScene(scene());
    const sb = uploadPackedScene(stub.device, packed);
    const pack = scenePackResultFromPacked(packed);
    const previous = tlasHandles(sb);
    const live = sb.tlasNodesBuffer as unknown as TrackedResource;
    const writesBefore = stub.queue.writeBuffer.mock.calls.length;
    stub.aliasBufferTo = live;

    expect(() => uploadScenePackTlasRealloc(stub.device, sb, pack)).toThrow(
      'aliased an existing GPU resource',
    );
    expect(tlasHandles(sb)).toEqual(previous);
    expect(stub.queue.writeBuffer).toHaveBeenCalledTimes(writesBefore);
    expect(live.destroy).not.toHaveBeenCalled();
  });

  it('deduplicates aliased old TLAS handles during successful retirement', () => {
    installGpuConstStubs();
    const stub = new FailureDevice();
    const packed = buildPackedScene(scene());
    const sb = uploadPackedScene(stub.device, packed);
    const pack = scenePackResultFromPacked(packed);
    const shared = sb.tlasNodesBuffer as unknown as TrackedResource;
    const displaced = sb.tlasInstanceIndicesBuffer as unknown as TrackedResource;
    (sb as unknown as { tlasInstanceIndicesBuffer: GPUBuffer }).tlasInstanceIndicesBuffer =
      sb.tlasNodesBuffer;

    uploadScenePackTlasRealloc(stub.device, sb, pack);

    expect(shared.destroy).toHaveBeenCalledOnce();
    expect(sb.tlasNodesBuffer).not.toBe(shared);
    displaced.destroy();
    sb.destroy();
    expect(shared.destroy).toHaveBeenCalledOnce();
  });

  it('rejects aliased same-size geometry handles before issuing any write', () => {
    installGpuConstStubs();
    const stub = new FailureDevice();
    const sb = uploadPackedScene(stub.device, buildPackedScene(scene()));
    const nextPack = scenePackResultFromPacked(buildPackedScene(scene()));
    nextPack.positions[0] = 42;
    const shared = sb.positionsBuffer as unknown as TrackedResource;
    (sb as unknown as { normalsBuffer: GPUBuffer }).normalsBuffer = sb.positionsBuffer;
    const writesBefore = stub.queue.writeBuffer.mock.calls.length;

    expect(() => uploadScenePackGeometry(stub.device, sb, nextPack)).toThrow(
      'transactional write set aliases one GPUBuffer',
    );
    expect(stub.queue.writeBuffer).toHaveBeenCalledTimes(writesBefore);
    expect(sb.positions[0]).toBe(0);
    expect(shared.destroy).not.toHaveBeenCalled();
  });

  it('rejects a CWBVH handle aliasing base geometry before touching either set', () => {
    installGpuConstStubs();
    const stub = new FailureDevice();
    const sb = uploadPackedScene(stub.device, buildPackedScene(scene()));
    const nextPack = scenePackResultFromPacked(buildPackedScene(scene()));
    nextPack.positions[0] = 42;
    const shared = sb.positionsBuffer as unknown as TrackedResource;
    (sb as unknown as { cwbvhChildMetaBuffer: GPUBuffer }).cwbvhChildMetaBuffer =
      sb.positionsBuffer;
    const writesBefore = stub.queue.writeBuffer.mock.calls.length;

    expect(() => uploadScenePackGeometry(stub.device, sb, nextPack)).toThrow(
      'live CWBVH resource aliases a protected transactional GPUBuffer',
    );
    expect(stub.queue.writeBuffer).toHaveBeenCalledTimes(writesBefore);
    expect(sb.positions[0]).toBe(0);
    expect(shared.destroy).not.toHaveBeenCalled();
  });

  it('rejects a CWBVH root aliasing binary TLAS before allocation or upload', () => {
    installGpuConstStubs();
    const stub = new FailureDevice();
    const packed = buildPackedScene(scene());
    const sb = uploadPackedScene(stub.device, packed);
    const pack = scenePackResultFromPacked(packed);
    const live = sb.tlasNodesBuffer as unknown as TrackedResource;
    (sb as unknown as { cwbvhTlasBlasRootsBuffer: GPUBuffer }).cwbvhTlasBlasRootsBuffer =
      sb.tlasNodesBuffer;
    const allocationsBefore = stub.allocationCount;
    const writesBefore = stub.queue.writeBuffer.mock.calls.length;

    expect(() => uploadScenePackTlasRealloc(stub.device, sb, pack)).toThrow(
      'live CWBVH TLAS-root buffer aliases a binary TLAS resource',
    );
    expect(stub.allocationCount).toBe(allocationsBefore);
    expect(stub.queue.writeBuffer).toHaveBeenCalledTimes(writesBefore);
    expect(live.destroy).not.toHaveBeenCalled();
  });

  it('rejects accumulation candidate aliases against both staged and live sets', () => {
    installGpuConstStubs();
    const stagedAliasStub = new FailureDevice();
    stagedAliasStub.aliasTextures = true;
    expect(() => new GpuResources(stagedAliasStub.device, 'full', false)
      .ensureAccumResources(4, 4)).toThrow('candidate aliased an existing GPU resource');
    expectCreatedDestroyed(stagedAliasStub, 0);

    const liveAliasStub = new FailureDevice();
    const gpu = new GpuResources(liveAliasStub.device, 'full', false);
    expect(gpu.ensureAccumResources(4, 4)).toBe(true);
    const previous = gpu.accumTexture as unknown as TrackedResource;
    liveAliasStub.aliasTextureTo = previous;
    expect(() => gpu.ensureAccumResources(8, 8)).toThrow(
      'candidate aliased an existing GPU resource',
    );
    expect(gpu.accumTexture).toBe(previous);
    expect(previous.destroy).not.toHaveBeenCalled();
  });

  it('keeps the previous accumulation set through every allocation/view/submit failure', () => {
    installGpuConstStubs();
    const stub = new FailureDevice();
    const gpu = new GpuResources(stub.device, 'full', false);
    expect(gpu.ensureAccumResources(4, 4)).toBe(true);
    const previous = [
      gpu.accumTexture, gpu.normalDepthTexture, gpu.albedoTexture,
      gpu.varianceTexture, gpu.motionVectorsTexture, gpu.accumBuffer,
      gpu.varianceMomentsBuffer, gpu.present.presentTexture,
    ] as const;

    for (let stage = 1; stage <= 8; stage += 1) {
      const start = stub.arm('allocation', stage);
      expect(() => gpu.ensureAccumResources(8, 8)).toThrow('injected allocation failure');
      expect([
        gpu.accumTexture, gpu.normalDepthTexture, gpu.albedoTexture,
        gpu.varianceTexture, gpu.motionVectorsTexture, gpu.accumBuffer,
        gpu.varianceMomentsBuffer, gpu.present.presentTexture,
      ]).toEqual(previous);
      expect(gpu.accumWidth).toBe(4);
      expectCreatedDestroyed(stub, start);
    }
    for (let stage = 1; stage <= 6; stage += 1) {
      const start = stub.arm('view', stage);
      expect(() => gpu.ensureAccumResources(8, 8)).toThrow('injected view failure');
      expect(gpu.accumTexture).toBe(previous[0]);
      expectCreatedDestroyed(stub, start);
    }
    const start = stub.arm('submit', 1);
    expect(() => gpu.ensureAccumResources(8, 8)).toThrow('injected submit failure');
    expect(gpu.accumTexture).toBe(previous[0]);
    expectCreatedDestroyed(stub, start);
    for (const handle of previous as readonly (TrackedResource | null)[]) {
      expect(handle?.destroy, handle?.label).not.toHaveBeenCalled();
    }
  });

  it('stages lite textures without touching the live trio and supports rollback', () => {
    installGpuConstStubs();
    const stub = new FailureDevice();
    const gpu = new GpuResources(stub.device, 'lite', false);
    const light = { width: 1, data: new Float32Array(4), lightCount: 0 };
    const env = { width: 1, height: 1, texels: new Float32Array(4) };
    const cdf = { width: 1, height: 1, data: new Float32Array(4) };
    const initial = gpu.stageLiteTextureReplacement(light, env, cdf);
    initial.commit();
    initial.finalize();
    const previous = [gpu.liteEnvTexture, gpu.liteEnvCdfTexture, gpu.liteLightTexture] as const;

    for (const kind of ['allocation', 'write', 'view'] as const) {
      for (let stage = 1; stage <= 3; stage += 1) {
        const start = stub.arm(kind, stage);
        expect(() => gpu.stageLiteTextureReplacement(light, env, cdf)).toThrow();
        expect([gpu.liteEnvTexture, gpu.liteEnvCdfTexture, gpu.liteLightTexture]).toEqual(previous);
        expectCreatedDestroyed(stub, start);
      }
    }
    stub.disarm();
    const liveEnv = previous[0] as unknown as TrackedResource;
    stub.aliasTextureTo = liveEnv;
    expect(() => gpu.stageLiteTextureReplacement(light, env, cdf)).toThrow(
      'candidate aliased an existing GPU texture',
    );
    expect([gpu.liteEnvTexture, gpu.liteEnvCdfTexture, gpu.liteLightTexture]).toEqual(previous);
    expect(liveEnv.destroy).not.toHaveBeenCalled();
    stub.aliasTextureTo = null;
    const external: TrackedResource = { label: 'scene-owned', size: 0, destroy: vi.fn() };
    stub.aliasTextureTo = external;
    expect(() => gpu.stageLiteTextureReplacement(light, env, cdf, [external])).toThrow(
      'candidate aliased an existing GPU texture',
    );
    expect(external.destroy).not.toHaveBeenCalled();
    stub.aliasTextureTo = null;


    const replacement = gpu.stageLiteTextureReplacement(light, env, cdf);
    replacement.commit();
    expect(gpu.liteEnvTexture).not.toBe(previous[0]);
    replacement.rollback();
    expect([gpu.liteEnvTexture, gpu.liteEnvCdfTexture, gpu.liteLightTexture]).toEqual(previous);
    for (const handle of previous as readonly (TrackedResource | null)[]) {
      expect(handle?.destroy, handle?.label).not.toHaveBeenCalled();
    }
  });

  it('clears every temporal history buffer through one submit boundary', () => {
    installGpuConstStubs();
    const stub = new FailureDevice();
    const gpu = new GpuResources(stub.device, 'full', false);
    gpu.ensureAccumResources(4, 4);
    const makeBuffer = (label: string): GPUBuffer => stub.device.createBuffer({
      label,
      size: 16,
      usage: GPUBufferUsage.STORAGE,
    });
    gpu.reservoir.rptReservoirCur = makeBuffer('reservoir-cur');
    gpu.reservoir.rptReservoirPrev = makeBuffer('reservoir-prev');
    gpu.sppm.sppmPixelStatsBuffer = makeBuffer('sppm-pixel-stats');
    gpu.sppm.sppmPixelStatsWidth = 4;
    const expected = new Set([
      gpu.accumBuffer,
      gpu.varianceMomentsBuffer,
      gpu.reservoir.rptReservoirCur,
      gpu.reservoir.rptReservoirPrev,
      gpu.sppm.sppmPixelStatsBuffer,
    ]);

    stub.arm('submit', 1);
    expect(() => gpu.clearTemporalBuffers()).toThrow('injected submit failure');
    expect(stub.queue.submit).toHaveBeenCalled();
    expect(stub.submitCount).toBe(1);
    expect(new Set(stub.clearCalls)).toEqual(expected);
    for (const resource of expected) {
      const tracked = resource as unknown as TrackedResource;
      expect(tracked.destroy, tracked.label).not.toHaveBeenCalled();
    }
  });
});
  it.each(['primitive', 'emitter', 'environment'] as const)(
    'rolls back every published scene field when %s reset fails',
    (kind) => {
      installGpuConstStubs();
      const liveScene: Scene = {
        primitives: [{
          kind: 'mesh',
          id: 'mesh',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [0.5, 0.5, 0.5],
            roughness: 0.5,
            metallic: 0,
          },
        }],
        emitters: [{
          kind: 'point',
          id: 'lamp',
          position: [0, 2, 0],
          color: [1, 0.8, 0.6],
          intensity: 2,
        }],
        environment: {
          kind: 'hdri',
          hdri: {
            width: 1,
            height: 1,
            data: new Float32Array([1, 1, 1]),
          },
          intensity: 1,
        },
      };
      const stub = new FailureDevice();
      const packed = buildPackedScene(liveScene);
      const sceneBuffers = uploadPackedScene(stub.device, packed);
      const originalGeoPack = scenePackResultFromPacked(packed);
      let sceneState = liveScene;
      let geoPack = originalGeoPack;
      const warn = vi.fn();
      const reset = vi.fn(() => {
        throw new Error('injected mutation reset failure');
      });
      const host: MutationHost = {
        device: stub.device,
        assertLive: vi.fn(),
        getScene: () => sceneState,
        setSceneState: vi.fn((next) => {
          sceneState = next;
        }),
        getSceneBuffers: () => sceneBuffers,
        getGeoPack: () => geoPack,
        setGeoPack: vi.fn((next) => {
          geoPack = next;
        }),
        invalidateBindGroups: vi.fn(),
        supportedAnalyticShapes: () => new Set<string>(),
        cameraVisibleEmitters: () => false,
        warn,
        repackScene: vi.fn(() => {
          throw new Error('unexpected repack');
        }),
        setScene: vi.fn(() => {
          throw new Error('unexpected setScene');
        }),
        reset,
      };
      const originalRegistry = SCENE_BUFFER_REGISTRY.map((entry) => ({
        entry,
        data: sceneBuffers[entry.key],
        buffer: sceneBuffers[entry.bufferField],
      }));
      const originalScalars = {
        triangleCount: sceneBuffers.triangleCount,
        tlasNodeCount: sceneBuffers.tlasNodeCount,
        primitiveTlasBindings: sceneBuffers.primitiveTlasBindings,
        pointLightCount: sceneBuffers.pointLightCount,
        lightTreeNodeCount: sceneBuffers.lightTreeNodeCount,
        environmentTint: sceneBuffers.environmentTint,
        environmentSunStrength: sceneBuffers.environmentSunStrength,
        environmentHdriIntensity: sceneBuffers.environmentHdriIntensity,
      };
      const candidateStart = stub.resources.length;
      const router = new SceneMutationRouter(host);

      const invoke = (): void => {
        if (kind === 'primitive') {
          router.updatePrimitive('mesh', {
            positions: new Float32Array([0.25, 0, 0, 1.25, 0, 0, 0.25, 1, 0]),
          });
        } else if (kind === 'emitter') {
          router.updateEmitter('lamp', { intensity: 7, position: [1, 3, 2] });
        } else {
          router.updateEnvironment({
            kind: 'hdri',
            hdri: {
              width: 1,
              height: 1,
              data: new Float32Array([2, 1, 0.5]),
            },
            intensity: 3,
          });
        }
      };

      expect(invoke).toThrow('injected mutation reset failure');
      expect(sceneState).toBe(liveScene);
      expect(geoPack).toBe(originalGeoPack);
      for (const snapshot of originalRegistry) {
        expect(sceneBuffers[snapshot.entry.key]).toBe(snapshot.data);
        expect(sceneBuffers[snapshot.entry.bufferField]).toBe(snapshot.buffer);
        expect((snapshot.buffer as unknown as TrackedResource).destroy).not.toHaveBeenCalled();
      }
      expect(sceneBuffers).toMatchObject(originalScalars);
      expectCreatedDestroyed(stub, candidateStart);
      // Same-size rollback restores bytes in the same handles, so both the
      // forward and rollback paths preserve cached bind-group identity.
      expect(host.invalidateBindGroups).not.toHaveBeenCalled();
      expect(reset).toHaveBeenCalledOnce();
      expect(warn).not.toHaveBeenCalled();
    },
  );
