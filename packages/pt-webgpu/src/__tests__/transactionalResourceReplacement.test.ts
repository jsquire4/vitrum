import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { GpuResources } from '../gpuResources.js';
import { SceneMutationRouter, type MutationHost } from '../sceneMutationRouter.js';
import {
  buildPackedScene,
  prepareSceneBufferMutation,
  scenePackResultFromPacked,
  SCENE_BUFFER_REGISTRY,
  uploadPackedScene,
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

function expectCreatedDestroyed(stub: FailureDevice, start: number): void {
  const candidates = stub.resources.slice(start);
  for (const candidate of candidates) {
    expect(candidate.destroy, candidate.label).toHaveBeenCalledOnce();
  }
}

interface SizeChangingFixture {
  readonly stub: FailureDevice;
  readonly sceneBuffers: UploadedSceneBuffers;
  readonly patch: {
    readonly positions: Float32Array;
    readonly normals: Float32Array;
    readonly triangleCount: number;
  };
  readonly previous: {
    readonly positions: Float32Array;
    readonly normals: Float32Array;
    readonly positionsBuffer: GPUBuffer;
    readonly normalsBuffer: GPUBuffer;
    readonly triangleCount: number;
  };
}

function sizeChangingFixture(): SizeChangingFixture {
  installGpuConstStubs();
  const stub = new FailureDevice();
  const sceneBuffers = uploadPackedScene(stub.device, buildPackedScene(scene()));
  const positions = new Float32Array(sceneBuffers.positions.length + 4);
  positions.set(sceneBuffers.positions);
  positions[positions.length - 4] = 2;
  const normals = new Float32Array(sceneBuffers.normals.length + 4);
  normals.set(sceneBuffers.normals);
  normals[normals.length - 2] = 1;
  return {
    stub,
    sceneBuffers,
    patch: {
      positions,
      normals,
      triangleCount: sceneBuffers.triangleCount + 1,
    },
    previous: {
      positions: sceneBuffers.positions,
      normals: sceneBuffers.normals,
      positionsBuffer: sceneBuffers.positionsBuffer,
      normalsBuffer: sceneBuffers.normalsBuffer,
      triangleCount: sceneBuffers.triangleCount,
    },
  };
}

function expectPreviousSceneGeneration(fixture: SizeChangingFixture): void {
  const { sceneBuffers, previous } = fixture;
  expect(sceneBuffers.positions).toBe(previous.positions);
  expect(sceneBuffers.normals).toBe(previous.normals);
  expect(sceneBuffers.positionsBuffer).toBe(previous.positionsBuffer);
  expect(sceneBuffers.normalsBuffer).toBe(previous.normalsBuffer);
  expect(sceneBuffers.triangleCount).toBe(previous.triangleCount);
  expect((previous.positionsBuffer as unknown as TrackedResource).destroy).not.toHaveBeenCalled();
  expect((previous.normalsBuffer as unknown as TrackedResource).destroy).not.toHaveBeenCalled();
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

describe('generic size-changing scene-buffer transactions', () => {
  it('preserves the live generation through every candidate allocation and upload failure', () => {
    const allocationProbe = sizeChangingFixture();
    const allocationProbeStart = allocationProbe.stub.arm('allocation', Number.MAX_SAFE_INTEGER);
    const allocationTransaction = prepareSceneBufferMutation(
      allocationProbe.stub.device,
      allocationProbe.sceneBuffers,
      allocationProbe.patch,
    );
    const allocationStages = allocationProbe.stub.allocationCount;
    expect(allocationStages).toBe(2);
    allocationTransaction.rollback();
    expectCreatedDestroyed(allocationProbe.stub, allocationProbeStart);
    expectPreviousSceneGeneration(allocationProbe);

    for (let stage = 1; stage <= allocationStages; stage += 1) {
      const fixture = sizeChangingFixture();
      const candidateStart = fixture.stub.arm('allocation', stage);
      expect(() => prepareSceneBufferMutation(
        fixture.stub.device,
        fixture.sceneBuffers,
        fixture.patch,
      )).toThrow('injected allocation failure');
      expectPreviousSceneGeneration(fixture);
      expectCreatedDestroyed(fixture.stub, candidateStart);
    }

    const writeProbe = sizeChangingFixture();
    const writeProbeStart = writeProbe.stub.arm('write', Number.MAX_SAFE_INTEGER);
    const writeTransaction = prepareSceneBufferMutation(
      writeProbe.stub.device,
      writeProbe.sceneBuffers,
      writeProbe.patch,
    );
    const writeStages = writeProbe.stub.writeCount;
    expect(writeStages).toBe(2);
    writeTransaction.rollback();
    expectCreatedDestroyed(writeProbe.stub, writeProbeStart);
    expectPreviousSceneGeneration(writeProbe);

    for (let stage = 1; stage <= writeStages; stage += 1) {
      const fixture = sizeChangingFixture();
      const candidateStart = fixture.stub.arm('write', stage);
      expect(() => prepareSceneBufferMutation(
        fixture.stub.device,
        fixture.sceneBuffers,
        fixture.patch,
      )).toThrow('injected upload failure');
      expectPreviousSceneGeneration(fixture);
      expectCreatedDestroyed(fixture.stub, candidateStart);
    }
  });

  it('rejects live and candidate-to-candidate aliases without destroying live resources', () => {
    const liveAlias = sizeChangingFixture();
    const liveCandidateStart = liveAlias.stub.arm('allocation', Number.MAX_SAFE_INTEGER);
    liveAlias.stub.aliasBufferTo =
      liveAlias.previous.positionsBuffer as unknown as TrackedResource;
    expect(() => prepareSceneBufferMutation(
      liveAlias.stub.device,
      liveAlias.sceneBuffers,
      liveAlias.patch,
    )).toThrow('candidate allocation for vitrum.pt-webgpu.scene.positions aliased an existing GPU resource');
    expectPreviousSceneGeneration(liveAlias);
    expectCreatedDestroyed(liveAlias.stub, liveCandidateStart);

    const stagedAlias = sizeChangingFixture();
    const stagedCandidateStart = stagedAlias.stub.arm('allocation', Number.MAX_SAFE_INTEGER);
    stagedAlias.stub.aliasBuffers = true;
    expect(() => prepareSceneBufferMutation(
      stagedAlias.stub.device,
      stagedAlias.sceneBuffers,
      stagedAlias.patch,
    )).toThrow('candidate allocation for vitrum.pt-webgpu.scene.normals aliased an existing GPU resource');
    expectPreviousSceneGeneration(stagedAlias);
    expect(stagedAlias.stub.resources.slice(stagedCandidateStart)).toHaveLength(1);
    expectCreatedDestroyed(stagedAlias.stub, stagedCandidateStart);
  });

  it('restores committed mirrors and handles on rollback and destroys candidates once', () => {
    const fixture = sizeChangingFixture();
    const candidateStart = fixture.stub.arm('allocation', Number.MAX_SAFE_INTEGER);
    const transaction = prepareSceneBufferMutation(
      fixture.stub.device,
      fixture.sceneBuffers,
      fixture.patch,
    );
    const candidatePositionsBuffer = transaction.preview.positionsBuffer;
    const candidateNormalsBuffer = transaction.preview.normalsBuffer;

    expectPreviousSceneGeneration(fixture);
    expect(transaction.replacesBufferHandles).toBe(true);
    expect(transaction.preview.positions).toEqual(fixture.patch.positions);
    expect(transaction.preview.positions).not.toBe(fixture.patch.positions);
    transaction.commit();
    expect(fixture.sceneBuffers.positions).toBe(transaction.preview.positions);
    expect(fixture.sceneBuffers.normals).toBe(transaction.preview.normals);
    expect(fixture.sceneBuffers.positionsBuffer).toBe(candidatePositionsBuffer);
    expect(fixture.sceneBuffers.normalsBuffer).toBe(candidateNormalsBuffer);
    expect(fixture.sceneBuffers.triangleCount).toBe(fixture.patch.triangleCount);

    transaction.rollback();
    transaction.rollback();
    expectPreviousSceneGeneration(fixture);
    expectCreatedDestroyed(fixture.stub, candidateStart);
  });

  it('finalizes a committed generation and retires duplicate old handles exactly once', () => {
    const fixture = sizeChangingFixture();
    const sharedPreviousBuffer = fixture.previous.positionsBuffer as unknown as TrackedResource;
    const displacedNormalsBuffer =
      fixture.previous.normalsBuffer as unknown as TrackedResource;
    (fixture.sceneBuffers as unknown as { normalsBuffer: GPUBuffer }).normalsBuffer =
      fixture.previous.positionsBuffer;
    const candidateStart = fixture.stub.arm('allocation', Number.MAX_SAFE_INTEGER);
    const transaction = prepareSceneBufferMutation(
      fixture.stub.device,
      fixture.sceneBuffers,
      fixture.patch,
    );
    const candidateBuffers = [
      transaction.preview.positionsBuffer,
      transaction.preview.normalsBuffer,
    ] as const;

    transaction.commit();
    transaction.finalize();
    transaction.finalize();

    expect(fixture.sceneBuffers.positions).toBe(transaction.preview.positions);
    expect(fixture.sceneBuffers.normals).toBe(transaction.preview.normals);
    expect(fixture.sceneBuffers.triangleCount).toBe(fixture.patch.triangleCount);
    expect(sharedPreviousBuffer.destroy).toHaveBeenCalledOnce();
    expect(displacedNormalsBuffer.destroy).not.toHaveBeenCalled();
    for (const candidate of candidateBuffers) {
      expect((candidate as unknown as TrackedResource).destroy).not.toHaveBeenCalled();
    }

    fixture.sceneBuffers.destroy();
    for (const candidate of fixture.stub.resources.slice(candidateStart)) {
      expect(candidate.destroy, candidate.label).toHaveBeenCalledOnce();
    }
    displacedNormalsBuffer.destroy();
    expect(displacedNormalsBuffer.destroy).toHaveBeenCalledOnce();
  });

  it('allocates 32-byte BVH and TLAS placeholder bindings for a primitive-less scene', () => {
    installGpuConstStubs();
    const stub = new FailureDevice();
    const packed = buildPackedScene({
      primitives: [],
      emitters: [],
      environment: { kind: 'none' },
    });
    expect(packed.bvhNodes.byteLength).toBe(0);
    expect(packed.tlasNodes.byteLength).toBe(0);

    const sceneBuffers = uploadPackedScene(stub.device, packed);
    const resource = (label: string): TrackedResource => {
      const found = stub.resources.find((candidate) => candidate.label === label);
      expect(found, label).toBeDefined();
      return found!;
    };
    expect(resource('vitrum.pt-webgpu.scene.bvhNodes').size).toBe(32);
    expect(resource('vitrum.pt-webgpu.scene.tlasNodes').size).toBe(32);
    sceneBuffers.destroy();
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
        validatePrimitiveCandidate: vi.fn(),
        validateEmitterCandidate: vi.fn(),
        validateEnvironmentCandidate: vi.fn(),
        validateEmittersCandidate: vi.fn(),
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
