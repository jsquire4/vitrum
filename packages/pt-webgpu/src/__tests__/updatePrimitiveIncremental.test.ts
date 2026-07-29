import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning, Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

function installWebGpuConstStubs(): void {
  installGpuConstStubs();
}

function makeScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0.1 },
      },
      {
        kind: 'mesh',
        id: 'mesh-b',
        positions: new Float32Array([0, 0, 1, 1, 0, 1, 0, 1, 1]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.1, 0.4, 0.9], roughness: 0.6, metallic: 0.2 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function makeInstancedScene(): Scene {
  return {
    primitives: [
      {
        kind: 'instanced-mesh',
        id: 'instanced-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.7, 0.7, 0.7], roughness: 0.4, metallic: 0.1 },
        instances: [
          asMat4(new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
          ])),
          asMat4(new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            2, 0, 0, 1,
          ])),
        ],
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function makeAnalyticScene(): Scene {
  return {
    primitives: [
      {
        kind: 'analytic',
        id: 'analytic-a',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 0.5]),
        material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.5, metallic: 0.1 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

interface StubBuffer {
  readonly label: string;
  readonly size: number;
  readonly usage: number;
  destroy: ReturnType<typeof vi.fn>;
}

function makeStubDevice() {
  const writeBuffer = vi.fn();
  const createBuffer = vi.fn((desc: GPUBufferDescriptor): StubBuffer => ({
    label: desc.label ?? '',
    size: Number(desc.size),
    usage: Number(desc.usage),
    destroy: vi.fn(),
  }));
  const copyBufferToBuffer = vi.fn();
  const finish = vi.fn(() => ({}));
  const copyTextureToTexture = vi.fn();
  const createCommandEncoder = vi.fn(() => ({
    copyBufferToBuffer,
    copyTextureToTexture,
    finish,
  }));
  const submit = vi.fn();
  const device = {
    queue: { writeBuffer, writeTexture: vi.fn(), submit },
    createBuffer,
    ...textureStubMethods(),
    createCommandEncoder,
    limits: {
      maxStorageBuffersPerShaderStage: 64,
      maxTextureDimension2D: 8192,
      maxTextureArrayLayers: 256,
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
  return {
    device,
    writeBuffer,
    createBuffer,
    createCommandEncoder,
    copyBufferToBuffer,
    finish,
    submit,
  };
}

function makeLiteStubDevice() {
  const out = makeStubDevice();
  (out.device as unknown as { limits: Record<string, number> }).limits = {
    maxStorageBuffersPerShaderStage: 8,
    maxStorageTexturesPerShaderStage: 4,
    maxTextureDimension2D: 8192,
    maxTextureArrayLayers: 256,
  };
  return out;
}

interface SpyLike {
  readonly mock: {
    readonly calls: readonly unknown[][];
    readonly results: readonly { readonly type: string; readonly value: unknown }[];
  };
}

/** Total `.destroy()` calls across every stub buffer created so far. */
function totalDestroyCalls(createBuffer: SpyLike): number {
  let total = 0;
  for (const result of createBuffer.mock.results) {
    if (result.type === 'return') {
      total += (result.value as StubBuffer).destroy.mock.calls.length;
    }
  }
  return total;
}

/** Labels of buffers created since `sinceIndex` (exclusive of earlier ones). */
function labelsCreatedSince(createBuffer: SpyLike, sinceIndex: number): string[] {
  return createBuffer.mock.results
    .slice(sinceIndex)
    .filter((r) => r.type === 'return')
    .map((r) => (r.value as StubBuffer).label);
}

interface TransferMetrics {
  readonly stagingAllocations: number;
  readonly stagingBytes: number;
  readonly writeCount: number;
  readonly writeBytes: number;
  readonly copyCount: number;
  readonly copyBytes: number;
  readonly submitCount: number;
  readonly copyRanges: readonly string[];
}

function transferMetrics(
  createBuffer: SpyLike,
  writeBuffer: SpyLike,
  copyBufferToBuffer: SpyLike,
  submit: SpyLike,
  starts: {
    readonly buffers: number;
    readonly writes: number;
    readonly copies: number;
    readonly submits: number;
  },
): TransferMetrics {
  const created = createBuffer.mock.calls.slice(starts.buffers);
  const writes = writeBuffer.mock.calls.slice(starts.writes);
  const copies = copyBufferToBuffer.mock.calls.slice(starts.copies);
  return {
    stagingAllocations: created.length,
    stagingBytes: created.reduce(
      (total, call) => total + Number((call[0] as GPUBufferDescriptor).size),
      0,
    ),
    writeCount: writes.length,
    writeBytes: writes.reduce(
      (total, call) => total + Number(call[4] ?? 0),
      0,
    ),
    copyCount: copies.length,
    copyBytes: copies.reduce(
      (total, call) => total + Number(call[4] ?? 0),
      0,
    ),
    submitCount: submit.mock.calls.length - starts.submits,
    copyRanges: copies.map((call) => {
      const destination = call[2] as StubBuffer;
      return `${destination.label}@${String(call[3])}+${String(call[4])}`;
    }),
  };
}

interface TransferStarts {
  readonly buffers: number;
  readonly writes: number;
  readonly copies: number;
  readonly submits: number;
}

function expectInPlaceTransfer(
  createBuffer: SpyLike,
  writeBuffer: SpyLike,
  copyBufferToBuffer: SpyLike,
  submit: SpyLike,
  starts: TransferStarts,
  expected: TransferMetrics,
): void {
  expect(transferMetrics(
    createBuffer,
    writeBuffer,
    copyBufferToBuffer,
    submit,
    starts,
  )).toEqual(expected);

  const created = createBuffer.mock.results
    .slice(starts.buffers)
    .filter((result) => result.type === 'return')
    .map((result) => result.value as StubBuffer);
  expect(created).toHaveLength(1);
  const staging = created[0]!;
  expect(staging.label).toBe('vitrum.pt-webgpu.scene.incremental-staging');
  // The allocation is exactly the packed dirty ranges: no whole-buffer copy
  // and no hidden padding beyond the already 4-byte-aligned ranges.
  expect(staging.size).toBe(expected.copyBytes);
  expect(staging.destroy).toHaveBeenCalledTimes(1);

  const writes = writeBuffer.mock.calls.slice(starts.writes);
  expect(writes).toHaveLength(1);
  expect(writes[0]?.[0]).toBe(staging);
  expect(writes[0]?.[1]).toBe(0);
  expect(writes[0]?.[4]).toBe(staging.size);

  const liveBuffers = new Set(
    createBuffer.mock.results
      .slice(0, starts.buffers)
      .filter((result) => result.type === 'return')
      .map((result) => result.value),
  );
  const previousEndByDestination = new Map<unknown, number>();
  let expectedStagingOffset = 0;
  for (const call of copyBufferToBuffer.mock.calls.slice(starts.copies)) {
    const [source, sourceOffset, destination, destinationOffset, byteLength] = call;
    expect(source).toBe(staging);
    expect(sourceOffset).toBe(expectedStagingOffset);
    expect(liveBuffers.has(destination)).toBe(true);
    expect(Number(sourceOffset) % 4).toBe(0);
    expect(Number(destinationOffset) % 4).toBe(0);
    expect(Number(byteLength) % 4).toBe(0);
    const previousEnd = previousEndByDestination.get(destination);
    if (previousEnd != null) {
      // Equal means two adjacent words escaped coalescing.
      expect(Number(destinationOffset)).toBeGreaterThan(previousEnd);
    }
    previousEndByDestination.set(
      destination,
      Number(destinationOffset) + Number(byteLength),
    );
    expectedStagingOffset += Number(byteLength);
  }
  expect(expectedStagingOffset).toBe(staging.size);

  // No initial/live buffer was retired; only the transient staging allocation
  // is destroyed after submission.
  for (const result of createBuffer.mock.results.slice(0, starts.buffers)) {
    if (result.type === 'return') {
      expect((result.value as StubBuffer).destroy).not.toHaveBeenCalled();
    }
  }
}

/** The most recent `writeBuffer` payload (as a typed array) for a buffer whose
 *  label contains `labelFragment`. Returns null if none. */
function lastWriteForLabel(
  _createBuffer: SpyLike,
  device: GPUDevice,
  labelFragment: string,
): Float32Array | null {
  const writeBuffer = (device.queue.writeBuffer as unknown) as SpyLike;
  for (let i = writeBuffer.mock.calls.length - 1; i >= 0; i -= 1) {
    const call = writeBuffer.mock.calls[i]!;
    const target = call[0] as StubBuffer;
    if (target?.label?.includes(labelFragment)) {
      const arrayBuffer = call[2] as ArrayBuffer;
      const byteOffset = (call[3] as number | undefined) ?? 0;
      const byteLength = (call[4] as number | undefined) ?? arrayBuffer.byteLength - byteOffset;
      return new Float32Array(arrayBuffer.slice(byteOffset, byteOffset + byteLength));
    }
  }
  return null;
}

const TLAS_LABELS = [
  'tlasNodes',
  'tlasInstanceIndices',
  'tlasBlasRoots',
  'tlasInstanceWorldToLocal',
  'tlasInstanceLocalToWorld',
  'cwbvhTlasBlasRoots',
];
const BLAS_LABELS = [
  'scene.positions',
  'scene.normals',
  'scene.uvs',
  'scene.tangents',
  'scene.colors',
  'scene.indices',
  'cwbvhNodeBounds',
  'cwbvhChildBoundsPacked',
  'cwbvhChildMeta',
  'cwbvhChildCount',
  'scene.triMaterialIds',
  'scene.bvhNodes',
];

describe('pt-webgpu incremental primitive updates', () => {
  it('advertises material-only incremental patch support', async () => {
    const { device } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    const patchSupport = engine.capabilities.incrementalPatchSupport;
    expect(engine.capabilities.supportsIncrementalScene).toBe(true);
    expect(patchSupport?.material).toBe(true);
    expect(patchSupport?.positions).toBe(true);
    expect(patchSupport?.transform).toBe(true);
  });

  it('copies only dirty geometry words into unchanged live buffers', async () => {
    installWebGpuConstStubs();
    const {
      device,
      writeBuffer,
      createBuffer,
      copyBufferToBuffer,
      submit,
    } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    const transferStarts = {
      buffers: buffersBefore,
      writes: writesBefore,
      copies: copyBufferToBuffer.mock.calls.length,
      submits: submit.mock.calls.length,
    };
    const shifted = new Float32Array([0.1, 0, 0, 1.1, 0, 0, 0.1, 1, 0]);
    engine.updatePrimitive?.('mesh-a', { positions: shifted });

    expectInPlaceTransfer(
      createBuffer,
      writeBuffer,
      copyBufferToBuffer,
      submit,
      transferStarts,
      {
        stagingAllocations: 1,
        stagingBytes: 68,
        writeCount: 1,
        writeBytes: 68,
        copyCount: 15,
        copyBytes: 68,
        submitCount: 1,
        copyRanges: [
          'vitrum.pt-webgpu.scene.positions@0+4',
          'vitrum.pt-webgpu.scene.positions@16+4',
          'vitrum.pt-webgpu.scene.positions@32+4',
          'vitrum.pt-webgpu.scene.bvhNodes@0+4',
          'vitrum.pt-webgpu.scene.bvhNodes@12+4',
          'vitrum.pt-webgpu.scene.cwbvhNodeBounds@0+4',
          'vitrum.pt-webgpu.scene.cwbvhNodeBounds@12+4',
          'vitrum.pt-webgpu.scene.tlasNodes@12+4',
          'vitrum.pt-webgpu.scene.tlasNodes@28+4',
          'vitrum.pt-webgpu.scene.tlasNodes@40+4',
          'vitrum.pt-webgpu.scene.tlasNodes@52+4',
          'vitrum.pt-webgpu.scene.tlasNodes@64+4',
          'vitrum.pt-webgpu.scene.tlasNodes@72+8',
          'vitrum.pt-webgpu.scene.tlasNodes@84+4',
          'vitrum.pt-webgpu.scene.tlasInstanceIndices@0+8',
        ],
      },
    );
  });

  it('copies only dirty material words into the unchanged live buffer', async () => {
    installWebGpuConstStubs();
    const {
      device,
      writeBuffer,
      createBuffer,
      copyBufferToBuffer,
      submit,
    } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    expect(typeof engine.updatePrimitive).toBe('function');
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    const transferStarts = {
      buffers: buffersBefore,
      writes: writesBefore,
      copies: copyBufferToBuffer.mock.calls.length,
      submits: submit.mock.calls.length,
    };

    engine.updatePrimitive?.('mesh-b', {
      material: { baseColor: [0.2, 0.7, 0.9], roughness: 0.05, metallic: 0.4 },
    });

    expectInPlaceTransfer(
      createBuffer, writeBuffer, copyBufferToBuffer, submit, transferStarts,
      {
        stagingAllocations: 1,
        stagingBytes: 28,
        writeCount: 1,
        writeBytes: 28,
        copyCount: 4,
        copyBytes: 28,
        submitCount: 1,
        copyRanges: [
          'vitrum.pt-webgpu.scene.materials@464+8',
          'vitrum.pt-webgpu.scene.materials@476+4',
          'vitrum.pt-webgpu.scene.materials@492+4',
          'vitrum.pt-webgpu.scene.materials@880+12',
        ],
      },
    );
  });

  it('rebuilds rather than warning unsupported-material-fields for layered normal patch descriptors', async () => {
    installWebGpuConstStubs();
    const { device } = makeStubDevice();
    const structured: EngineWarning[] = [];
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const engine = await createPTEngine_WebGPU({
        device,
        onWarning: (w) => structured.push(w),
      });
      const baseScene = makeScene();
      engine.setScene({
        ...baseScene,
        primitives: baseScene.primitives.map((primitive) =>
          primitive.id === 'mesh-a'
            ? { ...primitive, uvs: new Float32Array([0, 0, 1, 0, 0, 1]) }
            : primitive,
        ),
      });
      structured.length = 0;

      engine.updatePrimitive?.('mesh-a', {
        material: {
          baseColor: [0.2, 0.7, 0.9],
          roughness: 0.05,
          metallic: 0.4,
          frontLayer: {
            transmission: [0.9, 0.8, 0.7],
            roughness: 0.2,
            normalMap: {
              handle: {
                id: 'front-normal', width: 1, height: 1,
                data: new Uint8Array([128, 128, 255, 255]),
              },
            },
            normalScale: 0.75,
          },
          backLayer: {
            transmission: [0.7, 0.8, 0.9],
            roughness: 0.6,
            normalMap: {
              handle: {
                id: 'back-normal', width: 1, height: 1,
                data: new Uint8Array([128, 128, 255, 255]),
              },
            },
            normalScale: 0.5,
          },
        },
      });

      expect(structured.some((w) =>
        w.code === 'pt-webgpu.unsupported-material-fields' &&
        Array.isArray(w.details?.fields) &&
        (
          w.details.fields.includes('frontLayer.normalMap') ||
          w.details.fields.includes('frontLayer.normalScale') ||
          w.details.fields.includes('backLayer.normalMap') ||
          w.details.fields.includes('backLayer.normalScale')
        ),
      )).toBe(false);
    } finally {
      warn.mockRestore();
    }
  });

  it('lite tier material patches preserve the live material buffer', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeLiteStubDevice();
    const engine = await createPTEngine_WebGPU({ device, traceTier: 'lite' });
    expect(engine.capabilities.incrementalPatchSupport?.material).toBe(true);
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    engine.updatePrimitive?.('mesh-b', {
      material: { baseColor: [0.2, 0.7, 0.9], roughness: 0.05, metallic: 0.4 },
    });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore + 1);
    expect(writeBuffer.mock.calls.length).toBe(writesBefore + 1);

    const lastWrite = writeBuffer.mock.calls[writeBuffer.mock.calls.length - 1];
    const writeByteOffset = lastWrite?.[1];
    expect(writeByteOffset).toBe(0);
  });

  it('copies only dirty material and descriptor scalar words in place', async () => {
    installWebGpuConstStubs();
    const {
      device,
      writeBuffer,
      createBuffer,
      copyBufferToBuffer,
      submit,
    } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    const transferStarts = {
      buffers: buffersBefore,
      writes: writesBefore,
      copies: copyBufferToBuffer.mock.calls.length,
      submits: submit.mock.calls.length,
    };

    engine.updatePrimitive?.('mesh-b', {
      material: {
        baseColor: [0.2, 0.7, 0.9],
        roughness: 0.05,
        metallic: 0.4,
        alphaMode: 'blend',
        opacity: 0.5,
        normalScale: 0.6,
        envMapIntensity: 0.25,
        anisotropy: 0.4,
      },
    });

    expectInPlaceTransfer(
      createBuffer, writeBuffer, copyBufferToBuffer, submit, transferStarts,
      {
        stagingAllocations: 1,
        stagingBytes: 48,
        writeCount: 1,
        writeBytes: 48,
        copyCount: 8,
        copyBytes: 48,
        submitCount: 1,
        copyRanges: [
          'vitrum.pt-webgpu.scene.materials@464+8',
          'vitrum.pt-webgpu.scene.materials@476+4',
          'vitrum.pt-webgpu.scene.materials@492+4',
          'vitrum.pt-webgpu.scene.materials@880+12',
          'vitrum.pt-webgpu.scene.materialTexDescriptors@1616+4',
          'vitrum.pt-webgpu.scene.materialTexDescriptors@1624+4',
          'vitrum.pt-webgpu.scene.materialTexDescriptors@1676+8',
          'vitrum.pt-webgpu.scene.materialTexDescriptors@1692+4',
        ],
      },
    );
  });

  it('lite tier rejects full-tier-only material patches before mutating scene buffers', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeLiteStubDevice();
    const structured: EngineWarning[] = [];
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const engine = await createPTEngine_WebGPU({
        device,
        traceTier: 'lite',
        onWarning: (w) => structured.push(w),
      });
      engine.setScene(makeScene());
      structured.length = 0;

      const writesBefore = writeBuffer.mock.calls.length;
      const buffersBefore = createBuffer.mock.calls.length;

      expect(() => engine.updatePrimitive?.('mesh-b', {
        material: {
          baseColor: [0.2, 0.7, 0.9],
          roughness: 0.05,
          metallic: 0.4,
          alphaMode: 'blend',
          opacity: 0.5,
          normalScale: 0.6,
          envMapIntensity: 0.25,
          anisotropy: 0.4,
        },
      })).toThrow(/selected lite tier: alphaMode, anisotropy, envMapIntensity, normalScale, opacity/);

      expect(createBuffer.mock.calls.length).toBe(buffersBefore);
      expect(writeBuffer.mock.calls.length).toBe(writesBefore);
      expect(structured).toEqual([]);

      // A subsequent supported patch still updates the original live scene,
      // proving the rejected authored snapshot was never published.
      engine.updatePrimitive?.('mesh-b', {
        material: {
          baseColor: [0.25, 0.5, 0.75],
          roughness: 0.2,
          metallic: 0.1,
        },
      });
      expect(writeBuffer.mock.calls.length).toBeGreaterThan(writesBefore);
    } finally {
      warn.mockRestore();
    }
  });

  it('lite tier accepts same-count geometry patches via fallback merged-BLAS repack', async () => {
    installWebGpuConstStubs();
    const { device, createBuffer } = makeLiteStubDevice();
    const structured: EngineWarning[] = [];
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const engine = await createPTEngine_WebGPU({
        device,
        traceTier: 'lite',
        onWarning: (w) => structured.push(w),
      });
      expect(engine.capabilities.incrementalPatchSupport?.positions).toBe(true);
      expect(engine.capabilities.supportDetails?.mutations.positions).toBe('fallback-rebuild');
      engine.setScene(makeScene());
      const buffersBefore = createBuffer.mock.calls.length;

      engine.updatePrimitive?.('mesh-a', {
        positions: new Float32Array([0.1, 0, 0, 1.1, 0, 0, 0.1, 1, 0]),
      });

      expect(createBuffer.mock.calls.length).toBeGreaterThan(buffersBefore);
      expect(structured.some((w) =>
        w.code === 'pt-webgpu.lite-update-primitive-fallback-rebuild' &&
        w.details?.id === 'mesh-a' &&
        w.details?.fallbackReason === 'lite-merged-blas-geometry-rebuild',
      )).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  it('lite tier accepts mesh topology patches via fallback merged-BLAS repack', async () => {
    installWebGpuConstStubs();
    const { device, createBuffer } = makeLiteStubDevice();
    const structured: EngineWarning[] = [];
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const engine = await createPTEngine_WebGPU({
        device,
        traceTier: 'lite',
        onWarning: (w) => structured.push(w),
      });
      engine.setScene(makeScene());
      const buffersBefore = createBuffer.mock.calls.length;

      engine.updatePrimitive?.('mesh-b', {
        positions: new Float32Array([0, 0, 2, 1, 0, 2, 0, 1, 2, 1, 1, 2]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
        indices: new Uint32Array([0, 1, 2, 1, 3, 2]),
      });

      expect(createBuffer.mock.calls.length).toBeGreaterThan(buffersBefore);
      expect(structured.some((w) =>
        w.code === 'pt-webgpu.lite-update-primitive-fallback-rebuild' &&
        w.details?.id === 'mesh-b' &&
        w.details?.fallbackReason === 'lite-merged-blas-mesh-topology-rebuild',
      )).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  it('splices a vertex-count change, replacing only the 18 geometry/TLAS buffers', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    expect(typeof engine.updatePrimitive).toBe('function');
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;
    const destroysBefore = totalDestroyCalls(createBuffer);

    // Grow mesh-b from 3 verts / 1 tri to 4 verts / 2 tris (a quad). Slice-2
    // rebuilds ONLY mesh-b's BLAS, splices it into the concat buffers, and
    // replaces exactly 12 BLAS/CWBVH buffers (incl. UV/tangent/color) plus
    // 6 TLAS/CWBVH-root buffers — not material, analytic, or light buffers.
    engine.updatePrimitive?.('mesh-b', {
      positions: new Float32Array([0, 0, 2, 1, 0, 2, 0, 1, 2, 1, 1, 2]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2, 1, 3, 2]),
    });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore + 18);
    expect(totalDestroyCalls(createBuffer) - destroysBefore).toBe(18);
    const created = labelsCreatedSince(createBuffer, buffersBefore);
    expect(
      created.every(
        (l) => BLAS_LABELS.some((b) => l.includes(b)) || TLAS_LABELS.some((t) => l.includes(t)),
      ),
    ).toBe(true);
    expect(created.some((l) => l.includes('tangents'))).toBe(true);
    expect(created.some((l) => l.includes('colors'))).toBe(true);
    // No non-geometry buffer was recreated (materials/analytic/lights stay put).
    expect(created.some((l) => l.includes('materials'))).toBe(false);
    expect(created.some((l) => l.includes('analytic'))).toBe(false);
    expect(created.some((l) => l.includes('Lights'))).toBe(false);
    expect(writeBuffer.mock.calls.length).toBeGreaterThan(writesBefore);
  });

  it('copies only dirty TLAS words for transform-only mesh patches', async () => {
    installWebGpuConstStubs();
    const {
      device,
      writeBuffer,
      createBuffer,
      copyBufferToBuffer,
      submit,
    } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    const transferStarts = {
      buffers: buffersBefore,
      writes: writesBefore,
      copies: copyBufferToBuffer.mock.calls.length,
      submits: submit.mock.calls.length,
    };

    engine.updatePrimitive?.('mesh-b', {
      transform: asMat4(new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        2, 0, 0, 1,
      ])),
    });

    expectInPlaceTransfer(
      createBuffer, writeBuffer, copyBufferToBuffer, submit, transferStarts,
      {
        stagingAllocations: 1,
        stagingBytes: 20,
        writeCount: 1,
        writeBytes: 20,
        copyCount: 5,
        copyBytes: 20,
        submitCount: 1,
        copyRanges: [
          'vitrum.pt-webgpu.scene.tlasNodes@12+4',
          'vitrum.pt-webgpu.scene.tlasNodes@64+4',
          'vitrum.pt-webgpu.scene.tlasNodes@76+4',
          'vitrum.pt-webgpu.scene.tlasInstanceWorldToLocal@112+4',
          'vitrum.pt-webgpu.scene.tlasInstanceLocalToWorld@112+4',
        ],
      },
    );
  });

  it('lite tier accepts transform-only mesh patches via fallback merged-BLAS repack', async () => {
    installWebGpuConstStubs();
    const { device, createBuffer } = makeLiteStubDevice();
    const structured: EngineWarning[] = [];
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const engine = await createPTEngine_WebGPU({
        device,
        onWarning: (w) => structured.push(w),
      });
      engine.setScene(makeScene());
      const buffersBefore = createBuffer.mock.calls.length;

      engine.updatePrimitive?.('mesh-b', {
        transform: asMat4(new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          2, 0, 0, 1,
        ])),
      });

      expect(createBuffer.mock.calls.length).toBeGreaterThan(buffersBefore);
      expect(structured.some((w) =>
        w.code === 'pt-webgpu.lite-update-primitive-fallback-rebuild' &&
        w.details?.id === 'mesh-b' &&
        w.details?.fallbackReason === 'lite-merged-blas-transform-rebuild',
      )).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  it('copies only dirty TLAS words for instanced transform patches', async () => {
    installWebGpuConstStubs();
    const {
      device,
      writeBuffer,
      createBuffer,
      copyBufferToBuffer,
      submit,
    } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeInstancedScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    const transferStarts = {
      buffers: buffersBefore,
      writes: writesBefore,
      copies: copyBufferToBuffer.mock.calls.length,
      submits: submit.mock.calls.length,
    };

    engine.updatePrimitive?.('instanced-a', {
      instances: [
        asMat4(new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          1, 0, 0, 1,
        ])),
        asMat4(new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          3, 0, 0, 1,
        ])),
      ],
    });

    expectInPlaceTransfer(
      createBuffer, writeBuffer, copyBufferToBuffer, submit, transferStarts,
      {
        stagingAllocations: 1,
        stagingBytes: 40,
        writeCount: 1,
        writeBytes: 40,
        copyCount: 10,
        copyBytes: 40,
        submitCount: 1,
        copyRanges: [
          'vitrum.pt-webgpu.scene.tlasNodes@0+4',
          'vitrum.pt-webgpu.scene.tlasNodes@12+4',
          'vitrum.pt-webgpu.scene.tlasNodes@32+4',
          'vitrum.pt-webgpu.scene.tlasNodes@44+4',
          'vitrum.pt-webgpu.scene.tlasNodes@64+4',
          'vitrum.pt-webgpu.scene.tlasNodes@76+4',
          'vitrum.pt-webgpu.scene.tlasInstanceWorldToLocal@48+4',
          'vitrum.pt-webgpu.scene.tlasInstanceWorldToLocal@112+4',
          'vitrum.pt-webgpu.scene.tlasInstanceLocalToWorld@48+4',
          'vitrum.pt-webgpu.scene.tlasInstanceLocalToWorld@112+4',
        ],
      },
    );
  });

  it('lite tier accepts instanced topology patches via fallback merged-BLAS repack', async () => {
    installWebGpuConstStubs();
    const { device, createBuffer } = makeLiteStubDevice();
    const structured: EngineWarning[] = [];
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const engine = await createPTEngine_WebGPU({
        device,
        onWarning: (w) => structured.push(w),
      });
      engine.setScene(makeInstancedScene());
      const buffersBefore = createBuffer.mock.calls.length;

      engine.updatePrimitive?.('instanced-a', {
        instances: [
          asMat4(new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            1, 0, 0, 1,
          ])),
        ],
      });

      expect(createBuffer.mock.calls.length).toBeGreaterThan(buffersBefore);
      expect(structured.some((w) =>
        w.code === 'pt-webgpu.lite-update-primitive-fallback-rebuild' &&
        w.details?.id === 'instanced-a' &&
        w.details?.fallbackReason === 'lite-merged-blas-instanced-topology-rebuild',
      )).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  it('replaces ONLY the 6 TLAS/CWBVH-root buffers when instance count shrinks', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeInstancedScene());

    // Drop instance count 2 -> 1 (remove one instance).
    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;
    const destroysBefore = totalDestroyCalls(createBuffer);

    engine.updatePrimitive?.('instanced-a', {
      instances: [
        asMat4(new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          1, 0, 0, 1,
        ])),
      ],
    });

    // Five TLAS buffers plus the CWBVH TLAS-root mirror are one generation.
    expect(createBuffer.mock.calls.length).toBe(buffersBefore + 6);
    expect(totalDestroyCalls(createBuffer) - destroysBefore).toBe(6);
    // Every newly-created buffer is a TLAS buffer; NO BLAS buffer was recreated
    // (proves the per-triangle BLAS was reused verbatim — no buildArrayBvh).
    const created = labelsCreatedSince(createBuffer, buffersBefore);
    expect(created.every((l) => TLAS_LABELS.some((t) => l.includes(t)))).toBe(true);
    expect(created.some((l) => BLAS_LABELS.some((b) => l.includes(b)))).toBe(false);
    // The 5 TLAS buffers are written once each, and the CWBVH TLAS-root mirror
    // refreshes in-place when the byte size remains inside the 16-byte minimum.
    expect(writeBuffer.mock.calls.length).toBe(writesBefore + 6);
  });

  it('grows TLAS instance buffers correctly when instanced count increases', async () => {
    installWebGpuConstStubs();
    const { device, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeInstancedScene());

    const buffersBefore = createBuffer.mock.calls.length;
    const destroysBefore = totalDestroyCalls(createBuffer);

    // Grow instance count 2 -> 3 with distinct world transforms.
    engine.updatePrimitive?.('instanced-a', {
      instances: [
        asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
        asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1])),
        asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 4, 0, 0, 1])),
      ],
    });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore + 6);
    expect(totalDestroyCalls(createBuffer) - destroysBefore).toBe(6);
    const created = labelsCreatedSince(createBuffer, buffersBefore);
    expect(created.some((l) => BLAS_LABELS.some((b) => l.includes(b)))).toBe(false);

    // The local-to-world buffer carries 3 instances of 16 floats, each at its
    // real per-instance world transform.
    const l2w = lastWriteForLabel(createBuffer, device, 'tlasInstanceLocalToWorld');
    expect(l2w).not.toBeNull();
    expect((l2w as Float32Array).length).toBe(3 * 16);
    expect((l2w as Float32Array)[0 * 16 + 12]).toBe(0);
    expect((l2w as Float32Array)[1 * 16 + 12]).toBe(2);
    expect((l2w as Float32Array)[2 * 16 + 12]).toBe(4);
  });

  it('splices an UPSTREAM mesh resize without a full setScene (downstream rebased)', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;
    const destroysBefore = totalDestroyCalls(createBuffer);

    // Resize mesh-a (the FIRST primitive) — mesh-b is downstream, so its concat
    // offsets must rebase. The engine still replaces only the 18 geometry/TLAS buffers.
    engine.updatePrimitive?.('mesh-a', {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2, 1, 3, 2]),
    });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore + 18);
    expect(totalDestroyCalls(createBuffer) - destroysBefore).toBe(18);
    const created = labelsCreatedSince(createBuffer, buffersBefore);
    expect(created.some((l) => l.includes('tangents'))).toBe(true);
    expect(created.some((l) => l.includes('colors'))).toBe(true);
    expect(created.some((l) => l.includes('materials'))).toBe(false);
    expect(created.some((l) => l.includes('analytic'))).toBe(false);
    expect(writeBuffer.mock.calls.length).toBeGreaterThan(writesBefore);
  });

  it('copies only dirty analytic transform words in place', async () => {
    installWebGpuConstStubs();
    const {
      device,
      writeBuffer,
      createBuffer,
      copyBufferToBuffer,
      submit,
    } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeAnalyticScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    const transferStarts = {
      buffers: buffersBefore,
      writes: writesBefore,
      copies: copyBufferToBuffer.mock.calls.length,
      submits: submit.mock.calls.length,
    };

    engine.updatePrimitive?.('analytic-a', {
      transform: asMat4(new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        1, 0, 0, 1,
      ])),
    });

    expectInPlaceTransfer(
      createBuffer, writeBuffer, copyBufferToBuffer, submit, transferStarts,
      {
        stagingAllocations: 1,
        stagingBytes: 8,
        writeCount: 1,
        writeBytes: 8,
        copyCount: 2,
        copyBytes: 8,
        submitCount: 1,
        copyRanges: [
          'vitrum.pt-webgpu.scene.analyticLocalToWorld@48+4',
          'vitrum.pt-webgpu.scene.analyticWorldToLocal@48+4',
        ],
      },
    );
  });

  it('submits before publishing the CPU scene and preserves every live handle', async () => {
    installWebGpuConstStubs();
    const {
      device,
      writeBuffer,
      createBuffer,
      copyBufferToBuffer,
      submit,
    } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const sceneBefore = engine.getScene?.();
    const starts: TransferStarts = {
      buffers: createBuffer.mock.calls.length,
      writes: writeBuffer.mock.calls.length,
      copies: copyBufferToBuffer.mock.calls.length,
      submits: submit.mock.calls.length,
    };
    let sceneObservedDuringSubmit: Scene | null | undefined;
    submit.mockImplementationOnce(() => {
      sceneObservedDuringSubmit = engine.getScene?.();
    });

    engine.updatePrimitive?.('mesh-b', {
      material: { baseColor: [0.2, 0.7, 0.9], roughness: 0.05, metallic: 0.4 },
    });

    expect(sceneObservedDuringSubmit).toBe(sceneBefore);
    expect(engine.getScene?.()).not.toBe(sceneBefore);
    expect(engine.getScene?.()?.primitives[1]?.material.baseColor).toEqual([0.2, 0.7, 0.9]);
    expectInPlaceTransfer(
      createBuffer, writeBuffer, copyBufferToBuffer, submit, starts,
      {
        stagingAllocations: 1,
        stagingBytes: 28,
        writeCount: 1,
        writeBytes: 28,
        copyCount: 4,
        copyBytes: 28,
        submitCount: 1,
        copyRanges: [
          'vitrum.pt-webgpu.scene.materials@464+8',
          'vitrum.pt-webgpu.scene.materials@476+4',
          'vitrum.pt-webgpu.scene.materials@492+4',
          'vitrum.pt-webgpu.scene.materials@880+12',
        ],
      },
    );
  });

  it.each(['encoder', 'copy', 'submit'] as const)(
    'keeps CPU state and live handles unchanged on synchronous %s failure',
    async (failurePoint) => {
      installWebGpuConstStubs();
      const stub = makeStubDevice();
      const engine = await createPTEngine_WebGPU({ device: stub.device });
      engine.setScene(makeScene());

      const sceneBefore = engine.getScene?.();
      const buffersBefore = stub.createBuffer.mock.calls.length;
      const writesBefore = stub.writeBuffer.mock.calls.length;
      const copiesBefore = stub.copyBufferToBuffer.mock.calls.length;
      const submitsBefore = stub.submit.mock.calls.length;
      const liveBuffers = stub.createBuffer.mock.results.slice(0, buffersBefore)
        .filter((result) => result.type === 'return')
        .map((result) => result.value);
      const reset = vi.spyOn(engine, 'reset');
      const failure = new Error(`${failurePoint} failure`);
      if (failurePoint === 'encoder') {
        stub.createCommandEncoder.mockImplementationOnce(() => { throw failure; });
      } else if (failurePoint === 'copy') {
        stub.copyBufferToBuffer.mockImplementationOnce(() => { throw failure; });
      } else {
        stub.submit.mockImplementationOnce(() => { throw failure; });
      }

      expect(() => engine.updatePrimitive?.('mesh-b', {
        material: { baseColor: [0.2, 0.7, 0.9], roughness: 0.05, metallic: 0.4 },
      })).toThrow(failure);

      expect(engine.getScene?.()).toBe(sceneBefore);
      expect(engine.getScene?.()?.primitives[1]?.material.baseColor).toEqual([0.1, 0.4, 0.9]);
      expect(reset).not.toHaveBeenCalled();
      expect(stub.createBuffer.mock.calls.length).toBe(buffersBefore + 1);
      expect(stub.writeBuffer.mock.calls.length).toBe(writesBefore + 1);
      const staging = stub.createBuffer.mock.results[buffersBefore]!.value as StubBuffer;
      expect(staging.label).toBe('vitrum.pt-webgpu.scene.incremental-staging');
      expect(staging.size).toBe(28);
      expect(staging.destroy).toHaveBeenCalledTimes(1);
      for (const live of liveBuffers) expect(live.destroy).not.toHaveBeenCalled();

      expect(stub.copyBufferToBuffer.mock.calls.length - copiesBefore).toBe(
        failurePoint === 'encoder' ? 0 : failurePoint === 'copy' ? 1 : 4,
      );
      expect(stub.submit.mock.calls.length - submitsBefore).toBe(
        failurePoint === 'submit' ? 1 : 0,
      );
    },
  );

  it('does not allocate, encode, write, copy, or submit byte-identical data', async () => {
    installWebGpuConstStubs();
    const {
      device,
      writeBuffer,
      createBuffer,
      createCommandEncoder,
      copyBufferToBuffer,
      submit,
    } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());
    const scene = engine.getScene?.();
    if (scene == null) throw new Error('engine did not retain the test scene');
    const primitive = scene.primitives[1]!;
    if (primitive.kind !== 'mesh') throw new Error('test fixture changed');
    const buffersBefore = createBuffer.mock.calls.length;
    const writesBefore = writeBuffer.mock.calls.length;
    const encodersBefore = createCommandEncoder.mock.calls.length;
    const copiesBefore = copyBufferToBuffer.mock.calls.length;
    const submitsBefore = submit.mock.calls.length;

    engine.updatePrimitive?.('mesh-b', { material: primitive.material });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(writesBefore);
    expect(createCommandEncoder.mock.calls.length).toBe(encodersBefore);
    expect(copyBufferToBuffer.mock.calls.length).toBe(copiesBefore);
    expect(submit.mock.calls.length).toBe(submitsBefore);
  });

  // ── No-op / fall-through / throw characterization ─────────────────────────

  it('treats an empty patch as a validated constant-work no-op', async () => {
    installWebGpuConstStubs();
    const { device, createBuffer, writeBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const buffersBefore = createBuffer.mock.calls.length;
    const writesBefore = writeBuffer.mock.calls.length;
    const destroysBefore = totalDestroyCalls(createBuffer);
    const sceneBefore = engine.getScene?.();
    const reset = vi.spyOn(engine, 'reset');

    engine.updatePrimitive?.('mesh-a', {});

    expect(engine.getScene?.()).toBe(sceneBefore);
    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(writesBefore);
    expect(totalDestroyCalls(createBuffer)).toBe(destroysBefore);
    expect(reset).not.toHaveBeenCalled();

    expect(() => engine.updatePrimitive?.('missing', {})).toThrow(/not found/);
    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(writesBefore);
    expect(totalDestroyCalls(createBuffer)).toBe(destroysBefore);
    expect(reset).not.toHaveBeenCalled();
    expect(() =>
      engine.updatePrimitive?.('mesh-a', { id: null } as never),
    ).toThrow(/id cannot be changed/);
    expect(() =>
      engine.updatePrimitive?.('mesh-a', { kind: null } as never),
    ).toThrow(/kind cannot change/);
    expect(() =>
      engine.updatePrimitive?.('mesh-a', { legacyGeometryHint: true } as never),
    ).toThrow(/is not a known contract field/);
    expect(engine.getScene?.()).toBe(sceneBefore);
    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(writesBefore);

  });

  it('throws (does NOT silently setScene) on an illegal kind-change patch', async () => {
    installWebGpuConstStubs();
    const { device, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const buffersBefore = createBuffer.mock.calls.length;

    expect(() =>
      engine.updatePrimitive?.('mesh-a', { kind: 'analytic' }),
    ).toThrow(/kind cannot change/);
    // The throw happens in the shared preamble BEFORE any fast path or setScene,
    // so no buffer is recreated.
    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
  });

  it('throws when the primitive id is not present in the live scene', async () => {
    installWebGpuConstStubs();
    const { device, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const buffersBefore = createBuffer.mock.calls.length;

    expect(() =>
      engine.updatePrimitive?.('no-such-id', {
        material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
      }),
    ).toThrow(/not found in current scene/);
    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
  });

  it('rejects a non-invertible analytic transform before the GPU fast path', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const engine = await createPTEngine_WebGPU({ device });
      engine.setScene(makeAnalyticScene());
      warnSpy.mockClear();
      const writesBefore = writeBuffer.mock.calls.length;
      const buffersBefore = createBuffer.mock.calls.length;

      expect(() => engine.updatePrimitive?.('analytic-a', {
        transform: asMat4(new Float32Array([
          0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 1,
        ])),
      })).toThrow(/invertible linear transform/);

      expect(writeBuffer.mock.calls.length).toBe(writesBefore);
      expect(createBuffer.mock.calls.length).toBe(buffersBefore);
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });
});
