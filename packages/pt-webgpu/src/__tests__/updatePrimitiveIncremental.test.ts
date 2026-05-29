import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { MATERIAL_FLOAT_STRIDE } from '../scene/materialPacking.js';

function installWebGpuConstStubs(): void {
  const g = globalThis as unknown as { GPUBufferUsage?: Record<string, number> };
  if (g.GPUBufferUsage == null) {
    g.GPUBufferUsage = { STORAGE: 1 << 0, COPY_DST: 1 << 1 };
  }
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
  destroy: ReturnType<typeof vi.fn>;
}

function makeStubDevice() {
  const writeBuffer = vi.fn();
  const createBuffer = vi.fn((desc: { label?: string } | undefined): StubBuffer => ({
    label: desc?.label ?? '',
    destroy: vi.fn(),
  }));
  const device = {
    queue: { writeBuffer },
    createBuffer,
    createCommandEncoder: vi.fn(),
    limits: { maxStorageBuffersPerShaderStage: 64 },
  } as unknown as GPUDevice;
  return { device, writeBuffer, createBuffer };
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
];
const BLAS_LABELS = ['scene.positions', 'scene.normals', 'scene.indices', 'scene.triMaterialIds', 'scene.bvhNodes'];

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

  it('splices positions in-place without recreating scene buffers', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    const shifted = new Float32Array([0.1, 0, 0, 1.1, 0, 0, 0.1, 1, 0]);
    engine.updatePrimitive?.('mesh-a', { positions: shifted });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBeGreaterThan(writesBefore);
  });

  it('updates material slot in-place without rebuilding scene buffers', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    expect(typeof engine.updatePrimitive).toBe('function');
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    engine.updatePrimitive?.('mesh-b', {
      material: { baseColor: [0.2, 0.7, 0.9], roughness: 0.05, metallic: 0.4 },
    });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(writesBefore + 1);

    const lastWrite = writeBuffer.mock.calls[writeBuffer.mock.calls.length - 1];
    const writeByteOffset = lastWrite?.[1];
    expect(writeByteOffset).toBe(1 * MATERIAL_FLOAT_STRIDE * Float32Array.BYTES_PER_ELEMENT);
  });

  it('splices a vertex-count change, reallocating only the 10 geometry buffers', async () => {
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
    // reallocates exactly the 5 BLAS + 5 TLAS geometry buffers — NOT the
    // material / analytic / light buffers (those would prove a full setScene).
    engine.updatePrimitive?.('mesh-b', {
      positions: new Float32Array([0, 0, 2, 1, 0, 2, 0, 1, 2, 1, 1, 2]),
      indices: new Uint32Array([0, 1, 2, 1, 3, 2]),
    });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore + 10);
    expect(totalDestroyCalls(createBuffer) - destroysBefore).toBe(10);
    const created = labelsCreatedSince(createBuffer, buffersBefore);
    expect(
      created.every(
        (l) => BLAS_LABELS.some((b) => l.includes(b)) || TLAS_LABELS.some((t) => l.includes(t)),
      ),
    ).toBe(true);
    // No non-geometry buffer was recreated (materials/analytic/lights stay put).
    expect(created.some((l) => l.includes('materials'))).toBe(false);
    expect(created.some((l) => l.includes('analytic'))).toBe(false);
    expect(created.some((l) => l.includes('Lights'))).toBe(false);
    expect(writeBuffer.mock.calls.length).toBeGreaterThan(writesBefore);
  });

  it('updates TLAS buffers in-place for transform-only mesh patches', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    engine.updatePrimitive?.('mesh-b', {
      transform: asMat4(new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        2, 0, 0, 1,
      ])),
    });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(writesBefore + 5);
  });

  it('updates TLAS buffers in-place for instanced transform patches', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeInstancedScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

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

    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(writesBefore + 5);
  });

  it('reallocates ONLY the 5 TLAS buffers when instanced count shrinks (BLAS untouched)', async () => {
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

    // Exactly 5 fresh buffers created; 5 stale buffers destroyed.
    expect(createBuffer.mock.calls.length).toBe(buffersBefore + 5);
    expect(totalDestroyCalls(createBuffer) - destroysBefore).toBe(5);
    // Every newly-created buffer is a TLAS buffer; NO BLAS buffer was recreated
    // (proves the per-triangle BLAS was reused verbatim — no buildArrayBvh).
    const created = labelsCreatedSince(createBuffer, buffersBefore);
    expect(created.every((l) => TLAS_LABELS.some((t) => l.includes(t)))).toBe(true);
    expect(created.some((l) => BLAS_LABELS.some((b) => l.includes(b)))).toBe(false);
    // The 5 new TLAS buffers are written once each on (re)creation.
    expect(writeBuffer.mock.calls.length).toBe(writesBefore + 5);
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

    expect(createBuffer.mock.calls.length).toBe(buffersBefore + 5);
    expect(totalDestroyCalls(createBuffer) - destroysBefore).toBe(5);
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
    // offsets must rebase. The engine still takes the splice path: exactly the 10
    // geometry buffers reallocate (not the full ~21-buffer setScene set).
    engine.updatePrimitive?.('mesh-a', {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]),
      indices: new Uint32Array([0, 1, 2, 1, 3, 2]),
    });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore + 10);
    expect(totalDestroyCalls(createBuffer) - destroysBefore).toBe(10);
    const created = labelsCreatedSince(createBuffer, buffersBefore);
    expect(created.some((l) => l.includes('materials'))).toBe(false);
    expect(created.some((l) => l.includes('analytic'))).toBe(false);
    expect(writeBuffer.mock.calls.length).toBeGreaterThan(writesBefore);
  });

  it('updates analytic transform buffers in-place', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeAnalyticScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    engine.updatePrimitive?.('analytic-a', {
      transform: asMat4(new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        1, 0, 0, 1,
      ])),
    });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(writesBefore + 2);
  });
});
