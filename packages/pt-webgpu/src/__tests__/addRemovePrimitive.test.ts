import { describe, expect, it, vi } from 'vitest';
import type { Scene, ScenePrimitive } from '@vitrum/core';
import { buildTlas, tlasIntersect, type TlasInstance } from '@vitrum/shared-bvh';
import { createPTEngine_WebGPU } from '../index.js';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';
import { MATERIAL_FLOAT_STRIDE } from '../scene/materialPacking.js';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

function installWebGpuConstStubs(): void {
  installGpuConstStubs();
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
    queue: { writeBuffer, writeTexture: vi.fn() },
    createBuffer,
    ...textureStubMethods(),
    createCommandEncoder: vi.fn(),
    limits: { maxStorageBuffersPerShaderStage: 64, maxTextureDimension2D: 8192 },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
  return { device, writeBuffer, createBuffer };
}

/**
 * A mesh primitive whose single triangle sits in the z=0 plane around the
 * lateral centre `(cx, cy)`. Each primitive is given a DISTINCT lateral cell so
 * a straight -Z ray aimed at one primitive's centre crosses only that
 * primitive's TLAS leaf AABB — making the conservative `tlasIntersect`
 * candidate set a clean per-primitive hit/miss oracle (vs. stacking primitives
 * along the ray, where a -Z line would cross every parallel triangle's AABB).
 */
function triAt(id: string, cx: number, cy: number, baseColor: [number, number, number]): ScenePrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([cx, cy, 0, cx + 0.5, cy, 0, cx, cy + 0.5, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor, roughness: 0.4, metallic: 0.1 },
  };
}

function makeScene(): Scene {
  return {
    primitives: [
      triAt('mesh-a', 0, 0, [0.8, 0.2, 0.1]),
      triAt('mesh-b', 10, 0, [0.1, 0.4, 0.9]),
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

type TlasData = ReturnType<typeof buildTlas>;

/**
 * Reconstruct a `TlasData` from a packed scene's TLAS arrays so the CPU
 * reference traverser `tlasIntersect` can prove which primitive a ray hits.
 * The world-AABB of each instance is recovered by transforming each binding's
 * local AABB through its local-to-world matrix (the same transform the packer
 * used to build the TLAS).
 */
function tlasDataFromPacked(scene: Scene): TlasData {
  const packed = buildPackedScene(scene);
  const instances: TlasInstance[] = [];
  let flatInstance = 0;
  for (const binding of packed.primitiveTlasBindings) {
    for (let k = 0; k < binding.instanceCount; k += 1) {
      const l2w = packed.tlasInstanceLocalToWorld.subarray(flatInstance * 16, flatInstance * 16 + 16);
      const w2l = packed.tlasInstanceWorldToLocal.subarray(flatInstance * 16, flatInstance * 16 + 16);
      const world = transformAabb(binding.localAabbMin, binding.localAabbMax, l2w);
      instances.push({
        blasId: binding.blasRoot,
        aabbMin: world.min,
        aabbMax: world.max,
        worldToLocal: new Float32Array(w2l),
      });
      flatInstance += 1;
    }
  }
  return buildTlas(instances);
}

function transformAabb(
  min: readonly [number, number, number],
  max: readonly [number, number, number],
  l2w: Float32Array,
): { min: [number, number, number]; max: [number, number, number] } {
  let mnX = Infinity, mnY = Infinity, mnZ = Infinity;
  let mxX = -Infinity, mxY = -Infinity, mxZ = -Infinity;
  for (let c = 0; c < 8; c += 1) {
    const x = (c & 1) === 0 ? min[0] : max[0];
    const y = (c & 2) === 0 ? min[1] : max[1];
    const z = (c & 4) === 0 ? min[2] : max[2];
    const wx = (l2w[0] ?? 0) * x + (l2w[4] ?? 0) * y + (l2w[8] ?? 0) * z + (l2w[12] ?? 0);
    const wy = (l2w[1] ?? 0) * x + (l2w[5] ?? 0) * y + (l2w[9] ?? 0) * z + (l2w[13] ?? 0);
    const wz = (l2w[2] ?? 0) * x + (l2w[6] ?? 0) * y + (l2w[10] ?? 0) * z + (l2w[14] ?? 0);
    mnX = Math.min(mnX, wx); mnY = Math.min(mnY, wy); mnZ = Math.min(mnZ, wz);
    mxX = Math.max(mxX, wx); mxY = Math.max(mxY, wy); mxZ = Math.max(mxZ, wz);
  }
  return { min: [mnX, mnY, mnZ], max: [mxX, mxY, mxZ] };
}

/** Whether a straight -Z ray aimed at the triangle centred near (cx, cy) hits a
 *  TLAS leaf. Each test primitive lives in its own lateral cell, so this is a
 *  clean per-primitive hit/miss probe. */
function rayHitsTriAt(tlas: TlasData, cx: number, cy: number): boolean {
  const hits = tlasIntersect(tlas, [cx + 0.1, cy + 0.1, 5], [0, 0, -1]);
  return hits.length > 0;
}

const TLAS_LABELS = [
  'tlasNodes',
  'tlasInstanceIndices',
  'tlasBlasRoots',
  'tlasInstanceWorldToLocal',
  'tlasInstanceLocalToWorld',
];
const BLAS_LABELS = ['scene.positions', 'scene.normals', 'scene.indices', 'scene.triMaterialIds', 'scene.bvhNodes'];

describe('pt-webgpu add/remove primitive — capability', () => {
  it('advertises supportsAddRemovePrimitive=true and exposes both methods', async () => {
    const { device } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    expect(engine.capabilities.supportsAddRemovePrimitive).toBe(true);
    expect(typeof engine.addPrimitive).toBe('function');
    expect(typeof engine.removePrimitive).toBe('function');
  });
});

describe('pt-webgpu add/remove primitive — TLAS / packing correctness', () => {
  it('addPrimitive makes the new primitive hittable while leaving the others hittable', () => {
    const base = makeScene();
    const before = tlasDataFromPacked(base);
    // Baseline: mesh-a (cell 0,0) and mesh-b (cell 10,0) are hit; cell 20,0 empty.
    expect(rayHitsTriAt(before, 0, 0)).toBe(true);
    expect(rayHitsTriAt(before, 10, 0)).toBe(true);
    expect(rayHitsTriAt(before, 20, 0)).toBe(false);

    const added: Scene = {
      ...base,
      primitives: [...base.primitives, triAt('mesh-c', 20, 0, [0.2, 0.9, 0.3])],
    };
    const after = tlasDataFromPacked(added);
    // New primitive is now hittable AND the original two remain hittable.
    expect(rayHitsTriAt(after, 20, 0)).toBe(true);
    expect(rayHitsTriAt(after, 0, 0)).toBe(true);
    expect(rayHitsTriAt(after, 10, 0)).toBe(true);

    // Downstream offsets are correct: the appended primitive owns a fresh dense
    // material slot at the end, and the concat triangle count grew by one tri.
    const packed = buildPackedScene(added);
    expect(packed.triangleCount).toBe(3);
    expect(packed.materials.length).toBe(3 * MATERIAL_FLOAT_STRIDE);
    expect(packed.primitiveTlasBindings.map((b) => b.primitiveId)).toEqual([
      'mesh-a',
      'mesh-b',
      'mesh-c',
    ]);
    // mesh-c's BLAS slice is appended after mesh-a + mesh-b (dense, no gaps).
    const cBinding = packed.primitiveTlasBindings[2]!;
    expect(cBinding.vertexStart).toBe(6); // 3 verts each for a + b
    expect(cBinding.triStart).toBe(2); // 1 tri each for a + b
  });

  it('removePrimitive evicts the primitive (no longer hit) and rebases the survivors', () => {
    // Three primitives: remove the MIDDLE one so the survivors must rebase.
    const three: Scene = {
      primitives: [
        triAt('mesh-a', 0, 0, [0.8, 0.2, 0.1]),
        triAt('mesh-b', 10, 0, [0.1, 0.4, 0.9]),
        triAt('mesh-c', 20, 0, [0.2, 0.9, 0.3]),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const before = tlasDataFromPacked(three);
    expect(rayHitsTriAt(before, 10, 0)).toBe(true);

    const removed: Scene = {
      ...three,
      primitives: three.primitives.filter((p) => p.id !== 'mesh-b'),
    };
    const after = tlasDataFromPacked(removed);
    // mesh-b (cell 10,0) is gone; mesh-a (0,0) and mesh-c (20,0) survive.
    expect(rayHitsTriAt(after, 10, 0)).toBe(false);
    expect(rayHitsTriAt(after, 0, 0)).toBe(true);
    expect(rayHitsTriAt(after, 20, 0)).toBe(true);

    // Survivors are densely re-packed: mesh-c rebases to slot 1 (was slot 2).
    const packed = buildPackedScene(removed);
    expect(packed.triangleCount).toBe(2);
    expect(packed.materials.length).toBe(2 * MATERIAL_FLOAT_STRIDE);
    expect(packed.primitiveTlasBindings.map((b) => b.primitiveId)).toEqual(['mesh-a', 'mesh-c']);
    // mesh-c rebased to vertexStart 3 / triStart 1 (immediately after mesh-a).
    const cBinding = packed.primitiveTlasBindings[1]!;
    expect(cBinding.vertexStart).toBe(3);
    expect(cBinding.triStart).toBe(1);
    // mesh-c's per-triangle material id rebased to dense slot 1.
    expect(packed.triMaterialIds[cBinding.triStart]).toBe(1);
  });

  it('removing the last primitive yields a renderable empty scene', () => {
    const single: Scene = {
      primitives: [triAt('only', 0, 0, [1, 1, 1])],
      emitters: [],
      environment: { kind: 'none' },
    };
    const empty: Scene = { ...single, primitives: [] };
    const packed = buildPackedScene(empty);
    expect(packed.triangleCount).toBe(0);
    expect(packed.tlasNodes.length).toBe(0);
    expect(packed.primitiveTlasBindings.length).toBe(0);
  });
});

describe('pt-webgpu add/remove primitive — engine wiring + buffer lifecycle', () => {
  it('addPrimitive uploads the new geometry + material via a fresh scene-buffer set', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    engine.addPrimitive?.(triAt('mesh-c', 20, 0, [0.2, 0.9, 0.3]));

    // A full repack re-uploads the whole scene-buffer set (BLAS + TLAS +
    // materials + analytic + lights + environment), proving the new primitive's
    // geometry + material were uploaded.
    const newBuffers = createBuffer.mock.calls.length - buffersBefore;
    expect(newBuffers).toBeGreaterThan(BLAS_LABELS.length + TLAS_LABELS.length);
    expect(writeBuffer.mock.calls.length).toBeGreaterThan(writesBefore);
  });

  it('removePrimitive re-uploads a fresh scene-buffer set with the survivor', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const buffersBefore = createBuffer.mock.calls.length;
    engine.removePrimitive?.('mesh-a');
    expect(createBuffer.mock.calls.length).toBeGreaterThan(buffersBefore);
    expect(writeBuffer).toHaveBeenCalled();
  });

  it('addPrimitive then removePrimitive round-trips and leaves the engine ready', async () => {
    installWebGpuConstStubs();
    const { device } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    // A full repack resets accumulation (reset() is called at the end of
    // #repackScene, the same as setScene). Here we assert the round trip
    // completes without throwing and leaves a re-renderable state.
    expect(() => engine.addPrimitive?.(triAt('mesh-c', 20, 0, [0.2, 0.9, 0.3]))).not.toThrow();
    expect(() => engine.removePrimitive?.('mesh-c')).not.toThrow();
    expect(engine.state).toBe('ready');
  });
});

describe('pt-webgpu add/remove primitive — error semantics', () => {
  it('addPrimitive throws on a duplicate id and leaves the scene unchanged', async () => {
    installWebGpuConstStubs();
    const { device, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const buffersBefore = createBuffer.mock.calls.length;
    expect(() => engine.addPrimitive?.(triAt('mesh-a', 0, 0, [0, 0, 0]))).toThrow(/already exists/);
    // No repack happened — scene untouched.
    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
  });

  it('removePrimitive throws on a missing id and leaves the scene unchanged', async () => {
    installWebGpuConstStubs();
    const { device, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const buffersBefore = createBuffer.mock.calls.length;
    expect(() => engine.removePrimitive?.('does-not-exist')).toThrow(/no primitive with id/);
    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
  });

  it('addPrimitive before setScene throws (engine not yet initialized)', async () => {
    const { device } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    expect(() => engine.addPrimitive?.(triAt('x', 0, 0, [0, 0, 0]))).toThrow(/setScene/);
  });

  it('removePrimitive after dispose throws', async () => {
    installWebGpuConstStubs();
    const { device } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());
    engine.dispose();
    expect(() => engine.removePrimitive?.('mesh-a')).toThrow(/disposed/);
  });
});
