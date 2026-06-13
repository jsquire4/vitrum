/**
 * emissiveFold.test.ts — H10 emissive-fold desync tests.
 *
 * Verifies that the fold (emitter color * intensity re-attached to the
 * primitive's material) is:
 *   1. Applied by `buildPackedScene` with `cameraVisibleEmitters: true`.
 *   2. Preserved after `updateEmitter` patches the emitter color.
 *   3. Preserved after `updatePrimitive` patches a non-emissive material field
 *      (roughness) via the material fast path.
 *   4. Applied to InverseSession stubs (future) — covered by the packFoldedMaterialEntry
 *      unit test checking both fold and no-fold paths.
 *
 * The emissive channel lives at float offset 4..6 of the packed material
 * (vec4 #1: emissive.rgb + metallic — see materialPacking.ts line 14).
 */

import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { buildPackedScene, packFoldedMaterialEntry } from '../scene/uploadSceneBuffers.js';
import { MATERIAL_FLOAT_STRIDE } from '../scene/materialPacking.js';
import { SceneMutationRouter } from '../sceneMutationRouter.js';
import type { MutationHost } from '../sceneMutationRouter.js';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';

// ─── helpers ──────────────────────────────────────────────────────────────────

/** Smallest possible scene: one triangle mesh + one mesh-area emitter. */
function meshEmitterScene(
  emitterColor: [number, number, number],
  emitterIntensity: number,
): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'panel',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.8, metallic: 0,
                    emissive: [0, 0, 0], emissiveIntensity: 0 },
      },
    ],
    emitters: [
      {
        kind: 'mesh-area' as const,
        id: 'light',
        meshId: 'panel',
        color: emitterColor,
        intensity: emitterIntensity,
      },
    ],
    environment: { kind: 'none' },
  };
}

function implicitEmissiveScene(
  emissive: [number, number, number],
  emissiveIntensity: number,
): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'panel',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [0.5, 0.5, 0.5],
          roughness: 0.8,
          metallic: 0,
          emissive,
          emissiveIntensity,
        },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

/** Float offset of emissive.rgb in a packed material. */
const EMISSIVE_OFFSET = 4; // vec4 #1: emissive.rgb + metallic
const MESH_AREA_RADIANCE_OFFSET = 12; // first packed mesh-area triangle, vec4 #3

function emissiveFromMaterials(materials: Float32Array): [number, number, number] {
  return [materials[EMISSIVE_OFFSET]!, materials[EMISSIVE_OFFSET + 1]!, materials[EMISSIVE_OFFSET + 2]!];
}

// ─── packFoldedMaterialEntry unit tests ───────────────────────────────────────

describe('packFoldedMaterialEntry (H10 helper)', () => {
  const prim = {
    id: 'panel',
    material: { baseColor: [0.5, 0.5, 0.5] as [number,number,number], roughness: 0.8, metallic: 0,
                emissive: [0, 0, 0] as [number,number,number], emissiveIntensity: 0 },
  };

  const scene = meshEmitterScene([2, 0.5, 0.1], 3);

  it('applies the fold when cameraVisibleEmitters=true', () => {
    const packed = packFoldedMaterialEntry(prim, scene, true);
    expect(packed.length).toBe(MATERIAL_FLOAT_STRIDE);
    // Emissive = color * intensity = [2*3, 0.5*3, 0.1*3] = [6, 1.5, 0.3]
    expect(packed[EMISSIVE_OFFSET]).toBeCloseTo(6, 5);
    expect(packed[EMISSIVE_OFFSET + 1]).toBeCloseTo(1.5, 5);
    expect(packed[EMISSIVE_OFFSET + 2]).toBeCloseTo(0.3, 5);
  });

  it('does NOT apply the fold when cameraVisibleEmitters=false', () => {
    const packed = packFoldedMaterialEntry(prim, scene, false);
    expect(packed[EMISSIVE_OFFSET]).toBeCloseTo(0, 5);
    expect(packed[EMISSIVE_OFFSET + 1]).toBeCloseTo(0, 5);
    expect(packed[EMISSIVE_OFFSET + 2]).toBeCloseTo(0, 5);
  });

  it('does NOT apply the fold when the primitive has no mesh-area emitter backing', () => {
    const otherPrim = { id: 'other', material: prim.material };
    const packed = packFoldedMaterialEntry(otherPrim, scene, true);
    expect(packed[EMISSIVE_OFFSET]).toBeCloseTo(0, 5);
  });
});

// ─── buildPackedScene fold tests ─────────────────────────────────────────────

describe('buildPackedScene emissive fold (H10)', () => {
  it('fold present after setScene with cameraVisibleEmitters:true', () => {
    const scene = meshEmitterScene([1, 0.5, 0.25], 4);
    const packed = buildPackedScene(scene, { cameraVisibleEmitters: true });
    // mat slot 0 (first primitive = 'panel')
    const em = emissiveFromMaterials(packed.materials);
    expect(em[0]).toBeCloseTo(4, 5);     // 1 * 4
    expect(em[1]).toBeCloseTo(2, 5);     // 0.5 * 4
    expect(em[2]).toBeCloseTo(1, 5);     // 0.25 * 4
  });

  it('fold absent when cameraVisibleEmitters is off (default)', () => {
    const scene = meshEmitterScene([1, 0.5, 0.25], 4);
    const packed = buildPackedScene(scene, { cameraVisibleEmitters: false });
    const em = emissiveFromMaterials(packed.materials);
    expect(em[0]).toBeCloseTo(0, 5);
    expect(em[1]).toBeCloseTo(0, 5);
    expect(em[2]).toBeCloseTo(0, 5);
  });
});

// ─── SceneMutationRouter fold-preservation tests ─────────────────────────────

/** Build a minimal MutationHost with an in-memory scene + materials buffer. */
function makeHostWithScene(scene: Scene, cameraVisible: boolean): {
  host: MutationHost;
  sceneRef: { current: Scene };
  materialsBuffer: Float32Array;
  meshAreaLightsData: Float32Array;
  writeCalls: Array<{ byteOffset: number; data: Float32Array }>;
} {
  const packed = buildPackedScene(scene, { cameraVisibleEmitters: cameraVisible });
  const materialsBuffer = new Float32Array(packed.materials);
  const meshAreaLightsData = new Float32Array(packed.meshAreaLightsData);
  const writeCalls: Array<{ byteOffset: number; data: Float32Array }> = [];

  // Stub UploadedSceneBuffers — only the materials fields matter for H10 tests.
  const sceneBuffers = {
    ...packed,
    materials: materialsBuffer,
    meshAreaLightsData,
    materialsBuffer: {
      // Capture writes so we can assert the correct bytes were uploaded.
      destroy: vi.fn(),
    } as unknown as GPUBuffer,
    bvhNodeCount: 0,
    tlasNodeCount: 0,
    materialCount: 1,
    // GPU buffer stubs — not used by the material fast path.
    positionsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    normalsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    indicesBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    triMaterialIdsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    bvhNodesBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    analyticHeadersBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    analyticParamsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    analyticLocalToWorldBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    analyticWorldToLocalBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    environmentMapTexelsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    environmentMapCdfBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    pointLightsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    spotLightsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    rectAreaLightsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    meshAreaLightsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    lightTreeBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    tlasNodesBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    tlasInstanceIndicesBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    tlasBlasRootsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    tlasInstanceWorldToLocalBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    tlasInstanceLocalToWorldBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    uvsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    materialTexDescriptorsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    materialTexture: {} as GPUTexture,
    materialTextureView: {} as GPUTextureView,
    materialTextureSampler: {} as GPUSampler,
    materialLinearTexture: {} as GPUTexture,
    materialLinearTextureView: {} as GPUTextureView,
    gpuMemoryBytes: () => ({ bufferBytes: 0, textureBytesByFormat: {} }),
    destroy: vi.fn(),
  } as unknown as UploadedSceneBuffers;

  const sceneRef = { current: scene };

  const host: MutationHost = {
    device: {
      queue: {
        writeBuffer: vi.fn(
          (_buf: unknown, byteOffset: number, buffer: ArrayBuffer, offset: number, length: number) => {
            writeCalls.push({
              byteOffset,
              data: new Float32Array(buffer, offset, length / 4),
            });
          },
        ),
      },
    } as unknown as GPUDevice,
    assertLive: vi.fn(),
    getScene: () => sceneRef.current,
    setSceneState: vi.fn((s: Scene) => { sceneRef.current = s; }),
    getSceneBuffers: () => sceneBuffers,
    getGeoPack: () => null,
    setGeoPack: vi.fn(),
    invalidateBindGroups: vi.fn(),
    supportedAnalyticShapes: () => new Set<string>(),
    cameraVisibleEmitters: () => cameraVisible,
    repackScene: vi.fn((s: Scene) => { sceneRef.current = s; }),
    setScene: vi.fn((s: Scene) => { sceneRef.current = s; }),
    reset: vi.fn(),
  };

  return { host, sceneRef, materialsBuffer, meshAreaLightsData, writeCalls };
}

describe('SceneMutationRouter fold-preservation (H10)', () => {
  it('fold preserved after updateEmitter color patch (cameraVisibleEmitters=true)', () => {
    const scene = meshEmitterScene([1, 0, 0], 2);
    const { host, materialsBuffer } = makeHostWithScene(scene, true);
    // Initial fold = [2, 0, 0]
    expect(materialsBuffer[EMISSIVE_OFFSET]).toBeCloseTo(2, 5);

    const router = new SceneMutationRouter(host);
    // Patch the emitter to a new color
    router.updateEmitter('light', { color: [0, 1, 0] });

    // New fold = [0, 2, 0] (new color [0,1,0] * intensity 2)
    expect(materialsBuffer[EMISSIVE_OFFSET]).toBeCloseTo(0, 5);
    expect(materialsBuffer[EMISSIVE_OFFSET + 1]).toBeCloseTo(2, 5);
    expect(materialsBuffer[EMISSIVE_OFFSET + 2]).toBeCloseTo(0, 5);
  });

  it('fold preserved after updatePrimitive roughness patch (material fast path)', () => {
    const scene = meshEmitterScene([1, 0.5, 0], 4);
    const { host, materialsBuffer } = makeHostWithScene(scene, true);
    // Initial fold = [4, 2, 0]
    expect(materialsBuffer[EMISSIVE_OFFSET]).toBeCloseTo(4, 5);

    const router = new SceneMutationRouter(host);
    // Patch roughness (material-only fast path) — must NOT strip the fold
    router.updatePrimitive('panel', { material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.2, metallic: 0 } });

    // Fold must still be [4, 2, 0] after the roughness patch
    expect(materialsBuffer[EMISSIVE_OFFSET]).toBeCloseTo(4, 5);
    expect(materialsBuffer[EMISSIVE_OFFSET + 1]).toBeCloseTo(2, 5);
    expect(materialsBuffer[EMISSIVE_OFFSET + 2]).toBeCloseTo(0, 5);
  });

  it('emissive material patch re-packs the implicit mesh-area emitter', () => {
    const scene = implicitEmissiveScene([1, 0, 0], 2);
    const { host, materialsBuffer, meshAreaLightsData } = makeHostWithScene(scene, true);
    // Initial implicit NEE emitter radiance = emissive * intensity = [2, 0, 0].
    expect(meshAreaLightsData[MESH_AREA_RADIANCE_OFFSET]).toBeCloseTo(2, 5);
    expect(meshAreaLightsData[MESH_AREA_RADIANCE_OFFSET + 1]).toBeCloseTo(0, 5);

    const router = new SceneMutationRouter(host);
    router.updatePrimitive('panel', {
      material: {
        baseColor: [0.5, 0.5, 0.5],
        roughness: 0.8,
        metallic: 0,
        emissive: [0, 1, 0],
        emissiveIntensity: 3,
      },
    });

    // Material camera-hit emission and the synthesized mesh-area NEE emitter
    // both move to the new green radiance (material packing stores
    // emissive * emissiveIntensity).
    expect(materialsBuffer[EMISSIVE_OFFSET]).toBeCloseTo(0, 5);
    expect(materialsBuffer[EMISSIVE_OFFSET + 1]).toBeCloseTo(3, 5);
    expect(materialsBuffer[EMISSIVE_OFFSET + 2]).toBeCloseTo(0, 5);
    expect(meshAreaLightsData[MESH_AREA_RADIANCE_OFFSET]).toBeCloseTo(0, 5);
    expect(meshAreaLightsData[MESH_AREA_RADIANCE_OFFSET + 1]).toBeCloseTo(3, 5);
    expect(meshAreaLightsData[MESH_AREA_RADIANCE_OFFSET + 2]).toBeCloseTo(0, 5);
  });

  it('fold NOT applied when cameraVisibleEmitters=false (control case)', () => {
    const scene = meshEmitterScene([1, 0.5, 0], 4);
    const { host, materialsBuffer } = makeHostWithScene(scene, false);
    // No fold — emissive stays zero
    expect(materialsBuffer[EMISSIVE_OFFSET]).toBeCloseTo(0, 5);

    const router = new SceneMutationRouter(host);
    router.updateEmitter('light', { color: [0, 1, 0] });

    // Still no fold
    expect(materialsBuffer[EMISSIVE_OFFSET]).toBeCloseTo(0, 5);
  });
});
